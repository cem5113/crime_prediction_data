# scripts/update_311.py
# -*- coding: utf-8 -*-

import os
import re
import json
import time
import math
import requests
import numpy as np
import pandas as pd
import geopandas as gpd

from pathlib import Path
from datetime import datetime, timedelta, timezone

# ============================================================
# AYARLAR
# ============================================================

BASE_DIR = Path(os.getenv("CRIME_DATA_DIR", "crime_prediction_data")).resolve()
BASE_DIR.mkdir(parents=True, exist_ok=True)

RAW_OUT = BASE_DIR / "sf_311_last_5_years_raw.csv"
AGG_OUT = BASE_DIR / "sf_311_last_5_years.csv"

DATASET_URL = os.getenv("SF_311_DATASET_URL", "https://data.sfgov.org/resource/vw6y-z8j6.json")
APP_TOKEN = os.getenv("SOCRATA_APP_TOKEN", "").strip()

TRACT_ZIP_URL = os.getenv(
    "TRACT_ZIP_URL",
    "https://www2.census.gov/geo/tiger/TIGER2023/TRACT/tl_2023_06_tract.zip"
)

PAGE_LIMIT = int(os.getenv("SF311_PAGE_LIMIT", "50000"))
CHUNK_DAYS = int(os.getenv("SF311_CHUNK_DAYS", "31"))
TIMEOUT = int(os.getenv("SF311_TIMEOUT", "90"))
RETRIES = int(os.getenv("SF311_RETRIES", "5"))
SLEEP_SEC = float(os.getenv("SF311_SLEEP_SEC", "0.25"))
OVERLAP_DAYS = int(os.getenv("SF311_OVERLAP_DAYS", "14"))
KEEP_RAW = os.getenv("SF311_KEEP_RAW", "1") == "1"

SF_TZ = "America/Los_Angeles"
EPS = 1e-6

SELECT_COLS = [
    "service_request_id",
    "requested_datetime",
    "closed_date",
    "updated_datetime",
    "status_description",
    "agency_responsible",
    "service_name",
    "service_subtype",
    "service_details",
    "lat",
    "long",
    "point",
    "source",
]

SLOT_ORDER = [
    "00-03",
    "03-06",
    "06-09",
    "09-12",
    "12-15",
    "15-18",
    "18-21",
    "21-24",
]

SLOT_START_MAP = {
    "00-03": 0,
    "03-06": 3,
    "06-09": 6,
    "09-12": 9,
    "12-15": 12,
    "15-18": 15,
    "18-21": 18,
    "21-24": 21,
}

# ============================================================
# YARDIMCI FONKSİYONLAR
# ============================================================

def log(msg: str) -> None:
    print(msg, flush=True)


def safe_to_datetime(series: pd.Series, utc: bool = False) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=utc)


def safe_to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def zfill_geoid(series: pd.Series) -> pd.Series:
    return series.astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(11)


def make_hour_range(hour_series: pd.Series) -> pd.Series:
    start_h = ((hour_series // 3) * 3).astype("Int64")
    end_h = start_h + 3
    return (
        start_h.astype(str).str.zfill(2)
        + "-"
        + end_h.astype(str).str.zfill(2)
    )


def extract_latlon_from_point(val):
    lat_val, lon_val = None, None
    try:
        if isinstance(val, dict):
            lat_val = val.get("latitude")
            lon_val = val.get("longitude")
        elif isinstance(val, str):
            s = val.strip()
            if s.startswith("{") and s.endswith("}"):
                obj = json.loads(s)
                lat_val = obj.get("latitude")
                lon_val = obj.get("longitude")
            else:
                m = re.search(r"POINT\s*\(\s*([-0-9\.]+)\s+([-0-9\.]+)\s*\)", s, re.I)
                if m:
                    lon_val = m.group(1)
                    lat_val = m.group(2)
    except Exception:
        pass
    return pd.Series([lat_val, lon_val])


def socrata_get(session: requests.Session, params: dict):
    headers = {"Accept": "application/json"}
    if APP_TOKEN:
        headers["X-App-Token"] = APP_TOKEN

    last_err = None
    for i in range(RETRIES):
        try:
            r = session.get(DATASET_URL, params=params, headers=headers, timeout=TIMEOUT)
            if r.status_code in (408, 429) or 500 <= r.status_code < 600:
                raise requests.HTTPError(f"status={r.status_code}")
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            wait_s = max(SLEEP_SEC, SLEEP_SEC * (2 ** i))
            log(f"⚠️ Socrata retry {i+1}/{RETRIES}: {e} | {wait_s:.1f}s bekleniyor")
            time.sleep(wait_s)

    raise last_err


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, low_memory=False)
        log(f"📁 Mevcut dosya okundu: {path} | shape={df.shape}")
        return df
    except Exception as e:
        log(f"⚠️ Dosya okunamadı: {path} | {e}")
        return pd.DataFrame()


# ============================================================
# STANDARDIZATION
# ============================================================

def standardize_raw_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    rename_map = {
        "service_request_id": "id",
        "requested_datetime": "datetime",
        "service_name": "category",
        "service_subtype": "subcategory",
    }
    df = df.rename(columns=rename_map)

    needed = [
        "id",
        "datetime",
        "category",
        "subcategory",
        "agency_responsible",
        "lat",
        "long",
    ]
    for c in needed:
        if c not in df.columns:
            df[c] = pd.NA

    if "point" in df.columns:
        miss_mask = df["lat"].isna() | df["long"].isna()
        if miss_mask.any():
            parsed = df.loc[miss_mask, "point"].apply(extract_latlon_from_point)
            parsed.columns = ["_lat_from_point", "_lon_from_point"]
            df.loc[miss_mask, "lat"] = df.loc[miss_mask, "lat"].fillna(parsed["_lat_from_point"])
            df.loc[miss_mask, "long"] = df.loc[miss_mask, "long"].fillna(parsed["_lon_from_point"])

    df["lat"] = safe_to_numeric(df["lat"])
    df["long"] = safe_to_numeric(df["long"])
    df["latitude"] = df["lat"]
    df["longitude"] = df["long"]

    dt_utc = safe_to_datetime(df["datetime"], utc=True)
    dt_sf = dt_utc.dt.tz_convert(SF_TZ)

    df["datetime"] = dt_sf
    df["date"] = dt_sf.dt.strftime("%Y-%m-%d")
    df["time"] = dt_sf.dt.strftime("%H:%M:%S")

    for c in ["id", "category", "subcategory", "agency_responsible"]:
        if c in df.columns:
            df[c] = df[c].astype("string")

    return df


# ============================================================
# GEOID
# ============================================================

def load_sf_tracts() -> gpd.GeoDataFrame:
    log("🗺️ Census tract shapefile okunuyor...")
    tracts = gpd.read_file(TRACT_ZIP_URL)

    if "STATEFP" in tracts.columns:
        tracts = tracts[tracts["STATEFP"].astype(str) == "06"].copy()
    if "COUNTYFP" in tracts.columns:
        tracts = tracts[tracts["COUNTYFP"].astype(str) == "075"].copy()

    if "GEOID" not in tracts.columns:
        tracts["GEOID"] = (
            tracts["STATEFP"].astype(str).str.zfill(2)
            + tracts["COUNTYFP"].astype(str).str.zfill(3)
            + tracts["TRACTCE"].astype(str).str.zfill(6)
        )

    tracts = tracts[["GEOID", "geometry"]].copy()
    tracts["GEOID"] = zfill_geoid(tracts["GEOID"])

    if tracts.crs is None:
        tracts = tracts.set_crs(epsg=4269)
    tracts = tracts.to_crs(epsg=4326)

    log(f"✅ SF tract sayısı: {len(tracts):,}")
    return tracts


def attach_geoid(df: pd.DataFrame, tracts: gpd.GeoDataFrame) -> pd.DataFrame:
    df = df.copy()

    ok = df["latitude"].notna() & df["longitude"].notna()

    if "GEOID" not in df.columns:
        df["GEOID"] = pd.NA

    if ok.sum() == 0:
        log("⚠️ Koordinat yok, GEOID atanamadı.")
        return df

    cols_keep = [c for c in df.columns if c != "GEOID"]
    gdf_pts = gpd.GeoDataFrame(
        df.loc[ok, cols_keep].copy(),
        geometry=gpd.points_from_xy(df.loc[ok, "longitude"], df.loc[ok, "latitude"]),
        crs="EPSG:4326",
    )

    joined = gpd.sjoin(
        gdf_pts,
        tracts[["GEOID", "geometry"]].copy(),
        how="left",
        predicate="within",
    )

    geoid_col = None
    for cand in ["GEOID", "GEOID_right", "GEOID_r"]:
        if cand in joined.columns:
            geoid_col = cand
            break

    if geoid_col is None:
        raise KeyError(f"GEOID join sonrası bulunamadı. Kolonlar: {list(joined.columns)}")

    df.loc[ok, "GEOID"] = joined[geoid_col].astype("string").str.zfill(11).values
    df["GEOID"] = zfill_geoid(df["GEOID"])
    return df


# ============================================================
# DOWNLOAD
# ============================================================

def detect_incremental_start(existing_raw: pd.DataFrame) -> datetime.date:
    today_utc = datetime.now(timezone.utc).date()
    five_years_ago = today_utc - timedelta(days=5 * 365)

    if existing_raw is None or existing_raw.empty or "datetime" not in existing_raw.columns:
        log(f"📌 Mod: full | start={five_years_ago}")
        return five_years_ago

    existing_dt = pd.to_datetime(existing_raw["datetime"], errors="coerce")
    existing_dt = existing_dt.dropna()

    if existing_dt.empty:
        log(f"📌 Mod: full | start={five_years_ago}")
        return five_years_ago

    last_dt = existing_dt.max()
    start_date = last_dt.date() - timedelta(days=OVERLAP_DAYS)
    if start_date < five_years_ago:
        start_date = five_years_ago

    log(f"📌 Mod: incremental(raw)+overlap | start={start_date} | last={last_dt}")
    return start_date


def fetch_incremental_data(existing_raw: pd.DataFrame) -> pd.DataFrame:
    today_utc = datetime.now(timezone.utc).date()
    start_date = detect_incremental_start(existing_raw)

    session = requests.Session()
    all_parts = []

    current = start_date
    while current <= today_utc:
        chunk_end = min(current + timedelta(days=CHUNK_DAYS - 1), today_utc)

        start_iso = f"{current.isoformat()}T00:00:00.000"
        end_iso = f"{chunk_end.isoformat()}T23:59:59.999"

        log(f"⛏️  {current} → {chunk_end} aralığı çekiliyor…")

        offset = 0
        chunk_frames = []

        while True:
            params = {
                "$select": ",".join(SELECT_COLS),
                "$where": f"requested_datetime between '{start_iso}' and '{end_iso}'",
                "$order": "requested_datetime ASC",
                "$limit": PAGE_LIMIT,
                "$offset": offset,
            }

            data = socrata_get(session, params)
            df_part = pd.DataFrame(data)

            if df_part.empty:
                break

            chunk_frames.append(df_part)
            offset += len(df_part)
            log(f"   • +{len(df_part):,} satır | chunk toplam={offset:,}")

            if len(df_part) < PAGE_LIMIT:
                break

            time.sleep(SLEEP_SEC)

        if chunk_frames:
            chunk_df = pd.concat(chunk_frames, ignore_index=True)
            all_parts.append(chunk_df)
            log(f"✅ Chunk tamam: {len(chunk_df):,} satır")
        else:
            log("ℹ️  Bu aralıkta veri yok")

        current = chunk_end + timedelta(days=1)
        time.sleep(SLEEP_SEC)

    if not all_parts:
        return pd.DataFrame()

    inc = pd.concat(all_parts, ignore_index=True)
    log(f"📦 İndirilen ham incremental satır: {len(inc):,}")
    return inc


# ============================================================
# MERGE RAW
# ============================================================

def merge_existing_and_incremental(existing_raw: pd.DataFrame, inc_raw: pd.DataFrame) -> pd.DataFrame:
    frames = []

    if existing_raw is not None and not existing_raw.empty:
        frames.append(existing_raw.copy())

    if inc_raw is not None and not inc_raw.empty:
        frames.append(inc_raw.copy())

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = standardize_raw_columns(df)

    if "id" in df.columns:
        df = df.drop_duplicates(subset=["id"], keep="last").reset_index(drop=True)
    else:
        df = df.drop_duplicates().reset_index(drop=True)

    today_utc = datetime.now(timezone.utc).date()
    five_years_ago = today_utc - timedelta(days=5 * 365)

    dt_local = pd.to_datetime(df["datetime"], errors="coerce")
    keep = dt_local.notna() & (dt_local.dt.date >= five_years_ago)
    df = df.loc[keep].copy()

    return df


# ============================================================
# GRID HELPERS
# ============================================================

def build_full_daily_grid(geoids: pd.Series, date_min: pd.Timestamp, date_max: pd.Timestamp) -> pd.DataFrame:
    all_dates = pd.date_range(date_min, date_max, freq="D")
    idx = pd.MultiIndex.from_product([geoids.tolist(), all_dates], names=["GEOID", "date"])
    out = idx.to_frame(index=False)
    return out


def build_full_slot_grid(geoids: pd.Series, date_min: pd.Timestamp, date_max: pd.Timestamp) -> pd.DataFrame:
    all_dates = pd.date_range(date_min, date_max, freq="D")
    idx = pd.MultiIndex.from_product([geoids.tolist(), all_dates, SLOT_ORDER], names=["GEOID", "date", "hour_range"])
    out = idx.to_frame(index=False)
    return out


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def build_model_ready_agg(df_raw: pd.DataFrame, tracts: gpd.GeoDataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    if "GEOID" not in df.columns:
        df["GEOID"] = pd.NA

    need_geoid = df["GEOID"].isna()
    if need_geoid.any():
        log(f"🧭 GEOID atanacak satır: {int(need_geoid.sum()):,}")
        fixed = attach_geoid(df.loc[need_geoid].copy(), tracts)
        df.loc[need_geoid, "GEOID"] = fixed["GEOID"].values

    df["GEOID"] = zfill_geoid(df["GEOID"])
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    df = df.dropna(subset=["GEOID", "datetime", "date"]).copy()

    if "id" in df.columns:
        df = df.drop_duplicates(subset=["id"], keep="last").copy()
    else:
        df = df.drop_duplicates().copy()

    df["event_hour"] = df["datetime"].dt.hour
    df["hour_range"] = make_hour_range(df["event_hour"])

    slot_cur = (
        df.groupby(["GEOID", "date", "hour_range"], dropna=False)
        .agg(
            request_count_311=("id", "count"),
            unique_category_311=("category", "nunique"),
            unique_subcategory_311=("subcategory", "nunique"),
        )
        .reset_index()
    )

    daily_cur = (
        df.groupby(["GEOID", "date"], dropna=False)
        .agg(
            request_count_311_daily=("id", "count"),
        )
        .reset_index()
    )

    if slot_cur.empty:
        raise RuntimeError("311 aggregate üretilemedi: slot_cur boş.")

    date_min = slot_cur["date"].min()
    date_max = slot_cur["date"].max()

    geoids = tracts["GEOID"].astype("string").str.zfill(11).sort_values().drop_duplicates()

    # --------------------------------------------------------
    # DAILY FULL GRID
    # --------------------------------------------------------
    daily_full = build_full_daily_grid(geoids, date_min, date_max)
    daily_full["date"] = pd.to_datetime(daily_full["date"], errors="coerce")

    daily_full = daily_full.merge(daily_cur, on=["GEOID", "date"], how="left")
    daily_full["request_count_311_daily"] = daily_full["request_count_311_daily"].fillna(0).astype(np.int32)

    daily_full = daily_full.sort_values(["GEOID", "date"]).reset_index(drop=True)

    daily_full["request_count_311_prev_1d"] = (
        daily_full.groupby("GEOID")["request_count_311_daily"]
        .shift(1)
        .fillna(0)
    )

    daily_full["request_count_311_prev_3d"] = (
        daily_full.groupby("GEOID")["request_count_311_daily"]
        .transform(lambda s: s.shift(1).rolling(3, min_periods=1).sum())
        .fillna(0)
    )

    daily_full["request_count_311_prev_7d"] = (
        daily_full.groupby("GEOID")["request_count_311_daily"]
        .transform(lambda s: s.shift(1).rolling(7, min_periods=1).sum())
        .fillna(0)
    )

    daily_full["request_count_311_daily_roll7_mean"] = (
        daily_full.groupby("GEOID")["request_count_311_daily"]
        .transform(lambda s: s.shift(1).rolling(7, min_periods=2).mean())
    )

    daily_full["request_count_311_daily_roll7_std"] = (
        daily_full.groupby("GEOID")["request_count_311_daily"]
        .transform(lambda s: s.shift(1).rolling(7, min_periods=2).std())
    )

    daily_full["request_count_311_ratio_1d_7d"] = (
        daily_full["request_count_311_prev_1d"]
        / ((daily_full["request_count_311_prev_7d"] / 7.0) + EPS)
    )

    daily_full["request_count_311_zscore_7d"] = (
        (
            daily_full["request_count_311_prev_1d"]
            - daily_full["request_count_311_daily_roll7_mean"].fillna(0)
        )
        / (daily_full["request_count_311_daily_roll7_std"].fillna(0) + EPS)
    )

    daily_full["request_count_311_spike_flag"] = (
        daily_full["request_count_311_prev_1d"]
        > (1.5 * (daily_full["request_count_311_prev_7d"] / 7.0))
    ).astype(np.int8)

    daily_feats = daily_full[
        [
            "GEOID",
            "date",
            "request_count_311_prev_1d",
            "request_count_311_prev_3d",
            "request_count_311_prev_7d",
            "request_count_311_ratio_1d_7d",
            "request_count_311_zscore_7d",
            "request_count_311_spike_flag",
        ]
    ].copy()

    # --------------------------------------------------------
    # SLOT FULL GRID
    # --------------------------------------------------------
    slot_full = build_full_slot_grid(geoids, date_min, date_max)
    slot_full["date"] = pd.to_datetime(slot_full["date"], errors="coerce")

    slot_full = slot_full.merge(slot_cur, on=["GEOID", "date", "hour_range"], how="left")

    for c in ["request_count_311", "unique_category_311", "unique_subcategory_311"]:
        slot_full[c] = slot_full[c].fillna(0).astype(np.int32)

    slot_full = slot_full.merge(daily_feats, on=["GEOID", "date"], how="left")

    slot_full["slot_start_hour"] = slot_full["hour_range"].map(SLOT_START_MAP).astype(np.int16)
    slot_full["slot_dt"] = slot_full["date"] + pd.to_timedelta(slot_full["slot_start_hour"], unit="h")

    # --------------------------------------------------------
    # PREV SLOT
    # --------------------------------------------------------
    slot_full = slot_full.sort_values(["GEOID", "slot_dt"]).reset_index(drop=True)
    slot_full["request_count_311_prev_slot"] = (
        slot_full.groupby("GEOID")["request_count_311"]
        .shift(1)
        .fillna(0)
    )

    # --------------------------------------------------------
    # SAME SLOT FEATURES
    # --------------------------------------------------------
    slot_full = slot_full.sort_values(["GEOID", "hour_range", "date"]).reset_index(drop=True)

    slot_full["request_count_311_same_slot_prev_day"] = (
        slot_full.groupby(["GEOID", "hour_range"])["request_count_311"]
        .shift(1)
        .fillna(0)
    )

    slot_full["request_count_311_same_slot_prev_week"] = (
        slot_full.groupby(["GEOID", "hour_range"])["request_count_311"]
        .shift(7)
        .fillna(0)
    )

    slot_full["request_count_311_same_slot_roll7_mean"] = (
        slot_full.groupby(["GEOID", "hour_range"])["request_count_311"]
        .transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
        .fillna(0)
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------
    slot_full["date"] = slot_full["date"].dt.strftime("%Y-%m-%d")

    keep_cols = [
        "GEOID",
        "date",
        "hour_range",
        "request_count_311",
        "unique_category_311",
        "unique_subcategory_311",
        "request_count_311_prev_slot",
        "request_count_311_prev_1d",
        "request_count_311_prev_3d",
        "request_count_311_prev_7d",
        "request_count_311_same_slot_prev_day",
        "request_count_311_same_slot_prev_week",
        "request_count_311_same_slot_roll7_mean",
        "request_count_311_ratio_1d_7d",
        "request_count_311_zscore_7d",
        "request_count_311_spike_flag",
    ]

    out = slot_full[keep_cols].copy()

    int_like_cols = [
        "request_count_311",
        "unique_category_311",
        "unique_subcategory_311",
        "request_count_311_prev_1d",
        "request_count_311_prev_3d",
        "request_count_311_prev_7d",
        "request_count_311_spike_flag",
    ]
    for c in int_like_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)

    float_like_cols = [
        "request_count_311_prev_slot",
        "request_count_311_same_slot_prev_day",
        "request_count_311_same_slot_prev_week",
        "request_count_311_same_slot_roll7_mean",
        "request_count_311_ratio_1d_7d",
        "request_count_311_zscore_7d",
    ]
    for c in float_like_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)

    return out


# ============================================================
# SAVE
# ============================================================

def save_raw(df: pd.DataFrame, path: Path) -> None:
    raw_cols = [
        "id",
        "datetime",
        "date",
        "time",
        "lat",
        "long",
        "category",
        "subcategory",
        "agency_responsible",
        "latitude",
        "longitude",
        "GEOID",
    ]

    out = df.copy()
    for c in raw_cols:
        if c not in out.columns:
            out[c] = pd.NA

    out = out[raw_cols].copy()
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")

    out.to_csv(path, index=False, encoding="utf-8-sig")
    log(f"💾 RAW kaydedildi: {path} | shape={out.shape}")


def save_agg(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")
    log(f"💾 AGG kaydedildi: {path} | shape={df.shape}")


# ============================================================
# MAIN
# ============================================================

def main():
    log(f"🔎 CWD: {Path.cwd()}")
    log(f"🔎 SAVE_DIR: {BASE_DIR}")

    existing_raw = read_csv_if_exists(RAW_OUT)
    inc_raw = fetch_incremental_data(existing_raw)
    final_raw = merge_existing_and_incremental(existing_raw, inc_raw)

    if final_raw.empty:
        raise RuntimeError("311 verisi üretilemedi.")

    log(f"📦 Final raw satır: {len(final_raw):,}")

    tracts = load_sf_tracts()

    if "GEOID" not in final_raw.columns:
        final_raw["GEOID"] = pd.NA

    need_geoid = final_raw["GEOID"].isna()
    if need_geoid.any():
        log(f"🧭 Ana akışta GEOID atanacak satır: {int(need_geoid.sum()):,}")
        fixed = attach_geoid(final_raw.loc[need_geoid].copy(), tracts)
        final_raw.loc[need_geoid, "GEOID"] = fixed["GEOID"].values

    final_raw["GEOID"] = zfill_geoid(final_raw["GEOID"])

    if KEEP_RAW:
        save_raw(final_raw, RAW_OUT)

    agg = build_model_ready_agg(final_raw, tracts)
    save_agg(agg, AGG_OUT)

    log("✅ update_311.py tamamlandı")
    log("İlk 5 satır:")
    print(agg.head(), flush=True)


if __name__ == "__main__":
    main()
