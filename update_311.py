# scripts/update_311.py
from __future__ import annotations

import os
import re
import time
import json
import requests
import warnings
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import geopandas as gpd

warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
# TZ
# =============================================================================
try:
    import zoneinfo
    SF_TZ = zoneinfo.ZoneInfo("America/Los_Angeles")
except Exception:
    SF_TZ = None

UTC = "UTC"

# =============================================================================
# AYARLAR
# =============================================================================
BASE_DIR = os.getenv("CRIME_DATA_DIR", "crime_prediction_data")
os.makedirs(BASE_DIR, exist_ok=True)

DEFAULT_GEOID_LEN = int(os.getenv("GEOID_LEN", "11"))

# ---- Dosya isimleri ----------------------------------------------------------
RAW_311_PARQUET = os.getenv("RAW_311_PARQUET", "sf_311_last_5_years.parquet")
RAW_311_CSV     = os.getenv("RAW_311_CSV",     "sf_311_last_5_years.csv")

AGG_311_PARQUET = os.getenv("AGG_311_PARQUET", "sf_311_agg_3h.parquet")
AGG_311_CSV     = os.getenv("AGG_311_CSV",     "sf_311_agg_3h.csv")

FEAT_311_PARQUET = os.getenv("FEAT_311_PARQUET", "sf_311_features_3h.parquet")
FEAT_311_CSV     = os.getenv("FEAT_311_CSV",     "sf_311_features_3h.csv")

CRIME_IN_PARQUET = os.getenv("CRIME_IN_PARQUET", "sf_crime_01.parquet")
CRIME_IN_CSV     = os.getenv("CRIME_IN_CSV",     "sf_crime_01.csv")
CRIME_OUT_PARQUET = os.getenv("CRIME_OUT_PARQUET", "sf_crime_02.parquet")
CRIME_OUT_CSV     = os.getenv("CRIME_OUT_CSV",     "sf_crime_02.csv")

WRITE_CSV = os.getenv("WRITE_CSV", "1").strip() == "1"

# ---- Socrata ----------------------------------------------------------------
DATASET_BASE = os.getenv("SF311_DATASET", "https://data.sfgov.org/resource/vw6y-z8j6.json")
SOCRATA_APP_TOKEN = os.getenv("SOCS_APP_TOKEN", "").strip()

PAGE_LIMIT   = int(os.getenv("SF_SODA_PAGE_LIMIT", "50000"))
SODA_TIMEOUT = int(os.getenv("SF_SODA_TIMEOUT", "90"))
SODA_RETRIES = int(os.getenv("SF_SODA_RETRIES", "5"))
SLEEP_SEC    = float(os.getenv("SF_SODA_THROTTLE_SEC", "0.25"))

CHUNK_DAYS              = int(os.getenv("SF311_CHUNK_DAYS", "31"))
MAX_PAGES_PER_CHUNK     = int(os.getenv("SF311_MAX_PAGES_PER_CHUNK", "40"))
MAX_CONSEC_EMPTY_CHUNKS = int(os.getenv("SF311_MAX_EMPTY_CHUNKS", "8"))

TODAY = datetime.utcnow().date()
FIVE_YEARS = 5 * 365
DEFAULT_START = TODAY - timedelta(days=FIVE_YEARS)

BACKFILL_DAYS = int(os.getenv("BACKFILL_DAYS", "0"))
REINGEST_DAYS = int(os.getenv("SF311_REINGEST_DAYS", "14"))

# ---- GEOJSON ----------------------------------------------------------------
GEOJSON_NAME = os.getenv("SF_BLOCKS_GEOJSON", "sf_census_blocks.geojson")
GEOJSON_CANDIDATES = [
    os.path.join(BASE_DIR, GEOJSON_NAME),
    os.path.join("crime_prediction_data", GEOJSON_NAME),
    os.path.join(".", GEOJSON_NAME),
]

# ---- Kategori grup sözlüğü --------------------------------------------------
GROUP_RULES = {
    "disorder": [
        "encamp", "homeless", "noise", "disturb", "nuisance", "drug",
        "loiter", "behavior", "drinking", "graffiti", "vandal"
    ],
    "vehicle": [
        "vehicle", "parking", "abandoned", "tow", "car", "traffic"
    ],
    "street": [
        "street", "sidewalk", "curb", "pavement", "road", "alley", "crosswalk"
    ],
    "lighting": [
        "light", "lamp", "streetlight", "illumination"
    ],
    "sanitation": [
        "garbage", "trash", "dump", "debris", "clean", "feces", "waste", "sewer"
    ],
    "infrastructure": [
        "tree", "sign", "pole", "utility", "water", "drain", "pothole"
    ],
    "other": []
}

# =============================================================================
# YARDIMCILAR
# =============================================================================
def log(msg: str):
    print(msg, flush=True)

def log_shape(df: pd.DataFrame, label: str):
    r, c = df.shape
    log(f"📊 {label}: {r:,} satır × {c:,} sütun")

def normalize_geoid(series, target_len: int | None = None):
    L = int(target_len or DEFAULT_GEOID_LEN)
    s = pd.Series(series, dtype="string")
    s = s.str.extract(r"(\d+)", expand=False)
    s = s.str.slice(0, L)
    return s.str.zfill(L)

def normalize_geoid_11(x):
    if pd.isna(x):
        return pd.NA
    digits = re.sub(r"\D", "", str(x))
    return digits[:11] if len(digits) >= 11 else pd.NA

def safe_read_parquet(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_parquet(path)

def safe_read_csv(path: str, **kwargs) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)

def save_atomic_csv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)

def save_atomic_parquet(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp.parquet"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)

def save_both(df: pd.DataFrame, parquet_path: str | None = None, csv_path: str | None = None):
    if parquet_path:
        save_atomic_parquet(df, parquet_path)
    if WRITE_CSV and csv_path:
        save_atomic_csv(df, csv_path)

def parse_dt_utc(x):
    return pd.to_datetime(x, errors="coerce", utc=True)

def dt_to_sf(dt_ser: pd.Series) -> pd.Series:
    dt_ser = pd.to_datetime(dt_ser, errors="coerce", utc=True)
    if SF_TZ is not None:
        return dt_ser.dt.tz_convert(SF_TZ)
    return dt_ser

def to_sf_date(dt_ser: pd.Series) -> pd.Series:
    return dt_to_sf(dt_ser).dt.date

def to_sf_hour(dt_ser: pd.Series) -> pd.Series:
    return dt_to_sf(dt_ser).dt.hour

def make_hour_range_from_hour(hour_ser: pd.Series) -> pd.Series:
    hour_ser = pd.to_numeric(hour_ser, errors="coerce").fillna(0).astype(int)
    start_h = (hour_ser // 3) * 3
    end_h = start_h + 3
    end_h = np.where(end_h < 24, end_h, 24)
    return (
        pd.Series(start_h, index=hour_ser.index).astype(int).astype(str).str.zfill(2)
        + "-"
        + pd.Series(end_h, index=hour_ser.index).astype(int).astype(str).str.zfill(2)
    )

def hour_range_to_start(hour_range: pd.Series) -> pd.Series:
    hr = hour_range.astype(str).str.extract(r"(\d{1,2})")[0]
    return pd.to_numeric(hr, errors="coerce").fillna(0).astype(int)

def month_to_season(m):
    if pd.isna(m):
        return pd.NA
    m = int(m)
    if m in [12, 1, 2]:
        return "Winter"
    if m in [3, 4, 5]:
        return "Spring"
    if m in [6, 7, 8]:
        return "Summer"
    return "Fall"

def category_group(category, subcategory):
    txt = f"{category or ''} {subcategory or ''}".lower()
    for grp, kws in GROUP_RULES.items():
        if grp == "other":
            continue
        for kw in kws:
            if kw in txt:
                return grp
    return "other"

def is_closed_status(x: str) -> float:
    s = str(x).strip().lower()
    if not s or s == "nan":
        return np.nan
    closed_like = ["close", "closed", "resolved", "complete", "completed"]
    open_like   = ["open", "new", "pending", "in progress", "accepted"]
    if any(k in s for k in closed_like):
        return 1.0
    if any(k in s for k in open_like):
        return 0.0
    return np.nan

def infer_source_group(x: str) -> str:
    s = str(x).strip().lower()
    if not s or s == "nan":
        return "unknown"
    if "mobile" in s or "app" in s:
        return "mobile"
    if "phone" in s or "call" in s:
        return "phone"
    if "web" in s or "online" in s:
        return "web"
    return "other"

# =============================================================================
# SOCRATA
# =============================================================================
def socrata_get(session: requests.Session, url, params):
    headers = {"Accept": "application/json"}
    if SOCRATA_APP_TOKEN:
        headers["X-App-Token"] = SOCRATA_APP_TOKEN

    last_err = None
    for i in range(SODA_RETRIES):
        try:
            r = session.get(url, params=params, headers=headers, timeout=SODA_TIMEOUT)
            if r.status_code in (408, 429) or 500 <= r.status_code < 600:
                raise requests.HTTPError(f"status={r.status_code}")
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            sleep_s = max(SLEEP_SEC, SLEEP_SEC * (2 ** i))
            log(f"⚠️ Socrata retry {i+1}/{SODA_RETRIES} ({e}); {sleep_s:.1f}s bekleme…")
            time.sleep(sleep_s)
    raise last_err

# =============================================================================
# GEO
# =============================================================================
def ensure_blocks_gdf():
    for cand in GEOJSON_CANDIDATES:
        if os.path.exists(cand):
            gdf = gpd.read_file(cand)
            if "GEOID" not in gdf.columns:
                possible = [c for c in gdf.columns if str(c).upper().startswith("GEOID")]
                if not possible:
                    continue
                gdf["GEOID"] = gdf[possible[0]].astype(str)

            gdf["TRACT11"] = gdf["GEOID"].apply(normalize_geoid_11)
            gdf = gdf[["TRACT11", "geometry"]].dropna(subset=["TRACT11"]).copy()

            if gdf.crs is None:
                gdf = gdf.set_crs("EPSG:4326")
            elif str(gdf.crs).lower() not in ("epsg:4326", "wgs84", "wgs 84"):
                gdf = gdf.to_crs(epsg=4326)

            log(f"🧭 GEOJSON kullanılıyor: {os.path.abspath(cand)}")
            return gdf

    log("⚠️ GEOJSON bulunamadı; GEOID eşleme yapılamayacak.")
    return None

def geotag_to_geoid11(df_new: pd.DataFrame) -> pd.DataFrame:
    df_new = df_new.copy()

    if "latitude" not in df_new.columns and "lat" in df_new.columns:
        df_new["latitude"] = pd.to_numeric(df_new["lat"], errors="coerce")
    if "longitude" not in df_new.columns and "long" in df_new.columns:
        df_new["longitude"] = pd.to_numeric(df_new["long"], errors="coerce")

    df_new["latitude"] = pd.to_numeric(df_new.get("latitude"), errors="coerce")
    df_new["longitude"] = pd.to_numeric(df_new.get("longitude"), errors="coerce")

    df_ok = df_new.dropna(subset=["latitude", "longitude"]).copy()
    if df_ok.empty:
        df_new["GEOID"] = pd.NA
        return df_new

    gdf_blocks = ensure_blocks_gdf()
    if gdf_blocks is None:
        df_new["GEOID"] = pd.NA
        return df_new

    gdf_pts = gpd.GeoDataFrame(
        df_ok,
        geometry=gpd.points_from_xy(df_ok["longitude"], df_ok["latitude"]),
        crs="EPSG:4326",
    )

    try:
        gdf_join = gpd.sjoin(gdf_pts, gdf_blocks, how="left", predicate="within")
    except Exception:
        try:
            gdf_join = gpd.sjoin_nearest(gdf_pts, gdf_blocks, how="left", max_distance=0.001)
        except Exception:
            df_new["GEOID"] = pd.NA
            return df_new

    out = pd.DataFrame(gdf_join.drop(columns=["geometry"], errors="ignore"))
    out.rename(columns={"TRACT11": "GEOID"}, inplace=True)
    out["GEOID"] = normalize_geoid(out["GEOID"], DEFAULT_GEOID_LEN)

    back = df_new.copy()
    back["__rowid__"] = np.arange(len(back))
    out["__rowid__"] = out.index.values

    merged = back.merge(out[["__rowid__", "GEOID"]], on="__rowid__", how="left")
    merged.drop(columns="__rowid__", inplace=True)
    return merged

# =============================================================================
# DOSYA/SEED
# =============================================================================
def load_existing_raw_or_seed() -> pd.DataFrame:
    p_parquet = os.path.join(BASE_DIR, RAW_311_PARQUET)
    p_csv = os.path.join(BASE_DIR, RAW_311_CSV)

    if os.path.exists(p_parquet):
        df = pd.read_parquet(p_parquet)
        log(f"📁 Mevcut 311 parquet bulundu: {p_parquet}")
        return standardize_raw_schema(df)

    if os.path.exists(p_csv):
        df = pd.read_csv(p_csv, low_memory=False)
        log(f"📁 Mevcut 311 csv bulundu: {p_csv}")
        return standardize_raw_schema(df)

    log("ℹ️ Mevcut 311 ham dosyası yok; boş seed ile başlanıyor.")
    return pd.DataFrame()

def decide_start_date(df_existing: pd.DataFrame):
    if BACKFILL_DAYS > 0:
        start = TODAY - timedelta(days=BACKFILL_DAYS)
        log(f"📌 Mod: backfill | start={start}")
        return start, "backfill"

    if df_existing.empty or "datetime" not in df_existing.columns:
        log(f"📌 Mod: full-5y | window ≥ {DEFAULT_START}")
        return DEFAULT_START, "full-5y"

    dt = parse_dt_utc(df_existing["datetime"])
    last_dt = dt.max()

    if pd.isna(last_dt):
        log(f"📌 Mod: full-5y (datetime parse edilemedi) | window ≥ {DEFAULT_START}")
        return DEFAULT_START, "full-5y"

    last_date = last_dt.date()
    start = last_date - timedelta(days=max(1, REINGEST_DAYS))
    if start < DEFAULT_START:
        start = DEFAULT_START

    log(f"📌 Mod: incremental+overlap | start={start} | last={last_date} | reingest={REINGEST_DAYS}d")
    return start, "incremental+overlap"

# =============================================================================
# DOWNLOAD
# =============================================================================
def download_by_date_chunks(start_date: datetime.date) -> pd.DataFrame:
    log(f"🧩 İndirme modu: DATE-CHUNKS ({CHUNK_DAYS} gün) + paging")

    session = requests.Session()

    # Daha geniş sinyal için police-only yerine tüm 311 çekilebilir.
    # İstersen agency filtresi env ile ver.
    agency_like = os.getenv("SF311_AGENCY_FILTER", "").strip()
    extra_where = ""
    if agency_like:
        extra_where = f" AND ({agency_like})"

    cols = ",".join([
        "service_request_id",
        "requested_datetime",
        "closed_date",
        "updated_datetime",
        "status_description",
        "agency_responsible",
        "category",
        "subcategory",
        "service_details",
        "lat",
        "long",
        "point",
        "source"
    ])

    all_chunks = []
    consec_empty = 0
    cur = start_date
    end = TODAY

    while cur <= end:
        chunk_end = min(cur + timedelta(days=CHUNK_DAYS - 1), end)
        start_iso = f"{cur.isoformat()}T00:00:00.000"
        end_iso   = f"{chunk_end.isoformat()}T23:59:59.999"

        log(f"⛏️  {cur} → {chunk_end} aralığı çekiliyor…")

        offset = 0
        pages = 0
        chunk_rows = []

        while True:
            params = {
                "$select": cols,
                "$where": f"requested_datetime between '{start_iso}' and '{end_iso}'{extra_where}",
                "$order": "requested_datetime ASC",
                "$limit": PAGE_LIMIT,
                "$offset": offset,
            }

            try:
                data = socrata_get(session, DATASET_BASE, params)
            except Exception as e:
                log(f"❌ Chunk hata ({cur}→{chunk_end}, offset={offset}): {e} → chunk geçiliyor.")
                break

            df = pd.DataFrame(data)
            if df.empty:
                break

            if pages == 0:
                log(f"   • kolonlar: {list(df.columns)}")

            chunk_rows.append(df)
            offset += len(df)
            pages += 1
            log(f"   + {offset:,} kayıt (sayfa={pages})")

            if len(df) < PAGE_LIMIT or pages >= MAX_PAGES_PER_CHUNK:
                if pages >= MAX_PAGES_PER_CHUNK:
                    log(f"   ↪️ MAX_PAGES_PER_CHUNK={MAX_PAGES_PER_CHUNK} doldu, chunk kesildi.")
                break

            time.sleep(SLEEP_SEC)

        if chunk_rows:
            consec_empty = 0
            all_chunks.append(pd.concat(chunk_rows, ignore_index=True))
            log(f"✅ Chunk bitti: satır={sum(len(x) for x in chunk_rows):,}")
        else:
            consec_empty += 1
            log(f"ℹ️ Chunk boş döndü (ardışık boş={consec_empty}).")
            if consec_empty >= MAX_CONSEC_EMPTY_CHUNKS and cur > start_date:
                log("⏹️ Çok sayıda ardışık boş chunk; erken durdurma.")
                break

        cur = chunk_end + timedelta(days=1)
        time.sleep(SLEEP_SEC)

    return pd.concat(all_chunks, ignore_index=True) if all_chunks else pd.DataFrame()

# =============================================================================
# ŞEMA / TEMİZLİK
# =============================================================================
def standardize_raw_schema(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        cols = [
            "id", "datetime", "closed_date", "updated_datetime", "status_description",
            "agency_responsible", "category", "subcategory", "service_details",
            "lat", "long", "point", "source", "latitude", "longitude",
            "sf_date", "sf_time", "event_hour", "hour_range", "GEOID",
            "status_closed_flag", "resolution_hours", "category_group", "source_group"
        ]
        return pd.DataFrame(columns=cols)

    df = df.copy()
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]

    rename_map = {
        "service_request_id": "id",
        "requested_datetime": "datetime",
    }
    df.rename(columns=rename_map, inplace=True)

    for c in [
        "id", "datetime", "closed_date", "updated_datetime", "status_description",
        "agency_responsible", "category", "subcategory", "service_details",
        "lat", "long", "point", "source", "latitude", "longitude", "GEOID"
    ]:
        if c not in df.columns:
            df[c] = pd.NA

    # datetime
    df["datetime"] = parse_dt_utc(df["datetime"])
    df["closed_date"] = parse_dt_utc(df["closed_date"])
    df["updated_datetime"] = parse_dt_utc(df["updated_datetime"])

    # lat/long
    if "latitude" not in df.columns or df["latitude"].isna().all():
        df["latitude"] = pd.to_numeric(df["lat"], errors="coerce")
    else:
        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")

    if "longitude" not in df.columns or df["longitude"].isna().all():
        df["longitude"] = pd.to_numeric(df["long"], errors="coerce")
    else:
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    df["lat"] = df["latitude"]
    df["long"] = df["longitude"]

    # SF local türevleri
    dt_sf = dt_to_sf(df["datetime"])
    df["sf_date"] = dt_sf.dt.date
    df["sf_time"] = dt_sf.dt.time
    df["event_hour"] = dt_sf.dt.hour.astype("Int64")
    df["hour_range"] = make_hour_range_from_hour(df["event_hour"])

    # durumsal
    df["status_closed_flag"] = df["status_description"].map(is_closed_status)
    df["resolution_hours"] = (
        (df["closed_date"] - df["datetime"]).dt.total_seconds() / 3600.0
    )
    df.loc[df["resolution_hours"] < 0, "resolution_hours"] = np.nan
    df.loc[df["resolution_hours"] > 24 * 365, "resolution_hours"] = np.nan

    df["category_group"] = [
        category_group(c, s) for c, s in zip(df["category"], df["subcategory"])
    ]
    df["source_group"] = df["source"].map(infer_source_group)

    if "GEOID" in df.columns:
        df["GEOID"] = normalize_geoid(df["GEOID"], DEFAULT_GEOID_LEN)

    return df

def dedupe_raw(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    if "id" in df.columns:
        df["id"] = df["id"].astype("string")
    if "updated_datetime" in df.columns:
        df = df.sort_values(["id", "updated_datetime", "datetime"], na_position="last")
        df = df.drop_duplicates(subset=["id"], keep="last")
    else:
        df = df.sort_values(["id", "datetime"], na_position="last")
        df = df.drop_duplicates(subset=["id"], keep="last")

    df = df.sort_values("datetime").reset_index(drop=True)
    return df

# =============================================================================
# AGG
# =============================================================================
def build_agg_3h(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw.empty:
        return pd.DataFrame(columns=[
            "GEOID", "date", "hour_range", "311_request_count"
        ])

    d = df_raw.copy()
    d = d.dropna(subset=["GEOID", "sf_date", "hour_range"])
    d["date"] = pd.to_datetime(d["sf_date"], errors="coerce").dt.date

    agg = (
        d.groupby(["GEOID", "date", "hour_range"], as_index=False)
         .size()
         .rename(columns={"size": "311_request_count"})
    )
    agg["GEOID"] = normalize_geoid(agg["GEOID"], DEFAULT_GEOID_LEN)
    return agg.sort_values(["GEOID", "date", "hour_range"]).reset_index(drop=True)

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
def build_311_features_3h(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Leak-safe mantık:
    - feature slotu = GEOID x date x hour_range
    - aynı slot dahil edilmeden geçmişe dayalı hesap
    - rolling hesaplar groupby(GEOID) üzerinde, slot_time sıralı
    """
    base_cols = [
        "GEOID", "date", "hour_range",
        "311_request_count_3h",
        "311_request_count_24h",
        "311_request_count_7d",
        "311_request_count_14d",
        "311_unique_category_7d",
        "311_unique_subcategory_7d",
        "311_open_count_7d",
        "311_closed_count_7d",
        "311_open_ratio_7d",
        "311_avg_resolution_hours_30d",
        "311_median_resolution_hours_30d",
        "311_disorder_count_7d",
        "311_vehicle_count_7d",
        "311_street_count_7d",
        "311_lighting_count_7d",
        "311_sanitation_count_7d",
        "311_infrastructure_count_7d",
        "311_other_count_7d",
        "311_mobile_count_7d",
        "311_phone_count_7d",
        "311_web_count_7d",
        "311_other_source_count_7d",
    ]

    if df_raw.empty:
        return pd.DataFrame(columns=base_cols)

    d = df_raw.copy()
    d = d.dropna(subset=["GEOID", "datetime"]).copy()
    if d.empty:
        return pd.DataFrame(columns=base_cols)

    # Slot bilgisi SF local'den
    dt_sf = dt_to_sf(d["datetime"])
    d["date"] = dt_sf.dt.date
    d["event_hour"] = dt_sf.dt.hour.astype(int)
    d["hour_range"] = make_hour_range_from_hour(d["event_hour"])
    d["slot_start_hour"] = hour_range_to_start(d["hour_range"])
    d["slot_time"] = pd.to_datetime(d["date"].astype(str)) + pd.to_timedelta(d["slot_start_hour"], unit="h")

    # Bayraklar
    d["is_open"] = np.where(d["status_closed_flag"] == 0, 1, 0)
    d["is_closed"] = np.where(d["status_closed_flag"] == 1, 1, 0)

    for grp in ["disorder", "vehicle", "street", "lighting", "sanitation", "infrastructure", "other"]:
        d[f"grp_{grp}"] = (d["category_group"] == grp).astype(int)

    for sgrp in ["mobile", "phone", "web", "other"]:
        d[f"src_{sgrp}"] = (d["source_group"] == sgrp).astype(int)

    # Slot-level event aggregation
    # unique category / subcategory ve resolution istatistiklerini slot seviyesinde hazırla
    slot = (
        d.groupby(["GEOID", "date", "hour_range", "slot_time"], as_index=False)
         .agg(
             slot_count=("id", "count"),
             slot_open=("is_open", "sum"),
             slot_closed=("is_closed", "sum"),
             slot_unique_category=("category", lambda x: x.dropna().astype(str).nunique()),
             slot_unique_subcategory=("subcategory", lambda x: x.dropna().astype(str).nunique()),
             slot_avg_resolution_hours=("resolution_hours", "mean"),
             slot_median_resolution_hours=("resolution_hours", "median"),
             grp_disorder=("grp_disorder", "sum"),
             grp_vehicle=("grp_vehicle", "sum"),
             grp_street=("grp_street", "sum"),
             grp_lighting=("grp_lighting", "sum"),
             grp_sanitation=("grp_sanitation", "sum"),
             grp_infrastructure=("grp_infrastructure", "sum"),
             grp_other=("grp_other", "sum"),
             src_mobile=("src_mobile", "sum"),
             src_phone=("src_phone", "sum"),
             src_web=("src_web", "sum"),
             src_other=("src_other", "sum"),
         )
    )

    slot = slot.sort_values(["GEOID", "slot_time"]).reset_index(drop=True)

    # 3 saatlik tüm slot grid'i: var olan slotlar üzerinden ilerleyelim
    # rolling için timeseries index gerekecek
    results = []

    for geoid, g in slot.groupby("GEOID", sort=False):
        g = g.sort_values("slot_time").copy()

        # geçmişe kaydır: aynı slot leakage olmasın
        g["x_count_lag"] = g["slot_count"].shift(1)
        g["x_open_lag"] = g["slot_open"].shift(1)
        g["x_closed_lag"] = g["slot_closed"].shift(1)
        g["x_unique_category_lag"] = g["slot_unique_category"].shift(1)
        g["x_unique_subcategory_lag"] = g["slot_unique_subcategory"].shift(1)
        g["x_avg_resolution_lag"] = g["slot_avg_resolution_hours"].shift(1)
        g["x_median_resolution_lag"] = g["slot_median_resolution_hours"].shift(1)

        for c in [
            "grp_disorder", "grp_vehicle", "grp_street", "grp_lighting",
            "grp_sanitation", "grp_infrastructure", "grp_other",
            "src_mobile", "src_phone", "src_web", "src_other"
        ]:
            g[f"x_{c}_lag"] = g[c].shift(1)

        # rolling windows
        # 24h = 8 slot, 7d = 56 slot, 14d = 112 slot, 30d = 240 slot
        g["311_request_count_3h"]  = g["x_count_lag"].fillna(0)
        g["311_request_count_24h"] = g["x_count_lag"].rolling(8, min_periods=1).sum().fillna(0)
        g["311_request_count_7d"]  = g["x_count_lag"].rolling(56, min_periods=1).sum().fillna(0)
        g["311_request_count_14d"] = g["x_count_lag"].rolling(112, min_periods=1).sum().fillna(0)

        g["311_unique_category_7d"] = g["x_unique_category_lag"].rolling(56, min_periods=1).max().fillna(0)
        g["311_unique_subcategory_7d"] = g["x_unique_subcategory_lag"].rolling(56, min_periods=1).max().fillna(0)

        g["311_open_count_7d"]   = g["x_open_lag"].rolling(56, min_periods=1).sum().fillna(0)
        g["311_closed_count_7d"] = g["x_closed_lag"].rolling(56, min_periods=1).sum().fillna(0)

        denom = g["311_open_count_7d"] + g["311_closed_count_7d"]
        g["311_open_ratio_7d"] = np.where(denom > 0, g["311_open_count_7d"] / denom, 0.0)

        g["311_avg_resolution_hours_30d"] = g["x_avg_resolution_lag"].rolling(240, min_periods=1).mean()
        g["311_median_resolution_hours_30d"] = g["x_median_resolution_lag"].rolling(240, min_periods=1).median()

        grp_map = {
            "311_disorder_count_7d": "x_grp_disorder_lag",
            "311_vehicle_count_7d": "x_grp_vehicle_lag",
            "311_street_count_7d": "x_grp_street_lag",
            "311_lighting_count_7d": "x_grp_lighting_lag",
            "311_sanitation_count_7d": "x_grp_sanitation_lag",
            "311_infrastructure_count_7d": "x_grp_infrastructure_lag",
            "311_other_count_7d": "x_grp_other_lag",
            "311_mobile_count_7d": "x_src_mobile_lag",
            "311_phone_count_7d": "x_src_phone_lag",
            "311_web_count_7d": "x_src_web_lag",
            "311_other_source_count_7d": "x_src_other_lag",
        }
        for out_col, lag_col in grp_map.items():
            g[out_col] = g[lag_col].rolling(56, min_periods=1).sum().fillna(0)

        out = g[[
            "GEOID", "date", "hour_range",
            "311_request_count_3h",
            "311_request_count_24h",
            "311_request_count_7d",
            "311_request_count_14d",
            "311_unique_category_7d",
            "311_unique_subcategory_7d",
            "311_open_count_7d",
            "311_closed_count_7d",
            "311_open_ratio_7d",
            "311_avg_resolution_hours_30d",
            "311_median_resolution_hours_30d",
            "311_disorder_count_7d",
            "311_vehicle_count_7d",
            "311_street_count_7d",
            "311_lighting_count_7d",
            "311_sanitation_count_7d",
            "311_infrastructure_count_7d",
            "311_other_count_7d",
            "311_mobile_count_7d",
            "311_phone_count_7d",
            "311_web_count_7d",
            "311_other_source_count_7d",
        ]].copy()

        results.append(out)

    feat = pd.concat(results, ignore_index=True) if results else pd.DataFrame(columns=base_cols)
    feat["GEOID"] = normalize_geoid(feat["GEOID"], DEFAULT_GEOID_LEN)

    num_cols = [c for c in feat.columns if c not in ["GEOID", "date", "hour_range"]]
    for c in num_cols:
        feat[c] = pd.to_numeric(feat[c], errors="coerce").fillna(0)

    return feat.sort_values(["GEOID", "date", "hour_range"]).reset_index(drop=True)

# =============================================================================
# CRIME MERGE
# =============================================================================
def load_crime_input() -> pd.DataFrame:
    p_parquet = os.path.join(BASE_DIR, CRIME_IN_PARQUET)
    p_csv = os.path.join(BASE_DIR, CRIME_IN_CSV)

    if os.path.exists(p_parquet):
        log(f"📥 Crime input parquet: {p_parquet}")
        return pd.read_parquet(p_parquet)

    if os.path.exists(p_csv):
        log(f"📥 Crime input csv: {p_csv}")
        return pd.read_csv(p_csv, low_memory=False)

    raise FileNotFoundError(f"Crime input bulunamadı: {p_parquet} / {p_csv}")

def standardize_crime_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]

    if "GEOID" not in df.columns:
        raise ValueError("❌ Crime input içinde GEOID yok.")

    df["GEOID"] = normalize_geoid(df["GEOID"], DEFAULT_GEOID_LEN)

    # date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    elif "datetime" in df.columns:
        dt = parse_dt_utc(df["datetime"])
        if SF_TZ is not None:
            dt = dt.dt.tz_convert(SF_TZ)
        df["date"] = dt.dt.date
    else:
        raise ValueError("❌ Crime input içinde date/datetime yok.")

    # hour_range
    if "hour_range" not in df.columns:
        if "event_hour" in df.columns:
            hr = pd.to_numeric(df["event_hour"], errors="coerce").fillna(0).astype(int)
            df["hour_range"] = make_hour_range_from_hour(hr)
        else:
            raise ValueError("❌ Crime input içinde hour_range/event_hour yok.")

    df["hour_range"] = df["hour_range"].astype(str).str.replace(r"^21-00$", "21-24", regex=True)
    return df

def merge_311_features_to_crime(crime: pd.DataFrame, feat: pd.DataFrame) -> pd.DataFrame:
    crime = standardize_crime_keys(crime)
    feat = feat.copy()

    if feat.empty:
        log("⚠️ 311 feature boş → passthrough 0 feature uygulanacak.")
        feature_cols = [
            "311_request_count_3h",
            "311_request_count_24h",
            "311_request_count_7d",
            "311_request_count_14d",
            "311_unique_category_7d",
            "311_unique_subcategory_7d",
            "311_open_count_7d",
            "311_closed_count_7d",
            "311_open_ratio_7d",
            "311_avg_resolution_hours_30d",
            "311_median_resolution_hours_30d",
            "311_disorder_count_7d",
            "311_vehicle_count_7d",
            "311_street_count_7d",
            "311_lighting_count_7d",
            "311_sanitation_count_7d",
            "311_infrastructure_count_7d",
            "311_other_count_7d",
            "311_mobile_count_7d",
            "311_phone_count_7d",
            "311_web_count_7d",
            "311_other_source_count_7d",
        ]
        for c in feature_cols:
            if c not in crime.columns:
                crime[c] = 0
        return crime

    feat["date"] = pd.to_datetime(feat["date"], errors="coerce").dt.date
    feat["GEOID"] = normalize_geoid(feat["GEOID"], DEFAULT_GEOID_LEN)
    feat["hour_range"] = feat["hour_range"].astype(str)

    keys = ["GEOID", "date", "hour_range"]
    overlap = (set(crime.columns) & set(feat.columns)) - set(keys)
    if overlap:
        feat = feat.drop(columns=list(overlap), errors="ignore")

    before_shape = crime.shape
    merged = crime.merge(feat, on=keys, how="left")
    log(f"🔗 crime ⨯ 311 features: {before_shape} → {merged.shape}")

    # NaN -> 0
    feature_cols = [c for c in merged.columns if c.startswith("311_")]
    for c in feature_cols:
        merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0)

    return merged

# =============================================================================
# ANA
# =============================================================================
def main():
    log("============================================================")
    log("🚀 update_311.py başladı")
    log(f"📂 BASE_DIR = {os.path.abspath(BASE_DIR)}")

    raw_parquet_path = os.path.join(BASE_DIR, RAW_311_PARQUET)
    raw_csv_path     = os.path.join(BASE_DIR, RAW_311_CSV)
    agg_parquet_path = os.path.join(BASE_DIR, AGG_311_PARQUET)
    agg_csv_path     = os.path.join(BASE_DIR, AGG_311_CSV)
    feat_parquet_path = os.path.join(BASE_DIR, FEAT_311_PARQUET)
    feat_csv_path     = os.path.join(BASE_DIR, FEAT_311_CSV)
    crime_out_parquet = os.path.join(BASE_DIR, CRIME_OUT_PARQUET)
    crime_out_csv     = os.path.join(BASE_DIR, CRIME_OUT_CSV)

    # 1) Seed / mevcut ham
    df_existing = load_existing_raw_or_seed()
    df_existing = standardize_raw_schema(df_existing)
    df_existing = dedupe_raw(df_existing)
    log_shape(df_existing, "311 mevcut ham")

    # 2) Başlangıç tarihi
    start_date, mode = decide_start_date(df_existing)

    # 3) İndir
    df_new = download_by_date_chunks(start_date)
    if df_new.empty:
        log("ℹ️ Yeni 311 kaydı indirilemedi / boş döndü.")
    else:
        log_shape(df_new, "311 yeni indirilen ham")
        df_new = standardize_raw_schema(df_new)

        # 4) GEOID eşleme
        if "GEOID" not in df_new.columns or df_new["GEOID"].isna().all():
            log("🧭 GEOID eşleme başlıyor...")
            df_new = geotag_to_geoid11(df_new)

        df_new["GEOID"] = normalize_geoid(df_new["GEOID"], DEFAULT_GEOID_LEN)
        log_shape(df_new, "311 yeni indirilen + şema + GEOID")

    # 5) Birleştir / tekilleştir / 5y pencere
    if df_existing.empty:
        df_raw = df_new.copy() if not df_new.empty else pd.DataFrame()
    elif df_new.empty:
        df_raw = df_existing.copy()
    else:
        df_raw = pd.concat([df_existing, df_new], ignore_index=True)

    df_raw = standardize_raw_schema(df_raw)
    df_raw = dedupe_raw(df_raw)

    if not df_raw.empty:
        min_date = DEFAULT_START if BACKFILL_DAYS <= 0 else (TODAY - timedelta(days=BACKFILL_DAYS))
        df_raw = df_raw[pd.to_datetime(df_raw["sf_date"], errors="coerce") >= pd.Timestamp(min_date)].copy()

    log_shape(df_raw, "311 final ham (5y pencere sonrası)")

    # 6) Ham kaydet
    save_both(df_raw, raw_parquet_path, raw_csv_path)
    log(f"✅ Ham 311 yazıldı: {raw_parquet_path}")
    if WRITE_CSV:
        log(f"✅ Ham 311 CSV yazıldı: {raw_csv_path}")

    # 7) 3h agg
    agg = build_agg_3h(df_raw)
    log_shape(agg, "311 agg 3h")
    save_both(agg, agg_parquet_path, agg_csv_path)
    log(f"✅ 311 agg yazıldı: {agg_parquet_path}")

    # 8) Gelişmiş features
    feat = build_311_features_3h(df_raw)
    log_shape(feat, "311 features 3h")
    save_both(feat, feat_parquet_path, feat_csv_path)
    log(f"✅ 311 features yazıldı: {feat_parquet_path}")

    # 9) Crime merge
    try:
        crime = load_crime_input()
        log_shape(crime, "Crime input")
        merged = merge_311_features_to_crime(crime, feat)
        log_shape(merged, "Crime + 311 features")
        save_both(merged, crime_out_parquet, crime_out_csv)
        log(f"✅ Crime merge yazıldı: {crime_out_parquet}")
    except Exception as e:
        log(f"⚠️ Crime merge atlandı / hata: {e}")

    # 10) Kısa kalite özeti
    try:
        qa = {
            "mode": mode,
            "raw_rows": int(len(df_raw)),
            "agg_rows": int(len(agg)),
            "feat_rows": int(len(feat)),
            "raw_min_sf_date": str(pd.to_datetime(df_raw["sf_date"], errors="coerce").min().date()) if not df_raw.empty else None,
            "raw_max_sf_date": str(pd.to_datetime(df_raw["sf_date"], errors="coerce").max().date()) if not df_raw.empty else None,
            "geoid_nonnull_rate": float(df_raw["GEOID"].notna().mean()) if not df_raw.empty else 0.0,
        }
        qa_path = os.path.join(BASE_DIR, "sf_311_update_audit.json")
        with open(qa_path, "w", encoding="utf-8") as f:
            json.dump(qa, f, ensure_ascii=False, indent=2)
        log(f"📝 Audit yazıldı: {qa_path}")
        log(json.dumps(qa, ensure_ascii=False, indent=2))
    except Exception as e:
        log(f"⚠️ Audit yazılamadı: {e}")

    log("🏁 update_311.py tamamlandı")
    log("============================================================")

if __name__ == "__main__":
    main()
