# update_911.py
# =========================================================
# SAN FRANCISCO 911 -> DAILY UPDATE (INCREMENTAL / PATCHED)
# ---------------------------------------------------------
# GEREKLİ ÖN KOŞUL
#   initial_build_911.py en az 1 kez çalışmış olmalı.
#
# YAPTIĞI İŞ
#   1) Mevcut sf_911_last_5_year.* dosyasını okur
#   2) Son tarihi bulur
#   3) API'den sadece son kısmı tekrar indirir
#   4) GEOID üretir
#   5) Güncellenen pencere için summary patch üretir
#   6) Summary dosyasını patch eder
#   7) Past-only y feature dosyasını yeniden üretir
#   8) Crime ile GEOID + date + hour_range bazında birleştirir
#
# ÇIKTILAR
#   - sf_911_last_5_year.parquet / csv
#   - sf_911_last_5_year_y.parquet / csv
#   - sf_crime_01.parquet / csv
# =========================================================

from __future__ import annotations

import ast
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import geopandas as gpd


# =========================================================
# CONFIG
# =========================================================
DEFAULT_GEOID_LEN = 11

BASE_DIR = Path(".").resolve()
SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR

CRIME_IN_CANDIDATES = [
    BASE_DIR / "sf_crime_y.csv",
    BASE_DIR / "sf_crime_y.parquet",
    BASE_DIR / "sf_crime.csv",
    BASE_DIR / "sf_crime.parquet",
]

SUMMARY_IN_CANDIDATES = [
    BASE_DIR / "sf_911_last_5_year.parquet",
    BASE_DIR / "sf_911_last_5_year.csv",
]

SUMMARY_Y_IN_CANDIDATES = [
    BASE_DIR / "sf_911_last_5_year_y.parquet",
    BASE_DIR / "sf_911_last_5_year_y.csv",
]

SUMMARY_OUT_PARQUET = OUT_DIR / "sf_911_last_5_year.parquet"
SUMMARY_OUT_CSV = OUT_DIR / "sf_911_last_5_year.csv"
SUMMARY_Y_OUT_PARQUET = OUT_DIR / "sf_911_last_5_year_y.parquet"
SUMMARY_Y_OUT_CSV = OUT_DIR / "sf_911_last_5_year_y.csv"

CRIME_OUT_CSV = OUT_DIR / "sf_crime_01.csv"
CRIME_OUT_PARQUET = OUT_DIR / "sf_crime_01.parquet"

SF911_API_URL = "https://data.sfgov.org/resource/2zdj-bwza.json"
TRACT_SHP_URL = "https://www2.census.gov/geo/tiger/TIGER2023/TRACT/tl_2023_06_tract.zip"

APP_TOKEN = None
COUNTYFP_SF = "075"

# 5 yıllık pencere
TODAY = pd.Timestamp.today().normalize()
WINDOW_START = TODAY - pd.DateOffset(days=1825)

# update mantığı
REDOWNLOAD_BUFFER_DAYS = 14   # geç gelen / düzeltilen kayıtlar için
FEATURE_BUFFER_DAYS = 7       # lag/rolling/same-slot yeniden hesap için
API_LIMIT = 50000
MAX_RT_SECONDS = 60 * 60 * 24 * 3  # 3 gün üstü response time NaN

HOUR_ORDER = [
    "00-03", "03-06", "06-09", "09-12",
    "12-15", "15-18", "18-21", "21-24"
]
HOUR_TO_SLOT = {h: i for i, h in enumerate(HOUR_ORDER)}

KEEP_RAW_COLS = [
    "cad_number",
    "received_datetime",
    "entry_datetime",
    "dispatch_datetime",
    "enroute_datetime",
    "onscene_datetime",
    "close_datetime",
    "call_type_final",
    "call_type_final_desc",
    "priority_final",
    "priority_original",
    "agency",
    "disposition",
    "onview_flag",
    "sensitive_call",
    "intersection_point",
    "data_as_of",
    "data_updated_at",
    "data_loaded_at",
]


# =========================================================
# HELPERS
# =========================================================
def log(msg: str) -> None:
    print(msg, flush=True)


def pick_existing(candidates: List[Path]) -> Path:
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Gerekli base dosya bulunamadı. Önce initial_build_911.py çalıştırılmalı.\n"
        + "\n".join(str(x) for x in candidates)
    )


def read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def write_dual(df: pd.DataFrame, pq_path: Path, csv_path: Path) -> None:
    df.to_parquet(pq_path, index=False)
    df.to_csv(csv_path, index=False)


def normalize_geoid(s: pd.Series, length: int = DEFAULT_GEOID_LEN) -> pd.Series:
    out = s.astype("string").str.replace(r"\.0$", "", regex=True).str.strip()
    out = out.mask(out.isin(["", "nan", "None", "<NA>"]), pd.NA)
    out = out.str.zfill(length)
    return out


def normalize_hour_range_col(s: pd.Series) -> pd.Series:
    x = s.astype("string").str.strip()
    repl = {
        "0-3": "00-03",
        "3-6": "03-06",
        "6-9": "06-09",
        "9-12": "09-12",
        "12-15": "12-15",
        "15-18": "15-18",
        "18-21": "18-21",
        "21-24": "21-24",
        "00:00-03:00": "00-03",
        "03:00-06:00": "03-06",
        "06:00-09:00": "06-09",
        "09:00-12:00": "09-12",
        "12:00-15:00": "12-15",
        "15:00-18:00": "15-18",
        "18:00-21:00": "18-21",
        "21:00-24:00": "21-24",
    }
    x = x.replace(repl)
    x = x.where(x.isin(HOUR_ORDER), pd.NA)
    return x


def get_hour_range(dt_val) -> Optional[str]:
    if pd.isna(dt_val):
        return None
    h = int(pd.Timestamp(dt_val).hour)
    s = (h // 3) * 3
    e = s + 3
    return f"{s:02d}-{e:02d}"


def parse_datetime_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def extract_coords(x) -> Tuple[Optional[float], Optional[float]]:
    try:
        if pd.isna(x):
            return (None, None)

        if isinstance(x, dict):
            coords = x.get("coordinates", None)
            if isinstance(coords, list) and len(coords) >= 2:
                return coords[0], coords[1]
            return (None, None)

        d = ast.literal_eval(str(x))
        coords = d.get("coordinates", None)
        if isinstance(coords, list) and len(coords) >= 2:
            return coords[0], coords[1]
        return (None, None)
    except Exception:
        return (None, None)


def safe_seconds(delta_series: pd.Series) -> pd.Series:
    out = delta_series.dt.total_seconds()
    out = out.mask((out < 0) | (out > MAX_RT_SECONDS))
    return out.astype("float32")


def to_float32(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("float32")
    return df


def to_int32(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int32")
    return df


# =========================================================
# 1) EXISTING SUMMARY LOAD
# =========================================================
def load_existing_summary() -> pd.DataFrame:
    p = pick_existing(SUMMARY_IN_CANDIDATES)
    log(f"📥 mevcut 911 summary okunuyor: {p}")
    df = read_any(p)

    req = ["GEOID", "date", "hour_range", "slot_index", "call_count",
           "violent_count", "property_count", "disturbance_count",
           "traffic_count", "dispatch_to_onscene_mean", "dispatch_to_close_mean"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Mevcut summary dosyasında eksik kolonlar var: {missing}")

    df["GEOID"] = normalize_geoid(df["GEOID"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["hour_range"] = normalize_hour_range_col(df["hour_range"])
    df["slot_index"] = pd.to_numeric(df["slot_index"], errors="coerce").fillna(0).astype("int8")

    df = df[df["GEOID"].notna()].copy()
    df = df[df["date"].notna()].copy()
    df = df[df["hour_range"].isin(HOUR_ORDER)].copy()

    df = df.drop_duplicates(subset=["GEOID", "date", "hour_range"], keep="last").copy()
    df = df[df["date"] >= WINDOW_START].copy()

    df = to_int32(
        df,
        ["call_count", "violent_count", "property_count", "disturbance_count", "traffic_count"]
    )
    df = to_float32(
        df,
        ["dispatch_to_onscene_mean", "dispatch_to_close_mean"]
    )

    return df


# =========================================================
# 2) API DOWNLOAD (RECENT CHUNK)
# =========================================================
def download_recent_raw(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    log(f"🌐 API'den indiriliyor: {start_date.date()} -> {end_date.date()}")

    from urllib.parse import urlencode

    where_clause = (
        f"received_datetime >= '{start_date.strftime('%Y-%m-%dT00:00:00')}' "
        f"AND received_datetime <= '{end_date.strftime('%Y-%m-%dT23:59:59')}'"
    )

    dfs = []
    offset = 0

    while True:
        params = {
            "$select": ",".join(KEEP_RAW_COLS),
            "$where": where_clause,
            "$limit": API_LIMIT,
            "$offset": offset,
            "$order": "received_datetime ASC",
        }

        if APP_TOKEN:
            params["$$app_token"] = APP_TOKEN

        url = SF911_API_URL + "?" + urlencode(params)

        chunk = pd.read_json(url)

        if chunk.empty:
            break

        dfs.append(chunk)
        log(f"   ✅ çekildi: {len(chunk):,} satır | offset={offset:,}")
        offset += API_LIMIT

        if len(chunk) < API_LIMIT:
            break

    if not dfs:
        log("ℹ️ API son pencere için boş döndü.")
        return pd.DataFrame(columns=KEEP_RAW_COLS)

    raw = pd.concat(dfs, ignore_index=True)
    raw = raw[[c for c in KEEP_RAW_COLS if c in raw.columns]].copy()

    dedup_keys = [
        c for c in
        ["cad_number", "received_datetime", "call_type_final", "call_type_final_desc"]
        if c in raw.columns
    ]
    if dedup_keys:
        before = len(raw)
        raw = raw.drop_duplicates(subset=dedup_keys, keep="last").copy()
        log(f"🧹 raw duplicate temizliği: {before:,} -> {len(raw):,}")

    return raw

# =========================================================
# 3) RAW -> GEOID -> SUMMARY PATCH
# =========================================================
def build_geoid_from_raw(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame()

    df = raw_df.copy()

    dt_cols = [
        "received_datetime", "entry_datetime", "dispatch_datetime",
        "enroute_datetime", "onscene_datetime", "close_datetime",
        "data_as_of", "data_updated_at", "data_loaded_at"
    ]
    df = parse_datetime_cols(df, dt_cols)

    if "received_datetime" not in df.columns:
        raise ValueError("received_datetime kolonu yok.")

    df["date"] = pd.to_datetime(df["received_datetime"], errors="coerce").dt.normalize()
    df["hour_range"] = df["received_datetime"].apply(get_hour_range)
    df["hour_range"] = normalize_hour_range_col(df["hour_range"])
    df["slot_index"] = df["hour_range"].map(HOUR_TO_SLOT)

    coords = df["intersection_point"].apply(extract_coords)
    df["lon"] = coords.apply(lambda x: x[0])
    df["lat"] = coords.apply(lambda x: x[1])
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")

    df = df[df["lon"].between(-123.0, -122.0, inclusive="both") | df["lon"].isna()].copy()
    df = df[df["lat"].between(37.0, 38.5, inclusive="both") | df["lat"].isna()].copy()
    df = df.dropna(subset=["lon", "lat"]).copy()
    df = df[df["date"].notna()].copy()
    df = df[df["hour_range"].isin(HOUR_ORDER)].copy()

    if df.empty:
        return pd.DataFrame()

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326"
    )

    tracts = gpd.read_file(TRACT_SHP_URL)
    tracts = tracts[tracts["COUNTYFP"] == COUNTYFP_SF].copy()
    tracts = tracts.to_crs(gdf.crs)

    gdf_joined = gpd.sjoin(
        gdf,
        tracts[["GEOID", "geometry"]],
        how="left",
        predicate="intersects"
    )

    gdf_joined["GEOID"] = normalize_geoid(gdf_joined["GEOID"])
    geoid_df = gdf_joined.dropna(subset=["GEOID"]).copy()
    geoid_df = geoid_df[gdf_joined["GEOID"].str.len() == DEFAULT_GEOID_LEN].copy()
    geoid_df = geoid_df[geoid_df["GEOID"].str.startswith("06075")].copy()

    drop_cols = [c for c in ["geometry", "index_right"] if c in geoid_df.columns]
    geoid_df = geoid_df.drop(columns=drop_cols).copy()

    return geoid_df


def build_theme_flags(df: pd.DataFrame) -> pd.DataFrame:
    desc = pd.Series("", index=df.index, dtype="string")
    code = pd.Series("", index=df.index, dtype="string")

    if "call_type_final_desc" in df.columns:
        desc = df["call_type_final_desc"].astype("string").str.upper().fillna("")
    if "call_type_final" in df.columns:
        code = df["call_type_final"].astype("string").str.upper().fillna("")

    txt = (desc + " " + code).str.upper()

    df["is_violent_911"] = txt.str.contains(
        r"ASSAULT|BATTERY|FIGHT|ROBBERY|GUN|WEAPON|SHOT|SHOOT|KNIFE|STAB|DV|DOMESTIC",
        regex=True, na=False
    ).astype("int8")

    df["is_property_911"] = txt.str.contains(
        r"THEFT|LARCENY|BURGLARY|BURG|STOLEN|AUTO BOOST|AUTO THEFT|VANDAL|SHOPLIFT",
        regex=True, na=False
    ).astype("int8")

    df["is_disturbance_911"] = txt.str.contains(
        r"DISTURB|DISPUTE|NOISE|SUSP|TRESPASS|WELFARE|MENTAL|5150|SUIC",
        regex=True, na=False
    ).astype("int8")

    df["is_traffic_911"] = txt.str.contains(
        r"TRAF|TRAFFIC|VEH|ACCIDENT|COLLISION|TOW|DUI|RECKLESS",
        regex=True, na=False
    ).astype("int8")

    return df


def aggregate_recent_summary(geoid_df: pd.DataFrame) -> pd.DataFrame:
    if geoid_df.empty:
        return pd.DataFrame(columns=[
            "GEOID", "date", "hour_range", "slot_index",
            "call_count", "violent_count", "property_count",
            "disturbance_count", "traffic_count",
            "dispatch_to_onscene_mean", "dispatch_to_close_mean"
        ])

    df = geoid_df.copy()

    dt_cols = [
        "received_datetime", "entry_datetime", "dispatch_datetime",
        "enroute_datetime", "onscene_datetime", "close_datetime",
        "data_as_of", "data_updated_at", "data_loaded_at"
    ]
    df = parse_datetime_cols(df, dt_cols)

    df["GEOID"] = normalize_geoid(df["GEOID"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["hour_range"] = normalize_hour_range_col(df["hour_range"])
    df = df[df["GEOID"].notna()].copy()
    df = df[df["date"].notna()].copy()
    df = df[df["hour_range"].isin(HOUR_ORDER)].copy()
    df["slot_index"] = df["hour_range"].map(HOUR_TO_SLOT).astype("int8")

    if "cad_number" not in df.columns:
        df["cad_number"] = np.arange(len(df), dtype=np.int64)

    if {"dispatch_datetime", "onscene_datetime"}.issubset(df.columns):
        df["sec_dispatch_to_onscene"] = safe_seconds(df["onscene_datetime"] - df["dispatch_datetime"])
    else:
        df["sec_dispatch_to_onscene"] = np.nan

    if {"dispatch_datetime", "close_datetime"}.issubset(df.columns):
        df["sec_dispatch_to_close"] = safe_seconds(df["close_datetime"] - df["dispatch_datetime"])
    else:
        df["sec_dispatch_to_close"] = np.nan

    df = build_theme_flags(df)

    agg = (
        df.groupby(["GEOID", "date", "hour_range", "slot_index"], observed=True, sort=False)
          .agg(
              call_count=("cad_number", "count"),
              violent_count=("is_violent_911", "sum"),
              property_count=("is_property_911", "sum"),
              disturbance_count=("is_disturbance_911", "sum"),
              traffic_count=("is_traffic_911", "sum"),
              dispatch_to_onscene_mean=("sec_dispatch_to_onscene", "mean"),
              dispatch_to_close_mean=("sec_dispatch_to_close", "mean"),
          )
          .reset_index()
    )

    agg = to_int32(
        agg,
        ["call_count", "violent_count", "property_count", "disturbance_count", "traffic_count"]
    )
    agg = to_float32(
        agg,
        ["dispatch_to_onscene_mean", "dispatch_to_close_mean"]
    )

    return agg


# =========================================================
# 4) PATCH SUMMARY
# =========================================================
def patch_summary(existing_summary: pd.DataFrame, recent_summary: pd.DataFrame, patch_start: pd.Timestamp) -> pd.DataFrame:
    log(f"🔁 summary patch başlangıcı: {patch_start.date()}")

    keep_old = existing_summary[existing_summary["date"] < patch_start].copy()
    keep_new = recent_summary[recent_summary["date"] >= patch_start].copy()

    out = pd.concat([keep_old, keep_new], ignore_index=True)
    out = out.drop_duplicates(subset=["GEOID", "date", "hour_range"], keep="last").copy()
    out = out[out["date"] >= WINDOW_START].copy()
    out = out.sort_values(["GEOID", "date", "slot_index"]).reset_index(drop=True)

    return out


# =========================================================
# 5) REBUILD Y FEATURES
# =========================================================
def rebuild_y_features(summary_df: pd.DataFrame) -> pd.DataFrame:
    log("🧠 past-only y feature set yeniden üretiliyor...")

    panel = summary_df.copy()

    panel["GEOID"] = normalize_geoid(panel["GEOID"])
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce").dt.normalize()
    panel["hour_range"] = normalize_hour_range_col(panel["hour_range"])
    panel["slot_index"] = pd.to_numeric(panel["slot_index"], errors="coerce").fillna(0).astype("int8")

    panel = panel[panel["GEOID"].notna()].copy()
    panel = panel[panel["date"].notna()].copy()
    panel = panel[panel["hour_range"].isin(HOUR_ORDER)].copy()
    panel = panel[panel["date"] >= WINDOW_START].copy()

    panel = to_int32(
        panel,
        ["call_count", "violent_count", "property_count", "disturbance_count", "traffic_count"]
    )
    panel = to_float32(
        panel,
        ["dispatch_to_onscene_mean", "dispatch_to_close_mean"]
    )

    panel = panel.sort_values(["GEOID", "date", "slot_index"]).reset_index(drop=True)

    grp = panel.groupby("GEOID", sort=False)
    grp_slot = panel.groupby(["GEOID", "hour_range"], sort=False)

    panel["911_prev_slot"] = grp["call_count"].shift(1)
    panel["911_prev_2slot"] = grp["call_count"].shift(2)
    panel["911_prev_8slot"] = grp["call_count"].shift(8)

    panel["911_roll_1d"] = grp["call_count"].transform(
        lambda s: s.rolling(8, min_periods=1).sum().shift(1)
    )
    panel["911_roll_7d"] = grp["call_count"].transform(
        lambda s: s.rolling(56, min_periods=1).sum().shift(1)
    )

    panel["911_same_slot_prev_1d"] = grp_slot["call_count"].shift(1)
    panel["911_same_slot_prev_7d"] = grp_slot["call_count"].shift(7)
    panel["911_same_slot_roll_4"] = grp_slot["call_count"].transform(
        lambda s: s.rolling(4, min_periods=1).mean().shift(1)
    )

    roll_mean_1d = grp["call_count"].transform(
        lambda s: s.rolling(8, min_periods=2).mean().shift(1)
    )
    roll_std_1d = grp["call_count"].transform(
        lambda s: s.rolling(8, min_periods=2).std().shift(1)
    )
    panel["911_zscore_1d"] = (panel["call_count"] - roll_mean_1d) / (roll_std_1d + 1e-6)
    panel["911_spike_flag_1d"] = (panel["911_zscore_1d"] >= 2.0).astype("int8")

    for c in ["violent_count", "property_count", "disturbance_count", "traffic_count"]:
        out = "911_" + c.replace("_count", "_roll_1d")
        panel[out] = grp[c].transform(
            lambda s: s.rolling(8, min_periods=1).sum().shift(1)
        )

    panel["911_dispatch_to_onscene_roll_1d"] = grp["dispatch_to_onscene_mean"].transform(
        lambda s: s.rolling(8, min_periods=1).mean().shift(1)
    )
    panel["911_dispatch_to_close_roll_1d"] = grp["dispatch_to_close_mean"].transform(
        lambda s: s.rolling(8, min_periods=1).mean().shift(1)
    )

    float_cols = [
        "911_prev_slot",
        "911_prev_2slot",
        "911_prev_8slot",
        "911_roll_1d",
        "911_roll_7d",
        "911_same_slot_prev_1d",
        "911_same_slot_prev_7d",
        "911_same_slot_roll_4",
        "911_zscore_1d",
        "911_violent_roll_1d",
        "911_property_roll_1d",
        "911_disturbance_roll_1d",
        "911_traffic_roll_1d",
        "911_dispatch_to_onscene_roll_1d",
        "911_dispatch_to_close_roll_1d",
    ]

    for c in float_cols:
        panel[c] = (
            pd.to_numeric(panel[c], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
            .astype("float32")
        )

    panel["911_spike_flag_1d"] = panel["911_spike_flag_1d"].fillna(0).astype("int8")

    y_cols = [
        "GEOID",
        "date",
        "hour_range",
        "slot_index",
        "911_prev_slot",
        "911_prev_2slot",
        "911_prev_8slot",
        "911_roll_1d",
        "911_roll_7d",
        "911_same_slot_prev_1d",
        "911_same_slot_prev_7d",
        "911_same_slot_roll_4",
        "911_zscore_1d",
        "911_spike_flag_1d",
        "911_violent_roll_1d",
        "911_property_roll_1d",
        "911_disturbance_roll_1d",
        "911_traffic_roll_1d",
        "911_dispatch_to_onscene_roll_1d",
        "911_dispatch_to_close_roll_1d",
    ]

    y_df = panel[y_cols].copy()
    return y_df


# =========================================================
# 6) MERGE INTO CRIME
# =========================================================
def load_crime_input() -> Tuple[Path, pd.DataFrame]:
    p = pick_existing(CRIME_IN_CANDIDATES)
    log(f"📥 crime input okunuyor: {p}")
    df = read_any(p)

    req = ["GEOID", "date", "hour_range"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Crime input dosyasında eksik kolonlar var: {missing}")

    df["GEOID"] = normalize_geoid(df["GEOID"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["hour_range"] = normalize_hour_range_col(df["hour_range"])

    df = df[df["GEOID"].notna()].copy()
    df = df[df["date"].notna()].copy()
    df = df[df["hour_range"].isin(HOUR_ORDER)].copy()

    return p, df


def merge_911_into_crime(crime_df: pd.DataFrame, y_df: pd.DataFrame) -> pd.DataFrame:
    y_use_cols = [
        "GEOID",
        "date",
        "hour_range",
        "911_prev_slot",
        "911_prev_2slot",
        "911_prev_8slot",
        "911_roll_1d",
        "911_roll_7d",
        "911_same_slot_prev_1d",
        "911_same_slot_prev_7d",
        "911_same_slot_roll_4",
        "911_zscore_1d",
        "911_spike_flag_1d",
        "911_violent_roll_1d",
        "911_property_roll_1d",
        "911_disturbance_roll_1d",
        "911_traffic_roll_1d",
        "911_dispatch_to_onscene_roll_1d",
        "911_dispatch_to_close_roll_1d",
    ]
    y_small = y_df[y_use_cols].copy()

    before = len(crime_df)
    out = crime_df.merge(
        y_small,
        on=["GEOID", "date", "hour_range"],
        how="left",
        validate="m:1"
    )
    after = len(out)

    log(f"🔗 merge satır kontrolü: {before:,} -> {after:,}")

    added_cols = [c for c in y_use_cols if c not in ["GEOID", "date", "hour_range"]]
    for c in added_cols:
        if c in out.columns:
            if c == "911_spike_flag_1d":
                out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype("int8")
            else:
                out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype("float32")

    return out


# =========================================================
# MAIN
# =========================================================
def main():
    log("=========================================================")
    log("SF 911 DAILY UPDATE BAŞLADI")
    log("=========================================================")
    log(f"📂 BASE_DIR = {BASE_DIR}")
    log(f"📅 TODAY    = {TODAY.date()}")
    log(f"📅 WINDOW   = {WINDOW_START.date()} -> {TODAY.date()}")

    existing_summary = load_existing_summary()
    last_summary_date = existing_summary["date"].max()

    if pd.isna(last_summary_date):
        raise ValueError("Mevcut summary dosyasında geçerli tarih yok.")

    log(f"📌 mevcut summary son tarihi: {last_summary_date.date()}")

    patch_start = max(
        pd.Timestamp(last_summary_date) - pd.Timedelta(days=REDOWNLOAD_BUFFER_DAYS),
        pd.Timestamp(WINDOW_START)
    )
    feature_start = max(
        pd.Timestamp(patch_start) - pd.Timedelta(days=FEATURE_BUFFER_DAYS),
        pd.Timestamp(WINDOW_START)
    )

    log(f"🔁 patch_start   = {patch_start.date()}")
    log(f"🧠 feature_start = {feature_start.date()}")

    raw_recent = download_recent_raw(start_date=patch_start, end_date=TODAY)
    geoid_recent = build_geoid_from_raw(raw_recent)
    recent_summary = aggregate_recent_summary(geoid_recent)

    if recent_summary.empty:
        log("ℹ️ son pencere için yeni summary patch yok. Eski summary korunacak.")
        patched_summary = existing_summary.copy()
    else:
        patched_summary = patch_summary(existing_summary, recent_summary, patch_start=patch_start)

    patched_summary = patched_summary[patched_summary["date"] >= WINDOW_START].copy()
    patched_summary = patched_summary.sort_values(["GEOID", "date", "slot_index"]).reset_index(drop=True)

    write_dual(patched_summary, SUMMARY_OUT_PARQUET, SUMMARY_OUT_CSV)
    log(f"💾 summary güncellendi: {SUMMARY_OUT_PARQUET}")
    log(f"💾 summary güncellendi: {SUMMARY_OUT_CSV}")

    y_df = rebuild_y_features(patched_summary)
    write_dual(y_df, SUMMARY_Y_OUT_PARQUET, SUMMARY_Y_OUT_CSV)
    log(f"💾 y feature güncellendi: {SUMMARY_Y_OUT_PARQUET}")
    log(f"💾 y feature güncellendi: {SUMMARY_Y_OUT_CSV}")

    crime_path, crime_df = load_crime_input()
    crime_01 = merge_911_into_crime(crime_df, y_df)

    crime_01.to_csv(CRIME_OUT_CSV, index=False)
    crime_01.to_parquet(CRIME_OUT_PARQUET, index=False)
    log(f"💾 crime merge çıktı: {CRIME_OUT_CSV}")
    log(f"💾 crime merge çıktı: {CRIME_OUT_PARQUET}")

    log("=========================================================")
    log("✅ SF 911 DAILY UPDATE TAMAMLANDI")
    log("=========================================================")
    log(f"📏 existing_summary : {existing_summary.shape}")
    log(f"📏 recent_summary   : {recent_summary.shape}")
    log(f"📏 patched_summary  : {patched_summary.shape}")
    log(f"📏 y_df             : {y_df.shape}")
    log(f"📏 crime_df         : {crime_df.shape}")
    log(f"📏 crime_01         : {crime_01.shape}")
    log("=========================================================")


if __name__ == "__main__":
    main()
