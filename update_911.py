# -*- coding: utf-8 -*-
# =========================================================
# update_911.py
# PARQUET-FIRST + STACKING-GÜÇLÜ 911 FEATURE ENGINEERING
#
# ÜRETİR:
#   sf_911_last_5_year.parquet
#   sf_911_last_5_year_y.parquet
#   (opsiyonel) sf_911_last_5_year.csv
#   (opsiyonel) sf_911_last_5_year_y.csv
#   (opsiyonel) sf_crime_01.parquet / sf_crime_01.csv
#
# AMAÇ:
#   911 verisini GEOID-date-hour_range (3h slot) düzeyine getirip
#   stacking için base'i geçmeye yardımcı olacak temporal/anomaly/
#   same-slot/response-time/type-ratio feature'larını üretmek.
# =========================================================

import os
import re
import time
import math
import json
import requests
import warnings
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------
# LOG / HELPER
# ---------------------------------------------------------
def log(msg: str):
    print(msg, flush=True)

def load_neighbor_file() -> Optional[Path]:
    for p in NEIGHBOR_CANDIDATES:
        if p.exists() and p.is_file() and p.stat().st_size > 50:
            log(f"🧭 Neighbor file bulundu: {p}")
            return p
    log("ℹ️ neighbors.csv bulunamadı. Neighbor feature'lar fallback ile üretilecek.")
    return None

def add_neighbor_features_fallback(summary: pd.DataFrame) -> pd.DataFrame:
    summary = summary.copy()
    summary["date"] = pd.to_datetime(summary["date"], errors="coerce").dt.date
    log("ℹ️ Neighbor fallback: aynı date-hour_range'te diğer GEOID ortalaması kullanılacak.")

    base_cols = [
        "911_geo_last1h", "911_geo_last3h", "911_geo_last6h",
        "911_geo_last24h", "911_geo_last3d", "911_geo_last7d"
    ]

    grp = summary.groupby(["date", "hour_range"], observed=True)

    for c in base_cols:
        if c not in summary.columns:
            summary[c] = 0.0

        total = grp[c].transform("sum")
        cnt = grp[c].transform("count")
        neigh = np.where(cnt > 1, (total - summary[c]) / (cnt - 1), 0)

        summary[c.replace("911_geo_", "911_neighbors_")] = (
            pd.Series(neigh, index=summary.index).fillna(0).astype("float32")
        )

    return summary
    
def ensure_parent(path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def is_lfs_pointer_file(p: Path) -> bool:
    try:
        return "git-lfs.github.com/spec/v1" in p.read_text(errors="ignore")[:200]
    except Exception:
        return False

def safe_save_csv(df: pd.DataFrame, path: str | Path):
    path = Path(path)
    ensure_parent(path)
    df.to_csv(path, index=False, encoding="utf-8-sig")

def safe_save_parquet(df: pd.DataFrame, path: str | Path):
    path = Path(path)
    ensure_parent(path)
    try:
        df.to_parquet(path, index=False)
    except Exception as e:
        log(f"❌ Parquet kaydedilemedi: {path}\n{e}")
        fallback = path.with_suffix(".csv.bak")
        df.to_csv(fallback, index=False, encoding="utf-8-sig")
        log(f"📁 CSV fallback kaydedildi: {fallback}")

def safe_save_table(df: pd.DataFrame, path: str | Path):
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        safe_save_parquet(df, path)
    else:
        safe_save_csv(df, path)

def read_table_auto(path: str | Path, usecols=None) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path, columns=usecols)
    return pd.read_csv(path, low_memory=False, usecols=usecols)

def to_date(s) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.date

def normalize_geoid(s, width=11) -> pd.Series:
    x = pd.Series(s, copy=False).astype("string").str.strip()
    x = x.replace({"nan": pd.NA, "None": pd.NA, "": pd.NA})
    x = x.where(x.notna(), pd.NA)
    x = x.str.replace(r"\.0$", "", regex=True)
    x = x.str.replace(r"\D", "", regex=True)
    x = x.where(x.str.len() > 0, pd.NA)
    x = x.str.zfill(width)
    x = x.where(x.str.len() == width, pd.NA)
    return x

def log_shape(df: pd.DataFrame, name: str):
    log(f"📐 {name}: {df.shape}")

def log_nan_report(df: pd.DataFrame, name: str, top_n: int = 20):
    total_nan = int(df.isna().sum().sum())
    cols_with_nan = df.isna().sum()
    cols_with_nan = cols_with_nan[cols_with_nan > 0].sort_values(ascending=False)

    log(f"🧪 {name} toplam NaN sayısı: {total_nan:,}")
    log(f"🧪 {name} NaN içeren kolon sayısı: {len(cols_with_nan):,}")

    if len(cols_with_nan) == 0:
        log(f"✅ {name}: NaN yok.")
        return

    log(f"🧪 {name} en çok NaN içeren ilk {min(top_n, len(cols_with_nan))} kolon:")
    for col, cnt in cols_with_nan.head(top_n).items():
        pct = (cnt / len(df)) * 100 if len(df) else 0
        log(f"   - {col}: {cnt:,} (%{pct:.2f})")

def log_merge_quality(crime_before: pd.DataFrame, merged: pd.DataFrame):
    log(f"📏 crime input satır/sütun   : {crime_before.shape}")
    log(f"📏 sf_crime_01 satır/sütun   : {merged.shape}")

    row_diff = len(crime_before) - len(merged)
    if row_diff == 0:
        log("✅ Merge sonrası satır kaybı yok.")
    elif row_diff > 0:
        log(f"⚠️ Merge sonrası satır kaybı var: {row_diff:,}")
    else:
        log(f"ℹ️ Merge sonrası satır arttı: {abs(row_diff):,} (duplicate key olabilir)")
def safe_div(a, b):
    a = pd.Series(a)
    b = pd.Series(b)
    out = np.where((b.isna()) | (b == 0), 0, a / b)
    return pd.Series(out, index=a.index, dtype="float32")

# ---------------------------------------------------------
# CONFIG & PATHS
# ---------------------------------------------------------
DEFAULT_GEOID_LEN = int(os.getenv("GEOID_LEN", "11"))

SCRIPT_DIR = Path(__file__).resolve().parent

_raw_base = os.getenv("CRIME_DATA_DIR", "crime_prediction_data").strip().strip("/\\")
repo_leaf = Path.cwd().name
if not os.path.isabs(_raw_base) and Path(_raw_base).name == repo_leaf:
    _raw_base = "."
BASE_DIR = str(Path(_raw_base).resolve()) if _raw_base != "." else "."
Path(BASE_DIR).mkdir(parents=True, exist_ok=True)

OUT_DIR = Path(os.getenv("CRIME_DATA_DIR", str(Path(BASE_DIR)))).resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

log(f"📂 SCRIPT_DIR = {SCRIPT_DIR}")
log(f"📂 BASE_DIR   = {Path(BASE_DIR).resolve()}")
log(f"📂 OUT_DIR    = {OUT_DIR}")

# parquet-first 911 summary names
LOCAL_NAME = "sf_911_last_5_year.parquet"
Y_NAME = "sf_911_last_5_year_y.parquet"
LOCAL_NAME_CSV = "sf_911_last_5_year.csv"
Y_NAME_CSV = "sf_911_last_5_year_y.csv"

local_summary_path = OUT_DIR / LOCAL_NAME
y_summary_path = OUT_DIR / Y_NAME

# merge output
merged_output_path = Path(os.getenv("DAILY_OUT", str(OUT_DIR / "sf_crime_01.csv")))
if not merged_output_path.is_absolute():
    merged_output_path = OUT_DIR / merged_output_path.name
merged_output_parquet_path = merged_output_path.with_suffix(".parquet")

log(f"🧾 DAILY_OUT seen as: {os.getenv('DAILY_OUT', '(unset)')}")
log(f"📝 Writing sf_crime_01 → {merged_output_path}")
log(f"📁 911 summary target  → {local_summary_path}")
log(f"📁 911 y-summary target → {y_summary_path}")

# local source candidates
LOCAL_911_CANDIDATES = []
_seen = set()
for p in [
    SCRIPT_DIR / "sf_911_last_5_year.parquet",
    OUT_DIR / "sf_911_last_5_year.parquet",
    Path(BASE_DIR) / "sf_911_last_5_year.parquet",
]:
    p = p.resolve()
    if p not in _seen:
        LOCAL_911_CANDIDATES.append(p)
        _seen.add(p)

log("🔎 911 local candidate search order:")
for p in LOCAL_911_CANDIDATES:
    exists = "✅" if p.exists() else "❌"
    log(f"   {exists} {p}")

NEIGHBOR_CANDIDATES = [
    SCRIPT_DIR / "neighbors.csv",
    OUT_DIR / "neighbors.csv",
    Path(BASE_DIR) / "neighbors.csv",
    Path("./neighbors.csv"),
]

# optional release/raw URL
RAW_911_URL = ""

# crime input candidates
CRIME_IN_ENV = os.getenv("CRIME_IN", "").strip()
CRIME_INPUT_CANDIDATES = [
    Path(CRIME_IN_ENV) if CRIME_IN_ENV else None,

    SCRIPT_DIR / "sf_crime_grid_full_labeled.parquet",
    SCRIPT_DIR / "sf_crime_grid_full_labeled.csv",
    OUT_DIR / "sf_crime_grid_full_labeled.parquet",
    OUT_DIR / "sf_crime_grid_full_labeled.csv",
    Path(BASE_DIR) / "sf_crime_grid_full_labeled.parquet",
    Path(BASE_DIR) / "sf_crime_grid_full_labeled.csv",

    SCRIPT_DIR / "sf_crime_00.parquet",
    SCRIPT_DIR / "sf_crime_00.csv",
    OUT_DIR / "sf_crime_00.parquet",
    OUT_DIR / "sf_crime_00.csv",
    Path(BASE_DIR) / "sf_crime_00.parquet",
    Path(BASE_DIR) / "sf_crime_00.csv",
]
CRIME_INPUT_CANDIDATES = [p.resolve() for p in CRIME_INPUT_CANDIDATES if p is not None]

# ---------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------
HOUR_ORDER = ["00-03", "03-06", "06-09", "09-12", "12-15", "15-18", "18-21", "21-24"]
hour_map = {h: i for i, h in enumerate(HOUR_ORDER)}

# ---------------------------------------------------------
# INPUT DISCOVERY
# ---------------------------------------------------------
def ensure_local_911_base() -> Optional[Path]:
    def _ok(p: Path) -> bool:
        try:
            return (
                p.exists()
                and p.is_file()
                and p.suffix.lower() == ".parquet"
                and p.stat().st_size > 200
            )
        except Exception:
            return False

    for p in LOCAL_911_CANDIDATES:
        if _ok(p):
            log(f"📦 911 base bulundu: {p}")
            return p

    log("ℹ️ Sadece sf_911_last_5_year.parquet arandı ama bulunamadı.")
    return None

def ensure_crime_input() -> Optional[Path]:
    for p in CRIME_INPUT_CANDIDATES:
        if p.exists() and p.is_file() and p.stat().st_size > 200:
            log(f"📦 Crime input bulundu: {p}")
            return p
    log("ℹ️ Crime input bulunamadı. 911 summary yine üretilecek.")
    return None

# ---------------------------------------------------------
# RAW/EVENT -> STANDARD SUMMARY
# ---------------------------------------------------------
def make_standard_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Girdi:
      - ham 911 event-level geoid/raw tablo
      - veya zaten özet tablo
    Çıktı:
      - GEOID + date + hour_range bazlı full-grid summary
      - legacy + yeni stacking feature'lar
    """
    log_shape(df, "911 input raw/summary")

    # ----------------------------
    # zaten summary ise normalize et
    # ----------------------------
    if {"date", "hour_range"}.issubset(df.columns) and (
        "911_request_count_hour_range" in df.columns or "call_count" in df.columns
    ):
        log("ℹ️ Girdi zaten summary gibi görünüyor, kolonlar normalize edilecek.")
        if "call_count" not in df.columns and "911_request_count_hour_range" in df.columns:
            df["call_count"] = pd.to_numeric(df["911_request_count_hour_range"], errors="coerce").fillna(0)
        if "911_request_count_hour_range" not in df.columns and "call_count" in df.columns:
            df["911_request_count_hour_range"] = pd.to_numeric(df["call_count"], errors="coerce").fillna(0)

        df["date"] = to_date(df["date"])
        df["GEOID"] = normalize_geoid(df["GEOID"], DEFAULT_GEOID_LEN)
        df["hour_range"] = df["hour_range"].astype("string").str.strip()
        df = df[df["GEOID"].notna() & df["date"].notna() & df["hour_range"].isin(HOUR_ORDER)].copy()
        if "slot_index" not in df.columns:
            df["slot_index"] = df["hour_range"].map(hour_map)

        # günlük count yoksa üret
        if "911_request_count_daily(before_24_hours)" not in df.columns:
            day = (
                df.groupby(["GEOID", "date"], dropna=False, observed=True)["911_request_count_hour_range"]
                .sum()
                .reset_index(name="911_request_count_daily(before_24_hours)")
            )
            df = df.merge(day, on=["GEOID", "date"], how="left")

        return df

    # ----------------------------
    # ham event-level'dan özet üret
    # ----------------------------
    dt_cols = [
        "received_datetime", "entry_datetime", "dispatch_datetime",
        "enroute_datetime", "onscene_datetime", "close_datetime",
        "data_as_of", "data_updated_at", "data_loaded_at"
    ]
    for c in dt_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    # GEOID garanti
    if "GEOID" not in df.columns:
        raise ValueError("❌ 911 input'ta GEOID yok. Önce GEOID üretimi yapılmalı.")
    df["GEOID"] = normalize_geoid(df["GEOID"], DEFAULT_GEOID_LEN)

    # date/hour_range üret
    if "date" not in df.columns:
        df["date"] = pd.to_datetime(df["received_datetime"], errors="coerce").dt.date
    else:
        df["date"] = to_date(df["date"])

    def _fmt_hour_range(x):
        if pd.isna(x):
            return None
        x = str(x).strip()
        m = re.match(r"^\s*(\d{1,2})\s*-\s*(\d{1,2})\s*$", x)
        if m:
            a = int(m.group(1)) % 24
            b = int(m.group(2))
            b = b if b > a else min(a + 3, 24)
            return f"{a:02d}-{b:02d}"
        return None

    if "hour_range" in df.columns:
        df["hour_range"] = df["hour_range"].apply(_fmt_hour_range)
    else:
        if "received_datetime" not in df.columns:
            raise ValueError("❌ Ne hour_range ne received_datetime var.")
        hrs = pd.to_datetime(df["received_datetime"], errors="coerce").dt.hour
        starts = (hrs // 3) * 3
        df["hour_range"] = starts.apply(lambda h: f"{int(h):02d}-{min(int(h)+3,24):02d}" if pd.notna(h) else None)

    df["slot_index"] = df["hour_range"].map(hour_map)

    before = len(df)
    df = df[
        df["GEOID"].notna() &
        df["date"].notna() &
        df["hour_range"].isin(HOUR_ORDER)
    ].copy()
    dropped = before - len(df)
    if dropped:
        log(f"🧹 911 ham veri: key eksik/bozuk satır atıldı: {dropped:,}")

    # boolean / categorical cleanup
    if "sensitive_call" in df.columns:
        df["sensitive_call"] = (
            df["sensitive_call"]
            .astype(str).str.strip().str.lower()
            .map({"true": 1, "false": 0, "1": 1, "0": 0})
            .fillna(0).astype("int8")
        )
    else:
        df["sensitive_call"] = 0

    for c in ["agency", "onview_flag", "disposition", "priority_final", "priority_original",
              "call_type_final", "call_type_final_desc"]:
        if c not in df.columns:
            df[c] = ""
        df[c] = df[c].astype("string").fillna("").str.strip()

    # ----------------------------
    # event-level response time
    # ----------------------------
    df["sec_entry_to_dispatch"] = (df["dispatch_datetime"] - df["entry_datetime"]).dt.total_seconds()
    df["sec_dispatch_to_enroute"] = (df["enroute_datetime"] - df["dispatch_datetime"]).dt.total_seconds()
    df["sec_dispatch_to_onscene"] = (df["onscene_datetime"] - df["dispatch_datetime"]).dt.total_seconds()
    df["sec_enroute_to_onscene"] = (df["onscene_datetime"] - df["enroute_datetime"]).dt.total_seconds()
    df["sec_onscene_to_close"] = (df["close_datetime"] - df["onscene_datetime"]).dt.total_seconds()
    df["sec_dispatch_to_close"] = (df["close_datetime"] - df["dispatch_datetime"]).dt.total_seconds()

    event_time_cols = [
        "sec_entry_to_dispatch",
        "sec_dispatch_to_enroute",
        "sec_dispatch_to_onscene",
        "sec_enroute_to_onscene",
        "sec_onscene_to_close",
        "sec_dispatch_to_close",
    ]
    for c in event_time_cols:
        if c in df.columns:
            df.loc[(df[c] < 0) | (df[c] > 60 * 60 * 24 * 3), c] = np.nan

    # ----------------------------
    # event-level flags / types
    # ----------------------------
    df["is_police_agency"] = (df["agency"].str.lower() == "police").astype("int8")
    df["is_mta_agency"] = df["agency"].str.lower().str.contains("transportation", na=False).astype("int8")
    df["is_sheriff_agency"] = (df["agency"].str.lower() == "sheriff").astype("int8")

    df["is_onview"] = (df["onview_flag"].str.upper() == "Y").astype("int8")
    df["is_hsoc"] = (df["onview_flag"].str.upper() == "HSOC").astype("int8")

    df["priority_is_A"] = (df["priority_final"].str.upper() == "A").astype("int8")
    df["priority_is_B"] = (df["priority_final"].str.upper() == "B").astype("int8")
    df["priority_is_C"] = (df["priority_final"].str.upper() == "C").astype("int8")

    disp = df["disposition"].str.upper()
    df["disp_han"] = (disp == "HAN").astype("int8")
    df["disp_utl"] = (disp == "UTL").astype("int8")
    df["disp_adv"] = (disp == "ADV").astype("int8")
    df["disp_arr"] = (disp == "ARR").astype("int8")

    call_desc = df["call_type_final_desc"].str.upper().fillna("")
    df["type_traffic"] = call_desc.str.contains("TRAF|TRAFFIC|VEH|TOW|CITE", na=False).astype("int8")
    df["type_assault"] = call_desc.str.contains("ASSAULT|BATTERY|FIGHT", na=False).astype("int8")
    df["type_theft"] = call_desc.str.contains("THEFT|LARCENY|STOLEN|BURG|BURGLARY|ROBBERY|SHOPLIFT", na=False).astype("int8")
    df["type_fraud"] = call_desc.str.contains("FRAUD|SCAM|FORGERY", na=False).astype("int8")
    df["type_disturbance"] = call_desc.str.contains("DISTURB|NOISE|DISPUTE|SUSP|TRESPASS", na=False).astype("int8")
    df["type_domestic"] = call_desc.str.contains("DV|DOMESTIC", na=False).astype("int8")
    df["type_mental_health"] = call_desc.str.contains("MENTAL|5150|SUIC|WELFARE", na=False).astype("int8")
    df["type_weapon"] = call_desc.str.contains("GUN|WEAPON|SHOT|SHOOT|KNIFE", na=False).astype("int8")

    # ----------------------------
    # aggregate
    # ----------------------------
    agg = df.groupby(["GEOID", "date", "hour_range", "slot_index"], dropna=False).agg(
        call_count=("GEOID", "size"),
        unique_call_type=("call_type_final", "nunique"),
        unique_call_desc=("call_type_final_desc", "nunique"),
        unique_priority=("priority_final", "nunique"),
        unique_disposition=("disposition", "nunique"),

        police_calls=("is_police_agency", "sum"),
        mta_calls=("is_mta_agency", "sum"),
        sheriff_calls=("is_sheriff_agency", "sum"),

        sensitive_calls=("sensitive_call", "sum"),
        onview_calls=("is_onview", "sum"),
        hsoc_calls=("is_hsoc", "sum"),

        priority_A_calls=("priority_is_A", "sum"),
        priority_B_calls=("priority_is_B", "sum"),
        priority_C_calls=("priority_is_C", "sum"),

        disp_han_count=("disp_han", "sum"),
        disp_utl_count=("disp_utl", "sum"),
        disp_adv_count=("disp_adv", "sum"),
        disp_arr_count=("disp_arr", "sum"),

        type_traffic_count=("type_traffic", "sum"),
        type_assault_count=("type_assault", "sum"),
        type_theft_count=("type_theft", "sum"),
        type_fraud_count=("type_fraud", "sum"),
        type_disturbance_count=("type_disturbance", "sum"),
        type_domestic_count=("type_domestic", "sum"),
        type_mental_health_count=("type_mental_health", "sum"),
        type_weapon_count=("type_weapon", "sum"),

        sec_entry_to_dispatch_mean=("sec_entry_to_dispatch", "mean"),
        sec_dispatch_to_enroute_mean=("sec_dispatch_to_enroute", "mean"),
        sec_dispatch_to_onscene_mean=("sec_dispatch_to_onscene", "mean"),
        sec_enroute_to_onscene_mean=("sec_enroute_to_onscene", "mean"),
        sec_onscene_to_close_mean=("sec_onscene_to_close", "mean"),
        sec_dispatch_to_close_mean=("sec_dispatch_to_close", "mean"),

        sec_entry_to_dispatch_median=("sec_entry_to_dispatch", "median"),
        sec_dispatch_to_onscene_median=("sec_dispatch_to_onscene", "median"),
        sec_dispatch_to_close_median=("sec_dispatch_to_close", "median"),
    ).reset_index()

    log_shape(agg, "911 aggregate")

    # ratios
    ratio_pairs = {
        "police_ratio": "police_calls",
        "mta_ratio": "mta_calls",
        "sheriff_ratio": "sheriff_calls",
        "sensitive_ratio": "sensitive_calls",
        "onview_ratio": "onview_calls",
        "hsoc_ratio": "hsoc_calls",

        "priority_A_ratio": "priority_A_calls",
        "priority_B_ratio": "priority_B_calls",
        "priority_C_ratio": "priority_C_calls",

        "disp_han_ratio": "disp_han_count",
        "disp_utl_ratio": "disp_utl_count",
        "disp_adv_ratio": "disp_adv_count",
        "disp_arr_ratio": "disp_arr_count",

        "type_traffic_ratio": "type_traffic_count",
        "type_assault_ratio": "type_assault_count",
        "type_theft_ratio": "type_theft_count",
        "type_fraud_ratio": "type_fraud_count",
        "type_disturbance_ratio": "type_disturbance_count",
        "type_domestic_ratio": "type_domestic_count",
        "type_mental_health_ratio": "type_mental_health_count",
        "type_weapon_ratio": "type_weapon_count",
    }
    for new_col, base_col in ratio_pairs.items():
        agg[new_col] = safe_div(agg[base_col], agg["call_count"])

    # ----------------------------
    # full grid
    # ----------------------------
    all_geoids = sorted(agg["GEOID"].dropna().unique().tolist())
    all_dates = sorted(pd.to_datetime(agg["date"]).dt.date.unique().tolist())

    grid = pd.MultiIndex.from_product(
        [all_geoids, all_dates, HOUR_ORDER],
        names=["GEOID", "date", "hour_range"]
    ).to_frame(index=False)
    grid["slot_index"] = grid["hour_range"].map(hour_map)

    panel = grid.merge(agg, on=["GEOID", "date", "hour_range", "slot_index"], how="left")

    count_like_cols = [
        "call_count", "unique_call_type", "unique_call_desc", "unique_priority", "unique_disposition",
        "police_calls", "mta_calls", "sheriff_calls",
        "sensitive_calls", "onview_calls", "hsoc_calls",
        "priority_A_calls", "priority_B_calls", "priority_C_calls",
        "disp_han_count", "disp_utl_count", "disp_adv_count", "disp_arr_count",
        "type_traffic_count", "type_assault_count", "type_theft_count", "type_fraud_count",
        "type_disturbance_count", "type_domestic_count", "type_mental_health_count", "type_weapon_count",
    ]
    for col in count_like_cols:
        if col in panel.columns:
            panel[col] = panel[col].fillna(0)

    for col in ratio_pairs.keys():
        if col in panel.columns:
            panel[col] = panel[col].fillna(0.0)

    time_cols = [
        "sec_entry_to_dispatch_mean",
        "sec_dispatch_to_enroute_mean",
        "sec_dispatch_to_onscene_mean",
        "sec_enroute_to_onscene_mean",
        "sec_onscene_to_close_mean",
        "sec_dispatch_to_close_mean",
        "sec_entry_to_dispatch_median",
        "sec_dispatch_to_onscene_median",
        "sec_dispatch_to_close_median",
    ]
    for col in time_cols:
        if col in panel.columns:
            panel[col] = panel[col].fillna(0.0)

    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["GEOID", "date", "slot_index"]).reset_index(drop=True)

    # ----------------------------
    # new stacking features
    # ----------------------------
    grp = panel.groupby("GEOID", sort=False)

    panel["911_prev_slot"] = grp["call_count"].shift(1)
    panel["911_prev_2slot"] = grp["call_count"].shift(2)
    panel["911_prev_8slot"] = grp["call_count"].shift(8)
    panel["911_prev_16slot"] = grp["call_count"].shift(16)

    panel["911_roll_1d"] = grp["call_count"].transform(lambda s: s.rolling(8, min_periods=1).sum().shift(1))
    panel["911_roll_3d"] = grp["call_count"].transform(lambda s: s.rolling(24, min_periods=1).sum().shift(1))
    panel["911_roll_7d"] = grp["call_count"].transform(lambda s: s.rolling(56, min_periods=1).sum().shift(1))

    panel["911_unique_call_type_roll_1d"] = grp["unique_call_type"].transform(
        lambda s: s.rolling(8, min_periods=1).mean().shift(1)
    )

    panel["911_growth_prev_slot"] = (panel["call_count"] - panel["911_prev_slot"]) / (panel["911_prev_slot"] + 1.0)

    roll_mean_1d = grp["call_count"].transform(lambda s: s.rolling(8, min_periods=2).mean().shift(1))
    roll_std_1d = grp["call_count"].transform(lambda s: s.rolling(8, min_periods=2).std().shift(1))
    panel["911_zscore_1d"] = (panel["call_count"] - roll_mean_1d) / (roll_std_1d + 1e-6)
    panel["911_spike_flag_1d"] = (panel["911_zscore_1d"] >= 2.0).astype("int8")

    grp_slot = panel.groupby(["GEOID", "hour_range"], sort=False)
    panel["911_same_slot_prev_1d"] = grp_slot["call_count"].shift(1)
    panel["911_same_slot_prev_3d"] = grp_slot["call_count"].shift(3)
    panel["911_same_slot_prev_7d"] = grp_slot["call_count"].shift(7)
    panel["911_same_slot_roll_4"] = grp_slot["call_count"].transform(
        lambda s: s.rolling(4, min_periods=1).mean().shift(1)
    )

    for base_col in [
        "sec_entry_to_dispatch_mean",
        "sec_dispatch_to_enroute_mean",
        "sec_dispatch_to_onscene_mean",
        "sec_enroute_to_onscene_mean",
        "sec_onscene_to_close_mean",
        "sec_dispatch_to_close_mean",
    ]:
        if base_col in panel.columns:
            panel[f"{base_col}_prev_slot"] = grp[base_col].shift(1)
            panel[f"{base_col}_roll_1d"] = grp[base_col].transform(
                lambda s: s.rolling(8, min_periods=1).mean().shift(1)
            )

    # ----------------------------
    # legacy compatibility cols
    # ----------------------------
    # ana count
    panel["911_request_count_hour_range"] = panel["call_count"]

    # günlük toplam
    daily = (
        panel.groupby(["GEOID", "date"], dropna=False, observed=True)["call_count"]
        .sum()
        .reset_index(name="daily_cnt")
    )
    panel = panel.merge(daily, on=["GEOID", "date"], how="left")

    # önceki 24h günlük count (legacy isim)
    panel["911_request_count_daily(before_24_hours)"] = (
        panel.groupby("GEOID", sort=False)["daily_cnt"]
        .transform(lambda s: s.shift(1))
        .fillna(0)
    )

    # 3h slot bazlı legacy
    panel["911_geo_last3h"] = panel["911_prev_slot"]
    panel["911_geo_last6h"] = grp["call_count"].transform(lambda s: s.rolling(2, min_periods=1).sum().shift(1))
    panel["911_geo_last24h"] = panel["911_roll_1d"]
    panel["911_geo_last3d"] = panel["911_roll_3d"]
    panel["911_geo_last7d"] = panel["911_roll_7d"]

    # 1h legacy: 3h veriden birebir çıkmaz; backward compatibility için 3h sinyalinin hafif normalize hali
    panel["911_geo_last1h"] = (panel["911_geo_last3h"] / 3.0).astype("float32")

    # yardımcı
    panel["hr_cnt"] = panel["call_count"]

    # fill lag-like
    lag_like_cols = [
        "911_prev_slot", "911_prev_2slot", "911_prev_8slot", "911_prev_16slot",
        "911_roll_1d", "911_roll_3d", "911_roll_7d",
        "911_unique_call_type_roll_1d",
        "911_growth_prev_slot",
        "911_zscore_1d",
        "911_same_slot_prev_1d", "911_same_slot_prev_3d", "911_same_slot_prev_7d",
        "911_same_slot_roll_4",
        "911_geo_last1h", "911_geo_last3h", "911_geo_last6h", "911_geo_last24h", "911_geo_last3d", "911_geo_last7d",
        "911_request_count_daily(before_24_hours)", "daily_cnt", "hr_cnt",
    ]
    for c in lag_like_cols:
        if c in panel.columns:
            panel[c] = panel[c].replace([np.inf, -np.inf], np.nan).fillna(0)

    # response lag fill
    for c in panel.columns:
        if c.endswith("_prev_slot") or c.endswith("_roll_1d"):
            panel[c] = panel[c].replace([np.inf, -np.inf], np.nan).fillna(0)

    # final cleanup / dtypes
    int_like_cols = count_like_cols + ["911_spike_flag_1d"]
    for c in int_like_cols:
        if c in panel.columns:
            panel[c] = panel[c].astype("int32")

    float_like_cols = list(ratio_pairs.keys()) + time_cols + [
        "911_prev_slot", "911_prev_2slot", "911_prev_8slot", "911_prev_16slot",
        "911_roll_1d", "911_roll_3d", "911_roll_7d",
        "911_unique_call_type_roll_1d",
        "911_growth_prev_slot", "911_zscore_1d",
        "911_same_slot_prev_1d", "911_same_slot_prev_3d", "911_same_slot_prev_7d",
        "911_same_slot_roll_4",
        "911_geo_last1h", "911_geo_last3h", "911_geo_last6h", "911_geo_last24h", "911_geo_last3d", "911_geo_last7d",
        "911_request_count_daily(before_24_hours)", "daily_cnt", "hr_cnt",
    ]
    float_like_cols += [c for c in panel.columns if c.endswith("_prev_slot") or c.endswith("_roll_1d")]
    for c in sorted(set(float_like_cols)):
        if c in panel.columns:
            panel[c] = panel[c].astype("float32")

    panel["date"] = panel["date"].dt.date

    # kolon sırası
    tail_cols = [c for c in ["date", "hour_range", "GEOID"] if c in panel.columns]
    cols = [c for c in panel.columns if c not in tail_cols] + tail_cols
    panel = panel[cols]

    log_shape(panel, "911 standard summary final")
    return panel

# ---------------------------------------------------------
# OPTIONAL NEIGHBOR FEATURES
# ---------------------------------------------------------
def add_neighbor_features(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Önce neighbors.csv kullan.
    Dosya yoksa veya bozuksa fallback olarak
    aynı date-hour_range'te diğer GEOID ortalamasını kullan.
    """
    summary = summary.copy()
    summary["date"] = pd.to_datetime(summary["date"], errors="coerce").dt.date

    neighbor_path = load_neighbor_file()

    needed = [
        "GEOID", "date", "hour_range",
        "911_geo_last1h", "911_geo_last3h", "911_geo_last6h",
        "911_geo_last24h", "911_geo_last3d", "911_geo_last7d"
    ]
    for c in needed:
        if c not in summary.columns:
            summary[c] = 0.0

    if neighbor_path is None:
        return add_neighbor_features_fallback(summary)

    try:
        adj = pd.read_csv(neighbor_path, low_memory=False)

        cols_lower = {c.lower(): c for c in adj.columns}
        if "geoid" in cols_lower and "neighbor" in cols_lower:
            adj = adj.rename(columns={
                cols_lower["geoid"]: "GEOID",
                cols_lower["neighbor"]: "neighbor_GEOID",
            })
        elif "GEOID" in adj.columns and "neighbor_GEOID" in adj.columns:
            pass
        else:
            raise ValueError(
                f"neighbors.csv beklenen kolonları taşımıyor. Bulunan kolonlar: {adj.columns.tolist()}"
            )

        adj["GEOID"] = normalize_geoid(adj["GEOID"], DEFAULT_GEOID_LEN)
        adj["neighbor_GEOID"] = normalize_geoid(adj["neighbor_GEOID"], DEFAULT_GEOID_LEN)
        adj = adj.dropna(subset=["GEOID", "neighbor_GEOID"]).drop_duplicates().copy()

        log(f"🧭 Neighbor pair sayısı         : {len(adj):,}")
        log(f"🧭 Unique GEOID sayısı         : {adj['GEOID'].nunique():,}")
        log(f"🧭 Unique neighbor_GEOID sayısı: {adj['neighbor_GEOID'].nunique():,}")

        if adj.empty:
            log("⚠️ neighbors.csv boş/temizleme sonrası boş. Fallback neighbor kullanılacak.")
            return add_neighbor_features_fallback(summary)

        base = summary[[
            "GEOID", "date", "hour_range",
            "911_geo_last1h", "911_geo_last3h", "911_geo_last6h",
            "911_geo_last24h", "911_geo_last3d", "911_geo_last7d"
        ]].copy()

        base = base.groupby(
            ["GEOID", "date", "hour_range"],
            as_index=False,
            observed=True
        ).mean(numeric_only=True)

        merged = adj.merge(
            base.rename(columns={"GEOID": "neighbor_GEOID"}),
            on="neighbor_GEOID",
            how="left"
        )

        agg = merged.groupby(["GEOID", "date", "hour_range"], observed=True).agg(
            **{
                "911_neighbors_last1h": ("911_geo_last1h", "mean"),
                "911_neighbors_last3h": ("911_geo_last3h", "mean"),
                "911_neighbors_last6h": ("911_geo_last6h", "mean"),
                "911_neighbors_last24h": ("911_geo_last24h", "mean"),
                "911_neighbors_last3d": ("911_geo_last3d", "mean"),
                "911_neighbors_last7d": ("911_geo_last7d", "mean"),
            }
        ).reset_index()

        summary = summary.merge(agg, on=["GEOID", "date", "hour_range"], how="left")

        for c in [
            "911_neighbors_last1h", "911_neighbors_last3h", "911_neighbors_last6h",
            "911_neighbors_last24h", "911_neighbors_last3d", "911_neighbors_last7d"
        ]:
            if c in summary.columns:
                summary[c] = summary[c].fillna(0).astype("float32")

        return summary

    except Exception as e:
        log(f"⚠️ neighbors.csv okunamadı / işlenemedi, fallback kullanılacak: {e}")
        return add_neighbor_features_fallback(summary)

# ---------------------------------------------------------
# LOCAL / RELEASE LOADER
# ---------------------------------------------------------
def summary_from_local(path: Path | str, min_date=None) -> pd.DataFrame:
    log(f"📥 Yerel 911 tabanı okunuyor: {path}")
    df = read_table_auto(path)
    out = make_standard_summary(df)
    if min_date is not None:
        out = out[out["date"] >= min_date]
    cols_tail = [c for c in ["date", "hour_range", "GEOID"] if c in out.columns]
    cols = [c for c in out.columns if c not in cols_tail] + cols_tail
    return out[cols]

def summary_from_release(url: str, min_date=None) -> pd.DataFrame:
    log(f"⬇️ Release 911 özeti indiriliyor: {url}")
    r = requests.get(url, timeout=180)
    r.raise_for_status()

    tmp = OUT_DIR / "_tmp_911.csv"
    ensure_parent(tmp)
    tmp.write_bytes(r.content)

    df = pd.read_csv(tmp, low_memory=False)
    out = make_standard_summary(df)
    if min_date is not None:
        out = out[out["date"] >= min_date]
    cols_tail = [c for c in ["date", "hour_range", "GEOID"] if c in out.columns]
    cols = [c for c in out.columns if c not in cols_tail] + cols_tail
    return out[cols]

# ---------------------------------------------------------
# MERGE WITH CRIME GRID
# ---------------------------------------------------------
def merge_with_crime(crime_path: Path, summary: pd.DataFrame) -> pd.DataFrame:
    crime = read_table_auto(crime_path)
    crime_before = crime.copy()
    log_shape(crime, "crime input")

    if "GEOID" not in crime.columns:
        raise ValueError("❌ Crime input içinde GEOID yok.")
    crime["GEOID"] = normalize_geoid(crime["GEOID"], DEFAULT_GEOID_LEN)
    before = len(crime)
    crime = crime[crime["GEOID"].notna()].copy()
    dropped = before - len(crime)
    if dropped:
        log(f"🧹 crime grid: GEOID boş/bozuk satır atıldı: {dropped:,}")

    # hour_range / hr_key
    hr_pat = re.compile(r"^\s*(\d{1,2})\s*-\s*(\d{1,2})\s*$")
    def _hr_key_from_hr_range(x):
        m = hr_pat.match(str(x))
        return int(m.group(1)) % 24 if m else None

    if "hour_range" in crime.columns:
        crime["hour_range"] = crime["hour_range"].astype(str).str.strip()
        crime["hr_key"] = crime["hour_range"].apply(_hr_key_from_hr_range).astype("Int16")
    elif "event_hour" in crime.columns:
        crime["hr_key"] = ((pd.to_numeric(crime["event_hour"], errors="coerce").fillna(0).astype(int)) // 3) * 3
        crime["hr_key"] = crime["hr_key"].astype("Int16")
        crime["hour_range"] = crime["hr_key"].apply(lambda h: f"{int(h):02d}-{min(int(h)+3,24):02d}" if pd.notna(h) else None)
    else:
        raise ValueError("❌ Crime grid dosyasında ne 'hour_range' ne de 'event_hour' var.")

    # date
    has_date_col = ("date" in crime.columns) or ("datetime" in crime.columns)
    if has_date_col:
        if "date" not in crime.columns:
            crime["date"] = pd.to_datetime(crime["datetime"], errors="coerce").dt.date
        else:
            crime["date"] = to_date(crime["date"])

        keys = ["GEOID", "date", "hour_range"]
        overlap = (set(crime.columns) & set(summary.columns)) - set(keys)
        if overlap:
            log(f"🧹 Merge overlap (key dışı) summary'den düşürüldü: {sorted(overlap)}")
            summary = summary.drop(columns=list(overlap), errors="ignore")

        merged = crime.merge(summary, on=keys, how="left")
        log("🔗 Join modu: DATE-BASED (GEOID, date, hour_range)")
        
        if "911_request_count_hour_range" in merged.columns:
            matched = merged["911_request_count_hour_range"].notna().sum()
            unmatched = merged["911_request_count_hour_range"].isna().sum()
            log(f"🔍 911 match olan satır    : {matched:,}")
            log(f"🔍 911 match olmayan satır : {unmatched:,}")
            log(f"🔍 911 match oranı         : %{(matched / len(merged) * 100):.2f}")
    else:
        # takvim fallback
        cal_keys = ["GEOID", "hr_key", "day_of_week", "season"]
        summary2 = summary.copy()
        summary2["hr_key"] = summary2["hour_range"].apply(_hr_key_from_hr_range).astype("Int16")

        if "day_of_week" not in crime.columns:
            log("ℹ️ crime grid’de day_of_week yok → 0 atanıyor.")
            crime["day_of_week"] = 0
        if "season" not in crime.columns:
            if "month" in crime.columns:
                smap = {12:"Winter",1:"Winter",2:"Winter",3:"Spring",4:"Spring",5:"Spring",6:"Summer",7:"Summer",8:"Summer",9:"Fall",10:"Fall",11:"Fall"}
                crime["season"] = crime["month"].map(smap).fillna("Summer")
            else:
                crime["season"] = "Summer"

        agg_cols = [c for c in summary2.columns if c not in {"date", "hour_range"}]
        cal_agg = summary2.groupby(cal_keys, as_index=False, observed=True)[agg_cols].median(numeric_only=True)
        overlap = (set(crime.columns) & set(cal_agg.columns)) - set(cal_keys)
        if overlap:
            log(f"🧹 Merge overlap (calendar) summary'den düşürüldü: {sorted(overlap)}")
            cal_agg = cal_agg.drop(columns=list(overlap), errors="ignore")

        merged = crime.merge(cal_agg, on=cal_keys, how="left")
        log("🔗 Join modu: CALENDAR-BASED (GEOID, hr_key, day_of_week, season)")

    fill_cols = [
        "911_request_count_hour_range",
        "911_request_count_daily(before_24_hours)",
        "hr_cnt", "daily_cnt",
        "911_geo_last1h", "911_geo_last3h", "911_geo_last6h", "911_geo_last24h", "911_geo_last3d", "911_geo_last7d",
        "911_neighbors_last1h", "911_neighbors_last3h", "911_neighbors_last6h", "911_neighbors_last24h", "911_neighbors_last3d", "911_neighbors_last7d",
        "911_prev_slot", "911_prev_2slot", "911_prev_8slot", "911_prev_16slot",
        "911_roll_1d", "911_roll_3d", "911_roll_7d",
        "911_unique_call_type_roll_1d",
        "911_growth_prev_slot", "911_zscore_1d", "911_spike_flag_1d",
        "911_same_slot_prev_1d", "911_same_slot_prev_3d", "911_same_slot_prev_7d", "911_same_slot_roll_4",
        "police_ratio", "mta_ratio", "sheriff_ratio", "sensitive_ratio", "onview_ratio", "hsoc_ratio",
        "priority_A_ratio", "priority_B_ratio", "priority_C_ratio",
        "disp_han_ratio", "disp_utl_ratio", "disp_adv_ratio", "disp_arr_ratio",
        "type_traffic_ratio", "type_assault_ratio", "type_theft_ratio", "type_fraud_ratio",
        "type_disturbance_ratio", "type_domestic_ratio", "type_mental_health_ratio", "type_weapon_ratio",
        "sec_entry_to_dispatch_mean", "sec_dispatch_to_enroute_mean", "sec_dispatch_to_onscene_mean",
        "sec_enroute_to_onscene_mean", "sec_onscene_to_close_mean", "sec_dispatch_to_close_mean",
        "sec_entry_to_dispatch_median", "sec_dispatch_to_onscene_median", "sec_dispatch_to_close_median",
    ] + [c for c in merged.columns if c.endswith("_prev_slot") or c.endswith("_roll_1d")]

    for c in fill_cols:
        if c in merged.columns:
            merged[c] = merged[c].fillna(0)
            
    log_merge_quality(crime_before, merged)
    return merged

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    local_base = ensure_local_911_base()
    final_911 = summary_from_local(local_base)

    # neighbor features
    final_911 = add_neighbor_features(final_911)

    # save summary parquet-first
    safe_save_table(final_911, local_summary_path)
    safe_save_table(final_911, y_summary_path)

    # optional csv compatibility
    safe_save_csv(final_911, OUT_DIR / LOCAL_NAME_CSV)
    safe_save_csv(final_911, OUT_DIR / Y_NAME_CSV)

    log(f"✅ Yerel 911 özet kaydedildi → {local_summary_path}")
    log(f"✅ Y-özet kaydedildi        → {y_summary_path}")
    log_shape(final_911, "final_911")

    # merge with crime if possible
    crime_input = ensure_crime_input()
    if crime_input is None:
        log("ℹ️ Crime input bulunamadığı için sf_crime_01 üretilmedi.")
        return

    merged = merge_with_crime(crime_input, final_911)
    
    log_merge_quality(read_table_auto(crime_input), merged)
    log_shape(merged, "CRIME⨯911 merged")
    log_nan_report(merged, "sf_crime_01", top_n=25)
    
    safe_save_csv(merged, merged_output_path)
    safe_save_parquet(merged, merged_output_parquet_path)
    
    log(f"✅ sf_crime_01.csv yazıldı     → {merged_output_path}")
    log(f"✅ sf_crime_01.parquet yazıldı → {merged_output_parquet_path}")

if __name__ == "__main__":
    main()
