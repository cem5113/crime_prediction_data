#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# update_911.py — REVIZE / FEATURE-RICH / PANEL-SAFE / LEAK-AWARE
# Amaç:
#   1) sf_911_last_5_year.parquet / csv tabanını oku
#   2) Gerekirse API'den incremental çek
#   3) 911 özetini güçlü feature'larla üret
#   4) sf_crime_y.csv ile birleştirip sf_crime_01.csv yaz
#
# Not:
# - Bu sürüm özellikle stacking/base model ayrımını güçlendirmek için
#   semantic, priority, anomaly, spike, ratio ve response-time feature'ları ekler.
# - Panel-safe / geçmişe dayalı üretim yapılır; rolling feature'larda shift(1) kullanılır.

from __future__ import annotations

import os
import re
import io
import time
import json
import ast
import requests
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd

# =========================================================
# TZ / HELPERS
# =========================================================
try:
    import zoneinfo
    SF_TZ = zoneinfo.ZoneInfo("America/Los_Angeles")
except Exception:
    SF_TZ = None

def log(msg: str):
    print(msg, flush=True)

def ensure_parent(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def safe_save_csv(df: pd.DataFrame, path: str):
    ensure_parent(path)
    try:
        df.to_csv(path, index=False, encoding="utf-8-sig")
    except Exception as e:
        log(f"❌ CSV kayıt hatası: {path} | {e}")
        bak = path + ".bak"
        df.to_csv(bak, index=False, encoding="utf-8-sig")
        log(f"📁 Yedek kaydedildi: {bak}")

def safe_save_parquet(df: pd.DataFrame, path: str):
    ensure_parent(path)
    try:
        df.to_parquet(path, index=False)
    except Exception as e:
        log(f"⚠️ Parquet kayıt hatası: {path} | {e}")

def log_shape(df: pd.DataFrame, label: str):
    log(f"📊 {label}: {df.shape[0]:,} satır × {df.shape[1]} sütun")

def to_date(s):
    return pd.to_datetime(s, errors="coerce").dt.date

def is_lfs_pointer_file(p: Path) -> bool:
    try:
        return "git-lfs.github.com/spec/v1" in p.read_text(errors="ignore")[:200]
    except Exception:
        return False

def _to_sf_datetime(s):
    dt = pd.to_datetime(s, utc=True, errors="coerce")
    if SF_TZ is not None:
        try:
            dt = dt.dt.tz_convert(SF_TZ)
        except Exception:
            pass
    return dt

def _to_date_series(x):
    try:
        s = pd.to_datetime(x, utc=True, errors="coerce")
        if SF_TZ is not None:
            s = s.dt.tz_convert(SF_TZ)
        return s.dt.date.dropna()
    except Exception:
        return pd.to_datetime(x, errors="coerce").dt.date.dropna()

def log_date_range(df, date_col="date", label="DATA"):
    if date_col not in df.columns:
        log(f"⚠️ {label}: '{date_col}' yok.")
        return
    s = _to_date_series(df[date_col])
    if s.empty:
        log(f"⚠️ {label}: tarih parse edilemedi.")
        return
    log(f"🧭 {label} tarih aralığı: {s.min()} → {s.max()} | gün={s.nunique()}")

def normalize_geoid(s: pd.Series, target_len: int) -> pd.Series:
    s = s.astype(str).str.extract(r"(\d+)", expand=False)
    L = int(target_len)
    return s.str[:L].str.zfill(L)

def clean_text(x):
    if pd.isna(x):
        return ""
    x = str(x).strip().lower()
    x = re.sub(r"\s+", " ", x)
    return x

def robust_num(s, dtype="float32"):
    x = pd.to_numeric(s, errors="coerce")
    if dtype.startswith("int"):
        return x.fillna(0).astype(dtype)
    return x.astype(dtype)

def add_ratio(numer, denom):
    return numer / (denom.replace(0, np.nan) + 1e-6)

def safe_div(a, b, fill=0.0):
    out = np.divide(a, b, out=np.full_like(np.asarray(a, dtype="float64"), fill, dtype="float64"), where=(np.asarray(b) != 0))
    return out

# =========================================================
# CONFIG
# =========================================================
DEFAULT_GEOID_LEN = int(os.getenv("GEOID_LEN", "11"))

_raw_base = os.getenv("CRIME_DATA_DIR", "crime_prediction_data").strip().strip("/\\")
repo_leaf = Path.cwd().name
if not os.path.isabs(_raw_base) and Path(_raw_base).name == repo_leaf:
    _raw_base = "."
BASE_DIR = str(Path(_raw_base).resolve()) if _raw_base != "." else "."
Path(BASE_DIR).mkdir(parents=True, exist_ok=True)

OUT_DIR = Path(os.getenv("CRIME_DATA_DIR", str(Path(BASE_DIR)))).resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

log(f"📂 BASE_DIR = {Path(BASE_DIR).resolve()}")
log(f"📂 OUT_DIR  = {OUT_DIR}")

LOCAL_CSV_NAME     = "sf_911_last_5_year.csv"
LOCAL_PARQUET_NAME = "sf_911_last_5_year.parquet"
Y_NAME             = "sf_911_last_5_year_y.csv"

local_summary_csv_path     = OUT_DIR / LOCAL_CSV_NAME
local_summary_parquet_path = OUT_DIR / LOCAL_PARQUET_NAME
y_summary_path             = OUT_DIR / Y_NAME

merged_output_path = Path(os.getenv("DAILY_OUT", str(OUT_DIR / "sf_crime_01.csv")))
if not merged_output_path.is_absolute():
    merged_output_path = OUT_DIR / merged_output_path.name

log(f"📝 Writing sf_crime_01 → {merged_output_path}")

CENSUS_CANDIDATES = [
    OUT_DIR / "sf_census_blocks.geojson",
    Path(BASE_DIR) / "sf_census_blocks.geojson",
    Path("./sf_census_blocks.geojson"),
]

SF911_API_URL   = os.getenv("SF911_API_URL", "https://data.sfgov.org/resource/2zdj-bwza.json")
SF_APP_TOKEN    = os.getenv("SF911_API_TOKEN", "")
AGENCY_FILTER   = os.getenv("SF911_AGENCY_FILTER", "agency like '%Police%'")
REQUEST_TIMEOUT = int(os.getenv("SF911_REQUEST_TIMEOUT", "60"))
CHUNK_LIMIT     = int(os.getenv("SF911_CHUNK_LIMIT", "50000"))
MAX_RETRIES     = int(os.getenv("SF911_MAX_RETRIES", "2"))
SLEEP_BETWEEN_REQS = float(os.getenv("SF911_SLEEP", "0.2"))
BULK_RANGE      = os.getenv("SF911_BULK_RANGE", "1").lower() in ("1", "true", "yes", "on")
IS_V3           = "/api/v3/views/" in SF911_API_URL
V3_PAGE_LIMIT   = int(os.getenv("SF_V3_PAGE_LIMIT", "1000"))
SF911_RECENT_HOURS = int(os.getenv("SF911_RECENT_HOURS", "6"))
SF911_REINGEST_DAYS = int(os.getenv("SF911_REINGEST_DAYS", "14"))

RAW_911_URL_ENV = os.getenv("RAW_911_URL", "").strip()
RAW_911_URL_CANDIDATES = [
    RAW_911_URL_ENV or "",
    "https://github.com/cem5113/crime_prediction_data/releases/download/v1.0.1/sf_911_last_5_year.parquet",
    "https://github.com/cem5113/crime_prediction_data/releases/download/v1.0.1/sf_911_last_5_year.csv",
]

ENABLE_NEIGHBORS  = os.getenv("ENABLE_NEIGHBORS", "1").lower() in ("1", "true", "yes", "on")
NEIGHBOR_METHOD   = os.getenv("NEIGHBOR_METHOD", "touches")   # touches | radius
NEIGHBOR_RADIUS_M = float(os.getenv("NEIGHBOR_RADIUS_M", "500"))

SF_BBOX = (-123.2, 37.6, -122.3, 37.9)

# =========================================================
# GEO / BLOCKS
# =========================================================
def _load_blocks() -> tuple[gpd.GeoDataFrame, int]:
    census_path = next((p for p in CENSUS_CANDIDATES if p.exists()), None)
    if census_path is None:
        raise FileNotFoundError("❌ sf_census_blocks.geojson bulunamadı.")

    gdf_blocks = gpd.read_file(census_path)
    if "GEOID" not in gdf_blocks.columns:
        cand = [c for c in gdf_blocks.columns if str(c).upper().startswith("GEOID")]
        if not cand:
            raise ValueError("❌ GeoJSON içinde GEOID sütunu yok.")
        gdf_blocks = gdf_blocks.rename(columns={cand[0]: "GEOID"})

    tlen = gdf_blocks["GEOID"].astype(str).str.len().mode().iat[0]
    gdf_blocks["GEOID"] = normalize_geoid(gdf_blocks["GEOID"], tlen)

    if gdf_blocks.crs is None:
        gdf_blocks = gdf_blocks.set_crs("EPSG:4326")
    elif gdf_blocks.crs.to_epsg() != 4326:
        gdf_blocks = gdf_blocks.to_crs(4326)

    return gdf_blocks, tlen

def ensure_geoid(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "GEOID" in df.columns and df["GEOID"].notna().any():
        return df

    def _parse_intersection_point(x):
        """
        Desteklenen formatlar:
        1) dict: {'coordinates': [lon, lat], 'type': 'Point'}
        2) string-dict: "{'coordinates': [-122.49, 37.78], 'type': 'Point'}"
        3) json-string: '{"coordinates": [-122.49, 37.78], "type": "Point"}'
        4) fallback regex
        """
        if pd.isna(x):
            return (None, None)

        # dict ise direkt
        if isinstance(x, dict):
            coords = x.get("coordinates")
            if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                return (coords[1], coords[0])  # lat, lon
            return (None, None)

        # string ise önce literal_eval, sonra json, en son regex
        if isinstance(x, str):
            s = x.strip()

            # 1) ast.literal_eval -> tek tırnaklı dict için en güvenlisi
            try:
                obj = ast.literal_eval(s)
                if isinstance(obj, dict) and "coordinates" in obj:
                    coords = obj.get("coordinates")
                    if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                        return (coords[1], coords[0])  # lat, lon
            except Exception:
                pass

            # 2) json.loads -> çift tırnaklı json için
            try:
                obj = json.loads(s)
                if isinstance(obj, dict) and "coordinates" in obj:
                    coords = obj.get("coordinates")
                    if isinstance(coords, (list, tuple)) and len(coords) >= 2:
                        return (coords[1], coords[0])  # lat, lon
            except Exception:
                pass

            # 3) çok kaba fallback regex
            nums = re.findall(r"-?\d+(?:\.\d+)?", s)
            if len(nums) >= 2:
                lon, lat = float(nums[0]), float(nums[1])
                return (lat, lon)

        return (None, None)

    if "latitude" not in df.columns or "longitude" not in df.columns:
        if "intersection_point" in df.columns:
            parsed = df["intersection_point"].apply(_parse_intersection_point)
            df["latitude"] = [p[0] for p in parsed]
            df["longitude"] = [p[1] for p in parsed]

        for a, b in [("lat", "long"), ("y", "x")]:
            if a in df.columns and b in df.columns and (
                "latitude" not in df.columns or "longitude" not in df.columns
            ):
                df["latitude"]  = pd.to_numeric(df[a], errors="coerce")
                df["longitude"] = pd.to_numeric(df[b], errors="coerce")
                break

    df["latitude"] = pd.to_numeric(df.get("latitude"), errors="coerce")
    df["longitude"] = pd.to_numeric(df.get("longitude"), errors="coerce")

    if "latitude" in df.columns and "longitude" in df.columns:
        min_lon, min_lat, max_lon, max_lat = SF_BBOX
        df = df[
            df["latitude"].between(min_lat, max_lat) &
            df["longitude"].between(min_lon, max_lon)
        ].copy()

    df = df.dropna(subset=["latitude", "longitude"]).copy()
    if df.empty:
        return df

    gdf_blocks, tlen = _load_blocks()
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326"
    )
    gdf = gpd.sjoin(gdf, gdf_blocks[["GEOID", "geometry"]], how="left", predicate="within")
    out = pd.DataFrame(gdf.drop(columns=["geometry", "index_right"], errors="ignore"))

    if "GEOID" not in out.columns:
        out["GEOID"] = pd.NA
        return out

    out["GEOID"] = normalize_geoid(out["GEOID"], tlen)
    log(f"🧪 spatial join sonrası GEOID dolu: {out['GEOID'].notna().sum():,} / {len(out):,}")
    out = out.dropna(subset=["GEOID"]).copy()
    return out

# =========================================================
# 911 FEATURE ENGINEERING
# =========================================================
def add_semantic_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    call_final = df["call_type_final_desc"].map(clean_text) if "call_type_final_desc" in df.columns else pd.Series("", index=df.index)
    call_orig  = df["call_type_original_desc"].map(clean_text) if "call_type_original_desc" in df.columns else pd.Series("", index=df.index)
    call_all   = (call_final + " " + call_orig).str.strip()

    pri_final  = df["priority_final"].map(clean_text) if "priority_final" in df.columns else pd.Series("", index=df.index)
    pri_orig   = df["priority_original"].map(clean_text) if "priority_original" in df.columns else pd.Series("", index=df.index)
    agency     = df["agency"].map(clean_text) if "agency" in df.columns else pd.Series("", index=df.index)
    onview     = df["onview_flag"].map(clean_text) if "onview_flag" in df.columns else pd.Series("", index=df.index)
    sensitive  = df["sensitive_call"].map(clean_text) if "sensitive_call" in df.columns else pd.Series("", index=df.index)

    # --- semantic groups ---
    violent_re = r"(?:assault|battery|fight|stabbing|shot|shooting|gun|weapon|armed|robbery|homicide|person with a gun|shots fired)"
    property_re = r"(?:burglary|theft|larceny|auto burglary|vehicle theft|stolen|breaking|shoplifting|trespass)"
    disorder_re = r"(?:disturbance|noise|disorderly|party|drunk|loitering|suspicious|harassment)"
    traffic_re = r"(?:traffic|collision|accident|hit and run|vehicle|dui)"
    narcotics_re = r"(?:drug|narcotic|overdose)"
    mental_re = r"(?:mental|emotionally disturbed|5150|suicidal|crisis)"
    weapons_re = r"(?:gun|weapon|knife|shots fired|armed)"
    domestic_re = r"(?:domestic|family disturbance)"
    alarm_re = r"(?:alarm|silent alarm|audible alarm|burglar alarm)"

    df["is_violent_911"]   = call_all.str.contains(violent_re,  case=False, regex=True).astype("int8")
    df["is_property_911"]  = call_all.str.contains(property_re, case=False, regex=True).astype("int8")
    df["is_disorder_911"]  = call_all.str.contains(disorder_re, case=False, regex=True).astype("int8")
    df["is_traffic_911"]   = call_all.str.contains(traffic_re,  case=False, regex=True).astype("int8")
    df["is_narcotics_911"] = call_all.str.contains(narcotics_re, case=False, regex=True).astype("int8")
    df["is_mental_911"]    = call_all.str.contains(mental_re, case=False, regex=True).astype("int8")
    df["is_weapons_911"]   = call_all.str.contains(weapons_re, case=False, regex=True).astype("int8")
    df["is_domestic_911"]  = call_all.str.contains(domestic_re, case=False, regex=True).astype("int8")
    df["is_alarm_911"]     = call_all.str.contains(alarm_re, case=False, regex=True).astype("int8")

    # --- priority score ---
    def map_priority(x):
        x = clean_text(x)
        # harfli veya sayılı sistem için esnek
        if x in {"a", "alpha", "1", "priority 1", "high"}:
            return 4
        if x in {"b", "bravo", "2", "priority 2", "medium high"}:
            return 3
        if x in {"c", "charlie", "3", "priority 3", "medium"}:
            return 2
        if x in {"d", "delta", "4", "priority 4", "low"}:
            return 1
        m = re.search(r"(\d+)", x)
        if m:
            n = int(m.group(1))
            if n <= 1:
                return 4
            if n == 2:
                return 3
            if n == 3:
                return 2
            return 1
        return 0

    df["priority_score_final_911"] = pri_final.apply(map_priority).astype("int8")
    df["priority_score_orig_911"]  = pri_orig.apply(map_priority).astype("int8")
    df["priority_score_911"] = df[["priority_score_final_911", "priority_score_orig_911"]].max(axis=1).astype("int8")

    # --- flags ---
    df["onview_flag_911"]    = onview.isin({"y", "yes", "true", "1", "t"}).astype("int8")
    df["sensitive_flag_911"] = sensitive.isin({"y", "yes", "true", "1", "t"}).astype("int8")
    df["is_police_agency_911"] = agency.str.contains("police", case=False, regex=False).astype("int8")

    # --- weighted severity proxy ---
    df["call_severity_proxy_911"] = (
        2.0 * df["is_violent_911"] +
        1.5 * df["is_weapons_911"] +
        1.3 * df["is_domestic_911"] +
        1.2 * df["is_property_911"] +
        1.0 * df["is_narcotics_911"] +
        0.8 * df["is_disorder_911"] +
        0.5 * df["is_alarm_911"] +
        0.3 * df["is_traffic_911"] +
        0.2 * df["sensitive_flag_911"] +
        0.5 * df["priority_score_911"]
    ).astype("float32")

    return df

def add_time_features(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()

    # zaman kolonunu seç
    ts_col = None
    for cand in [
        "received_datetime", "received_time", "entry_datetime", "dispatch_datetime",
        "datetime", "date", "call_datetime"
    ]:
        if cand in df.columns:
            ts_col = cand
            break
    if ts_col is None:
        raise ValueError("❌ 911 zaman kolonu bulunamadı.")

    df["ts"] = _to_sf_datetime(df[ts_col])
    df = df[df["ts"].notna()].copy()

    df["date"] = df["ts"].dt.date
    df["event_hour"] = df["ts"].dt.hour.astype("int8")
    start = ((df["event_hour"] // 3) * 3).astype("int8")
    df["hour_range"] = start.apply(lambda s: f"{int(s):02d}-{int(min(s+3, 24)):02d}")
    df["hr_key"] = start.astype("int8")

    df["day_of_week"] = df["ts"].dt.weekday.astype("int8")
    df["month"] = df["ts"].dt.month.astype("int8")
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype("int8")
    df["is_night_911"] = df["event_hour"].isin([0,1,2,3,4,5,22,23]).astype("int8")

    season_map = {12:"Winter",1:"Winter",2:"Winter",3:"Spring",4:"Spring",5:"Spring",
                  6:"Summer",7:"Summer",8:"Summer",9:"Fall",10:"Fall",11:"Fall"}
    df["season"] = df["month"].map(season_map).astype("category")

    return df

def add_response_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    pairs = [
        ("dispatch_datetime", "onscene_datetime", "dispatch_to_onscene_min_911"),
        ("entry_datetime", "dispatch_datetime", "entry_to_dispatch_min_911"),
        ("received_datetime", "close_datetime", "received_to_close_min_911"),
        ("dispatch_datetime", "close_datetime", "dispatch_to_close_min_911"),
    ]

    for a, b, out_col in pairs:
        if a in df.columns and b in df.columns:
            ta = _to_sf_datetime(df[a])
            tb = _to_sf_datetime(df[b])
            delta = (tb - ta).dt.total_seconds() / 60.0
            delta = delta.where((delta >= 0) & (delta <= 60 * 24 * 3))  # saçma uçları kırp
            df[out_col] = delta.astype("float32")
        else:
            df[out_col] = np.nan

    return df

def build_event_level_911(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df = add_time_features(df)
    df = add_semantic_flags(df)
    df = add_response_time_features(df)

    if "GEOID" not in df.columns:
        df["GEOID"] = pd.NA

    need_geoid = df["GEOID"].isna()
    if need_geoid.any():
        try:
            missing_idx = df.index[need_geoid]
            subset = df.loc[missing_idx].copy()

            filled = ensure_geoid(subset)

            if filled is not None and not filled.empty and "GEOID" in filled.columns:
                # index hizalı güvenli atama
                common_idx = filled.index.intersection(df.index)
                df.loc[common_idx, "GEOID"] = filled.loc[common_idx, "GEOID"]
        except Exception as e:
            log(f"⚠️ ensure_geoid başarısız: {e}")

    if "GEOID" not in df.columns:
        df["GEOID"] = pd.NA

    if "GEOID" in df.columns:
        df["GEOID"] = normalize_geoid(df["GEOID"], DEFAULT_GEOID_LEN)

    return df

def make_standard_summary(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=[
            "GEOID", "date", "hour_range",
            "911_request_count_hour_range",
            "911_request_count_daily(before_24_hours)"
        ])

    df = build_event_level_911(raw)
    log(f"🧪 latitude dolu: {df['latitude'].notna().sum():,} | longitude dolu: {df['longitude'].notna().sum():,}")
    df = df.dropna(subset=["date", "hour_range"]).copy()

    has_geoid = "GEOID" in df.columns and df["GEOID"].notna().any()
    grp_hr  = (["GEOID"] if has_geoid else []) + ["date", "hour_range", "hr_key"]
    grp_day = (["GEOID"] if has_geoid else []) + ["date"]

    # saatlik / slot bazlı agregasyon
    agg_dict = {
        "ts": "size",
        "priority_score_911": ["mean", "max"],
        "call_severity_proxy_911": ["mean", "max", "sum"],
        "is_violent_911": "sum",
        "is_property_911": "sum",
        "is_disorder_911": "sum",
        "is_traffic_911": "sum",
        "is_narcotics_911": "sum",
        "is_mental_911": "sum",
        "is_weapons_911": "sum",
        "is_domestic_911": "sum",
        "is_alarm_911": "sum",
        "onview_flag_911": "sum",
        "sensitive_flag_911": "sum",
        "is_police_agency_911": "sum",
        "is_weekend": "max",
        "is_night_911": "max",
        "dispatch_to_onscene_min_911": ["mean", "median"],
        "entry_to_dispatch_min_911": ["mean", "median"],
        "received_to_close_min_911": ["mean", "median"],
        "dispatch_to_close_min_911": ["mean", "median"],
    }

    hr = df.groupby(grp_hr, dropna=False, observed=True).agg(agg_dict)
    hr.columns = [
        "911_request_count_hour_range" if c[0] == "ts" else f"{c[0]}_{c[1]}"
        for c in hr.columns.to_flat_index()
    ]
    hr = hr.reset_index()

    rename_map = {
        "priority_score_911_mean": "911_priority_mean_hr",
        "priority_score_911_max": "911_priority_max_hr",
        "call_severity_proxy_911_mean": "911_severity_mean_hr",
        "call_severity_proxy_911_max": "911_severity_max_hr",
        "call_severity_proxy_911_sum": "911_severity_sum_hr",

        "is_violent_911_sum": "911_violent_count_hr",
        "is_property_911_sum": "911_property_count_hr",
        "is_disorder_911_sum": "911_disorder_count_hr",
        "is_traffic_911_sum": "911_traffic_count_hr",
        "is_narcotics_911_sum": "911_narcotics_count_hr",
        "is_mental_911_sum": "911_mental_count_hr",
        "is_weapons_911_sum": "911_weapons_count_hr",
        "is_domestic_911_sum": "911_domestic_count_hr",
        "is_alarm_911_sum": "911_alarm_count_hr",

        "onview_flag_911_sum": "911_onview_count_hr",
        "sensitive_flag_911_sum": "911_sensitive_count_hr",
        "is_police_agency_911_sum": "911_police_agency_count_hr",

        "dispatch_to_onscene_min_911_mean": "911_dispatch_to_onscene_mean_hr",
        "dispatch_to_onscene_min_911_median": "911_dispatch_to_onscene_median_hr",
        "entry_to_dispatch_min_911_mean": "911_entry_to_dispatch_mean_hr",
        "entry_to_dispatch_min_911_median": "911_entry_to_dispatch_median_hr",
        "received_to_close_min_911_mean": "911_received_to_close_mean_hr",
        "received_to_close_min_911_median": "911_received_to_close_median_hr",
        "dispatch_to_close_min_911_mean": "911_dispatch_to_close_mean_hr",
        "dispatch_to_close_min_911_median": "911_dispatch_to_close_median_hr",
    }
    hr = hr.rename(columns=rename_map)

    # günlük baz
    daily_agg = df.groupby(grp_day, dropna=False, observed=True).agg({
        "ts": "size",
        "is_violent_911": "sum",
        "is_property_911": "sum",
        "is_disorder_911": "sum",
        "is_weapons_911": "sum",
        "is_narcotics_911": "sum",
        "priority_score_911": "mean",
        "call_severity_proxy_911": ["mean", "sum"],
        "dispatch_to_onscene_min_911": "mean",
    })
    daily_agg.columns = [
        "911_request_count_daily(before_24_hours)" if c[0] == "ts" else f"{c[0]}_{c[1]}"
        for c in daily_agg.columns.to_flat_index()
    ]
    daily_agg = daily_agg.reset_index()

    daily_agg = daily_agg.rename(columns={
        "is_violent_911_sum": "911_violent_count_day",
        "is_property_911_sum": "911_property_count_day",
        "is_disorder_911_sum": "911_disorder_count_day",
        "is_weapons_911_sum": "911_weapons_count_day",
        "is_narcotics_911_sum": "911_narcotics_count_day",
        "priority_score_911_mean": "911_priority_mean_day",
        "call_severity_proxy_911_mean": "911_severity_mean_day",
        "call_severity_proxy_911_sum": "911_severity_sum_day",
        "dispatch_to_onscene_min_911_mean": "911_dispatch_to_onscene_mean_day",
    })

    out = hr.merge(daily_agg, on=grp_day, how="left")

    # intensity / ratio feature'lar
    out["911_intensity_hr_over_day"] = safe_div(
        out["911_request_count_hour_range"].astype(float),
        out["911_request_count_daily(before_24_hours)"].astype(float) + 1.0
    ).astype("float32")

    for c_num, c_den, out_name in [
        ("911_violent_count_hr", "911_request_count_hour_range", "911_violent_ratio_hr"),
        ("911_property_count_hr", "911_request_count_hour_range", "911_property_ratio_hr"),
        ("911_weapons_count_hr", "911_request_count_hour_range", "911_weapons_ratio_hr"),
        ("911_narcotics_count_hr", "911_request_count_hour_range", "911_narcotics_ratio_hr"),
        ("911_sensitive_count_hr", "911_request_count_hour_range", "911_sensitive_ratio_hr"),
    ]:
        if c_num in out.columns and c_den in out.columns:
            out[out_name] = safe_div(
                out[c_num].astype(float),
                out[c_den].astype(float) + 1.0
            ).astype("float32")

    tail = [c for c in ["date", "hour_range", "GEOID"] if c in out.columns]
    cols = [c for c in out.columns if c not in tail] + tail
    return out[cols]

# =========================================================
# SOURCE READERS
# =========================================================
def _pick_working_release_url(candidates: list[str]) -> str:
    for u in candidates:
        if not u:
            continue
        try:
            r = requests.get(u, timeout=20)
            if r.ok and r.content and len(r.content) > 200 and b"git-lfs" not in r.content[:200].lower():
                log(f"⬇️ Release kaynağı seçildi: {u}")
                return u
        except Exception as e:
            log(f"⚠️ Release URL okunamadı: {u} | {e}")
    raise RuntimeError("❌ Hiçbir release URL çalışmadı.")

def summary_from_local(path: Path | str, min_date=None) -> pd.DataFrame:
    path = Path(path)
    log(f"📥 Yerel 911 tabanı okunuyor: {path}")

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, low_memory=False, dtype={"GEOID": "string"})

    # zaten summary ise
    is_already_summary = (
        {"date", "hour_range"}.issubset(df.columns) and
        "911_request_count_hour_range" in df.columns
    )

    if is_already_summary:
        df["date"] = to_date(df["date"])
        if "GEOID" in df.columns:
            df["GEOID"] = normalize_geoid(df["GEOID"], DEFAULT_GEOID_LEN)
        if min_date is not None:
            df = df[df["date"] >= min_date].copy()
        return df

    std = make_standard_summary(df)
    if min_date is not None:
        std = std[std["date"] >= min_date].copy()
    return std

def summary_from_release(url: str, min_date=None) -> pd.DataFrame:
    log(f"⬇️ Release 911 indiriliyor: {url}")
    r = requests.get(url, timeout=120)
    r.raise_for_status()

    if url.lower().endswith(".parquet"):
        tmp = OUT_DIR / "_tmp_911.parquet"
        tmp.write_bytes(r.content)
        df = pd.read_parquet(tmp)
    else:
        tmp = OUT_DIR / "_tmp_911.csv"
        tmp.write_bytes(r.content)
        df = pd.read_csv(tmp, low_memory=False, dtype={"GEOID": "string"})

    is_already_summary = (
        {"date", "hour_range"}.issubset(df.columns) and
        "911_request_count_hour_range" in df.columns
    )

    if is_already_summary:
        df["date"] = to_date(df["date"])
        if "GEOID" in df.columns:
            df["GEOID"] = normalize_geoid(df["GEOID"], DEFAULT_GEOID_LEN)
        if min_date is not None:
            df = df[df["date"] >= min_date].copy()
        return df

    std = make_standard_summary(df)
    if min_date is not None:
        std = std[std["date"] >= min_date].copy()
    return std

def ensure_local_911_base() -> Optional[Path]:
    prefer_names = [
        "sf_911_last_5_year.parquet",
        "sf_911_last_5_year.csv",
        "sf_911_last_5_year_y.csv",
    ]

    roots = [OUT_DIR, Path(BASE_DIR), Path.cwd()]

    def _ok(p: Path) -> bool:
        if not p.exists() or p.is_dir():
            return False
        if p.suffix.lower() not in {".csv", ".parquet"}:
            return False
        if p.suffix.lower() == ".csv" and is_lfs_pointer_file(p):
            return False
        try:
            if p.stat().st_size < 200:
                return False
        except Exception:
            return False
        return True

    for nm in prefer_names:
        for rt in roots:
            cand = rt / nm
            if _ok(cand):
                log(f"📦 911 base bulundu: {cand}")
                return cand
            cand2 = rt / "crime_prediction_data" / nm
            if _ok(cand2):
                log(f"📦 911 base bulundu: {cand2}")
                return cand2

    for nm in prefer_names:
        for rt in roots:
            try:
                for found in rt.rglob(nm):
                    if _ok(found):
                        log(f"📦 911 base bulundu (rglob): {found}")
                        return found
            except Exception:
                continue

    return None

# =========================================================
# INCREMENTAL API
# =========================================================
def try_small_request(params, headers):
    p = dict(params)
    p["$limit"], p["$offset"] = 1, 0
    r = requests.get(SF911_API_URL, headers=headers, params=p, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r

def fetch_range_all_chunks(start_day, end_day) -> Optional[pd.DataFrame]:
    dt_candidates = ["received_datetime", "received_time", "entry_datetime", "dispatch_datetime", "datetime", "date"]
    headers = {"X-App-Token": SF_APP_TOKEN} if SF_APP_TOKEN else {}
    rng_start = f"{start_day}T00:00:00"
    rng_end   = f"{end_day}T23:59:59"

    chosen_dt, last_err = None, None
    for dt_col in dt_candidates:
        base_where = f"{dt_col} between '{rng_start}' and '{rng_end}'"
        where_try = [
            base_where + (f" AND {AGENCY_FILTER}" if AGENCY_FILTER else ""),
            base_where
        ]
        for wc in where_try:
            try:
                try_small_request({"$where": wc}, headers)
                chosen_dt = dt_col
                break
            except Exception as e:
                last_err = e
        if chosen_dt:
            break

    if chosen_dt is None:
        log(f"❌ API datetime kolonu bulunamadı: {last_err}")
        return None

    pieces, offset, page = [], 0, 1
    where_with_agency = f"{chosen_dt} between '{rng_start}' and '{rng_end}'" + (f" AND {AGENCY_FILTER}" if AGENCY_FILTER else "")
    where_without_agency = f"{chosen_dt} between '{rng_start}' and '{rng_end}'"

    while True:
        df = None
        for attempt in range(MAX_RETRIES):
            try:
                r = requests.get(
                    SF911_API_URL,
                    headers=headers,
                    params={"$where": where_with_agency, "$limit": CHUNK_LIMIT, "$offset": offset},
                    timeout=REQUEST_TIMEOUT
                )
                if r.status_code == 400:
                    r = requests.get(
                        SF911_API_URL,
                        headers=headers,
                        params={"$where": where_without_agency, "$limit": CHUNK_LIMIT, "$offset": offset},
                        timeout=REQUEST_TIMEOUT
                    )
                r.raise_for_status()
                df = pd.read_json(io.BytesIO(r.content))
                break
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    log(f"❌ API sayfa hatası page={page} offset={offset} | {e}")
                time.sleep(1 + attempt * 0.5)

        if df is None or df.empty:
            if page == 1:
                log("ℹ️ Bu aralıkta API boş döndü.")
            break

        log(f"    + {len(df):,} satır | page={page} offset={offset}")
        pieces.append(df)

        if len(df) < CHUNK_LIMIT:
            break

        offset += CHUNK_LIMIT
        page += 1
        time.sleep(SLEEP_BETWEEN_REQS)

    if not pieces:
        return None
    return pd.concat(pieces, ignore_index=True)

def write_recent_csv(raw: pd.DataFrame, hours: int = SF911_RECENT_HOURS):
    ts_col = next((c for c in ["received_datetime", "received_time", "entry_datetime", "dispatch_datetime", "datetime", "date"] if c in raw.columns), None)
    if not ts_col:
        return

    tmp = raw.copy()
    tmp["ts"] = pd.to_datetime(tmp[ts_col], errors="coerce", utc=True)
    tmp = tmp[tmp["ts"].notna()].copy()
    if tmp.empty:
        return

    lat_col = next((c for c in ["latitude", "lat", "y"] if c in raw.columns), None)
    lon_col = next((c for c in ["longitude", "long", "x"] if c in raw.columns), None)

    tmax = tmp["ts"].max()
    cutoff = tmax - pd.Timedelta(hours=hours)

    out = pd.DataFrame({"ts": tmp["ts"]})
    if lat_col:
        out["lat"] = pd.to_numeric(tmp[lat_col], errors="coerce")
    if lon_col:
        out["lon"] = pd.to_numeric(tmp[lon_col], errors="coerce")

    out = out[out["ts"] >= cutoff].copy()
    safe_save_csv(out, str(OUT_DIR / "sf_911_recent.csv"))
    log(f"ℹ️ sf_911_recent.csv yazıldı: {len(out):,} satır")

def incremental_summary(start_day, end_day) -> pd.DataFrame:
    if start_day is None or end_day is None or end_day < start_day:
        return pd.DataFrame()

    log(f"🌐 API incremental 911: {start_day} → {end_day}")
    raw = fetch_range_all_chunks(start_day, end_day) if BULK_RANGE else None

    try:
        if raw is not None and not raw.empty:
            write_recent_csv(raw, hours=SF911_RECENT_HOURS)
    except Exception as e:
        log(f"⚠️ recent write atlandı: {e}")

    if raw is None or raw.empty:
        return pd.DataFrame()

    try:
        raw = ensure_geoid(raw)
    except Exception as e:
        log(f"⚠️ API ensure_geoid başarısız: {e}; ham devam edilecek.")

    return make_standard_summary(raw)

# =========================================================
# ROLLING / SPIKE / ZSCORE / TREND
# =========================================================
ROLLING_SLOT_WINDOWS = {
    "1h": 1,
    "3h": 1,
    "6h": 2,
    "24h": 8,
    "3d": 24,
    "7d": 56,
}

def add_rolling_features(summary_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    need_cols = [
        "GEOID", "date", "hour_range", "hr_key",
        "911_request_count_hour_range",
        "911_request_count_daily(before_24_hours)",
        "911_violent_count_hr", "911_property_count_hr", "911_weapons_count_hr", "911_severity_sum_hr",
        "911_violent_count_day", "911_property_count_day", "911_weapons_count_day", "911_severity_sum_day",
    ]
    need_cols = [c for c in need_cols if c in summary_df.columns]
    final_911 = summary_df[need_cols].copy()

    # günlük unique
    day_keep = ["GEOID", "date"]
    day_keep += [c for c in [
        "911_request_count_daily(before_24_hours)",
        "911_violent_count_day",
        "911_property_count_day",
        "911_weapons_count_day",
        "911_severity_sum_day",
    ] if c in final_911.columns]

    day_unique = (
        final_911[day_keep]
        .groupby(["GEOID", "date"], as_index=False, observed=True)
        .sum(numeric_only=True)
        .sort_values(["GEOID", "date"])
        .reset_index(drop=True)
    )

    # slot unique
    hr_keep = ["GEOID", "hr_key", "date", "911_request_count_hour_range"]
    hr_keep += [c for c in [
        "911_violent_count_hr", "911_property_count_hr", "911_weapons_count_hr", "911_severity_sum_hr"
    ] if c in final_911.columns]

    hr_unique = (
        final_911[hr_keep]
        .groupby(["GEOID", "hr_key", "date"], as_index=False, observed=True)
        .sum(numeric_only=True)
        .sort_values(["GEOID", "hr_key", "date"])
        .reset_index(drop=True)
    )

    # 1) daily rolling
    base_daily_map = {
        "911_request_count_daily(before_24_hours)": "911_geo_daily_cnt",
        "911_violent_count_day": "911_geo_violent_day",
        "911_property_count_day": "911_geo_property_day",
        "911_weapons_count_day": "911_geo_weapons_day",
        "911_severity_sum_day": "911_geo_severity_day",
    }
    for src, stem in base_daily_map.items():
        if src in day_unique.columns:
            for label, n in ROLLING_SLOT_WINDOWS.items():
                col = f"{stem}_last{label}"
                day_unique[col] = (
                    day_unique.groupby("GEOID")[src]
                    .transform(lambda s: s.rolling(n, min_periods=1).sum().shift(1))
                    .astype("float32")
                )

    # günlük trend/spike/anomaly
    if "911_request_count_daily(before_24_hours)" in day_unique.columns:
        s = day_unique.groupby("GEOID")["911_request_count_daily(before_24_hours)"]

        day_unique["911_geo_day_trend_1"] = s.diff().astype("float32")
        day_unique["911_geo_day_roll7_mean"] = s.transform(lambda x: x.rolling(7, min_periods=3).mean().shift(1)).astype("float32")
        day_unique["911_geo_day_roll7_std"]  = s.transform(lambda x: x.rolling(7, min_periods=3).std().shift(1)).astype("float32")
        day_unique["911_geo_day_roll14_mean"] = s.transform(lambda x: x.rolling(14, min_periods=5).mean().shift(1)).astype("float32")

        cur = day_unique["911_request_count_daily(before_24_hours)"].astype("float32")
        r7  = day_unique["911_geo_day_roll7_mean"].astype("float32")
        r7s = day_unique["911_geo_day_roll7_std"].astype("float32")
        r14 = day_unique["911_geo_day_roll14_mean"].astype("float32")

        day_unique["911_geo_spike_vs_roll7"] = (cur / (r7 + 1.0)).astype("float32")
        day_unique["911_geo_spike_vs_roll14"] = (cur / (r14 + 1.0)).astype("float32")
        day_unique["911_geo_zscore_day"] = ((cur - r7) / (r7s + 1.0)).astype("float32")
        day_unique["911_geo_above_roll7_flag"] = (cur > (r7 + r7s)).fillna(False).astype("int8")

    # 2) hr rolling
    base_hr_map = {
        "911_request_count_hour_range": "911_geo_hr_cnt",
        "911_violent_count_hr": "911_geo_violent_hr",
        "911_property_count_hr": "911_geo_property_hr",
        "911_weapons_count_hr": "911_geo_weapons_hr",
        "911_severity_sum_hr": "911_geo_severity_hr",
    }
    for src, stem in base_hr_map.items():
        if src in hr_unique.columns:
            for label, n in ROLLING_SLOT_WINDOWS.items():
                col = f"{stem}_last{label}"
                hr_unique[col] = (
                    hr_unique.groupby(["GEOID", "hr_key"])[src]
                    .transform(lambda s: s.rolling(n, min_periods=1).sum().shift(1))
                    .astype("float32")
                )

    # slot anomaly
    if "911_request_count_hour_range" in hr_unique.columns:
        g = hr_unique.groupby(["GEOID", "hr_key"])["911_request_count_hour_range"]
        hr_unique["911_geo_hr_roll8_mean"] = g.transform(lambda x: x.rolling(8, min_periods=3).mean().shift(1)).astype("float32")
        hr_unique["911_geo_hr_roll8_std"]  = g.transform(lambda x: x.rolling(8, min_periods=3).std().shift(1)).astype("float32")

        cur = hr_unique["911_request_count_hour_range"].astype("float32")
        m8  = hr_unique["911_geo_hr_roll8_mean"].astype("float32")
        s8  = hr_unique["911_geo_hr_roll8_std"].astype("float32")

        hr_unique["911_geo_hr_spike_vs_roll8"] = (cur / (m8 + 1.0)).astype("float32")
        hr_unique["911_geo_hr_zscore"] = ((cur - m8) / (s8 + 1.0)).astype("float32")

    # summary merge
    enriched = final_911.merge(hr_unique, on=["GEOID", "hr_key", "date"], how="left", suffixes=("", "_dup1"))
    enriched = enriched.merge(day_unique, on=["GEOID", "date"], how="left", suffixes=("", "_dup2"))

    dup_cols = [c for c in enriched.columns if c.endswith("_dup1") or c.endswith("_dup2")]
    if dup_cols:
        enriched = enriched.drop(columns=dup_cols, errors="ignore")

    return enriched, day_unique, hr_unique

# =========================================================
# NEIGHBORS
# =========================================================
def build_neighbors(method: str = "touches", radius_m: float = 500.0) -> pd.DataFrame:
    gdf_blocks, _ = _load_blocks()
    tracts = gdf_blocks.dissolve(by="GEOID", as_index=False)

    if method == "radius":
        tr_utm = tracts.to_crs("EPSG:26910")
        buf = tr_utm.buffer(radius_m)
        g_buf = gpd.GeoDataFrame(tr_utm[["GEOID"]].copy(), geometry=buf, crs=tr_utm.crs)
        join = gpd.sjoin(g_buf, tr_utm[["GEOID", "geometry"]].rename(columns={"GEOID": "nbr"}), predicate="intersects")
        edges = join[["GEOID", "nbr"]]
    else:
        join = gpd.sjoin(
            tracts[["GEOID", "geometry"]],
            tracts[["GEOID", "geometry"]].rename(columns={"GEOID": "nbr"}),
            predicate="touches"
        )
        edges = join[["GEOID", "nbr"]]

    edges = edges[edges["GEOID"] != edges["nbr"]].copy()
    edges["pair"] = edges.apply(lambda r: tuple(sorted((r["GEOID"], r["nbr"]))), axis=1)
    edges = edges.drop_duplicates("pair").drop(columns=["pair"])

    edges["GEOID"] = normalize_geoid(edges["GEOID"], DEFAULT_GEOID_LEN)
    edges["nbr"]   = normalize_geoid(edges["nbr"], DEFAULT_GEOID_LEN)
    return edges.reset_index(drop=True)

def add_neighbor_features(enriched: pd.DataFrame, day_unique: pd.DataFrame) -> pd.DataFrame:
    if not ENABLE_NEIGHBORS:
        return enriched

    try:
        neighbors_df = build_neighbors(NEIGHBOR_METHOD, NEIGHBOR_RADIUS_M)
        log_shape(neighbors_df, f"neighbors ({NEIGHBOR_METHOD})")
    except Exception as e:
        log(f"⚠️ neighbor build başarısız: {e}")
        return enriched

    if neighbors_df.empty:
        return enriched

    cols_for_nbr = ["GEOID", "date"]
    for c in [
        "911_request_count_daily(before_24_hours)",
        "911_violent_count_day",
        "911_property_count_day",
        "911_weapons_count_day",
        "911_severity_sum_day",
    ]:
        if c in day_unique.columns:
            cols_for_nbr.append(c)

    tmp = day_unique[cols_for_nbr].rename(columns={"GEOID": "nbr"})
    day_nbr = neighbors_df.merge(tmp, on="nbr", how="left")

    agg_map = {}
    if "911_request_count_daily(before_24_hours)" in day_nbr.columns:
        agg_map["911_request_count_daily(before_24_hours)"] = "sum"
    if "911_violent_count_day" in day_nbr.columns:
        agg_map["911_violent_count_day"] = "sum"
    if "911_property_count_day" in day_nbr.columns:
        agg_map["911_property_count_day"] = "sum"
    if "911_weapons_count_day" in day_nbr.columns:
        agg_map["911_weapons_count_day"] = "sum"
    if "911_severity_sum_day" in day_nbr.columns:
        agg_map["911_severity_sum_day"] = "sum"

    nbr_daily = day_nbr.groupby(["GEOID", "date"], as_index=False, observed=True).agg(agg_map)

    rename_nbr = {
        "911_request_count_daily(before_24_hours)": "911_neighbors_daily_cnt",
        "911_violent_count_day": "911_neighbors_violent_day",
        "911_property_count_day": "911_neighbors_property_day",
        "911_weapons_count_day": "911_neighbors_weapons_day",
        "911_severity_sum_day": "911_neighbors_severity_day",
    }
    nbr_daily = nbr_daily.rename(columns=rename_nbr)
    nbr_daily = nbr_daily.sort_values(["GEOID", "date"]).reset_index(drop=True)

    base_neighbor_map = [
        "911_neighbors_daily_cnt",
        "911_neighbors_violent_day",
        "911_neighbors_property_day",
        "911_neighbors_weapons_day",
        "911_neighbors_severity_day",
    ]

    for base_col in base_neighbor_map:
        if base_col in nbr_daily.columns:
            for label, n in ROLLING_SLOT_WINDOWS.items():
                nbr_daily[f"{base_col}_last{label}"] = (
                    nbr_daily.groupby("GEOID")[base_col]
                    .transform(lambda s: s.rolling(n, min_periods=1).sum().shift(1))
                    .astype("float32")
                )

    if "911_neighbors_daily_cnt" in nbr_daily.columns:
        g = nbr_daily.groupby("GEOID")["911_neighbors_daily_cnt"]
        nbr_daily["911_neighbors_roll7_mean"] = g.transform(lambda x: x.rolling(7, min_periods=3).mean().shift(1)).astype("float32")
        nbr_daily["911_neighbors_roll7_std"]  = g.transform(lambda x: x.rolling(7, min_periods=3).std().shift(1)).astype("float32")

        cur = nbr_daily["911_neighbors_daily_cnt"].astype("float32")
        m7  = nbr_daily["911_neighbors_roll7_mean"].astype("float32")
        s7  = nbr_daily["911_neighbors_roll7_std"].astype("float32")

        nbr_daily["911_neighbors_spike_vs_roll7"] = (cur / (m7 + 1.0)).astype("float32")
        nbr_daily["911_neighbors_zscore_day"] = ((cur - m7) / (s7 + 1.0)).astype("float32")

    out = enriched.merge(nbr_daily, on=["GEOID", "date"], how="left")
    return out

# =========================================================
# MAIN
# =========================================================
def main():
    five_years_ago = datetime.now(timezone.utc).date() - timedelta(days=5 * 365)

    log(f"📁 Yerel 911 summary yolu: {local_summary_csv_path}")
    base_path = ensure_local_911_base()
    
    if base_path is not None:
        final_911 = summary_from_local(base_path, min_date=five_years_ago)
        safe_save_parquet(final_911, str(local_summary_parquet_path))
        log("✅ Yerel 911 summary hazırlandı (erken aşamada yalnız parquet yazıldı).")
    else:
        if os.getenv("ALLOW_911_RELEASE_FALLBACK", "0").strip().lower() in ("1", "true", "yes", "on"):
            release_url = _pick_working_release_url(RAW_911_URL_CANDIDATES)
            final_911 = summary_from_release(release_url, min_date=five_years_ago)
            safe_save_parquet(final_911, str(local_summary_parquet_path))
            log("✅ Release fallback ile 911 summary hazırlandı (erken aşamada yalnız parquet yazıldı).")
        else:
            raise FileNotFoundError(
                "❌ Yerel 911 base bulunamadı ve ALLOW_911_RELEASE_FALLBACK kapalı."
            )
    base_max_date = to_date(final_911["date"]).max() if not final_911.empty else None
    today_sf = (datetime.now(SF_TZ) if SF_TZ is not None else datetime.now()).date()

    if base_max_date is None:
        fetch_start, fetch_end = today_sf, today_sf
    else:
        fetch_start = base_max_date - timedelta(days=max(1, SF911_REINGEST_DAYS))
        fetch_end = today_sf
        if fetch_start < five_years_ago:
            fetch_start = five_years_ago
        if fetch_start > fetch_end:
            fetch_start = fetch_end

    log(f"🗓️ Incremental fetch aralığı: {fetch_start} → {fetch_end}")

    inc = incremental_summary(fetch_start, fetch_end)
    if inc is not None and not inc.empty:
        if "GEOID" in inc.columns:
            inc["GEOID"] = normalize_geoid(inc["GEOID"], DEFAULT_GEOID_LEN)
        inc["date"] = to_date(inc["date"])

        before = len(final_911)
        final_911 = pd.concat([final_911, inc], ignore_index=True)

        dedup_keys = [c for c in ["GEOID", "date", "hour_range"] if c in final_911.columns]
        if dedup_keys:
            final_911 = (
                final_911.dropna(subset=["date"])
                .sort_values(dedup_keys)
                .drop_duplicates(subset=dedup_keys, keep="last")
                .reset_index(drop=True)
            )
        else:
            final_911 = final_911.drop_duplicates().reset_index(drop=True)

        final_911 = final_911[final_911["date"] >= five_years_ago].copy()

        # feature
        enriched, day_unique, hr_unique = add_rolling_features(final_911)
        safe_save_csv(enriched, str(local_summary_csv_path))
        safe_save_parquet(enriched, str(local_summary_parquet_path))
        safe_save_csv(enriched, str(y_summary_path))
        
        log(f"💾 911 summary güncellendi (+{len(final_911) - before:,} satır net fark)")
    else:
        log("ℹ️ API incremental boş döndü; mevcut 911 summary korunuyor.")

    if final_911 is None or final_911.empty:
        log("⚠️ final_911 boş. Çıkılıyor.")
        raise SystemExit(0)

    final_911 = final_911.dropna(subset=["GEOID", "date", "hour_range"]).copy()
    final_911["GEOID"] = normalize_geoid(final_911["GEOID"], DEFAULT_GEOID_LEN)
    final_911["date"] = to_date(final_911["date"])

    hr_pat = re.compile(r"^\s*(\d{1,2})\s*-\s*(\d{1,2})\s*$")
    def _hr_key_from_range(hr):
        m = hr_pat.match(str(hr))
        return int(m.group(1)) % 24 if m else None

    if "hr_key" not in final_911.columns:
        final_911["hr_key"] = final_911["hour_range"].apply(_hr_key_from_range).astype("int16")

    if "day_of_week" not in final_911.columns:
        final_911["day_of_week"] = pd.to_datetime(final_911["date"]).dt.weekday.astype("int8")
    if "month" not in final_911.columns:
        final_911["month"] = pd.to_datetime(final_911["date"]).dt.month.astype("int8")
    if "season" not in final_911.columns:
        season_map = {12:"Winter",1:"Winter",2:"Winter",3:"Spring",4:"Spring",5:"Spring",
                      6:"Summer",7:"Summer",8:"Summer",9:"Fall",10:"Fall",11:"Fall"}
        final_911["season"] = final_911["month"].map(season_map).astype("category")

    log_shape(final_911, "911 summary (base)")
    log_date_range(final_911, "date", "911")

    enriched, day_unique, hr_unique = add_rolling_features(final_911)
    enriched = add_neighbor_features(enriched, day_unique)

    # kritik derived ratios
    if "911_request_count_hour_range" in enriched.columns and "911_geo_hr_cnt_last24h" in enriched.columns:
        enriched["911_hr_vs_last24h_ratio"] = safe_div(
            enriched["911_request_count_hour_range"].astype(float),
            enriched["911_geo_hr_cnt_last24h"].astype(float) + 1.0
        ).astype("float32")

    if "911_request_count_daily(before_24_hours)" in enriched.columns and "911_geo_daily_cnt_last7d" in enriched.columns:
        enriched["911_day_vs_last7d_ratio"] = safe_div(
            enriched["911_request_count_daily(before_24_hours)"].astype(float),
            enriched["911_geo_daily_cnt_last7d"].astype(float) + 1.0
        ).astype("float32")

    # kaydet
    safe_save_csv(enriched, str(local_summary_csv_path))
    safe_save_parquet(enriched, str(local_summary_parquet_path))
    safe_save_csv(enriched, str(y_summary_path))
    log_shape(enriched, "911 summary (final enriched)")

    # crime grid yükle
    CRIME_GRID_CANDIDATES = [
        OUT_DIR / "sf_crime_y.csv",
        Path(BASE_DIR) / "sf_crime_y.csv",
        Path("./sf_crime_y.csv"),
    ]
    crime_grid_path = next((p for p in CRIME_GRID_CANDIDATES if p.exists()), None)
    if crime_grid_path is None:
        raise FileNotFoundError("❌ sf_crime_y.csv bulunamadı.")

    crime = pd.read_csv(crime_grid_path, dtype={"GEOID": str}, low_memory=False)
    log(f"📥 crime grid yüklendi: {crime_grid_path}")
    log_shape(crime, "crime grid")

    crime["GEOID"] = normalize_geoid(crime["GEOID"], DEFAULT_GEOID_LEN)

    if "hour_range" in crime.columns:
        crime["hr_key"] = crime["hour_range"].apply(_hr_key_from_range).astype("Int16")
    elif "event_hour" in crime.columns:
        crime["hr_key"] = (((pd.to_numeric(crime["event_hour"], errors="coerce").fillna(0).astype(int)) // 3) * 3).astype("Int16")
        crime["hour_range"] = crime["hr_key"].apply(lambda s: f"{int(s):02d}-{int(min(int(s)+3, 24)):02d}")
    else:
        raise ValueError("❌ crime grid içinde ne hour_range ne event_hour var.")

    if "date" not in crime.columns and "datetime" in crime.columns:
        crime["date"] = pd.to_datetime(crime["datetime"], errors="coerce").dt.date
    else:
        crime["date"] = to_date(crime["date"])

    keys = ["GEOID", "date", "hour_range"]

    overlap = (set(crime.columns) & set(enriched.columns)) - set(keys)
    if overlap:
        log(f"🧹 merge overlap bulundu, enriched'ten düşülüyor: {sorted(overlap)}")
        enriched = enriched.drop(columns=list(overlap), errors="ignore")

    merged = crime.merge(enriched, on=keys, how="left")
    log("🔗 Join modu: DATE-BASED (GEOID, date, hour_range)")

    # fill numeric
    for c in merged.columns:
        if c.startswith("911_") and merged[c].dtype == "O":
            merged[c] = pd.to_numeric(merged[c], errors="ignore")

    zero_fill_prefixes = ("911_",)
    for c in merged.columns:
        if c.startswith(zero_fill_prefixes):
            if pd.api.types.is_numeric_dtype(merged[c]):
                if str(merged[c].dtype).startswith("float"):
                    merged[c] = merged[c].fillna(0).astype("float32")
                elif str(merged[c].dtype).startswith(("int", "Int")):
                    merged[c] = merged[c].fillna(0)
                else:
                    merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0).astype("float32")

    # nan raporu
    nan_counts = merged.isna().sum()
    nan_counts = nan_counts[nan_counts > 0].sort_values(ascending=False)
    log(f"🧪 NaN içeren sütun sayısı = {len(nan_counts)}")
    if len(nan_counts) > 0:
        nan_report = nan_counts.rename("nan_count").reset_index().rename(columns={"index": "column"})
        nan_report_path = OUT_DIR / "nan_report_sf_crime_01.csv"
        nan_report.to_csv(nan_report_path, index=False)
        log(f"📄 NaN raporu kaydedildi: {nan_report_path}")

    # çıktı
    safe_save_csv(merged, str(merged_output_path))
    safe_save_parquet(merged, str(merged_output_path).replace(".csv", ".parquet"))
    log_shape(merged, "CRIME × 911")
    log(f"✅ sf_crime_01 tamamlandı → {merged_output_path}")

    # ============================================================
    # 🔎 DEBUG — 911 FEATURE DOLULUK KONTROLÜ
    # ============================================================
    try:
        cols_911 = [c for c in final_df.columns if "911_" in c]
    
        if cols_911:
            nan_stats = final_df[cols_911].isna().mean().sort_values(ascending=False)
    
            # 🚨 BURAYA EKLE
            if nan_stats.mean() > 0.9:
                raise ValueError("❌ 911 feature tamamen boş → pipeline durduruldu")
    
            log("🔎 911 NaN oranı (top 10):")
            log(nan_stats.head(10).to_string())
    
            log(f"📊 Ortalama NaN oranı: {nan_stats.mean():.4f}")
    
            if nan_stats.mean() > 0.5:
                log("🚨 UYARI: 911 feature'ların çoğu boş! (GEOID / merge sorunu olabilir)")
    
        if "GEOID" in final_df.columns:
            geoid_nan = final_df["GEOID"].isna().mean()
            log(f"📊 GEOID NaN oranı: {geoid_nan:.4f}")
    
    except Exception as e:
        log(f"⚠️ 911 debug başarısız: {e}")
    
    log(f"✅ sf_crime_01 tamamlandı → {merged_output_path}")
    
    try:
        print(merged.head(5).to_string(index=False))
    except Exception:
        pass


if __name__ == "__main__":
    main()
