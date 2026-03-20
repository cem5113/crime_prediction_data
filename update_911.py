#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================
# update_911.py — SON REVİZE
# 911 özetini üretir/günceller ve sf_crime_y.csv ile birleştirip
# sf_crime_01.csv yazar.
#
# STACKING / suç tahmini için EKLER:
# - priority / onview / sensitive feature'ları
# - semantic call-type group count'ları
# - response-time duration feature'ları
# - slot/day rolling feature'ları
# - neighbor rolling feature'ları
# ============================================================

from __future__ import annotations
import os
import re
import io
import time
import requests
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import geopandas as gpd

# ============================================================
# TZ / LOG / HELPERS
# ============================================================
try:
    import zoneinfo
    SF_TZ = zoneinfo.ZoneInfo("America/Los_Angeles")
except Exception:
    SF_TZ = None

def normalize_hour_range(hr):
    import re
    import numpy as np

    m = re.match(r"^\s*(\d{1,2})\s*[-:]\s*(\d{1,2})\s*$", str(hr))
    if not m:
        return np.nan

    a = int(m.group(1)) % 24
    b = int(m.group(2))

    if b <= a:
        b = min(a + 3, 24)

    return f"{a:02d}-{b:02d}"
    
def log(msg: str):
    print(msg, flush=True)


def ensure_parent(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def safe_save_csv(df: pd.DataFrame, path: str):
    try:
        ensure_parent(path)
        df.to_csv(path, index=False, encoding="utf-8-sig")
    except Exception as e:
        log(f"❌ Kaydetme hatası: {path}\n{e}")
        df.to_csv(path + ".bak", index=False, encoding="utf-8-sig")
        log(f"📁 Yedek oluşturuldu: {path}.bak")


def _to_date_series(x):
    try:
        s = pd.to_datetime(x, utc=True, errors="coerce")
        if SF_TZ is not None:
            s = s.dt.tz_convert(SF_TZ)
        return s.dt.date.dropna()
    except Exception:
        return pd.to_datetime(x, errors="coerce").dt.date.dropna()


def log_shape(df, label):
    r, c = df.shape
    log(f"📊 {label}: {r} satır × {c} sütun")


def log_date_range(df, date_col="date", label="911"):
    if date_col not in df.columns:
        log(f"⚠️ {label}: '{date_col}' kolonu yok.")
        return
    s = _to_date_series(df[date_col])
    if s.empty:
        log(f"⚠️ {label}: tarih parse edilemedi.")
        return
    log(f"🧭 {label} tarihi aralığı: {s.min()} → {s.max()} (gün={s.nunique()})")


def normalize_geoid(s: pd.Series, target_len: int) -> pd.Series:
    s = s.astype(str).str.extract(r"(\d+)", expand=False)
    L = int(target_len)
    return s.str[:L].str.zfill(L)


def to_date(s):
    return pd.to_datetime(s, errors="coerce").dt.date


def is_lfs_pointer_file(p: Path) -> bool:
    try:
        return "git-lfs.github.com/spec/v1" in p.read_text(errors="ignore")[:200]
    except Exception:
        return False


def _first_existing_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _num_minutes(a, b):
    aa = pd.to_datetime(a, errors="coerce")
    bb = pd.to_datetime(b, errors="coerce")
    return (bb - aa).dt.total_seconds() / 60.0


def _safe_bool01(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.strip().str.lower()
    return x.isin(["1", "true", "t", "yes", "y"]).astype("int8")


# ============================================================
# CONFIG & PATHS
# ============================================================
DEFAULT_GEOID_LEN = int(os.getenv("GEOID_LEN", "11"))

_raw_base = os.getenv("CRIME_DATA_DIR", "crime_prediction_data").strip().strip("/\\")
repo_leaf = Path.cwd().name
if not os.path.isabs(_raw_base) and Path(_raw_base).name == repo_leaf:
    _raw_base = "."
BASE_DIR = str(Path(_raw_base).resolve()) if _raw_base != "." else "."
Path(BASE_DIR).mkdir(parents=True, exist_ok=True)
log(f"📂 BASE_DIR = {Path(BASE_DIR).resolve()}")

OUT_DIR = Path(os.getenv("CRIME_DATA_DIR", str(Path(BASE_DIR)))).resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOCAL_NAME = "sf_911_last_5_year.csv"
local_summary_path = OUT_DIR / LOCAL_NAME
Y_NAME = "sf_911_last_5_year_y.csv"
y_summary_path = OUT_DIR / Y_NAME

merged_output_path = Path(os.getenv("DAILY_OUT", str(OUT_DIR / "sf_crime_01.csv")))
if not merged_output_path.is_absolute():
    merged_output_path = OUT_DIR / merged_output_path.name
log(f"🧾 DAILY_OUT seen as: {os.getenv('DAILY_OUT', '(unset)')}")
log(f"📝 Writing sf_crime_01 → {merged_output_path}")

CENSUS_CANDIDATES = [
    OUT_DIR / "sf_census_blocks.geojson",
    Path(BASE_DIR) / "sf_census_blocks.geojson",
    Path("./sf_census_blocks.geojson"),
]

SF911_API_URL = os.getenv("SF911_API_URL", "https://data.sfgov.org/resource/2zdj-bwza.json")
SF_APP_TOKEN = os.getenv("SF911_API_TOKEN", "")
AGENCY_FILTER = os.getenv("SF911_AGENCY_FILTER", "agency like '%Police%'")
REQUEST_TIMEOUT = int(os.getenv("SF911_REQUEST_TIMEOUT", "60"))
CHUNK_LIMIT = int(os.getenv("SF911_CHUNK_LIMIT", "50000"))
MAX_RETRIES = int(os.getenv("SF911_MAX_RETRIES", "4"))
SLEEP_BETWEEN_REQS = float(os.getenv("SF911_SLEEP", "0.2"))
BULK_RANGE = os.getenv("SF911_BULK_RANGE", "1").lower() in ("1", "true", "yes", "on")
IS_V3 = "/api/v3/views/" in SF911_API_URL
V3_PAGE_LIMIT = int(os.getenv("SF_V3_PAGE_LIMIT", "1000"))
SF911_RECENT_HOURS = int(os.getenv("SF911_RECENT_HOURS", "6"))
SF911_REINGEST_DAYS = int(os.getenv("SF911_REINGEST_DAYS", "14"))

RAW_911_URL_ENV = os.getenv("RAW_911_URL", "").strip()
RAW_911_URL_CANDIDATES = [
    RAW_911_URL_ENV or "",
    "https://github.com/cem5113/crime_prediction_data/releases/download/v1.0.1/sf_911_last_5_year_y.csv",
    "https://github.com/cem5113/crime_prediction_data/releases/download/v1.0.1/sf_911_last_5_year.csv",
]

ENABLE_NEIGHBORS = os.getenv("ENABLE_NEIGHBORS", "1").lower() in ("1", "true", "yes", "on")
NEIGHBOR_METHOD = os.getenv("NEIGHBOR_METHOD", "touches")
NEIGHBOR_RADIUS_M = float(os.getenv("NEIGHBOR_RADIUS_M", "500"))

SF_BBOX = (-123.2, 37.6, -122.3, 37.9)

# ============================================================
# IO HELPERS
# ============================================================
def read_large_csv_in_chunks(path, usecols=None, chunksize=200_000):
    try:
        it = pd.read_csv(path, low_memory=False, dtype={"GEOID": "string"}, usecols=usecols, chunksize=chunksize)
        return pd.concat(it, ignore_index=True)
    except ValueError:
        it = pd.read_csv(path, low_memory=False, dtype={"GEOID": "string"}, chunksize=chunksize)
        return pd.concat(it, ignore_index=True)


def _pick_working_release_url(candidates: list[str]) -> str:
    for u in candidates:
        if not u:
            continue
        try:
            r = requests.get(u, timeout=20)
            if r.ok and r.content and len(r.content) > 200 and b"git-lfs" not in r.content[:200].lower():
                log(f"⬇️ Release kaynağı seçildi: {u}")
                return u
            else:
                log(f"⚠️ Uygun değil (boş/küçük/LFS pointer olabilir): {u}")
        except Exception as e:
            log(f"⚠️ Ulaşılamadı: {u} ({e})")
    raise RuntimeError("❌ Hiçbir release 911 URL’i erişilebilir değil.")


# ============================================================
# GEO / BLOCKS
# ============================================================
def _load_blocks() -> tuple[gpd.GeoDataFrame, int]:
    census_path = next((p for p in CENSUS_CANDIDATES if p.exists()), None)
    if census_path is None:
        raise FileNotFoundError("❌ Nüfus blokları GeoJSON yok (OUT_DIR/BASE_DIR/kök).")
    gdf_blocks = gpd.read_file(census_path)
    if "GEOID" not in gdf_blocks.columns:
        cand = [c for c in gdf_blocks.columns if str(c).upper().startswith("GEOID")]
        if not cand:
            raise ValueError("GeoJSON içinde GEOID benzeri bir sütun yok.")
        gdf_blocks = gdf_blocks.rename(columns={cand[0]: "GEOID"})
    tlen = gdf_blocks["GEOID"].astype(str).str.len().mode().iat[0]
    gdf_blocks["GEOID"] = normalize_geoid(gdf_blocks["GEOID"], tlen)
    if gdf_blocks.crs is None:
        gdf_blocks.set_crs("EPSG:4326", inplace=True)
    elif gdf_blocks.crs.to_epsg() != 4326:
        gdf_blocks = gdf_blocks.to_crs(4326)
    return gdf_blocks, tlen


def ensure_geoid(df: pd.DataFrame) -> pd.DataFrame:
    if "GEOID" in df.columns and df["GEOID"].notna().any():
        return df

    if "latitude" not in df.columns or "longitude" not in df.columns:
        if "intersection_point" in df.columns:
            def _lon(x):
                if isinstance(x, dict) and "coordinates" in x:
                    return x["coordinates"][0]
                if isinstance(x, str):
                    m = re.search(r"[-\d\.]+,\s*[-\d\.]+", x)
                    if m:
                        lo, la = m.group(0).split(",")
                        return float(lo)
                return None

            def _lat(x):
                if isinstance(x, dict) and "coordinates" in x:
                    return x["coordinates"][1]
                if isinstance(x, str):
                    m = re.search(r"[-\d\.]+,\s*[-\d\.]+", x)
                    if m:
                        lo, la = m.group(0).split(",")
                        return float(la)
                return None

            df["longitude"] = df["intersection_point"].apply(_lon)
            df["latitude"] = df["intersection_point"].apply(_lat)

        for a, b in (("y", "x"), ("lat", "long")):
            if a in df.columns and b in df.columns and "latitude" not in df.columns:
                df["latitude"] = pd.to_numeric(df[a], errors="coerce")
                df["longitude"] = pd.to_numeric(df[b], errors="coerce")
                break

    if "latitude" in df.columns and "longitude" in df.columns:
        min_lon, min_lat, max_lon, max_lat = SF_BBOX
        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
        df = df[
            df["latitude"].between(min_lat, max_lat) &
            df["longitude"].between(min_lon, max_lon)
        ]

    df = df.dropna(subset=["latitude", "longitude"]).copy()

    gdf_blocks, tlen = _load_blocks()
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs="EPSG:4326")
    gdf = gpd.sjoin(gdf, gdf_blocks[["GEOID", "geometry"]], how="left", predicate="within")
    out = pd.DataFrame(gdf.drop(columns=["geometry", "index_right"], errors="ignore"))
    out["GEOID"] = normalize_geoid(out["GEOID"], tlen)
    out = out.dropna(subset=["GEOID"]).copy()
    return out


# ============================================================
# 911 FEATURE ENGINEERING
# ============================================================
def _build_call_group_flags(df: pd.DataFrame) -> pd.DataFrame:
    desc_col = _first_existing_col(df, [
        "call_type_final_desc",
        "call_type_original_desc",
        "call_type_final_notes",
        "call_type_original_notes",
        "call_type_final",
        "call_type_original",
    ])

    if desc_col is None:
        txt = pd.Series("", index=df.index, dtype="string")
    else:
        txt = df[desc_col].astype(str).str.lower().fillna("")

    violent_pat = r"(?:assault|battery|fight|violence|domestic|abuse|rape|sexual|homicide|stab|shoot|shots fired)"
    property_pat = r"(?:burglary|robbery|theft|larceny|shoplift|stolen|break[- ]?in|embezz|vandal|property)"
    weapon_pat = r"(?:weapon|gun|firearm|knife|shot|shots fired|armed)"
    disturbance_pat = r"(?:disturb|noise|party|dispute|trespass|harass|loiter|suspicious person)"
    vehicle_pat = r"(?:vehicle|traffic|collision|accident|parking|tow|carjacking|auto|dui)"
    suspicious_pat = r"(?:suspicious|welfare check|person down|unknown trouble|check well[- ]?being)"
    medical_pat = r"(?:medical|overdose|ambulance|injured|unconscious|breathing|sick)"

    df["is_violent_call"] = txt.str.contains(violent_pat, regex=True, na=False).astype("int8")
    df["is_property_call"] = txt.str.contains(property_pat, regex=True, na=False).astype("int8")
    df["is_weapon_call"] = txt.str.contains(weapon_pat, regex=True, na=False).astype("int8")
    df["is_disturbance_call"] = txt.str.contains(disturbance_pat, regex=True, na=False).astype("int8")
    df["is_vehicle_call"] = txt.str.contains(vehicle_pat, regex=True, na=False).astype("int8")
    df["is_suspicious_call"] = txt.str.contains(suspicious_pat, regex=True, na=False).astype("int8")
    df["is_medical_call"] = txt.str.contains(medical_pat, regex=True, na=False).astype("int8")
    return df


def _prepare_911_raw(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    if df.empty:
        return df

    ts_col = _first_existing_col(df, [
        "received_time",
        "received_datetime",
        "date",
        "datetime",
        "timestamp",
        "call_received_datetime",
    ])
    if ts_col is None:
        raise ValueError("Zaman kolonu bulunamadı (received_time / received_datetime / date).")

    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df[df[ts_col].notna()].copy()

    df["date"] = df[ts_col].dt.date
    df["event_hour"] = df[ts_col].dt.hour.astype("Int16")

    eh = pd.to_numeric(df["event_hour"], errors="coerce").fillna(0).astype(int) % 24
    start = (eh // 3) * 3
    df["hour_range"] = start.map(lambda s: f"{int(s):02d}-{int(min(s + 3, 24)):02d}")
    df["hr_key"] = start.astype("int16")

    pr_col = _first_existing_col(df, ["priority_final", "priority_original"])
    if pr_col is not None:
        pr_txt = df[pr_col].astype(str).str.strip().str.lower()
        df["is_priority_high"] = pr_txt.isin(["a", "1", "2", "high", "highest", "priority 1", "priority 2"]).astype("int8")
    else:
        df["is_priority_high"] = 0

    ov_col = _first_existing_col(df, ["onview_flag"])
    if ov_col is not None:
        ov_txt = df[ov_col].astype(str).str.strip().str.lower()
        df["is_onview"] = ov_txt.isin(["y", "yes", "true", "1"]).astype("int8")
    else:
        df["is_onview"] = 0

    sen_col = _first_existing_col(df, ["sensitive_call"])
    if sen_col is not None:
        df["is_sensitive"] = _safe_bool01(df[sen_col])
    else:
        df["is_sensitive"] = 0

    ag_col = _first_existing_col(df, ["agency"])
    if ag_col is not None:
        ag_txt = df[ag_col].astype(str).str.lower()
        df["is_police_agency"] = ag_txt.str.contains("police", na=False).astype("int8")
    else:
        df["is_police_agency"] = 0

    df = _build_call_group_flags(df)

    entry_col = _first_existing_col(df, ["entry_datetime"])
    dispatch_col = _first_existing_col(df, ["dispatch_datetime"])
    enroute_col = _first_existing_col(df, ["enroute_datetime"])
    onscene_col = _first_existing_col(df, ["onscene_datetime"])
    close_col = _first_existing_col(df, ["close_datetime"])

    if entry_col and dispatch_col:
        df["dispatch_delay_min"] = _num_minutes(df[entry_col], df[dispatch_col])
    else:
        df["dispatch_delay_min"] = np.nan

    if enroute_col and onscene_col:
        df["travel_time_min"] = _num_minutes(df[enroute_col], df[onscene_col])
    else:
        df["travel_time_min"] = np.nan

    if ts_col and onscene_col:
        df["total_response_min"] = _num_minutes(df[ts_col], df[onscene_col])
    else:
        df["total_response_min"] = np.nan

    if ts_col and close_col:
        df["close_time_min"] = _num_minutes(df[ts_col], df[close_col])
    else:
        df["close_time_min"] = np.nan

    for c in ["dispatch_delay_min", "travel_time_min", "total_response_min", "close_time_min"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df.loc[(df[c] < 0) | (df[c] > 24 * 60), c] = np.nan

    return df


# ============================================================
# SUMMARY BUILDERS
# ============================================================
def make_standard_summary(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=[
            "GEOID", "date", "hour_range", "hr_key",
            "911_request_count_hour_range",
            "911_request_count_daily(before_24_hours)"
        ])

    df = _prepare_911_raw(raw)
    has_geoid = "GEOID" in df.columns

    grp_hr = (["GEOID"] if has_geoid else []) + ["date", "hour_range", "hr_key"]
    grp_day = (["GEOID"] if has_geoid else []) + ["date"]

    hr_agg = df.groupby(grp_hr, dropna=False, observed=True).agg(
        **{
            "911_request_count_hour_range": ("hour_range", "size"),
            "911_priority_high_count_hour_range": ("is_priority_high", "sum"),
            "911_onview_count_hour_range": ("is_onview", "sum"),
            "911_sensitive_count_hour_range": ("is_sensitive", "sum"),
            "911_police_agency_count_hour_range": ("is_police_agency", "sum"),

            "911_violent_call_count_hour_range": ("is_violent_call", "sum"),
            "911_property_call_count_hour_range": ("is_property_call", "sum"),
            "911_weapon_call_count_hour_range": ("is_weapon_call", "sum"),
            "911_disturbance_call_count_hour_range": ("is_disturbance_call", "sum"),
            "911_vehicle_call_count_hour_range": ("is_vehicle_call", "sum"),
            "911_suspicious_call_count_hour_range": ("is_suspicious_call", "sum"),
            "911_medical_call_count_hour_range": ("is_medical_call", "sum"),

            "911_mean_dispatch_delay_min_hour_range": ("dispatch_delay_min", "mean"),
            "911_mean_travel_time_min_hour_range": ("travel_time_min", "mean"),
            "911_mean_total_response_min_hour_range": ("total_response_min", "mean"),
            "911_mean_close_time_min_hour_range": ("close_time_min", "mean"),
        }
    ).reset_index()

    day_agg = df.groupby(grp_day, dropna=False, observed=True).agg(
        **{
            "911_request_count_daily(before_24_hours)": ("date", "size"),
            "911_priority_high_count_daily": ("is_priority_high", "sum"),
            "911_onview_count_daily": ("is_onview", "sum"),
            "911_sensitive_count_daily": ("is_sensitive", "sum"),

            "911_violent_call_count_daily": ("is_violent_call", "sum"),
            "911_property_call_count_daily": ("is_property_call", "sum"),
            "911_weapon_call_count_daily": ("is_weapon_call", "sum"),
            "911_disturbance_call_count_daily": ("is_disturbance_call", "sum"),
            "911_vehicle_call_count_daily": ("is_vehicle_call", "sum"),
            "911_suspicious_call_count_daily": ("is_suspicious_call", "sum"),
            "911_medical_call_count_daily": ("is_medical_call", "sum"),

            "911_mean_dispatch_delay_min_daily": ("dispatch_delay_min", "mean"),
            "911_mean_travel_time_min_daily": ("travel_time_min", "mean"),
            "911_mean_total_response_min_daily": ("total_response_min", "mean"),
            "911_mean_close_time_min_daily": ("close_time_min", "mean"),
        }
    ).reset_index()

    out = hr_agg.merge(day_agg, on=grp_day, how="left")

    denom = out["911_request_count_hour_range"].replace(0, np.nan)
    out["911_priority_high_share_hour_range"] = out["911_priority_high_count_hour_range"] / denom
    out["911_onview_share_hour_range"] = out["911_onview_count_hour_range"] / denom
    out["911_sensitive_share_hour_range"] = out["911_sensitive_count_hour_range"] / denom
    out["911_violent_share_hour_range"] = out["911_violent_call_count_hour_range"] / denom
    out["911_property_share_hour_range"] = out["911_property_call_count_hour_range"] / denom
    out["911_weapon_share_hour_range"] = out["911_weapon_call_count_hour_range"] / denom
    out["911_disturbance_share_hour_range"] = out["911_disturbance_call_count_hour_range"] / denom
    out["911_vehicle_share_hour_range"] = out["911_vehicle_call_count_hour_range"] / denom
    out["911_suspicious_share_hour_range"] = out["911_suspicious_call_count_hour_range"] / denom
    out["911_medical_share_hour_range"] = out["911_medical_call_count_hour_range"] / denom

    for c in [c for c in out.columns if c.endswith("_share_hour_range")]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).clip(0, 1)

    cols_tail = [c for c in ["date", "hour_range", "GEOID"] if c in out.columns]
    cols = [c for c in out.columns if c not in cols_tail] + cols_tail
    return out[cols]


def summary_from_local(path: Path | str, min_date=None) -> pd.DataFrame:
    log(f"📥 Yerel 911 tabanı okunuyor: {path}")
    df = pd.read_csv(path, low_memory=False, dtype={"GEOID": "string"})

    is_already_summary = {"date", "hour_range"}.issubset(df.columns) and (
        "911_request_count_hour_range" in df.columns or
        "call_count" in df.columns or
        "count" in df.columns or
        "requests" in df.columns or
        "n" in df.columns
    )

    if is_already_summary:
        cnt_col = _first_existing_col(df, ["911_request_count_hour_range", "call_count", "count", "requests", "n"])
        if cnt_col != "911_request_count_hour_range":
            df = df.rename(columns={cnt_col: "911_request_count_hour_range"})

        df["date"] = to_date(df["date"])
        if "GEOID" in df.columns:
            df["GEOID"] = normalize_geoid(df["GEOID"], DEFAULT_GEOID_LEN)

        df["hour_range"] = df["hour_range"].apply(normalize_hour_range)
        
        bad_hr = df["hour_range"].isna().sum()
        if bad_hr:
            log(f"⚠️ LOCAL hour_range parse edilemeyen: {bad_hr:,}")

        if "hr_key" not in df.columns:
            df["hr_key"] = df["hour_range"].astype(str).str.extract(r"^(\d{2})").astype(float)

        if "911_request_count_daily(before_24_hours)" not in df.columns:
            keys = (["GEOID"] if "GEOID" in df.columns else []) + ["date"]
            day = (
                df.groupby(keys, dropna=False, observed=True)["911_request_count_hour_range"]
                .sum()
                .reset_index(name="911_request_count_daily(before_24_hours)")
            )
            df = df.merge(day, on=keys, how="left")

        if min_date is not None:
            df = df[df["date"] >= min_date]

        cols_tail = [c for c in ["date", "hour_range", "GEOID"] if c in df.columns]
        cols = [c for c in df.columns if c not in cols_tail] + cols_tail
        return df[cols]

    std = make_standard_summary(df)
    if min_date is not None:
        std = std[std["date"] >= min_date]
    return std


def summary_from_release(url: str, min_date=None) -> pd.DataFrame:
    log(f"⬇️ Release 911 özeti indiriliyor: {url}")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    tmp = OUT_DIR / "_tmp_911.csv"
    ensure_parent(str(tmp))
    tmp.write_bytes(r.content)

    df = pd.read_csv(tmp, low_memory=False, dtype={"GEOID": "string"})
    is_already_summary = {"date", "hour_range"}.issubset(df.columns) and (
        "911_request_count_hour_range" in df.columns or
        "call_count" in df.columns or
        "count" in df.columns or
        "requests" in df.columns or
        "n" in df.columns
    )

    if is_already_summary:
        cnt_col = _first_existing_col(df, ["911_request_count_hour_range", "call_count", "count", "requests", "n"])
        if cnt_col != "911_request_count_hour_range":
            df = df.rename(columns={cnt_col: "911_request_count_hour_range"})

        df["date"] = to_date(df["date"])
        if "GEOID" in df.columns:
            df["GEOID"] = normalize_geoid(df["GEOID"], DEFAULT_GEOID_LEN)

        df["hour_range"] = df["hour_range"].apply(normalize_hour_range)
        
        bad_hr = df["hour_range"].isna().sum()
        if bad_hr:
            log(f"⚠️ RELEASE hour_range parse edilemeyen: {bad_hr:,}")

        if "hr_key" not in df.columns:
            df["hr_key"] = df["hour_range"].astype(str).str.extract(r"^(\d{2})").astype(float)

        if "911_request_count_daily(before_24_hours)" not in df.columns:
            keys = (["GEOID"] if "GEOID" in df.columns else []) + ["date"]
            day = (
                df.groupby(keys, dropna=False, observed=True)["911_request_count_hour_range"]
                .sum()
                .reset_index(name="911_request_count_daily(before_24_hours)")
            )
            df = df.merge(day, on=keys, how="left")

        if min_date is not None:
            df = df[df["date"] >= min_date]

        cols_tail = [c for c in ["date", "hour_range", "GEOID"] if c in df.columns]
        cols = [c for c in df.columns if c not in cols_tail] + cols_tail
        return df[cols]

    std = make_standard_summary(df)
    if min_date is not None:
        std = std[std["date"] >= min_date]
    return std


def ensure_local_911_base() -> Optional[Path]:
    ARTIFACT_NAME = os.getenv("ARTIFACT_NAME", "sf-crime-pipeline-output").strip()
    prefer_names = ["sf_911_last_5_year_y.csv", "sf_911_last_5_year.csv"]

    crime_grid_candidates = [
        OUT_DIR / "sf_crime_y.parquet",
        Path(BASE_DIR) / "sf_crime_y.parquet",
        Path("./sf_crime_y.parquet"),
        Path("crime_prediction_data/sf_crime_y.parquet"),
        OUT_DIR / "crime_prediction_data/sf_crime_y.parquet",
        Path(BASE_DIR) / "crime_prediction_data/sf_crime_y.parquet",
        Path(ARTIFACT_NAME) / "sf_crime_y.parquet",
        Path(ARTIFACT_NAME) / "crime_prediction_data/sf_crime_y.parquet",
    ]

    crime_grid_path = next(
        (p for p in crime_grid_candidates if p.exists()),
        None
    )
    crime_grid_dir = crime_grid_path.parent if crime_grid_path else None

    def _ok(p: Path) -> bool:
        if not p or not p.exists() or p.is_dir():
            return False
        if p.suffix.lower() != ".csv":
            return False
        if is_lfs_pointer_file(p):
            return False
        try:
            if p.stat().st_size < 200:
                return False
        except Exception:
            return False
        return True

    roots = [OUT_DIR, Path(BASE_DIR), Path.cwd()]
    if crime_grid_dir:
        roots.insert(0, crime_grid_dir)

    artifact_dir = Path(ARTIFACT_NAME)
    if artifact_dir.exists() and artifact_dir.is_dir():
        roots.insert(0, artifact_dir)

    for r in [Path.cwd(), Path(BASE_DIR), OUT_DIR]:
        try:
            for d in r.glob("sf-crime-pipeline-output*"):
                if d.is_dir():
                    roots.append(d)
        except Exception:
            pass

    for nm in prefer_names:
        for rt in roots:
            for cand in [rt / nm, rt / "crime_prediction_data" / nm, rt / "outputs" / nm]:
                if _ok(cand):
                    log(f"📦 911 base bulundu: {cand}")
                    return cand

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

    def _ok(p: Path) -> bool:
        if not p or not p.exists() or p.is_dir():
            return False
        if p.suffix.lower() != ".csv":
            return False
        if is_lfs_pointer_file(p):
            return False
        try:
            if p.stat().st_size < 200:
                return False
        except Exception:
            return False
        return True

    roots = [OUT_DIR, Path(BASE_DIR), Path.cwd()]
    if crime_grid_dir:
        roots.insert(0, crime_grid_dir)

    artifact_dir = Path(ARTIFACT_NAME)
    if artifact_dir.exists() and artifact_dir.is_dir():
        roots.insert(0, artifact_dir)

    for r in [Path.cwd(), Path(BASE_DIR), OUT_DIR]:
        try:
            for d in r.glob("sf-crime-pipeline-output*"):
                if d.is_dir():
                    roots.append(d)
        except Exception:
            pass

    for nm in prefer_names:
        for rt in roots:
            for cand in [rt / nm, rt / "crime_prediction_data" / nm, rt / "outputs" / nm]:
                if _ok(cand):
                    log(f"📦 911 base bulundu: {cand}")
                    return cand

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


# ============================================================
# INCREMENTAL FETCH
# ============================================================
def try_small_request(params, headers):
    p = dict(params)
    p["$limit"], p["$offset"] = 1, 0
    r = requests.get(SF911_API_URL, headers=headers, params=p, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return r


def fetch_range_all_chunks(start_day, end_day) -> Optional[pd.DataFrame]:
    dt_candidates = ["received_time", "received_datetime", "date", "datetime", "call_datetime", "received_dttm", "call_date"]
    headers = {"X-App-Token": SF_APP_TOKEN} if SF_APP_TOKEN else {}
    rng_start = f"{start_day}T00:00:00"
    rng_end = f"{end_day}T23:59:59"
    chosen_dt, last_err = None, None

    for dt_col in dt_candidates:
        base_where = f"{dt_col} between '{rng_start}' and '{rng_end}'"
        for wc in [base_where + (f" AND {AGENCY_FILTER}" if AGENCY_FILTER else ""), base_where]:
            try:
                try_small_request({"$where": wc}, headers)
                chosen_dt = dt_col
                break
            except Exception as e:
                last_err = e
                continue
        if chosen_dt:
            break

    if chosen_dt is None:
        log(f"    ❌ Aralık için uygun datetime kolonu bulunamadı. Son hata: {last_err}")
        return None

    pieces, offset, page = [], 0, 1
    where_list = [
        f"{chosen_dt} between '{rng_start}' and '{rng_end}'" + (f" AND {AGENCY_FILTER}" if AGENCY_FILTER else ""),
        f"{chosen_dt} between '{rng_start}' and '{rng_end}'"
    ]

    while True:
        df = None
        for attempt in range(MAX_RETRIES):
            try:
                r = requests.get(
                    SF911_API_URL,
                    headers=headers,
                    params={"$where": where_list[0], "$limit": CHUNK_LIMIT, "$offset": offset},
                    timeout=REQUEST_TIMEOUT
                )
                if r.status_code == 400:
                    r = requests.get(
                        SF911_API_URL,
                        headers=headers,
                        params={"$where": where_list[1], "$limit": CHUNK_LIMIT, "$offset": offset},
                        timeout=REQUEST_TIMEOUT
                    )
                r.raise_for_status()
                df = pd.read_json(io.BytesIO(r.content))
                break
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    log(f"    ❌ range page {page} (offset={offset}) hata: {e}")
                    df = None
                    break
                time.sleep(1.0 + attempt * 0.5)

        if df is None or df.empty:
            if page == 1:
                log("    (bu aralıkta veri yok)")
            break

        log(f"    + {len(df)} satır (range-page={page}, offset={offset})")
        pieces.append(df)

        if len(df) < CHUNK_LIMIT:
            break

        offset += CHUNK_LIMIT
        page += 1
        time.sleep(SLEEP_BETWEEN_REQS)

    if not pieces:
        return None

    return pd.concat(pieces, ignore_index=True)


def fetch_v3_range_all_chunks(start_day, end_day) -> Optional[pd.DataFrame]:
    from requests.adapters import HTTPAdapter, Retry

    sess = requests.Session()
    retries = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    sess.mount("https://", HTTPAdapter(max_retries=retries))
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    headers = {"Accept": "application/json"}
    if SF_APP_TOKEN:
        headers["X-App-Token"] = SF_APP_TOKEN

    dt_candidates = ["received_time", "received_datetime", "date", "datetime", "call_datetime", "received_dttm", "call_date"]
    rng_start = f"{start_day}T00:00:00"
    rng_end = f"{end_day}T23:59:59"

    chosen_dt, cols = None, None
    for dtc in dt_candidates:
        where = f"{dtc} between '{rng_start}' and '{rng_end}'"
        if AGENCY_FILTER:
            where += f" AND {AGENCY_FILTER}"
        q = f"SELECT * WHERE {where} LIMIT 1 OFFSET 0"
        try:
            r = sess.get(SF911_API_URL, params={"query": q}, headers=headers, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            obj = r.json()
            if obj.get("data"):
                chosen_dt = dtc
                cols = [c.get("fieldName") or c.get("name") or f"c{i}" for i, c in enumerate(obj.get("meta", {}).get("view", {}).get("columns", []))]
                break
        except Exception:
            continue

    if not chosen_dt:
        log("    ❌ v3: uygun datetime kolonu bulunamadı.")
        return None

    all_rows, offset, page = [], 0, 1
    while True:
        where = f"{chosen_dt} between '{rng_start}' and '{rng_end}'"
        if AGENCY_FILTER:
            where += f" AND {AGENCY_FILTER}"
        q = f"SELECT * WHERE {where} LIMIT {V3_PAGE_LIMIT} OFFSET {offset}"

        got = 0
        for attempt in range(MAX_RETRIES):
            try:
                r = sess.get(SF911_API_URL, params={"query": q}, headers=headers, timeout=REQUEST_TIMEOUT)
                r.raise_for_status()
                obj = r.json()
                data = obj.get("data", [])
                if data:
                    for row in data:
                        if isinstance(row, list):
                            all_rows.append({cols[i]: (row[i] if i < len(cols) else None) for i in range(len(cols))})
                        elif isinstance(row, dict):
                            all_rows.append(row)
                    got = len(data)
                break
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    log(f"    ❌ v3 range page {page} (offset={offset}) hata: {e}")
                time.sleep(1.0 + attempt * 0.5)

        if got < V3_PAGE_LIMIT:
            break

        offset += V3_PAGE_LIMIT
        page += 1
        time.sleep(SLEEP_BETWEEN_REQS)

    return pd.DataFrame(all_rows)


def write_recent_csv(raw: pd.DataFrame, hours: int = SF911_RECENT_HOURS):
    ts_col = _first_existing_col(raw, [
        "received_time",
        "received_datetime",
        "date",
        "datetime",
        "timestamp",
        "call_received_datetime",
        "ts",
    ])
    if not ts_col:
        return

    tmp = raw.copy()
    tmp["ts"] = pd.to_datetime(tmp[ts_col], errors="coerce")
    tmp = tmp[tmp["ts"].notna()]
    if tmp.empty:
        return

    lat_col = _first_existing_col(raw, ["latitude", "lat", "y"])
    lon_col = _first_existing_col(raw, ["longitude", "lon", "x"])

    tmax = tmp["ts"].max()
    cutoff = tmax - pd.Timedelta(hours=hours)

    out = pd.DataFrame({"ts": tmp["ts"]})
    if lat_col:
        out["lat"] = pd.to_numeric(tmp[lat_col], errors="coerce")
    if lon_col:
        out["lon"] = pd.to_numeric(tmp[lon_col], errors="coerce")

    out = out[out["ts"] >= cutoff].copy()

    path = OUT_DIR / "sf_911_recent.csv"
    safe_save_csv(out, str(path))
    log(f"ℹ️ sf_911_recent.csv yazıldı (son {hours} saat): {len(out)} satır")


def incremental_summary(start_day: datetime.date, end_day: datetime.date) -> pd.DataFrame:
    if start_day is None or end_day is None or end_day < start_day:
        return pd.DataFrame()

    log(f"🌐 API artımlı: {start_day} → {end_day} ({(end_day - start_day).days + 1} gün)")
    raw = None
    if BULK_RANGE:
        raw = fetch_v3_range_all_chunks(start_day, end_day) if IS_V3 else fetch_range_all_chunks(start_day, end_day)

    try:
        if raw is not None and not raw.empty:
            write_recent_csv(raw, hours=SF911_RECENT_HOURS)
    except Exception as e:
        log(f"⚠️ recent yazımı atlandı: {e}")

    if raw is None or raw.empty:
        return pd.DataFrame()

    try:
        raw = ensure_geoid(raw)
    except Exception as e:
        log(f"⚠️ ensure_geoid sırasında hata: {e}; GEOID’siz özet üretilecek")

    return make_standard_summary(raw)


# ============================================================
# MAIN — LOCAL/RELEASE → INCREMENT → ENRICH → MERGE
# ============================================================
five_years_ago = datetime.now(timezone.utc).date() - timedelta(days=1825)

log(f"📁 911 yerel özet yolu: {local_summary_path}")

base_csv_path = ensure_local_911_base()
if base_csv_path is not None:
    final_911 = summary_from_local(base_csv_path, min_date=five_years_ago)
    safe_save_csv(final_911, str(local_summary_path))
    safe_save_csv(final_911, str(y_summary_path))
    log(f"✅ Yerel 911 özet kaydedildi → {local_summary_path} & {y_summary_path} (satır: {len(final_911)})")
else:
    release_url = _pick_working_release_url(RAW_911_URL_CANDIDATES)
    final_911 = summary_from_release(release_url, min_date=five_years_ago)
    safe_save_csv(final_911, str(local_summary_path))
    safe_save_csv(final_911, str(y_summary_path))
    log(f"✅ Release özet kaydedildi → {local_summary_path} & {y_summary_path} (satır: {len(final_911)})")

base_max_date = to_date(final_911["date"]).max() if not final_911.empty else None
today_sf = (datetime.now(SF_TZ) if SF_TZ is not None else datetime.now()).date()

if base_max_date is None:
    fetch_start, fetch_end = today_sf, today_sf
else:
    fetch_end = today_sf
    fetch_start = max(five_years_ago, base_max_date - timedelta(days=max(1, SF911_REINGEST_DAYS)))
    fetch_start = min(fetch_start, fetch_end)

log(f"🗓️ İndirme aralığı: {fetch_start} → {fetch_end} ({(fetch_end - fetch_start).days + 1} gün)")

inc = incremental_summary(fetch_start, fetch_end)
if inc is not None and not inc.empty:
    if "GEOID" in inc.columns:
        inc["GEOID"] = normalize_geoid(inc["GEOID"], DEFAULT_GEOID_LEN)
    inc["date"] = to_date(inc["date"])

    before = len(final_911)
    final_911 = pd.concat([final_911, inc], ignore_index=True)

    subset_cols = [c for c in ["GEOID", "date", "hour_range"] if c in final_911.columns]
    if subset_cols:
        final_911 = (
            final_911.dropna(subset=["date"])
            .sort_values(subset_cols)
            .drop_duplicates(subset=subset_cols, keep="last")
        )
    else:
        final_911 = (
            final_911
            .dropna(subset=["date"])
            .sort_values(["date"])
            .drop_duplicates(keep="last")
        )
    
    final_911 = final_911[final_911["date"] >= five_years_ago].copy()

    safe_save_csv(final_911, str(local_summary_path))
    safe_save_csv(final_911, str(y_summary_path))
    log(f"💾 911 özet GÜNCELLENDİ (base+API) → {local_summary_path} & {y_summary_path} (+{len(final_911) - before:,} satır)")
else:
    log("ℹ️ API tarafında yeni gün yok veya boş döndü; taban veri geçerli.")

if final_911 is None or final_911.empty:
    log("⚠️ 911 özeti üretilemedi (boş). Çıkılıyor.")
    raise SystemExit(0)

# ============================================================
# STANDARDIZE + DERIVED KEYS
# ============================================================
hr_pat = re.compile(r"^\s*(\d{1,2})\s*-\s*(\d{1,2})\s*$")


def _hr_key_from_range(hr):
    m = hr_pat.match(str(hr))
    return int(m.group(1)) % 24 if m else None


final_911 = final_911.dropna(subset=["GEOID", "date", "hour_range"]).copy()
final_911["GEOID"] = normalize_geoid(final_911["GEOID"], DEFAULT_GEOID_LEN)
final_911["date"] = to_date(final_911["date"])

if "hr_key" not in final_911.columns:
    final_911["hr_key"] = final_911["hour_range"].apply(_hr_key_from_range)

final_911["hr_key"] = pd.to_numeric(final_911["hr_key"], errors="coerce").fillna(0).astype("int16")
_date_ts = pd.to_datetime(final_911["date"], errors="coerce")
final_911["day_of_week"] = _date_ts.dt.weekday.astype("int8")
final_911["month"] = _date_ts.dt.month.astype("int8")

_season_map = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Fall", 10: "Fall", 11: "Fall"
}
final_911["season"] = final_911["month"].map(_season_map).astype("category")

log_shape(final_911, "911 summary (normalize)")
log_date_range(final_911, "date", "911")

# ============================================================
# SLOT / DAY TABLES
# ============================================================
slot_keep = [c for c in [
    "GEOID", "date", "hr_key", "hour_range",
    "911_request_count_hour_range",
    "911_priority_high_count_hour_range",
    "911_onview_count_hour_range",
    "911_sensitive_count_hour_range",
    "911_police_agency_count_hour_range",
    "911_violent_call_count_hour_range",
    "911_property_call_count_hour_range",
    "911_weapon_call_count_hour_range",
    "911_disturbance_call_count_hour_range",
    "911_vehicle_call_count_hour_range",
    "911_suspicious_call_count_hour_range",
    "911_medical_call_count_hour_range",
    "911_priority_high_share_hour_range",
    "911_onview_share_hour_range",
    "911_sensitive_share_hour_range",
    "911_violent_share_hour_range",
    "911_property_share_hour_range",
    "911_weapon_share_hour_range",
    "911_disturbance_share_hour_range",
    "911_vehicle_share_hour_range",
    "911_suspicious_share_hour_range",
    "911_medical_share_hour_range",
    "911_mean_dispatch_delay_min_hour_range",
    "911_mean_travel_time_min_hour_range",
    "911_mean_total_response_min_hour_range",
    "911_mean_close_time_min_hour_range",
] if c in final_911.columns]

_slot_unique = final_911[slot_keep].drop_duplicates(subset=["GEOID", "date", "hr_key"]).copy()
_slot_unique = _slot_unique.sort_values(["GEOID", "date", "hr_key"]).reset_index(drop=True)

day_keep = [c for c in [
    "GEOID", "date",
    "911_request_count_daily(before_24_hours)",
    "911_priority_high_count_daily",
    "911_onview_count_daily",
    "911_sensitive_count_daily",
    "911_violent_call_count_daily",
    "911_property_call_count_daily",
    "911_weapon_call_count_daily",
    "911_disturbance_call_count_daily",
    "911_vehicle_call_count_daily",
    "911_suspicious_call_count_daily",
    "911_medical_call_count_daily",
    "911_mean_dispatch_delay_min_daily",
    "911_mean_travel_time_min_daily",
    "911_mean_total_response_min_daily",
    "911_mean_close_time_min_daily",
] if c in final_911.columns]

_day_unique = final_911[day_keep].drop_duplicates(subset=["GEOID", "date"]).copy()
_day_unique = _day_unique.sort_values(["GEOID", "date"]).reset_index(drop=True)

# ============================================================
# SLOT ROLLING
# 3 saatlik grid üzerinde:
# 1h alias = son 1 slot (uyumluluk için)
# 3h = son 1 slot
# 6h = son 2 slot
# 24h = son 8 slot
# 3d = son 24 slot
# 7d = son 56 slot
# ============================================================
ROLLING_SLOT_WINDOWS = {
    "1h": 1,
    "3h": 1,
    "6h": 2,
    "24h": 8,
    "3d": 24,
    "7d": 56,
}

slot_roll_sources = {
    "911_request_count_hour_range": "911_geo_last",
    "911_priority_high_count_hour_range": "911_geo_highprio_last",
    "911_violent_call_count_hour_range": "911_geo_violent_last",
    "911_property_call_count_hour_range": "911_geo_property_last",
    "911_weapon_call_count_hour_range": "911_geo_weapon_last",
}

for src_col, prefix in slot_roll_sources.items():
    if src_col not in _slot_unique.columns:
        continue
    for label, n_slots in ROLLING_SLOT_WINDOWS.items():
        _slot_unique[f"{prefix}{label}"] = (
            _slot_unique.groupby("GEOID")[src_col]
            .transform(lambda s: s.rolling(n_slots, min_periods=1).sum().shift(1))
            .astype("float32")
        )

# hr_cnt backward compatibility
if "911_request_count_hour_range" in _slot_unique.columns:
    _slot_unique["hr_cnt"] = pd.to_numeric(_slot_unique["911_request_count_hour_range"], errors="coerce").fillna(0).astype("float32")

# ============================================================
# DAY ROLLING
# ============================================================
day_roll_sources = {
    "911_request_count_daily(before_24_hours)": "911_day_total_last",
    "911_priority_high_count_daily": "911_day_highprio_last",
    "911_violent_call_count_daily": "911_day_violent_last",
    "911_property_call_count_daily": "911_day_property_last",
    "911_weapon_call_count_daily": "911_day_weapon_last",
}

ROLLING_DAY_WINDOWS = {
    "1d": 1,
    "3d": 3,
    "7d": 7,
}

for src_col, prefix in day_roll_sources.items():
    if src_col not in _day_unique.columns:
        continue
    for label, n_days in ROLLING_DAY_WINDOWS.items():
        _day_unique[f"{prefix}{label}"] = (
            _day_unique.groupby("GEOID")[src_col]
            .transform(lambda s: s.rolling(n_days, min_periods=1).sum().shift(1))
            .astype("float32")
        )

# daily_cnt backward compatibility
if "911_request_count_daily(before_24_hours)" in _day_unique.columns:
    _day_unique["daily_cnt"] = pd.to_numeric(_day_unique["911_request_count_daily(before_24_hours)"], errors="coerce").fillna(0).astype("float32")

# ============================================================
# NEIGHBOR FEATURES
# ============================================================
def build_neighbors(method: str = "touches", radius_m: float = 500.0) -> pd.DataFrame:
    gdf_blocks, _ = _load_blocks()
    tracts = gdf_blocks.dissolve(by="GEOID", as_index=False)

    if method == "radius":
        tr_utm = tracts.to_crs("EPSG:26910")
        buf = tr_utm.buffer(radius_m)
        g_buf = gpd.GeoDataFrame(
            tr_utm[["GEOID"]].copy(),
            geometry=buf,
            crs=tr_utm.crs
        )

        join = gpd.sjoin(
            g_buf,
            tr_utm[["GEOID", "geometry"]].rename(columns={"GEOID": "nbr"}),
            predicate="intersects",
            how="left"
        )
        edges = join[["GEOID", "nbr"]].copy()

    else:
        join = gpd.sjoin(
            tracts[["GEOID", "geometry"]],
            tracts[["GEOID", "geometry"]].rename(columns={"GEOID": "nbr"}),
            predicate="touches",
            how="left"
        )
        edges = join[["GEOID", "nbr"]].copy()

    edges = edges.dropna(subset=["GEOID", "nbr"]).copy()
    edges = edges[edges["GEOID"] != edges["nbr"]].copy()

    edges["pair"] = edges.apply(
        lambda r: tuple(sorted((str(r["GEOID"]), str(r["nbr"])))),
        axis=1
    )
    edges = edges.drop_duplicates("pair").drop(columns=["pair"])

    edges["GEOID"] = normalize_geoid(edges["GEOID"], DEFAULT_GEOID_LEN)
    edges["nbr"] = normalize_geoid(edges["nbr"], DEFAULT_GEOID_LEN)

    return pd.DataFrame(edges)


neighbors_df = None
if ENABLE_NEIGHBORS:
    try:
        neighbors_df = build_neighbors(NEIGHBOR_METHOD, NEIGHBOR_RADIUS_M)
        log_shape(neighbors_df, f"Komşu haritası ({NEIGHBOR_METHOD})")
    except Exception as e:
        log(f"⚠️ Komşu haritası üretilemedi: {e}")
        neighbors_df = None

_neighbor_roll = None
if neighbors_df is not None and not neighbors_df.empty:
    nbr_src = _day_unique.rename(columns={"GEOID": "nbr"}).copy()

    nbr_use_cols = [c for c in [
        "nbr", "date",
        "911_request_count_daily(before_24_hours)",
        "911_violent_call_count_daily",
        "911_property_call_count_daily",
        "911_weapon_call_count_daily",
    ] if c in nbr_src.columns]

    day_nbr = neighbors_df.merge(nbr_src[nbr_use_cols], on="nbr", how="left")

    agg_map = {}
    if "911_request_count_daily(before_24_hours)" in day_nbr.columns:
        agg_map["nbr_daily_cnt"] = ("911_request_count_daily(before_24_hours)", "sum")
    if "911_violent_call_count_daily" in day_nbr.columns:
        agg_map["nbr_violent_daily_cnt"] = ("911_violent_call_count_daily", "sum")
    if "911_property_call_count_daily" in day_nbr.columns:
        agg_map["nbr_property_daily_cnt"] = ("911_property_call_count_daily", "sum")
    if "911_weapon_call_count_daily" in day_nbr.columns:
        agg_map["nbr_weapon_daily_cnt"] = ("911_weapon_call_count_daily", "sum")

    _neighbor_roll = day_nbr.groupby(["GEOID", "date"], as_index=False, observed=True).agg(**agg_map)
    _neighbor_roll = _neighbor_roll.sort_values(["GEOID", "date"]).reset_index(drop=True)

    nbr_roll_sources = {
        "nbr_daily_cnt": "911_neighbors_last",
        "nbr_violent_daily_cnt": "911_neighbors_violent_last",
        "nbr_property_daily_cnt": "911_neighbors_property_last",
        "nbr_weapon_daily_cnt": "911_neighbors_weapon_last",
    }

    for src_col, prefix in nbr_roll_sources.items():
        if src_col not in _neighbor_roll.columns:
            continue
        for label, n_days in ROLLING_DAY_WINDOWS.items():
            _neighbor_roll[f"{prefix}{label}"] = (
                _neighbor_roll.groupby("GEOID")[src_col]
                .transform(lambda s: s.rolling(n_days, min_periods=1).sum().shift(1))
                .astype("float32")
            )

# ============================================================
# MERGE STRATEJİSİ
# ============================================================
_enriched = final_911.copy()

slot_merge_cols = [c for c in _slot_unique.columns if c not in ["hour_range"]]
_enriched = _enriched.merge(
    _slot_unique[slot_merge_cols],
    on=["GEOID", "date", "hr_key"],
    how="left",
    suffixes=("", "_slot")
)

day_merge_cols = [c for c in _day_unique.columns]
_enriched = _enriched.merge(
    _day_unique[day_merge_cols],
    on=["GEOID", "date"],
    how="left",
    suffixes=("", "_day")
)

if _neighbor_roll is not None:
    _enriched = _enriched.merge(_neighbor_roll, on=["GEOID", "date"], how="left")

KEEP_911_COLS = [c for c in [

    # KEYS
    "GEOID", "date", "hour_range", "hr_key",

    # BASE COUNTS
    "911_request_count_hour_range",
    "911_request_count_daily(before_24_hours)",

    # PRIORITY / FLAGS
    "911_priority_high_count_hour_range",
    "911_onview_count_hour_range",
    "911_sensitive_count_hour_range",
    "911_police_agency_count_hour_range",

    # CALL TYPE COUNTS
    "911_violent_call_count_hour_range",
    "911_property_call_count_hour_range",
    "911_weapon_call_count_hour_range",
    "911_disturbance_call_count_hour_range",
    "911_vehicle_call_count_hour_range",
    "911_suspicious_call_count_hour_range",
    "911_medical_call_count_hour_range",

    # SHARES
    "911_priority_high_share_hour_range",
    "911_onview_share_hour_range",
    "911_sensitive_share_hour_range",
    "911_violent_share_hour_range",
    "911_property_share_hour_range",
    "911_weapon_share_hour_range",
    "911_disturbance_share_hour_range",
    "911_vehicle_share_hour_range",
    "911_suspicious_share_hour_range",
    "911_medical_share_hour_range",

    # RESPONSE TIMES (HOUR)
    "911_mean_dispatch_delay_min_hour_range",
    "911_mean_travel_time_min_hour_range",
    "911_mean_total_response_min_hour_range",
    "911_mean_close_time_min_hour_range",

    # DAILY COUNTS
    "911_priority_high_count_daily",
    "911_onview_count_daily",
    "911_sensitive_count_daily",
    "911_violent_call_count_daily",
    "911_property_call_count_daily",
    "911_weapon_call_count_daily",
    "911_disturbance_call_count_daily",
    "911_vehicle_call_count_daily",
    "911_suspicious_call_count_daily",
    "911_medical_call_count_daily",

    # DAILY RESPONSE TIMES
    "911_mean_dispatch_delay_min_daily",
    "911_mean_travel_time_min_daily",
    "911_mean_total_response_min_daily",
    "911_mean_close_time_min_daily",

    # GEO ROLLING (NO 1h!)
    "911_geo_last3h",
    "911_geo_last6h",
    "911_geo_last24h",
    "911_geo_last3d",
    "911_geo_last7d",

    "911_geo_highprio_last3h",
    "911_geo_highprio_last6h",
    "911_geo_highprio_last24h",
    "911_geo_highprio_last3d",
    "911_geo_highprio_last7d",

    "911_geo_violent_last3h",
    "911_geo_violent_last6h",
    "911_geo_violent_last24h",
    "911_geo_violent_last3d",
    "911_geo_violent_last7d",

    "911_geo_property_last3h",
    "911_geo_property_last6h",
    "911_geo_property_last24h",
    "911_geo_property_last3d",
    "911_geo_property_last7d",

    "911_geo_weapon_last3h",
    "911_geo_weapon_last6h",
    "911_geo_weapon_last24h",
    "911_geo_weapon_last3d",
    "911_geo_weapon_last7d",

    # DAY ROLLING
    "911_day_total_last1d",
    "911_day_total_last3d",
    "911_day_total_last7d",

    "911_day_highprio_last1d",
    "911_day_highprio_last3d",
    "911_day_highprio_last7d",

    "911_day_violent_last1d",
    "911_day_violent_last3d",
    "911_day_violent_last7d",

    "911_day_property_last1d",
    "911_day_property_last3d",
    "911_day_property_last7d",

    "911_day_weapon_last1d",
    "911_day_weapon_last3d",
    "911_day_weapon_last7d",

    # NEIGHBORS
    "911_neighbors_last1d",
    "911_neighbors_last3d",
    "911_neighbors_last7d",

    "911_neighbors_violent_last1d",
    "911_neighbors_violent_last3d",
    "911_neighbors_violent_last7d",

    "911_neighbors_property_last1d",
    "911_neighbors_property_last3d",
    "911_neighbors_property_last7d",

    "911_neighbors_weapon_last1d",
    "911_neighbors_weapon_last3d",
    "911_neighbors_weapon_last7d",

] if c in _enriched.columns]

_enriched = _enriched[KEEP_911_COLS].copy()
log(f"📌 KEEP_911_COLS kept: {len(KEEP_911_COLS)} kolon")

# ============================================================
# CRIME GRID MERGE
# ============================================================
CRIME_GRID_CANDIDATES = [
    OUT_DIR / "sf_crime_y.parquet",
    Path(BASE_DIR) / "sf_crime_y.parquet",
    Path("./sf_crime_y.parquet"),
]

crime_grid_path = next((p for p in CRIME_GRID_CANDIDATES if p.exists()), None)
if crime_grid_path is None:
    raise FileNotFoundError("❌ Suç grid yok: OUT_DIR/BASE_DIR/kök'te sf_crime_y.parquet.")

crime = pd.read_parquet(crime_grid_path)
if "GEOID" in crime.columns:
    crime["GEOID"] = crime["GEOID"].astype(str)
    
log(f"📥 Suç grid yüklendi: {len(crime)} satır ({crime_grid_path})")
log_shape(crime, "CRIME grid — ham")

before = len(crime)
crime["GEOID"] = normalize_geoid(crime["GEOID"], DEFAULT_GEOID_LEN)
crime = crime[crime["GEOID"].notna()].copy()
dropped = before - len(crime)
if dropped:
    log(f"🧹 crime grid: GEOID boş/bozuk satır atıldı: {dropped}")

if "hour_range" in crime.columns:
    def _hr_key_from_hr_range(x):
        m = hr_pat.match(str(x))
        return int(m.group(1)) % 24 if m else None
    crime["hr_key"] = crime["hour_range"].apply(_hr_key_from_hr_range).astype("Int16")
elif "event_hour" in crime.columns:
    crime["hr_key"] = ((pd.to_numeric(crime["event_hour"], errors="coerce").fillna(0).astype(int)) // 3) * 3
    crime["hr_key"] = crime["hr_key"].astype("Int16")
else:
    raise ValueError("❌ Suç grid dosyasında ne 'hour_range' ne de 'event_hour' var.")

has_date_col = ("date" in crime.columns) or ("datetime" in crime.columns)

if has_date_col:
    if "date" not in crime.columns:
        crime["date"] = pd.to_datetime(crime["datetime"], errors="coerce").dt.date
    else:
        crime["date"] = to_date(crime["date"])

    keys = ["GEOID", "date", "hour_range"]

    overlap = (set(crime.columns) & set(_enriched.columns)) - set(keys)
    if overlap:
        log(f"🧹 Merge overlap (key dışı) bulundu, _enriched'ten düşürüldü: {sorted(overlap)}")
        _enriched = _enriched.drop(columns=list(overlap), errors="ignore")
    merged = crime.merge(_enriched, on=keys, how="left")
    probe_cols = [c for c in [
        "911_request_count_hour_range",
        "911_request_count_daily(before_24_hours)",
        "911_geo_last24h",
        "911_day_total_last7d"
    ] if c in merged.columns]
    for c in probe_cols:
        log(f"🔎 merge doluluk | {c}: nonzero={(pd.to_numeric(merged[c], errors='coerce').fillna(0) > 0).sum():,}")
    log("🔗 Join modu: DATE-BASED (GEOID, date, hour_range)")

else:
    cal_keys = ["GEOID", "hr_key", "day_of_week", "season"]

    if "hr_key" not in crime.columns or crime["hr_key"].isna().all():
        if "hour_range" in crime.columns:
            crime["hr_key"] = crime["hour_range"].apply(_hr_key_from_range).astype("Int16")

    agg_cols = [c for c in _enriched.columns if c not in ["GEOID", "date", "hour_range", "hr_key", "day_of_week", "season", "month"]]
    cal_src = _enriched.copy()

    if "day_of_week" not in cal_src.columns:
        cal_src["day_of_week"] = pd.to_datetime(cal_src["date"]).dt.weekday.astype("int8")
    if "season" not in cal_src.columns:
        cal_src["month"] = pd.to_datetime(cal_src["date"]).dt.month.astype("int8")
        cal_src["season"] = cal_src["month"].map(_season_map).fillna("Summer")

    cal_agg = cal_src.groupby(cal_keys, as_index=False, observed=True)[agg_cols].median(numeric_only=True)

    if "day_of_week" not in crime.columns:
        log("ℹ️ crime grid’de day_of_week yok → 0 atanıyor.")
        crime["day_of_week"] = 0
    if "season" not in crime.columns:
        if "month" in crime.columns:
            crime["season"] = crime["month"].map(_season_map).fillna("Summer")
        else:
            crime["season"] = "Summer"

    merged = crime.merge(cal_agg, on=cal_keys, how="left")
    log("🔗 Join modu: CALENDAR-BASED (GEOID, hr_key, day_of_week, season)")

# ============================================================
# FILL / TYPE FIX
# ============================================================
fill_cols = [c for c in merged.columns if c.startswith("911_") or c in ["hr_cnt", "daily_cnt"]]

FLOAT_COLS = set([
    c for c in fill_cols
    if (
        "share" in c or
        "mean_" in c or
        c.endswith("1h") or c.endswith("3h") or c.endswith("6h") or
        c.endswith("24h") or c.endswith("3d") or c.endswith("7d") or
        c.endswith("1d")
    )
])

for c in fill_cols:
    if c in merged.columns:
        v = pd.to_numeric(merged[c], errors="coerce").fillna(0)
        merged[c] = v.astype("float32") if c in FLOAT_COLS else v.astype("int32")

# ============================================================
# NaN RAPORU
# ============================================================
try:
    nan_counts = merged.isna().sum()
    nan_counts = nan_counts[nan_counts > 0].sort_values(ascending=False)

    log(f"🧪 NaN raporu: NaN içeren sütun sayısı = {len(nan_counts)}")

    if len(nan_counts) > 0:
        for col, cnt in nan_counts.items():
            log(f"   - {col}: {int(cnt):,} NaN")

        nan_report_path = OUT_DIR / "nan_report_sf_crime_01.csv"
        (
            nan_counts.rename("nan_count")
            .reset_index()
            .rename(columns={"index": "column"})
            .to_csv(nan_report_path, index=False)
        )
        log(f"📄 NaN raporu kaydedildi → {nan_report_path}")
    else:
        log("✅ NaN yok.")
except Exception as e:
    log(f"⚠️ NaN raporu üretilemedi: {e}")

# ============================================================
# SAVE
# ============================================================
safe_save_csv(merged, str(merged_output_path))
log_shape(merged, "CRIME⨯911 (kayıt öncesi)")
log(f"✅ Suç + 911 birleştirmesi tamamlandı → {merged_output_path}")

try:
    for p in [
        local_summary_path,
        y_summary_path,
        OUT_DIR / "sf_crime_01.csv",
        merged_output_path,
    ]:
        if p.exists():
            dst = OUT_DIR / p.name
            if p.resolve() != dst.resolve():
                dst.write_bytes(p.read_bytes())
                log(f"📦 Normalize kopya: {p} → {dst}")
    log("📦 911 çıktıları OUT_DIR altında hazır.")
except Exception as e:
    log(f"⚠️ Normalize skip: {e}")

try:
    print(merged.head(5).to_string(index=False))
except Exception:
    pass
