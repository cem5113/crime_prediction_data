#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# update_911.py — 911 özetini üretir/günceller ve sf_crime_y.csv ile birleştirip sf_crime_01.csv yazar.
# TÜM ÇIKTILAR OUT_DIR (= CRIME_DATA_DIR) ALTINA DÜŞER.
#
# REVIZE NOTLARI
# --------------
# 1) 1 saatlik feature ETIKETLERI kaldırıldı. Veri 3 saatlik hour_range üstünden yürür.
# 2) Eski current-count kolonları backward compatibility için korunur:
#       - 911_request_count_hour_range
#       - 911_request_count_daily(before_24_hours)
#    Ama stacking/model için asıl önerilenler aşağıdaki past-only feature'lardır:
#       - 911_geo_prev_1d / 3d / 7d
#       - 911_geo_daily_ratio_1d_7d
#       - 911_geo_daily_zscore_1d_7d
#       - 911_geo_daily_spike_flag
#       - 911_same_slot_prev_day
#       - 911_same_slot_prev_week
#       - 911_same_slot_roll7_mean / std
#       - 911_same_slot_ratio_prevday_7d
#       - 911_same_slot_zscore_prevday_7d
#       - 911_same_slot_spike_flag
#       - 911_prev_slot
#       - 911_prev_2slot
#       - 911_neighbors_prev_1d / 3d / 7d
#       - 911_geo_to_neighbors_ratio_prev_1d
#       - 911_slot_share_prev_day_of_daily_prev1d
#
# 3) hour_range daima 3 saatlik string format:
#       00-03, 03-06, 06-09, 09-12, 12-15, 15-18, 18-21, 21-24
#
# 4) Zaman uyumu:
#    - raw gelen 911 verisi received_time/datetime üstünden okunur
#    - özet date/hour_range kolonları ile panel tarafına merge edilir
#
# 5) Neighbor feature'lar günlük bazda tutulur; 3h slot paneline date üzerinden taşınır.
#
# 6) Leakage notu:
#    - current count kolonları bilgi amaçlı tutuluyor
#    - model eğitimi için past-only feature set tercih edilmelidir

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
    """UTC -> SF yerel tarihe dönüştürmeyi dener; olmazsa naive tarihe düşer."""
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
    """Sadece rakamları al, soldan L karaktere kes ve zfill(L) yap."""
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


# ============================================================
# CONFIG & PATHS
# ============================================================

DEFAULT_GEOID_LEN = int(os.getenv("GEOID_LEN", "11"))
EPS = 1e-6

# BASE_DIR
_raw_base = os.getenv("CRIME_DATA_DIR", "crime_prediction_data").strip().strip("/\\")
repo_leaf = Path.cwd().name
if not os.path.isabs(_raw_base) and Path(_raw_base).name == repo_leaf:
    _raw_base = "."
BASE_DIR = str(Path(_raw_base).resolve()) if _raw_base != "." else "."
Path(BASE_DIR).mkdir(parents=True, exist_ok=True)
log(f"📂 BASE_DIR = {Path(BASE_DIR).resolve()}")

# OUT_DIR
OUT_DIR = Path(os.getenv("CRIME_DATA_DIR", str(Path(BASE_DIR)))).resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 911 summary dosyaları
LOCAL_NAME = "sf_911_last_5_year.csv"
local_summary_path = OUT_DIR / LOCAL_NAME

Y_NAME = "sf_911_last_5_year_y.csv"
y_summary_path = OUT_DIR / Y_NAME

# Crime merge output
merged_output_path = Path(os.getenv("DAILY_OUT", str(OUT_DIR / "sf_crime_01.csv")))
if not merged_output_path.is_absolute():
    merged_output_path = OUT_DIR / merged_output_path.name
log(f"🧾 DAILY_OUT seen as: {os.getenv('DAILY_OUT', '(unset)')}")
log(f"📝 Writing sf_crime_01 → {merged_output_path}")

# Census blocks
CENSUS_CANDIDATES = [
    OUT_DIR / "sf_census_blocks.geojson",
    Path(BASE_DIR) / "sf_census_blocks.geojson",
    Path("./sf_census_blocks.geojson"),
]

# API / kaynak
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

# Release URL adayları
RAW_911_URL_ENV = os.getenv("RAW_911_URL", "").strip()
RAW_911_URL_CANDIDATES = [
    RAW_911_URL_ENV or "",
    "https://github.com/cem5113/crime_prediction_data/releases/download/v1.0.1/sf_911_last_5_year_y.csv",
    "https://github.com/cem5113/crime_prediction_data/releases/download/v1.0.1/sf_911_last_5_year.csv",
]

# Neighbor ayarları
ENABLE_NEIGHBORS = os.getenv("ENABLE_NEIGHBORS", "1").lower() in ("1", "true", "yes", "on")
NEIGHBOR_METHOD = os.getenv("NEIGHBOR_METHOD", "touches")  # touches | radius
NEIGHBOR_RADIUS_M = float(os.getenv("NEIGHBOR_RADIUS_M", "500"))

# SF BBOX
SF_BBOX = (-123.2, 37.6, -122.3, 37.9)

# Slot düzeni: 3 saatlik
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
# IO HELPERS
# ============================================================

def read_large_csv_in_chunks(path, usecols=None, chunksize=200_000):
    try:
        it = pd.read_csv(
            path,
            low_memory=False,
            dtype={"GEOID": "string"},
            usecols=usecols,
            chunksize=chunksize,
        )
        return pd.concat(it, ignore_index=True)
    except ValueError:
        it = pd.read_csv(
            path,
            low_memory=False,
            dtype={"GEOID": "string"},
            chunksize=chunksize,
        )
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

            df["longitude"], df["latitude"] = (
                df["intersection_point"].apply(_lon),
                df["intersection_point"].apply(_lat),
            )

        for a, b in (("y", "x"), ("lat", "long")):
            if a in df.columns and b in df.columns and "latitude" not in df.columns:
                df["latitude"], df["longitude"] = (
                    pd.to_numeric(df[a], errors="coerce"),
                    pd.to_numeric(df[b], errors="coerce"),
                )
                break

    if "latitude" in df.columns and "longitude" in df.columns:
        min_lon, min_lat, max_lon, max_lat = SF_BBOX
        df = df[
            (df["latitude"].between(min_lat, max_lat))
            & (df["longitude"].between(min_lon, max_lon))
        ]

    df = df.dropna(subset=["latitude", "longitude"]).copy()

    gdf_blocks, tlen = _load_blocks()
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    )
    gdf = gpd.sjoin(gdf, gdf_blocks[["GEOID", "geometry"]], how="left", predicate="within")

    out = pd.DataFrame(gdf.drop(columns=["geometry", "index_right"], errors="ignore"))
    out["GEOID"] = normalize_geoid(out["GEOID"], tlen)
    out = out.dropna(subset=["GEOID"]).copy()
    return out


def build_neighbors(method: str = "touches", radius_m: float = 500.0) -> pd.DataFrame:
    gdf_blocks, _ = _load_blocks()
    tracts = gdf_blocks.dissolve(by="GEOID", as_index=False)

    if method == "radius":
        tr_utm = tracts.to_crs("EPSG:26910")
        buf = tr_utm.buffer(radius_m)
        g_buf = gpd.GeoDataFrame(tr_utm[["GEOID"]].copy(), geometry=buf, crs=tr_utm.crs)
        join = gpd.sjoin(
            g_buf,
            tr_utm[["GEOID", "geometry"]].rename(columns={"GEOID": "nbr"}),
            predicate="intersects",
        )
        edges = join[["GEOID", "nbr"]]
    else:
        join = gpd.sjoin(
            tracts[["GEOID", "geometry"]],
            tracts[["GEOID", "geometry"]].rename(columns={"GEOID": "nbr"}),
            predicate="touches",
        )
        edges = join[["GEOID", "nbr"]]

    edges = edges[edges["GEOID"] != edges["nbr"]].copy()
    edges["pair"] = edges.apply(lambda r: tuple(sorted((r["GEOID"], r["nbr"]))), axis=1)
    edges = edges.drop_duplicates("pair").drop(columns=["pair"])
    edges["GEOID"] = normalize_geoid(edges["GEOID"], DEFAULT_GEOID_LEN)
    edges["nbr"] = normalize_geoid(edges["nbr"], DEFAULT_GEOID_LEN)
    return pd.DataFrame(edges)


# ============================================================
# SLOT / GRID HELPERS
# ============================================================

hr_pat = re.compile(r"^\s*(\d{1,2})\s*-\s*(\d{1,2})\s*$")


def _hr_key_from_range(hr):
    m = hr_pat.match(str(hr))
    return int(m.group(1)) % 24 if m else None


def make_hour_range_from_hour(hour_series: pd.Series) -> pd.Series:
    h = pd.to_numeric(hour_series, errors="coerce").fillna(0).astype(int) % 24
    start = (h // 3) * 3
    return start.apply(lambda s: f"{int(s):02d}-{int(min(s + 3, 24)):02d}")


def build_full_daily_grid(geoids: pd.Series, date_min, date_max) -> pd.DataFrame:
    all_dates = pd.date_range(date_min, date_max, freq="D")
    idx = pd.MultiIndex.from_product([geoids.tolist(), all_dates], names=["GEOID", "date"])
    return idx.to_frame(index=False)


def build_full_slot_grid(geoids: pd.Series, date_min, date_max) -> pd.DataFrame:
    all_dates = pd.date_range(date_min, date_max, freq="D")
    idx = pd.MultiIndex.from_product(
        [geoids.tolist(), all_dates, SLOT_ORDER],
        names=["GEOID", "date", "hour_range"],
    )
    return idx.to_frame(index=False)


# ============================================================
# SUMMARY BUILDERS
# ============================================================

def make_standard_summary(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Raw 911 -> standart özet
    Çıktı:
        GEOID, date, hour_range,
        911_request_count_hour_range,
        911_request_count_daily(before_24_hours)
    """
    if raw is None or raw.empty:
        return pd.DataFrame(
            columns=[
                "GEOID",
                "date",
                "hour_range",
                "911_request_count_hour_range",
                "911_request_count_daily(before_24_hours)",
            ]
        )

    df = raw.copy()

    ts_col = None
    for cand in [
        "received_time",
        "received_datetime",
        "date",
        "datetime",
        "timestamp",
        "call_received_datetime",
    ]:
        if cand in df.columns:
            ts_col = cand
            break

    if ts_col is None:
        raise ValueError("Zaman kolonu bulunamadı (received_time/received_datetime/date).")

    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df["date"] = df[ts_col].dt.date
    df["event_hour"] = df[ts_col].dt.hour
    df["hour_range"] = make_hour_range_from_hour(df["event_hour"])

    has_geoid = "GEOID" in df.columns
    grp_hr = (["GEOID"] if has_geoid else []) + ["date", "hour_range"]
    grp_day = (["GEOID"] if has_geoid else []) + ["date"]

    hr_agg = (
        df.groupby(grp_hr, dropna=False, observed=True)
        .size()
        .reset_index(name="911_request_count_hour_range")
    )

    day_agg = (
        df.groupby(grp_day, dropna=False, observed=True)
        .size()
        .reset_index(name="911_request_count_daily(before_24_hours)")
    )

    out = hr_agg.merge(day_agg, on=grp_day, how="left")

    cols_tail = [c for c in ["date", "hour_range", "GEOID"] if c in out.columns]
    cols = [c for c in out.columns if c not in cols_tail] + cols_tail
    return out[cols]


def summary_from_local(path: Path | str, min_date=None) -> pd.DataFrame:
    log(f"📥 Yerel 911 tabanı okunuyor: {path}")
    df = pd.read_csv(path, low_memory=False, dtype={"GEOID": "string"})

    is_already_summary = (
        {"date", "hour_range"}.issubset(df.columns)
        and any(
            c in df.columns
            for c in ["911_request_count_hour_range", "call_count", "count", "requests", "n"]
        )
    )

    if is_already_summary:
        cnt_col = next(
            c
            for c in ["911_request_count_hour_range", "call_count", "count", "requests", "n"]
            if c in df.columns
        )
        if cnt_col != "911_request_count_hour_range":
            df = df.rename(columns={cnt_col: "911_request_count_hour_range"})

        df["date"] = to_date(df["date"])

        if "GEOID" in df.columns:
            df["GEOID"] = normalize_geoid(df["GEOID"], DEFAULT_GEOID_LEN)

        def _fmt_hr(hr):
            m = re.match(r"^\s*(\d{1,2})\s*-\s*(\d{1,2})\s*$", str(hr))
            if not m:
                return None
            a = int(m.group(1)) % 24
            b = int(m.group(2))
            b = b if b > a else min(a + 3, 24)
            return f"{a:02d}-{b:02d}"

        df["hour_range"] = df["hour_range"].apply(_fmt_hr)

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

    is_already_summary = (
        {"date", "hour_range"}.issubset(df.columns)
        and any(
            c in df.columns
            for c in ["911_request_count_hour_range", "call_count", "count", "requests", "n"]
        )
    )

    if is_already_summary:
        cnt_col = next(
            c
            for c in ["911_request_count_hour_range", "call_count", "count", "requests", "n"]
            if c in df.columns
        )
        if cnt_col != "911_request_count_hour_range":
            df = df.rename(columns={cnt_col: "911_request_count_hour_range"})

        df["date"] = to_date(df["date"])

        if "GEOID" in df.columns:
            df["GEOID"] = normalize_geoid(df["GEOID"], DEFAULT_GEOID_LEN)

        def _fmt_hr(hr):
            m = re.match(r"^\s*(\d{1,2})\s*-\s*(\d{1,2})\s*$", str(hr))
            if not m:
                return None
            a = int(m.group(1)) % 24
            b = int(m.group(2))
            b = b if b > a else min(a + 3, 24)
            return f"{a:02d}-{b:02d}"

        df["hour_range"] = df["hour_range"].apply(_fmt_hr)

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
        OUT_DIR / "sf_crime_y.csv",
        Path(BASE_DIR) / "sf_crime_y.csv",
        Path("./sf_crime_y.csv"),
        Path("crime_prediction_data/sf_crime_y.csv"),
        OUT_DIR / "crime_prediction_data/sf_crime_y.csv",
        Path(BASE_DIR) / "crime_prediction_data/sf_crime_y.csv",
        Path(ARTIFACT_NAME) / "sf_crime_y.csv",
        Path(ARTIFACT_NAME) / "crime_prediction_data/sf_crime_y.csv",
    ]

    crime_grid_path = next(
        (p for p in crime_grid_candidates if p.exists() and not is_lfs_pointer_file(p)),
        None,
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
            cand = rt / nm
            if _ok(cand):
                log(f"📦 911 base bulundu: {cand}")
                return cand

            cand2 = rt / "crime_prediction_data" / nm
            if _ok(cand2):
                log(f"📦 911 base bulundu: {cand2}")
                return cand2

            cand3 = rt / "outputs" / nm
            if _ok(cand3):
                log(f"📦 911 base bulundu: {cand3}")
                return cand3

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
    dt_candidates = [
        "received_time",
        "received_datetime",
        "date",
        "datetime",
        "call_datetime",
        "received_dttm",
        "call_date",
    ]
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
        f"{chosen_dt} between '{rng_start}' and '{rng_end}'"
        + (f" AND {AGENCY_FILTER}" if AGENCY_FILTER else ""),
        f"{chosen_dt} between '{rng_start}' and '{rng_end}'",
    ]

    while True:
        df = None
        for attempt in range(MAX_RETRIES):
            try:
                r = requests.get(
                    SF911_API_URL,
                    headers=headers,
                    params={"$where": where_list[0], "$limit": CHUNK_LIMIT, "$offset": offset},
                    timeout=REQUEST_TIMEOUT,
                )
                if r.status_code == 400:
                    r = requests.get(
                        SF911_API_URL,
                        headers=headers,
                        params={"$where": where_list[1], "$limit": CHUNK_LIMIT, "$offset": offset},
                        timeout=REQUEST_TIMEOUT,
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
        allowed_methods=["GET"],
    )
    sess.mount("https://", HTTPAdapter(max_retries=retries))
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    headers = {"Accept": "application/json"}
    if SF_APP_TOKEN:
        headers["X-App-Token"] = SF_APP_TOKEN

    dt_candidates = [
        "received_time",
        "received_datetime",
        "date",
        "datetime",
        "call_datetime",
        "received_dttm",
        "call_date",
    ]
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
                cols = [
                    c.get("fieldName") or c.get("name") or f"c{i}"
                    for i, c in enumerate(obj.get("meta", {}).get("view", {}).get("columns", []))
                ]
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
                            all_rows.append(
                                {cols[i]: (row[i] if i < len(cols) else None) for i in range(len(cols))}
                            )
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
    ts_col = next(
        (
            c
            for c in [
                "received_time",
                "received_datetime",
                "date",
                "datetime",
                "timestamp",
                "call_received_datetime",
                "ts",
            ]
            if c in raw.columns
        ),
        None,
    )
    if not ts_col:
        return

    tmp = raw.copy()
    tmp["ts"] = pd.to_datetime(tmp[ts_col], errors="coerce")
    tmp = tmp[tmp["ts"].notna()]
    if tmp.empty:
        return

    lat_col = next((c for c in ["latitude", "lat", "y"] if c in raw.columns), None)
    lon_col = next((c for c in ["longitude", "lon", "x"] if c in raw.columns), None)

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
# FEATURE ENGINEERING
# ============================================================

def build_enriched_911_features(final_911: pd.DataFrame) -> pd.DataFrame:
    """
    final_911 (summary-level) -> enriched summary
    Çıktı: date + GEOID + hour_range üstünden merge edilecek feature set.
    """

    if final_911 is None or final_911.empty:
        return pd.DataFrame()

    df = final_911.copy()

    df = df.dropna(subset=["GEOID", "date", "hour_range"]).copy()
    df["GEOID"] = normalize_geoid(df["GEOID"], DEFAULT_GEOID_LEN)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["hr_key"] = df["hour_range"].apply(_hr_key_from_range).astype("Int16")
    df["day_of_week"] = df["date"].dt.weekday.astype("int8")
    df["month"] = df["date"].dt.month.astype("int8")

    _season_map = {
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Fall", 10: "Fall", 11: "Fall",
    }
    df["season"] = df["month"].map(_season_map).astype("category")

    log_shape(df, "911 summary (normalize)")
    log_date_range(df, "date", "911")

    # --------------------------------------------------------
    # CURRENT AGG TABLES
    # --------------------------------------------------------
    day_cur = (
        df[["GEOID", "date", "911_request_count_daily(before_24_hours)"]]
        .drop_duplicates(subset=["GEOID", "date"])
        .rename(columns={"911_request_count_daily(before_24_hours)": "daily_cnt"})
        .sort_values(["GEOID", "date"])
        .reset_index(drop=True)
    )

    slot_cur = (
        df[["GEOID", "date", "hour_range", "911_request_count_hour_range"]]
        .groupby(["GEOID", "date", "hour_range"], as_index=False, observed=True)["911_request_count_hour_range"]
        .sum()
        .rename(columns={"911_request_count_hour_range": "slot_cnt"})
        .sort_values(["GEOID", "date", "hour_range"])
        .reset_index(drop=True)
    )

    if day_cur.empty or slot_cur.empty:
        log("⚠️ 911 feature üretimi için day_cur/slot_cur boş.")
        return pd.DataFrame()

    # --------------------------------------------------------
    # FULL DAILY GRID
    # --------------------------------------------------------
    try:
        gdf_blocks, _ = _load_blocks()
        all_geoids = gdf_blocks["GEOID"].astype("string").sort_values().drop_duplicates()
    except Exception:
        all_geoids = day_cur["GEOID"].astype("string").sort_values().drop_duplicates()

    date_min = min(day_cur["date"].min(), slot_cur["date"].min())
    date_max = max(day_cur["date"].max(), slot_cur["date"].max())

    daily_full = build_full_daily_grid(all_geoids, date_min, date_max)
    daily_full["date"] = pd.to_datetime(daily_full["date"], errors="coerce")
    daily_full = daily_full.merge(day_cur, on=["GEOID", "date"], how="left")
    daily_full["daily_cnt"] = pd.to_numeric(daily_full["daily_cnt"], errors="coerce").fillna(0).astype("float32")
    daily_full = daily_full.sort_values(["GEOID", "date"]).reset_index(drop=True)

    # --------------------------------------------------------
    # DAILY PAST-ONLY FEATURE'lar
    # --------------------------------------------------------
    daily_full["911_geo_prev_1d"] = (
        daily_full.groupby("GEOID")["daily_cnt"]
        .shift(1)
        .fillna(0)
        .astype("float32")
    )

    daily_full["911_geo_prev_3d"] = (
        daily_full.groupby("GEOID")["daily_cnt"]
        .transform(lambda s: s.shift(1).rolling(3, min_periods=1).sum())
        .fillna(0)
        .astype("float32")
    )

    daily_full["911_geo_prev_7d"] = (
        daily_full.groupby("GEOID")["daily_cnt"]
        .transform(lambda s: s.shift(1).rolling(7, min_periods=1).sum())
        .fillna(0)
        .astype("float32")
    )

    daily_full["911_geo_daily_roll7_mean"] = (
        daily_full.groupby("GEOID")["daily_cnt"]
        .transform(lambda s: s.shift(1).rolling(7, min_periods=2).mean())
        .fillna(0)
        .astype("float32")
    )

    daily_full["911_geo_daily_roll7_std"] = (
        daily_full.groupby("GEOID")["daily_cnt"]
        .transform(lambda s: s.shift(1).rolling(7, min_periods=2).std())
        .fillna(0)
        .astype("float32")
    )

    daily_full["911_geo_daily_ratio_1d_7d"] = (
        daily_full["911_geo_prev_1d"]
        / ((daily_full["911_geo_prev_7d"] / 7.0) + EPS)
    ).astype("float32")

    daily_full["911_geo_daily_zscore_1d_7d"] = (
        (daily_full["911_geo_prev_1d"] - daily_full["911_geo_daily_roll7_mean"])
        / (daily_full["911_geo_daily_roll7_std"] + EPS)
    ).astype("float32")

    daily_full["911_geo_daily_spike_flag"] = (
        daily_full["911_geo_prev_1d"]
        > (1.5 * (daily_full["911_geo_prev_7d"] / 7.0))
    ).astype("int8")

    # --------------------------------------------------------
    # NEIGHBOR DAILY FEATURE'lar
    # --------------------------------------------------------
    neighbors_df = None
    if ENABLE_NEIGHBORS:
        try:
            neighbors_df = build_neighbors(NEIGHBOR_METHOD, NEIGHBOR_RADIUS_M)
            log_shape(neighbors_df, f"Komşu haritası ({NEIGHBOR_METHOD})")
        except Exception as e:
            log(f"⚠️ Komşu haritası üretilemedi: {e}")
            neighbors_df = None

    if neighbors_df is not None and not neighbors_df.empty:
        day_nbr = neighbors_df.merge(
            daily_full[["GEOID", "date", "daily_cnt"]].rename(columns={"GEOID": "nbr"}),
            on="nbr",
            how="left",
        )
        day_nbr = (
            day_nbr.groupby(["GEOID", "date"], as_index=False, observed=True)["daily_cnt"]
            .sum()
            .rename(columns={"daily_cnt": "nbr_daily_cnt"})
        )
        day_nbr = day_nbr.sort_values(["GEOID", "date"]).reset_index(drop=True)

        day_nbr["911_neighbors_prev_1d"] = (
            day_nbr.groupby("GEOID")["nbr_daily_cnt"]
            .shift(1)
            .fillna(0)
            .astype("float32")
        )

        day_nbr["911_neighbors_prev_3d"] = (
            day_nbr.groupby("GEOID")["nbr_daily_cnt"]
            .transform(lambda s: s.shift(1).rolling(3, min_periods=1).sum())
            .fillna(0)
            .astype("float32")
        )

        day_nbr["911_neighbors_prev_7d"] = (
            day_nbr.groupby("GEOID")["nbr_daily_cnt"]
            .transform(lambda s: s.shift(1).rolling(7, min_periods=1).sum())
            .fillna(0)
            .astype("float32")
        )

        day_nbr = day_nbr[[
            "GEOID",
            "date",
            "911_neighbors_prev_1d",
            "911_neighbors_prev_3d",
            "911_neighbors_prev_7d",
        ]].copy()

        daily_full = daily_full.merge(day_nbr, on=["GEOID", "date"], how="left")
    else:
        daily_full["911_neighbors_prev_1d"] = 0.0
        daily_full["911_neighbors_prev_3d"] = 0.0
        daily_full["911_neighbors_prev_7d"] = 0.0

    daily_full["911_neighbors_prev_1d"] = pd.to_numeric(daily_full["911_neighbors_prev_1d"], errors="coerce").fillna(0).astype("float32")
    daily_full["911_neighbors_prev_3d"] = pd.to_numeric(daily_full["911_neighbors_prev_3d"], errors="coerce").fillna(0).astype("float32")
    daily_full["911_neighbors_prev_7d"] = pd.to_numeric(daily_full["911_neighbors_prev_7d"], errors="coerce").fillna(0).astype("float32")

    daily_full["911_geo_to_neighbors_ratio_prev_1d"] = (
        daily_full["911_geo_prev_1d"] / (daily_full["911_neighbors_prev_1d"] + EPS)
    ).astype("float32")

    # --------------------------------------------------------
    # FULL SLOT GRID
    # --------------------------------------------------------
    slot_full = build_full_slot_grid(all_geoids, date_min, date_max)
    slot_full["date"] = pd.to_datetime(slot_full["date"], errors="coerce")

    slot_full = slot_full.merge(slot_cur, on=["GEOID", "date", "hour_range"], how="left")
    slot_full["slot_cnt"] = pd.to_numeric(slot_full["slot_cnt"], errors="coerce").fillna(0).astype("float32")
    slot_full["hr_key"] = slot_full["hour_range"].apply(_hr_key_from_range).astype("Int16")
    slot_full["slot_start_hour"] = slot_full["hour_range"].map(SLOT_START_MAP).astype("int16")
    slot_full["slot_dt"] = slot_full["date"] + pd.to_timedelta(slot_full["slot_start_hour"], unit="h")

    # --------------------------------------------------------
    # SLOT PAST-ONLY FEATURE'lar
    # --------------------------------------------------------
    slot_full = slot_full.sort_values(["GEOID", "slot_dt"]).reset_index(drop=True)

    slot_full["911_prev_slot"] = (
        slot_full.groupby("GEOID")["slot_cnt"]
        .shift(1)
        .fillna(0)
        .astype("float32")
    )

    slot_full["911_prev_2slot"] = (
        slot_full.groupby("GEOID")["slot_cnt"]
        .shift(2)
        .fillna(0)
        .astype("float32")
    )

    slot_full = slot_full.sort_values(["GEOID", "hour_range", "date"]).reset_index(drop=True)

    slot_full["911_same_slot_prev_day"] = (
        slot_full.groupby(["GEOID", "hour_range"])["slot_cnt"]
        .shift(1)
        .fillna(0)
        .astype("float32")
    )

    slot_full["911_same_slot_prev_week"] = (
        slot_full.groupby(["GEOID", "hour_range"])["slot_cnt"]
        .shift(7)
        .fillna(0)
        .astype("float32")
    )

    slot_full["911_same_slot_roll7_mean"] = (
        slot_full.groupby(["GEOID", "hour_range"])["slot_cnt"]
        .transform(lambda s: s.shift(1).rolling(7, min_periods=2).mean())
        .fillna(0)
        .astype("float32")
    )

    slot_full["911_same_slot_roll7_std"] = (
        slot_full.groupby(["GEOID", "hour_range"])["slot_cnt"]
        .transform(lambda s: s.shift(1).rolling(7, min_periods=2).std())
        .fillna(0)
        .astype("float32")
    )

    slot_full["911_same_slot_ratio_prevday_7d"] = (
        slot_full["911_same_slot_prev_day"]
        / (slot_full["911_same_slot_roll7_mean"] + EPS)
    ).astype("float32")

    slot_full["911_same_slot_zscore_prevday_7d"] = (
        (slot_full["911_same_slot_prev_day"] - slot_full["911_same_slot_roll7_mean"])
        / (slot_full["911_same_slot_roll7_std"] + EPS)
    ).astype("float32")

    slot_full["911_same_slot_spike_flag"] = (
        slot_full["911_same_slot_prev_day"]
        > (1.5 * slot_full["911_same_slot_roll7_mean"])
    ).astype("int8")

    # --------------------------------------------------------
    # DAILY FEATURE'lari SLOT grid'e taşı
    # --------------------------------------------------------
    daily_keep = [
        "GEOID",
        "date",
        "daily_cnt",
        "911_geo_prev_1d",
        "911_geo_prev_3d",
        "911_geo_prev_7d",
        "911_geo_daily_roll7_mean",
        "911_geo_daily_roll7_std",
        "911_geo_daily_ratio_1d_7d",
        "911_geo_daily_zscore_1d_7d",
        "911_geo_daily_spike_flag",
        "911_neighbors_prev_1d",
        "911_neighbors_prev_3d",
        "911_neighbors_prev_7d",
        "911_geo_to_neighbors_ratio_prev_1d",
    ]

    slot_full = slot_full.merge(daily_full[daily_keep], on=["GEOID", "date"], how="left")

    # slot vs daily geçmiş baskı oranı
    slot_full["911_slot_share_prev_day_of_daily_prev1d"] = (
        slot_full["911_same_slot_prev_day"] / (slot_full["911_geo_prev_1d"] + EPS)
    ).astype("float32")

    # --------------------------------------------------------
    # BACKWARD COMPATIBILITY: mevcut count kolonlarını koru
    # --------------------------------------------------------
    slot_full["911_request_count_hour_range"] = slot_full["slot_cnt"].astype("float32")
    slot_full["911_request_count_daily(before_24_hours)"] = slot_full["daily_cnt"].astype("float32")

    slot_full["date"] = slot_full["date"].dt.date

    enriched = slot_full[[
        "GEOID",
        "date",
        "hour_range",
        "hr_key",

        # mevcut count kolonları
        "911_request_count_hour_range",
        "911_request_count_daily(before_24_hours)",

        # yeni günlük past-only feature'lar
        "911_geo_prev_1d",
        "911_geo_prev_3d",
        "911_geo_prev_7d",
        "911_geo_daily_roll7_mean",
        "911_geo_daily_roll7_std",
        "911_geo_daily_ratio_1d_7d",
        "911_geo_daily_zscore_1d_7d",
        "911_geo_daily_spike_flag",

        # yeni slot past-only feature'lar
        "911_prev_slot",
        "911_prev_2slot",
        "911_same_slot_prev_day",
        "911_same_slot_prev_week",
        "911_same_slot_roll7_mean",
        "911_same_slot_roll7_std",
        "911_same_slot_ratio_prevday_7d",
        "911_same_slot_zscore_prevday_7d",
        "911_same_slot_spike_flag",
        "911_slot_share_prev_day_of_daily_prev1d",

        # komşu feature'lar
        "911_neighbors_prev_1d",
        "911_neighbors_prev_3d",
        "911_neighbors_prev_7d",
        "911_geo_to_neighbors_ratio_prev_1d",
    ]].copy()

    # numerik kolonları temizle
    int_cols = [
        "911_geo_daily_spike_flag",
        "911_same_slot_spike_flag",
    ]
    float_cols = [c for c in enriched.columns if c not in ["GEOID", "date", "hour_range"] + int_cols]

    for c in float_cols:
        enriched[c] = pd.to_numeric(enriched[c], errors="coerce").fillna(0).astype("float32")

    for c in int_cols:
        enriched[c] = pd.to_numeric(enriched[c], errors="coerce").fillna(0).astype("int8")

    return enriched


# ============================================================
# MAIN — LOCAL/RELEASE → INCREMENT → ENRICH → MERGE
# ============================================================

five_years_ago = datetime.now(timezone.utc).date() - timedelta(days=5 * 365)

log(f"📁 911 yerel özet yolu: {local_summary_path}")

# 1) Önce yerel tabanı dene
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

# 2) Max tarihten bugüne artımlı aralık
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

log(f"🗓️ İndirme aralığı: {fetch_start} → {fetch_end} ({(fetch_end - fetch_start).days + 1} gün)")

# 3) Artımlı API çek ve birleştir
inc = incremental_summary(fetch_start, fetch_end)

if inc is not None and not inc.empty:
    if "GEOID" in inc.columns:
        inc["GEOID"] = normalize_geoid(inc["GEOID"], DEFAULT_GEOID_LEN)

    inc["date"] = to_date(inc["date"])
    before = len(final_911)

    final_911 = pd.concat([final_911, inc], ignore_index=True)

    subset_cols = [c for c in ["GEOID", "date", "hour_range"] if c in final_911.columns]
    final_911 = (
        final_911.dropna(subset=["date"])
        .sort_values(subset_cols if subset_cols else ["date"])
        .drop_duplicates(subset=subset_cols if subset_cols else ["date"], keep="last")
    )

    final_911 = final_911[final_911["date"] >= five_years_ago]

    safe_save_csv(final_911, str(local_summary_path))
    safe_save_csv(final_911, str(y_summary_path))

    log(f"💾 911 özet GÜNCELLENDİ (base+API) → {local_summary_path} & {y_summary_path} (+{len(final_911) - before:,} satır)")
else:
    log("ℹ️ API tarafında yeni gün yok veya boş döndü; taban veri geçerli.")

if final_911 is None or final_911.empty:
    log("⚠️ 911 özeti üretilemedi (boş). Çıkılıyor.")
    raise SystemExit(0)

# 4) Enriched feature set oluştur
_enriched = build_enriched_911_features(final_911)

if _enriched is None or _enriched.empty:
    log("⚠️ Enriched 911 feature set boş. Çıkılıyor.")
    raise SystemExit(0)

# 5) Crime grid yükle
CRIME_GRID_CANDIDATES = [
    OUT_DIR / "sf_crime_y.csv",
    Path(BASE_DIR) / "sf_crime_y.csv",
    Path("./sf_crime_y.csv"),
]

crime_grid_path = next((p for p in CRIME_GRID_CANDIDATES if p.exists()), None)
if crime_grid_path is None:
    raise FileNotFoundError("❌ Suç grid yok: OUT_DIR/BASE_DIR/kök'te sf_crime_y.csv.")

crime = pd.read_csv(crime_grid_path, dtype={"GEOID": str}, low_memory=False)
log(f"📥 Suç grid yüklendi: {len(crime)} satır ({crime_grid_path})")
log_shape(crime, "CRIME grid — ham")

before = len(crime)
crime["GEOID"] = normalize_geoid(crime["GEOID"], DEFAULT_GEOID_LEN)
crime = crime[crime["GEOID"].notna()].copy()
dropped = before - len(crime)
if dropped:
    log(f"🧹 crime grid: GEOID boş/bozuk satır atıldı: {dropped}")

# hour_range / hr_key kontratı
if "hour_range" in crime.columns:
    hr_pat2 = re.compile(r"^\s*(\d{1,2})\s*-\s*(\d{1,2})\s*$")

    def _hr_key_from_hr_range(x):
        m = hr_pat2.match(str(x))
        return int(m.group(1)) % 24 if m else None

    crime["hr_key"] = crime["hour_range"].apply(_hr_key_from_hr_range).astype("Int16")

elif "event_hour" in crime.columns:
    crime["hour_range"] = make_hour_range_from_hour(crime["event_hour"])
    crime["hr_key"] = crime["hour_range"].apply(_hr_key_from_range).astype("Int16")
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
    log("🔗 Join modu: DATE-BASED (GEOID, date, hour_range)")

else:
    # Tarih yoksa daha kaba fallback
    cal_keys = ["GEOID", "hr_key"]
    agg_cols = [c for c in _enriched.columns if c not in ["GEOID", "date", "hour_range", "hr_key"]]
    cal_agg = _enriched.groupby(cal_keys, as_index=False, observed=True)[agg_cols].median(numeric_only=True)
    merged = crime.merge(cal_agg, on=cal_keys, how="left")
    log("🔗 Join modu: FALLBACK (GEOID, hr_key)")

# 6) Eksikleri doldur
fill_cols = [
    "911_request_count_hour_range",
    "911_request_count_daily(before_24_hours)",

    "911_geo_prev_1d",
    "911_geo_prev_3d",
    "911_geo_prev_7d",
    "911_geo_daily_roll7_mean",
    "911_geo_daily_roll7_std",
    "911_geo_daily_ratio_1d_7d",
    "911_geo_daily_zscore_1d_7d",
    "911_geo_daily_spike_flag",

    "911_prev_slot",
    "911_prev_2slot",
    "911_same_slot_prev_day",
    "911_same_slot_prev_week",
    "911_same_slot_roll7_mean",
    "911_same_slot_roll7_std",
    "911_same_slot_ratio_prevday_7d",
    "911_same_slot_zscore_prevday_7d",
    "911_same_slot_spike_flag",
    "911_slot_share_prev_day_of_daily_prev1d",

    "911_neighbors_prev_1d",
    "911_neighbors_prev_3d",
    "911_neighbors_prev_7d",
    "911_geo_to_neighbors_ratio_prev_1d",
]

INT_COLS = {
    "911_geo_daily_spike_flag",
    "911_same_slot_spike_flag",
}

for c in fill_cols:
    if c in merged.columns:
        v = pd.to_numeric(merged[c], errors="coerce").fillna(0)
        merged[c] = v.astype("int8") if c in INT_COLS else v.astype("float32")

# 7) NaN raporu
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

# 8) Yaz
safe_save_csv(merged, str(merged_output_path))
log_shape(merged, "CRIME⨯911 (kayıt öncesi)")
log(f"✅ Suç + 911 birleştirmesi tamamlandı → {merged_output_path}")

# 9) Normalize
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

# 10) Preview
try:
    print(merged.head(5).to_string(index=False))
except Exception:
    pass
