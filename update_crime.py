# scripts/update_crime.py
from __future__ import annotations

import io
import os
import re
import time
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import holidays
import requests
import zoneinfo

# ============================================================
# CONFIG
# ============================================================

SF_TZ = zoneinfo.ZoneInfo("America/Los_Angeles")
SF_TZ_NAME = "America/Los_Angeles"

DEFAULT_GEOID_LEN = int(os.getenv("GEOID_LEN", "11"))
SF_BBOX = (-123.2, 37.6, -122.3, 37.9)

PREFER_REMOTE_BASE = os.getenv("PREFER_REMOTE_BASE", "1").lower() in ("1", "true", "yes", "on")
WRITE_BASE_TO_REPO = os.getenv("WRITE_BASE_TO_REPO", "0").lower() in ("1", "true", "yes", "on")

RUN_TMP_DIR = Path(os.getenv("RUNNER_TEMP", "/tmp")) / "sfcrime_runtime"
RUN_TMP_DIR.mkdir(parents=True, exist_ok=True)

TMP_BASE_CSV = RUN_TMP_DIR / "sf_crime.csv"
TMP_BASE_GZ = RUN_TMP_DIR / "sf_crime.csv.gz"

CRIME_BASE_URL = os.getenv(
    "CRIME_CSV_URL",
    "https://github.com/cem5113/crime_prediction_data/releases/latest/download/sf_crime.csv",
)
CRIME_API_URL = os.getenv("CRIME_API_URL", "https://data.sfgov.org/resource/wg3w-h783.json")
SFCRIME_APP_TOKEN = os.getenv("SFCRIME_API_TOKEN", "")

CHUNK_LIMIT = int(os.getenv("SFCRIME_CHUNK_LIMIT", "50000"))
MAX_RETRIES = int(os.getenv("SFCRIME_MAX_RETRIES", "4"))
SLEEP_BETWEEN_REQS = float(os.getenv("SFCRIME_SLEEP", "0.2"))
BULK_RANGE = os.getenv("SFCRIME_BULK_RANGE", "1").lower() in ("1", "true", "yes", "on")
CRIME_REINGEST_DAYS = int(os.getenv("SFCRIME_REINGEST_DAYS", "14"))
PUBLISH_LAG_FALLBACK_DAYS = int(os.getenv("PUBLISH_LAG_FALLBACK_DAYS", "2"))
FORCE_FULL = os.getenv("CRIME_FORCE_FULL", "0").lower() in ("1", "true", "yes", "on")

EVENT_CSV_NAME = os.getenv("EVENT_CSV_NAME", "sf_crime_x.csv")
PANEL_CSV_NAME = os.getenv("PANEL_CSV_NAME", "sf_crime_y.csv")

SAVE_DIR = Path(os.getenv("CRIME_DATA_DIR", "."))
SAVE_DIR.mkdir(parents=True, exist_ok=True)

event_csv_path = SAVE_DIR / EVENT_CSV_NAME
panel_csv_path = SAVE_DIR / PANEL_CSV_NAME
blocks_path = SAVE_DIR / "sf_census_blocks.geojson"

GITHUB_REPO = os.getenv("GITHUB_REPO", "cem5113/crime_prediction_data")
GH_TOKEN = os.getenv("GH_TOKEN", "")
ARTIFACT_NAME = os.getenv("ARTIFACT_NAME", "sf-crime-pipeline-output")

headers = {"X-App-Token": SFCRIME_APP_TOKEN} if SFCRIME_APP_TOKEN else {}


# ============================================================
# HELPERS
# ============================================================

def log(msg: str) -> None:
    print(msg, flush=True)


def safe_save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    tmp.replace(path)


def is_lfs_pointer(p: Path) -> bool:
    try:
        head = p.read_text(errors="ignore")[:200]
        return "git-lfs.github.com/spec/v1" in head
    except Exception:
        return False


def _is_valid_csv_bytes(b: bytes, min_bytes: int = 5000) -> bool:
    if not b or len(b) < min_bytes:
        return False
    head = b[:300].decode("utf-8", errors="ignore")
    if "git-lfs.github.com/spec/v1" in head:
        return False
    return True


def _is_valid_local_csv(p: Path, min_bytes: int = 5000) -> bool:
    if not p.exists():
        return False
    if p.stat().st_size < min_bytes:
        return False
    if p.suffix == ".csv" and is_lfs_pointer(p):
        return False
    return True


def _write_bytes_atomic(dst: Path, content: bytes) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    tmp.write_bytes(content)
    tmp.replace(dst)


def _gh_headers():
    if not GH_TOKEN:
        return None
    return {
        "Authorization": f"Bearer {GH_TOKEN}",
        "Accept": "application/vnd.github+json",
    }


def fetch_file_from_latest_artifact(
    pick_names: List[str],
    artifact_name: str = ARTIFACT_NAME,
) -> Optional[bytes]:
    hdr = _gh_headers()
    if not hdr:
        return None

    try:
        runs_url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/runs?per_page=20"
        runs = requests.get(runs_url, headers=hdr, timeout=30).json()
        run_ids = [r["id"] for r in runs.get("workflow_runs", []) if r.get("conclusion") == "success"]

        for rid in run_ids:
            arts_url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/runs/{rid}/artifacts"
            arts = requests.get(arts_url, headers=hdr, timeout=30).json().get("artifacts", [])

            for a in arts:
                if a.get("name") != artifact_name or a.get("expired", False):
                    continue

                dl = requests.get(a["archive_download_url"], headers=hdr, timeout=60)
                import zipfile
                zf = zipfile.ZipFile(io.BytesIO(dl.content))
                names = zf.namelist()

                for pick in pick_names:
                    for candidate in (pick, f"crime_prediction_data/{pick}"):
                        if candidate in names:
                            return zf.read(candidate)

                for n in names:
                    if any(n.endswith(p) for p in pick_names):
                        return zf.read(n)

        return None
    except Exception:
        return None


def download_url_to_file(url: str, dst: Path, timeout: int = 120) -> bool:
    try:
        r = requests.get(url, timeout=timeout)
        log(f"⬇️ Release HTTP {r.status_code} → {url}")
        r.raise_for_status()
        if not _is_valid_csv_bytes(r.content):
            return False
        _write_bytes_atomic(dst, r.content)
        return True
    except Exception:
        return False


def ensure_base_csv_remote_first() -> Optional[Path]:
    # 1) artifact
    if PREFER_REMOTE_BASE and GH_TOKEN:
        blob = fetch_file_from_latest_artifact(
            pick_names=[EVENT_CSV_NAME, "sf_crime_x.csv", "sf_crime.csv", PANEL_CSV_NAME, "sf_crime_y.csv"],
            artifact_name=ARTIFACT_NAME,
        )
        if blob and _is_valid_csv_bytes(blob):
            _write_bytes_atomic(TMP_BASE_CSV, blob)
            log(f"📦 Base (artifact) indirildi → {TMP_BASE_CSV}")
            return TMP_BASE_CSV
        log("⚠️ Artifact base bulunamadı/uygun değil.")

    # 2) release
    if PREFER_REMOTE_BASE and CRIME_BASE_URL:
        dst = TMP_BASE_GZ if CRIME_BASE_URL.endswith(".gz") else TMP_BASE_CSV
        ok = download_url_to_file(CRIME_BASE_URL, dst)
        if ok:
            log(f"⬇️ Base (release latest) indirildi → {dst}")
            return dst
        log(f"⚠️ Release latest base indirilemedi/uygun değil: {CRIME_BASE_URL}")

    # 3) local fallback
    local_candidates = [
        SAVE_DIR / "sf_crime_x.csv",
        Path("sf_crime_x.csv"),
        SAVE_DIR / "sf_crime.csv",
        Path("sf_crime.csv"),
        SAVE_DIR / "sf_crime.csv.gz",
        SAVE_DIR / "sf_crime_y.csv",
        Path("sf_crime_y.csv"),
    ]

    for p in local_candidates:
        if _is_valid_local_csv(p):
            log(f"📦 Base (local fallback) bulundu: {p}")
            return p

    log("❌ Base bulunamadı (artifact/release/local).")
    return None


def read_existing_crime_csv(p: Path) -> Optional[pd.DataFrame]:
    if p is None or not p.exists():
        return None

    try:
        compression = "gzip" if p.suffix == ".gz" else None
        df = pd.read_csv(p, dtype={"GEOID": str}, low_memory=False, compression=compression)

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        elif "datetime" in df.columns:
            df["date"] = pd.to_datetime(df["datetime"], errors="coerce").dt.date
        else:
            raise ValueError("CSV içinde 'date' veya 'datetime' sütunu yok.")

        if "id" in df.columns:
            df["id"] = df["id"].astype(str)

        if "GEOID" in df.columns:
            df["GEOID"] = normalize_geoid_series(df["GEOID"])

        log(f"📂 Mevcut veri yüklendi: {len(df):,} satır | son tarih={df['date'].max()}")
        return df

    except Exception as e:
        log(f"⚠️ Mevcut sf_crime okunamadı: {e}")
        return None


def normalize_geoid_series(s: pd.Series, width: int = DEFAULT_GEOID_LEN) -> pd.Series:
    x = s.astype(str).str.extract(r"(\d+)")[0]
    x = x.str[:width]
    x = x.where(x.notna(), np.nan)
    return x


def safe_zfill_geoid(x, width: int = DEFAULT_GEOID_LEN):
    try:
        s = str(x)
        s = re.sub(r"\.0$", "", s)
        s = re.sub(r"\D", "", s)
        if not s:
            return np.nan
        return s.zfill(width)
    except Exception:
        return np.nan

def parse_dt_to_sf(s: pd.Series) -> pd.Series:
    dt = pd.to_datetime(s, errors="coerce", utc=True)
    return dt.dt.tz_convert(SF_TZ)

def to_slot_start_hour(hour_s: pd.Series) -> pd.Series:
    h = pd.to_numeric(hour_s, errors="coerce").fillna(0).astype(int) % 24
    return ((h // 3) * 3).astype(int)


def hour_range_from_start(start_h: pd.Series) -> pd.Series:
    return start_h.map(lambda x: f"{int(x):02d}-{int(x + 3):02d}")


def add_calendar_flags(df: pd.DataFrame, date_col: str = "date", hour_col: str = "event_hour") -> pd.DataFrame:
    out = df.copy()

    dt_date = pd.to_datetime(out[date_col], errors="coerce")
    out["day_of_week"] = dt_date.dt.weekday.astype("Int8")
    out["month"] = dt_date.dt.month.astype("Int8")

    season_map = {
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Fall", 10: "Fall", 11: "Fall",
    }
    out["season"] = out["month"].map(season_map)

    out["is_weekend"] = (out["day_of_week"] >= 5).astype("Int8")
    h = pd.to_numeric(out[hour_col], errors="coerce").fillna(0).astype(int)
    out["is_night"] = ((h >= 22) | (h <= 5)).astype("Int8")
    out["is_school_hour"] = h.between(8, 15).astype("Int8")
    out["is_business_hour"] = (h.between(9, 17) & (out["day_of_week"] < 5)).astype("Int8")

    d_norm = dt_date.dt.normalize()
    if d_norm.notna().any():
        min_year = int(d_norm.dt.year.min())
        max_year = int(d_norm.dt.year.max())
        us_hol = holidays.US(years=range(min_year, max_year + 1))
        hol_idx = pd.DatetimeIndex(pd.to_datetime(list(us_hol.keys()))).normalize()
        out["is_holiday"] = d_norm.isin(hol_idx).astype("Int8")
    else:
        out["is_holiday"] = 0

    return out


# ============================================================
# GEOID BLOCKS
# ============================================================

def load_blocks_geojson(path: Path) -> Optional[gpd.GeoDataFrame]:
    if not path.exists():
        log(f"ℹ️ {path} bulunamadı; GEOID eşlemesi atlanacak.")
        return None

    try:
        gdf = gpd.read_file(path)

        if "GEOID" not in gdf.columns:
            log(f"⚠️ blocks dosyasında 'GEOID' kolonu yok: {path}")
            return None

        gdf["GEOID"] = normalize_geoid_series(gdf["GEOID"])
        gdf = gdf.set_crs("EPSG:4326") if gdf.crs is None else gdf.to_crs("EPSG:4326")

        log(f"📊 BLOCKS geojson: {len(gdf):,} satır")
        return gdf

    except Exception as e:
        log(f"⚠️ Blok dosyası okunamadı ({path}): {e}")
        return None


# ============================================================
# SOCrata CRIME FETCH
# ============================================================

def _try_small_crime_request(params: dict):
    p = dict(params)
    p["$limit"] = 1
    p["$offset"] = 0
    r = requests.get(CRIME_API_URL, headers=headers, params=p, timeout=60)
    r.raise_for_status()
    return r


def detect_datetime_column_for_range(start_day: datetime.date, end_day: datetime.date) -> Optional[str]:
    dt_candidates = ["incident_datetime", "incident_date", "datetime"]
    rng_start = f"{start_day}T00:00:00"
    rng_end = f"{end_day}T23:59:59"

    for dt_col in dt_candidates:
        try:
            _try_small_crime_request({"$where": f"{dt_col} between '{rng_start}' and '{rng_end}'"})
            return dt_col
        except Exception:
            continue
    return None


def fetch_crime_range_all_chunks(start_day: datetime.date, end_day: datetime.date) -> Optional[pd.DataFrame]:
    chosen_dt = detect_datetime_column_for_range(start_day, end_day)
    if chosen_dt is None:
        log("❌ Aralık için datetime alanı bulunamadı.")
        return None

    rng_start = f"{start_day}T00:00:00"
    rng_end = f"{end_day}T23:59:59"

    pieces = []
    offset = 0
    page = 1

    while True:
        params = {
            "$where": f"{chosen_dt} between '{rng_start}' and '{rng_end}'",
            "$limit": CHUNK_LIMIT,
            "$offset": offset,
        }

        df = None
        for attempt in range(MAX_RETRIES):
            try:
                r = requests.get(CRIME_API_URL, headers=headers, params=params, timeout=60)
                r.raise_for_status()
                df = pd.read_json(io.BytesIO(r.content))
                break
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    log(f"❌ range page {page} (offset={offset}) hata: {e}")
                else:
                    time.sleep(1.0 + attempt * 0.5)

        if df is None or df.empty:
            if page == 1:
                log("ℹ️ Bu aralıkta veri yok.")
            break

        log(f"   + {len(df):,} satır (range-page={page}, offset={offset})")
        pieces.append(df)

        if len(df) < CHUNK_LIMIT:
            break

        offset += CHUNK_LIMIT
        page += 1
        time.sleep(SLEEP_BETWEEN_REQS)

    if not pieces:
        return None
    return pd.concat(pieces, ignore_index=True)


def get_latest_available_date() -> Optional[datetime.date]:
    dt_candidates = ["incident_datetime", "incident_date", "datetime"]
    for dt_col in dt_candidates:
        try:
            params = {"$select": f"max({dt_col}) as max_dt", "$limit": 1}
            r = requests.get(CRIME_API_URL, headers=headers, params=params, timeout=60)
            r.raise_for_status()
            js = r.json()
            if js and js[0].get("max_dt"):
                dt = pd.to_datetime(js[0]["max_dt"], errors="coerce")
                if pd.notna(dt):
                    return dt.date()
        except Exception:
            continue
    return None


# ============================================================
# RAW -> EVENT
# ============================================================

def parse_point_like(s):
    try:
        if isinstance(s, dict) and "coordinates" in s:
            lon, lat = s["coordinates"]
            return pd.Series({"longitude": float(lon), "latitude": float(lat)})

        txt = str(s)
        m = re.search(r"(-?\d+\.\d+)[ ,]+(-?\d+\.\d+)", txt)
        if m:
            a, b = float(m.group(1)), float(m.group(2))
            lon, lat = (a, b) if abs(a) > abs(b) else (b, a)
            return pd.Series({"longitude": lon, "latitude": lat})
    except Exception:
        pass

    return pd.Series({"longitude": np.nan, "latitude": np.nan})


def prepare_raw_crime_to_event(raw_new: pd.DataFrame, gdf_blocks: Optional[gpd.GeoDataFrame]) -> pd.DataFrame:
    df = raw_new.copy()

    # datetime üret
    if "incident_datetime" not in df.columns:
        if "incident_date" in df.columns and "incident_time" in df.columns:
            df["incident_datetime"] = pd.to_datetime(
                df["incident_date"].astype(str) + " " + df["incident_time"].astype(str),
                errors="coerce",
            )
        elif "incident_date" in df.columns:
            df["incident_datetime"] = pd.to_datetime(df["incident_date"], errors="coerce")
        elif "datetime" in df.columns:
            df["incident_datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    # lat/lon onar
    if ("latitude" not in df.columns) or ("longitude" not in df.columns):
        if "point" in df.columns:
            coords = df["point"].apply(parse_point_like)
            if "latitude" not in df.columns:
                df["latitude"] = coords["latitude"]
            if "longitude" not in df.columns:
                df["longitude"] = coords["longitude"]
        elif "location" in df.columns:
            coords = df["location"].apply(parse_point_like)
            if "latitude" not in df.columns:
                df["latitude"] = coords["latitude"]
            if "longitude" not in df.columns:
                df["longitude"] = coords["longitude"]

    df["datetime"] = parse_dt_to_sf(df["incident_datetime"])
    df["date"] = pd.to_datetime(df["datetime"], errors="coerce").dt.date
    df["time"] = pd.to_datetime(df["datetime"], errors="coerce").dt.strftime("%H:%M:%S")
    df["event_hour"] = pd.to_datetime(df["datetime"], errors="coerce").dt.hour

    # id üret
    id_cols = [c for c in ["row_id", "incident_id", "incident_number", "cad_number"] if c in df.columns]
    if id_cols:
        s = df[id_cols[0]].astype(str)
        for c in id_cols[1:]:
            s = s.where(s.notna() & (s.astype(str) != "nan"), df[c].astype(str))
        df["id"] = s
    else:
        df["id"] = np.nan

    mask = df["id"].isna() | (df["id"].astype(str).str.lower().isin(["nan", "none", ""]))
    if mask.any():
        if {"latitude", "longitude"}.issubset(df.columns):
            df.loc[mask, "id"] = (
                df.loc[mask, "datetime"].astype(str)
                + "_"
                + df.loc[mask, "latitude"].round(6).astype(str)
                + "_"
                + df.loc[mask, "longitude"].round(6).astype(str)
            )
        else:
            base = df.loc[mask, "datetime"].astype(str)
            if "incident_number" in df.columns:
                base = base + "_" + df.loc[mask, "incident_number"].astype(str)
            elif "incident_id" in df.columns:
                base = base + "_" + df.loc[mask, "incident_id"].astype(str)
            if "category" in df.columns:
                base = base + "_" + df.loc[mask, "category"].astype(str)
            df.loc[mask, "id"] = base

    df["id"] = df["id"].astype(str)

    # rename
    df = df.rename(columns={
        "incident_category": "category",
        "incident_subcategory": "subcategory",
    })

    keep_cols = [c for c in [
        "id", "datetime", "date", "time", "event_hour",
        "latitude", "longitude", "category", "subcategory"
    ] if c in df.columns]
    df = df[keep_cols].copy()

    subset_cols = [c for c in ["id", "date", "latitude", "longitude"] if c in df.columns]
    if subset_cols:
        df = df.dropna(subset=subset_cols)

    if {"latitude", "longitude"}.issubset(df.columns):
        min_lon, min_lat, max_lon, max_lat = SF_BBOX
        df = df[df["latitude"].between(min_lat, max_lat)]
        df = df[df["longitude"].between(min_lon, max_lon)]

    # GEOID
    if gdf_blocks is not None and {"latitude", "longitude"}.issubset(df.columns):
        gdf_points = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
            crs="EPSG:4326",
        )
        gdf_points = gpd.sjoin(
            gdf_points,
            gdf_blocks[["GEOID", "geometry"]],
            how="left",
            predicate="within",
        )
        gdf_points = gdf_points.drop(columns=["geometry", "index_right"], errors="ignore")
        gdf_points["GEOID"] = normalize_geoid_series(gdf_points["GEOID"])
        df = pd.DataFrame(gdf_points)
    else:
        if "GEOID" not in df.columns:
            df["GEOID"] = np.nan

    # category cleanup
    if {"category", "subcategory"}.issubset(df.columns):
        for col in ["category", "subcategory"]:
            df[col] = df[col].astype(str).replace({
                "nan": np.nan, "None": np.nan, "none": np.nan, "": np.nan
            })

        df["is_category_missing"] = df["category"].isna().astype("Int8")
        df["is_subcategory_missing"] = df["subcategory"].isna().astype("Int8")

        both_nan = df["category"].isna() & df["subcategory"].isna()
        df.loc[both_nan, ["category", "subcategory"]] = "Unknown"

        only_sub_nan = df["subcategory"].isna() & df["category"].notna()
        df.loc[only_sub_nan, "subcategory"] = "Unknown"
    else:
        df["category"] = "Unknown"
        df["subcategory"] = "Unknown"
        df["is_category_missing"] = 1
        df["is_subcategory_missing"] = 1

    # 5 yıl filtresi
    today_sf = datetime.now(SF_TZ).date()
    start_date_5y = (pd.Timestamp(today_sf) - pd.DateOffset(years=5)).date()
    df = df[pd.to_datetime(df["date"], errors="coerce").notna()].copy()
    df = df[df["date"] >= start_date_5y].copy()

    # datetime floor
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"]).copy()
    df["datetime"] = df["datetime"].dt.floor("h")

    try:
        if getattr(df["datetime"].dt, "tz", None) is None:
            df["datetime"] = df["datetime"].dt.tz_localize(SF_TZ, nonexistent="shift_forward", ambiguous="NaT")
        else:
            df["datetime"] = df["datetime"].dt.tz_convert(SF_TZ)
    except Exception:
        pass

    # duplicates
    if "id" in df.columns:
        before = len(df)
        df = df.drop_duplicates(["id"], keep="last")
        log(f"🧹 Duplicate (id) temizlendi: {before - len(df)} satır")
    else:
        key_cols = [c for c in ["datetime", "latitude", "longitude"] if c in df.columns]
        if len(key_cols) == 3:
            before = len(df)
            df = df.drop_duplicates(key_cols, keep="last")
            log(f"🧹 Duplicate (datetime+lat+lon) temizlendi: {before - len(df)} satır")

    # std kolonlar
    df["GEOID_std"] = df["GEOID"].apply(safe_zfill_geoid)
    df["category_std"] = (
        df["category"].astype(str)
        .str.strip()
        .str.replace(r"[?]+$", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .replace({"None": np.nan, "none": np.nan, "nan": np.nan, "NaN": np.nan, "": np.nan})
    )
    df["subcategory_std"] = (
        df["subcategory"].astype(str)
        .replace({"nan": np.nan, "None": np.nan, "": np.nan})
    )

    both_nan_std = df["category_std"].isna() & df["subcategory_std"].isna()
    df.loc[both_nan_std, ["category_std", "subcategory_std"]] = "Unknown"

    only_sub_nan_std = df["subcategory_std"].isna() & df["category_std"].notna()
    df.loc[only_sub_nan_std, "subcategory_std"] = "Unknown"

    df["is_category_valid"] = (df["category_std"].notna() & (df["category_std"] != "Unknown")).astype("Int8")

    # time flags
    df = add_calendar_flags(df, date_col="date", hour_col="event_hour")

    # same-slot key
    df["slot_start_hour"] = to_slot_start_hour(df["event_hour"])
    df["hour_range"] = hour_range_from_start(df["slot_start_hour"])
    df["slot_start_dt"] = (
        pd.to_datetime(df["date"], errors="coerce")
        + pd.to_timedelta(df["slot_start_hour"], unit="h")
    )
    try:
        if getattr(df["slot_start_dt"].dt, "tz", None) is None:
            df["slot_start_dt"] = df["slot_start_dt"].dt.tz_localize(
                SF_TZ, nonexistent="shift_forward", ambiguous="NaT"
            )
        else:
            df["slot_start_dt"] = df["slot_start_dt"].dt.tz_convert(SF_TZ)
    except Exception:
        pass

    return df


# ============================================================
# PANEL FEATURES FROM CRIME ONLY
# ============================================================

def add_last_crime_anchor_features(panel_df: pd.DataFrame, event_df: pd.DataFrame) -> pd.DataFrame:
    out = panel_df.copy()

    feature_cols = [
        "crime_count_last_1d_from_last_crime",
        "crime_count_last_3d_from_last_crime",
        "crime_count_last_7d_from_last_crime",
        "last_crime_dt",
        "hours_since_last_crime",
        "days_since_last_crime",
        "exp_decay_last_crime_24h",
        "exp_decay_last_crime_72h",
    ]

    if out.empty:
        for c in feature_cols:
            out[c] = 0 if c != "last_crime_dt" else pd.NaT
        return out

    if "slot_start_dt" not in out.columns:
        raise ValueError("panel_df içinde 'slot_start_dt' kolonu yok.")

    ev = event_df.copy()
    if ev.empty or ("GEOID" not in ev.columns) or ("datetime" not in ev.columns):
        for c in feature_cols:
            out[c] = 0 if c != "last_crime_dt" else pd.NaT
        return out

    ev["GEOID"] = normalize_geoid_series(ev["GEOID"])
    ev["datetime"] = pd.to_datetime(ev["datetime"], errors="coerce")
    ev = ev.dropna(subset=["GEOID", "datetime"]).copy()

    try:
        if getattr(ev["datetime"].dt, "tz", None) is None:
            ev["datetime"] = ev["datetime"].dt.tz_localize(
                SF_TZ, nonexistent="shift_forward", ambiguous="NaT"
            )
        else:
            ev["datetime"] = ev["datetime"].dt.tz_convert(SF_TZ)
    except Exception:
        pass

    ev = ev.sort_values(["GEOID", "datetime"]).copy()

    for c in feature_cols:
        out[c] = 0 if c != "last_crime_dt" else pd.NaT

    out["_row_id_tmp"] = np.arange(len(out))
    out = out.sort_values(["GEOID", "slot_start_dt"]).copy()

    ev_groups = {
        g: x["datetime"].astype("int64").to_numpy()
        for g, x in ev.groupby("GEOID", sort=False)
    }

    windows = {
        "crime_count_last_1d_from_last_crime": pd.Timedelta(days=1).value,
        "crime_count_last_3d_from_last_crime": pd.Timedelta(days=3).value,
        "crime_count_last_7d_from_last_crime": pd.Timedelta(days=7).value,
    }

    for geoid, idx in out.groupby("GEOID", sort=False).groups.items():
        slot_vals = pd.to_datetime(out.loc[idx, "slot_start_dt"], errors="coerce")
        try:
            if getattr(slot_vals.dt, "tz", None) is None:
                slot_vals = slot_vals.dt.tz_localize(SF_TZ, nonexistent="shift_forward", ambiguous="NaT")
            else:
                slot_vals = slot_vals.dt.tz_convert(SF_TZ)
        except Exception:
            pass

        slot_ns = slot_vals.astype("int64").to_numpy()
        ev_ns = ev_groups.get(str(geoid))

        if ev_ns is None or len(ev_ns) == 0:
            continue

        pos = np.searchsorted(ev_ns, slot_ns, side="right") - 1
        valid = pos >= 0
        if not valid.any():
            continue

        last_ns = np.full(len(slot_ns), np.nan, dtype="float64")
        last_ns[valid] = ev_ns[pos[valid]]

        last_dt = pd.to_datetime(last_ns, errors="coerce", utc=True).tz_convert(SF_TZ)
        out.loc[idx, "last_crime_dt"] = pd.Series(last_dt, index=idx)

        hours_since = np.full(len(slot_ns), np.nan, dtype="float64")
        hours_since[valid] = (slot_ns[valid] - ev_ns[pos[valid]]) / 3_600_000_000_000
        out.loc[idx, "hours_since_last_crime"] = hours_since
        out.loc[idx, "days_since_last_crime"] = hours_since / 24.0
        out.loc[idx, "exp_decay_last_crime_24h"] = np.exp(-hours_since / 24.0)
        out.loc[idx, "exp_decay_last_crime_72h"] = np.exp(-hours_since / 72.0)

        for col, win_ns in windows.items():
            counts = np.zeros(len(slot_ns), dtype=np.int32)
            valid_pos = np.where(valid)[0]
            if len(valid_pos) > 0:
                anchors = ev_ns[pos[valid]]
                left = np.searchsorted(ev_ns, anchors - win_ns, side="left")
                right = np.searchsorted(ev_ns, anchors, side="right")
                counts[valid] = (right - left).astype(np.int32)
            out.loc[idx, col] = counts

    out["hours_since_last_crime"] = pd.to_numeric(out["hours_since_last_crime"], errors="coerce").fillna(9999.0)
    out["days_since_last_crime"] = pd.to_numeric(out["days_since_last_crime"], errors="coerce").fillna(9999.0)
    out["exp_decay_last_crime_24h"] = pd.to_numeric(out["exp_decay_last_crime_24h"], errors="coerce").fillna(0.0)
    out["exp_decay_last_crime_72h"] = pd.to_numeric(out["exp_decay_last_crime_72h"], errors="coerce").fillna(0.0)

    out = out.sort_values("_row_id_tmp").drop(columns=["_row_id_tmp"])
    return out


def add_cell_rolling_history(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out = out.sort_values(["GEOID", "hour_range", "date"]).copy()

    grp = out.groupby(["GEOID", "hour_range"], sort=False)["y_count"]

    out["same_slot_prev_1"]  = grp.shift(1).fillna(0).astype("float32")
    out["same_slot_prev_2"]  = grp.shift(2).fillna(0).astype("float32")
    out["same_slot_prev_7"]  = grp.shift(7).fillna(0).astype("float32")
    out["same_slot_prev_14"] = grp.shift(14).fillna(0).astype("float32")
    out["same_slot_prev_28"] = grp.shift(28).fillna(0).astype("float32")

    shifted = grp.shift(1)

    roll_grp = shifted.groupby([out["GEOID"], out["hour_range"]], sort=False)

    out["same_slot_mean_7"]  = (
        roll_grp.rolling(7, min_periods=1)
        .mean()
        .reset_index(level=[0, 1], drop=True)
        .fillna(0)
        .astype("float32")
    )
    out["same_slot_mean_14"] = (
        roll_grp.rolling(14, min_periods=1)
        .mean()
        .reset_index(level=[0, 1], drop=True)
        .fillna(0)
        .astype("float32")
    )
    out["same_slot_mean_28"] = (
        roll_grp.rolling(28, min_periods=1)
        .mean()
        .reset_index(level=[0, 1], drop=True)
        .fillna(0)
        .astype("float32")
    )

    out["same_slot_std_7"] = (
        roll_grp.rolling(7, min_periods=2)
        .std()
        .reset_index(level=[0, 1], drop=True)
        .fillna(0)
        .astype("float32")
    )
    out["same_slot_std_14"] = (
        roll_grp.rolling(14, min_periods=2)
        .std()
        .reset_index(level=[0, 1], drop=True)
        .fillna(0)
        .astype("float32")
    )
    out["same_slot_std_28"] = (
        roll_grp.rolling(28, min_periods=2)
        .std()
        .reset_index(level=[0, 1], drop=True)
        .fillna(0)
        .astype("float32")
    )

    return out


def add_daily_geoid_history(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()

    day_counts = (
        out.groupby(["GEOID", "date"], as_index=False, observed=True)["y_count"]
        .sum()
        .sort_values(["GEOID", "date"])
    )

    grp = day_counts.groupby("GEOID", sort=False)["y_count"]

    day_counts["geoid_day_prev_1"] = grp.shift(1).fillna(0).astype("float32")
    day_counts["geoid_day_prev_7"] = grp.shift(7).fillna(0).astype("float32")
    day_counts["geoid_day_mean_7"] = grp.shift(1).rolling(7, min_periods=1).mean().reset_index(level=0, drop=True).fillna(0).astype("float32")
    day_counts["geoid_day_mean_14"] = grp.shift(1).rolling(14, min_periods=1).mean().reset_index(level=0, drop=True).fillna(0).astype("float32")
    day_counts["geoid_day_std_7"] = grp.shift(1).rolling(7, min_periods=2).std().reset_index(level=0, drop=True).fillna(0).astype("float32")

    keep_cols = [
        "GEOID", "date",
        "geoid_day_prev_1", "geoid_day_prev_7",
        "geoid_day_mean_7", "geoid_day_mean_14", "geoid_day_std_7",
    ]
    out = out.merge(day_counts[keep_cols], on=["GEOID", "date"], how="left")

    for c in keep_cols[2:]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype("float32")

    return out


def add_simple_interactions(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()

    out["night_x_weekend"] = (out["is_night"].fillna(0).astype(float) * out["is_weekend"].fillna(0).astype(float)).astype("float32")
    out["holiday_x_night"] = (out["is_holiday"].fillna(0).astype(float) * out["is_night"].fillna(0).astype(float)).astype("float32")
    out["business_x_weekday"] = (out["is_business_hour"].fillna(0).astype(float) * (1 - out["is_weekend"].fillna(0).astype(float))).astype("float32")
    out["recent_crime_x_weekend"] = (out["exp_decay_last_crime_24h"].fillna(0).astype(float) * out["is_weekend"].fillna(0).astype(float)).astype("float32")

    return out


# ============================================================
# BUILD PANEL
# ============================================================

def build_panel_from_event(df_all: pd.DataFrame) -> pd.DataFrame:
    panel_evt = df_all.copy()
    panel_evt["date"] = pd.to_datetime(panel_evt["date"], errors="coerce").dt.date

    panel_evt["slot_start_hour"] = to_slot_start_hour(panel_evt["event_hour"])
    panel_evt["hour_range"] = hour_range_from_start(panel_evt["slot_start_hour"])

    panel_evt["slot_start_dt"] = (
        pd.to_datetime(panel_evt["date"], errors="coerce")
        + pd.to_timedelta(panel_evt["slot_start_hour"], unit="h")
    )

    try:
        if getattr(panel_evt["slot_start_dt"].dt, "tz", None) is None:
            panel_evt["slot_start_dt"] = panel_evt["slot_start_dt"].dt.tz_localize(
                SF_TZ, nonexistent="shift_forward", ambiguous="NaT"
            )
        else:
            panel_evt["slot_start_dt"] = panel_evt["slot_start_dt"].dt.tz_convert(SF_TZ)
    except Exception:
        pass

    slot_y = (
        panel_evt.dropna(subset=["GEOID", "date", "hour_range"])
        .groupby(["GEOID", "date", "hour_range"], as_index=False, observed=True)
        .size()
        .rename(columns={"size": "y_count"})
    )

    slot_y["y_count"] = pd.to_numeric(slot_y["y_count"], errors="coerce").fillna(0).astype("int16")
    slot_y["y_event"] = (slot_y["y_count"] > 0).astype("int8")
    slot_y["Y_label"] = slot_y["y_event"].astype("int8")

    all_geoids = (
        panel_evt["GEOID"]
        .dropna()
        .astype(str)
        .str.extract(r"(\d+)")[0]
        .str[:DEFAULT_GEOID_LEN]
        .dropna()
        .unique()
    )

    dmin = panel_evt["date"].min()
    dmax = panel_evt["date"].max()
    all_dates = pd.date_range(dmin, dmax, freq="D").date
    hour_starts = list(range(0, 24, 3))

    grid = pd.MultiIndex.from_product(
        [all_geoids, all_dates, hour_starts],
        names=["GEOID", "date", "slot_start_hour"]
    ).to_frame(index=False)

    grid["hour_range"] = grid["slot_start_hour"].map(lambda h: f"{int(h):02d}-{int(h + 3):02d}")
    grid["slot_start_dt"] = (
        pd.to_datetime(grid["date"], errors="coerce")
        + pd.to_timedelta(grid["slot_start_hour"], unit="h")
    )

    try:
        if getattr(grid["slot_start_dt"].dt, "tz", None) is None:
            grid["slot_start_dt"] = grid["slot_start_dt"].dt.tz_localize(
                SF_TZ, nonexistent="shift_forward", ambiguous="NaT"
            )
        else:
            grid["slot_start_dt"] = grid["slot_start_dt"].dt.tz_convert(SF_TZ)
    except Exception:
        pass

    latest_published_dt = parse_dt_to_sf(panel_evt["datetime"]).max()
    if pd.notna(latest_published_dt):
        latest_published_dt = pd.Timestamp(latest_published_dt)
        try:
            if latest_published_dt.tz is None:
                latest_published_dt = latest_published_dt.tz_localize(
                    SF_TZ, nonexistent="shift_forward", ambiguous="NaT"
                )
            else:
                latest_published_dt = latest_published_dt.tz_convert(SF_TZ)
        except Exception:
            pass

        latest_anchor_slot = latest_published_dt.floor("3h")
        grid = grid[grid["slot_start_dt"] <= latest_anchor_slot].copy()
        log(f"🕒 Grid son yayınlanan slot'a göre kesildi: {latest_anchor_slot}")

    panel = grid.merge(
        slot_y[["GEOID", "date", "hour_range", "y_count", "y_event", "Y_label"]],
        on=["GEOID", "date", "hour_range"],
        how="left",
    )

    panel["y_count"] = panel["y_count"].fillna(0).astype("int16")
    panel["y_event"] = panel["y_event"].fillna(0).astype("int8")
    panel["Y_label"] = panel["Y_label"].fillna(0).astype("int8")

    panel["event_hour"] = panel["slot_start_hour"].astype("int8")
    panel = add_calendar_flags(panel, date_col="date", hour_col="event_hour")

    panel = add_last_crime_anchor_features(panel, df_all)
    panel = add_cell_rolling_history(panel)
    panel = add_daily_geoid_history(panel)
    panel = add_simple_interactions(panel)

    panel["GEOID_std"] = panel["GEOID"].apply(safe_zfill_geoid)

    panel = panel.drop(columns=["slot_start_hour"], errors="ignore")
    return panel


# ============================================================
# MAIN
# ============================================================

def main():
    today = datetime.now(SF_TZ).date()

    base_path = ensure_base_csv_remote_first()
    if base_path is None:
        raise SystemExit(1)

    log(f"📦 Base path seçildi: {base_path}")
    df_old = read_existing_crime_csv(base_path)
    if df_old is None:
        raise SystemExit(1)

    if "date" not in df_old.columns:
        raise SystemExit("❌ Base veri içinde 'date' kolonu yok.")

    latest_date = pd.to_datetime(df_old["date"], errors="coerce").dt.date.max()

    api_latest = get_latest_available_date()
    if api_latest:
        latest_available = api_latest
        log(f"🛰️ API latest available date: {latest_available}")
    else:
        latest_available = today - timedelta(days=PUBLISH_LAG_FALLBACK_DAYS)
        log(f"⚠️ API latest alınamadı → fallback latest_available: {latest_available}")

    start_date_5y = (pd.Timestamp(today) - pd.DateOffset(years=5)).date()

    if FORCE_FULL:
        start_missing = start_date_5y
        end_missing = latest_available
    else:
        start_missing = latest_date - timedelta(days=max(1, CRIME_REINGEST_DAYS))
        if start_missing < start_date_5y:
            start_missing = start_date_5y
        end_missing = latest_available

    missing_dates = []
    if latest_date < latest_available:
        date_range = pd.date_range(start=start_missing, end=end_missing)
        missing_dates = [d.date() for d in date_range]
    else:
        log(f"ℹ️ Base zaten güncel görünüyor: latest_date={latest_date} ≥ latest_available={latest_available}")

    log(f"📆 Eksik tarihler: {len(missing_dates)} | end={latest_available}")

    gdf_blocks = load_blocks_geojson(blocks_path)

    raw_new = None
    if missing_dates or FORCE_FULL:
        log(f"📥 CRIME indirme penceresi: {start_missing} → {end_missing} | BULK={BULK_RANGE} | CHUNK={CHUNK_LIMIT}")
        if BULK_RANGE:
            raw_new = fetch_crime_range_all_chunks(start_missing, end_missing)
        else:
            pieces = []
            cur = start_missing
            while cur <= end_missing:
                part = fetch_crime_range_all_chunks(cur, cur)
                if part is not None and not part.empty:
                    pieces.append(part)
                cur += timedelta(days=1)
                time.sleep(SLEEP_BETWEEN_REQS)
            if pieces:
                raw_new = pd.concat(pieces, ignore_index=True)

    if raw_new is not None and not raw_new.empty:
        df_new = prepare_raw_crime_to_event(raw_new, gdf_blocks)
        log(f"📊 Yeni indirilen event: {df_new.shape}")
    else:
        df_new = pd.DataFrame()
        log("ℹ️ Yeni crime verisi indirilemedi veya boş.")

    # old + new birleştir
    if FORCE_FULL and not df_new.empty:
        df_all = df_new.copy()
    else:
        if df_new.empty:
            df_all = df_old.copy()
        else:
            common_cols = sorted(set(df_old.columns).union(set(df_new.columns)))
            df_old2 = df_old.reindex(columns=common_cols)
            df_new2 = df_new.reindex(columns=common_cols)
            df_all = pd.concat([df_old2, df_new2], ignore_index=True)

    # standard cleanup
    if "id" not in df_all.columns:
        df_all["id"] = np.nan

    df_all["id"] = df_all["id"].astype(str)
    if "GEOID" in df_all.columns:
        df_all["GEOID"] = normalize_geoid_series(df_all["GEOID"])

    if "id" in df_all.columns:
        before = len(df_all)
        df_all = df_all.drop_duplicates(["id"], keep="last")
        log(f"🧹 Birleşim sonrası duplicate (id) temizlendi: {before - len(df_all)} satır")

    if "date" in df_all.columns:
        df_all["date"] = pd.to_datetime(df_all["date"], errors="coerce").dt.date
        df_all = df_all[df_all["date"].notna()].copy()
        df_all = df_all[df_all["date"] >= start_date_5y].copy()

    if "datetime" in df_all.columns:
        df_all["datetime"] = parse_dt_to_sf(df_all["datetime"])
    else:
        if {"date", "time"}.issubset(df_all.columns):
            df_all["datetime"] = parse_dt_to_sf(
                pd.to_datetime(
                    pd.to_datetime(df_all["date"], errors="coerce").astype(str) + " " + df_all["time"].astype(str),
                    errors="coerce",
                )
            )

    if "event_hour" not in df_all.columns and "datetime" in df_all.columns:
        df_all["event_hour"] = pd.to_datetime(df_all["datetime"], errors="coerce").dt.hour

    if "category" not in df_all.columns:
        df_all["category"] = "Unknown"
    if "subcategory" not in df_all.columns:
        df_all["subcategory"] = "Unknown"

    # event tarafında eksik std flag varsa tamamla
    if "GEOID_std" not in df_all.columns:
        df_all["GEOID_std"] = df_all["GEOID"].apply(safe_zfill_geoid)

    if "category_std" not in df_all.columns:
        df_all["category_std"] = (
            df_all["category"].astype(str)
            .replace({"nan": np.nan, "None": np.nan, "": np.nan})
            .fillna("Unknown")
        )

    if "subcategory_std" not in df_all.columns:
        df_all["subcategory_std"] = (
            df_all["subcategory"].astype(str)
            .replace({"nan": np.nan, "None": np.nan, "": np.nan})
            .fillna("Unknown")
        )

    if "is_category_valid" not in df_all.columns:
        df_all["is_category_valid"] = (df_all["category_std"] != "Unknown").astype("Int8")

    # time flags eksikse ekle
    need_time_flags = any(c not in df_all.columns for c in [
        "day_of_week", "month", "season",
        "is_weekend", "is_night", "is_school_hour",
        "is_business_hour", "is_holiday"
    ])
    if need_time_flags and {"date", "event_hour"}.issubset(df_all.columns):
        df_all = add_calendar_flags(df_all, date_col="date", hour_col="event_hour")

    # event save
    safe_save_csv(df_all.drop(columns=["date_only"], errors="ignore"), event_csv_path)
    log(f"💾 Event-level yazıldı → {event_csv_path}")

    # panel build
    panel = build_panel_from_event(df_all)
    safe_save_csv(panel, panel_csv_path)
    log(f"💾 Panel yazıldı → {panel_csv_path} | rows={len(panel):,}")

    # artifact copies
    try:
        artifact_dir = Path("crime_prediction_data")
        artifact_dir.mkdir(exist_ok=True)

        shutil.copy2(event_csv_path, artifact_dir / event_csv_path.name)
        shutil.copy2(panel_csv_path, artifact_dir / panel_csv_path.name)
        log("✅ artifact outputs: sf_crime_x.csv + sf_crime_y.csv")

        if WRITE_BASE_TO_REPO:
            shutil.copy2(panel_csv_path, artifact_dir / "sf_crime.csv")
            shutil.copy2(panel_csv_path, Path("sf_crime.csv"))
            log("📝 WRITE_BASE_TO_REPO=1 → sf_crime.csv panel ile güncellendi.")
        else:
            log("ℹ️ WRITE_BASE_TO_REPO=0 → repo sf_crime.csv ezilmedi.")

    except Exception as e:
        log(f"⚠️ Kopya uyarısı: {e}")

    log("✅ update_crime.py tamamlandı.")


if __name__ == "__main__":
    main()
