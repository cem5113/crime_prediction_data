# update_crime.py
from __future__ import annotations
import os
import time
import itertools
from datetime import datetime, timedelta
from pathlib import Path
import shutil
import io
import re
from typing import List

import numpy as np
import pandas as pd
import geopandas as gpd
import holidays
import requests
import zoneinfo
import json

# --------------------------
# ENV / Config
# --------------------------
SF_TZ = zoneinfo.ZoneInfo("America/Los_Angeles")
SF_TZ_NAME = "America/Los_Angeles"

AGGR_Y_THRESHOLD   = int(os.getenv("AGGR_Y_THRESHOLD", "6"))     # takvim grid label eşiği
DAILY_Y_THRESHOLD  = int(os.getenv("DAILY_Y_THRESHOLD", "1"))    # günlük grid label eşiği
DEFAULT_GEOID_LEN  = int(os.getenv("GEOID_LEN", "11"))           # tract GEOID uzunluğu
SF_BBOX            = (-123.2, 37.6, -122.3, 37.9)

# Günlük arşiv yazımı (opsiyonel)
WRITE_DAILY_ARCHIVE = os.getenv("WRITE_DAILY_ARCHIVE", "0").lower() in ("1","true","yes","on")
DAILY_WINDOW_DAYS   = int(os.getenv("DAILY_WINDOW_DAYS", "60"))
OVERWRITE_DAILY     = os.getenv("OVERWRITE_DAILY", "0").lower() in ("1","true","yes","on")

# Komşuluk hesapları
NEIGHBOR_METHOD   = os.getenv("NEIGHBOR_METHOD", "touches")  # touches | radius
NEIGHBOR_RADIUS_M = float(os.getenv("NEIGHBOR_RADIUS_M", "500"))

# --- Remote-first base davranışı ---
PREFER_REMOTE_BASE = os.getenv("PREFER_REMOTE_BASE", "1").lower() in ("1","true","yes","on")

# base veriyi repo içine yazma (default kapalı → statik dosyayı ezme)
WRITE_BASE_TO_REPO = os.getenv("WRITE_BASE_TO_REPO", "0").lower() in ("1","true","yes","on")

# sadece run içinde kullanılacak geçici klasör
RUN_TMP_DIR = Path(os.getenv("RUNNER_TEMP", "/tmp")) / "sfcrime_runtime"
RUN_TMP_DIR.mkdir(parents=True, exist_ok=True)

# remote base dosya isimleri (temp)
TMP_BASE_Y = RUN_TMP_DIR / "sf_crime_y.csv"
TMP_BASE_CSV = RUN_TMP_DIR / "sf_crime.csv"
TMP_BASE_GZ = RUN_TMP_DIR / "sf_crime.csv.gz"

# --------------------------
# Kaynak URL/Token
# --------------------------
# ➜ İstediğin akış: 1) Artifact'tan sf_crime_y.csv, 2) releases/latest sf_crime.csv
CRIME_BASE_URL = os.getenv(
    "CRIME_CSV_URL",
    "https://github.com/cem5113/crime_prediction_data/releases/latest/download/sf_crime.csv"  # Fallback (auto-latest)
)
CRIME_API_URL = os.getenv("CRIME_API_URL", "https://data.sfgov.org/resource/wg3w-h783.json")
SFCRIME_APP_TOKEN = os.getenv("SFCRIME_API_TOKEN", "")

CHUNK_LIMIT         = int(os.getenv("SFCRIME_CHUNK_LIMIT", "50000"))
MAX_RETRIES         = int(os.getenv("SFCRIME_MAX_RETRIES", "4"))
SLEEP_BETWEEN_REQS  = float(os.getenv("SFCRIME_SLEEP", "0.2"))
BULK_RANGE          = os.getenv("SFCRIME_BULK_RANGE", "1").lower() in ("1","true","yes","on")

# Yol/çıktılar
save_dir   = "."
csv_path   = os.path.join(save_dir, "sf_crime.csv")
sum_path   = os.path.join(save_dir, "sf_crime_grid_summary_labeled.csv")
full_path  = os.path.join(save_dir, "sf_crime_grid_full_labeled.csv")
blocks_path = os.path.join(save_dir, "sf_census_blocks_with_population.geojson")
CATMAP_PATH = Path(os.getenv("CATEGORY_MAP_PATH", "crime_prediction_data/category_map.json"))
DAILY_OUT_BASE = os.getenv("DAILY_OUT_BASE", "sf_crime_grid_daily_labels.parquet")
DAILY_PARTITION = True
USE_PARQUET = True

# ---- CACHE/Y-only çıkış ----
CACHE_WRITE_Y_ONLY = os.getenv("CACHE_WRITE_Y_ONLY", "1").lower() in ("1","true","yes","on")
Y_CSV_NAME = os.getenv("Y_CSV_NAME", "sf_crime_y.csv")
y_csv_path = os.path.join(save_dir, Y_CSV_NAME)

# ---- GitHub Actions artifact (sf_crime_y.csv) ayarları ----
GITHUB_REPO = os.getenv("GITHUB_REPO", "cem5113/crime_prediction_data")   # owner/repo
GH_TOKEN = os.getenv("GH_TOKEN", "")
ARTIFACT_NAME = os.getenv("ARTIFACT_NAME", "sf-crime-pipeline-output")

# --------------------------
# Helpers
# --------------------------
def _to_date_series(x):
    try:
        s = pd.to_datetime(x, utc=True, errors="coerce").dt.tz_convert(SF_TZ).dt.date
    except Exception:
        s = pd.to_datetime(x, errors="coerce").dt.date
    return pd.Series(s).dropna()

def log_shape(df, label):
    r, c = df.shape
    print(f"\U0001F4CA {label}: {r} satır × {c} sütun")

def log_date_range(df, date_col="date", label="Suç"):
    if date_col not in df.columns:
        print(f"\u26A0\ufe0f {label}: '{date_col}' kolonu yok.")
        return
    s = _to_date_series(df[date_col])
    if s.empty:
        print(f"\u26A0\ufe0f {label}: tarih parse edilemedi.")
        return
    print(f"\U0001F9ED {label} tarihi aralığı: {s.min()} → {s.max()} (gün={s.nunique()})")

def log_delta(before_shape, after_shape, label):
    br, bc = before_shape
    ar, ac = after_shape
    print(f"\U0001F517 {label}: {br}×{bc} → {ar}×{ac} (Δr={ar-br}, Δc={ac-bc})")

def safe_save(df: pd.DataFrame, path: str) -> None:
    try:
        Path(os.path.dirname(path) or ".").mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    except Exception as e:
        print(f"❌ Kaydedilemedi: {path}\n{e}")
        backup_path = path + ".bak"
        df.to_csv(backup_path, index=False)
        print(f"\U0001F4C1 Yedek dosya: {backup_path}")

def normalize_geoid(series: pd.Series, target_len: int) -> pd.Series:
    s = series.astype(str).str.extract(r"(\d+)", expand=False)
    L = int(target_len)
    return s.str[:L].str.zfill(L)

def is_lfs_pointer(p: Path) -> bool:
    try:
        head = p.read_text(errors="ignore")[:200]
        return "git-lfs.github.com/spec/v1" in head
    except Exception:
        return False

def fetch_all_categories_from_api() -> List[str]:
    def _norm(v: str) -> str: return str(v).strip().title()
    api = CRIME_API_URL
    headers = {"X-App-Token": SFCRIME_APP_TOKEN} if SFCRIME_APP_TOKEN else {}
    for field in ("incident_category", "category"):
        try:
            params = {"$select": field, "$group": field, "$limit": 50000}
            r = requests.get(api, headers=headers, params=params, timeout=60)
            r.raise_for_status()
            data = r.json() if r.content else []
            vals = [d.get(field) for d in data if d.get(field)]
            vals = sorted({_norm(v) for v in vals if v})
            if vals: return vals
        except Exception:
            continue
    return []

def _mix(df, keys):
    g = (df.groupby(keys + ["category"], dropna=False)
           .size().rename("cnt").reset_index())
    g["share"] = g["cnt"] / g.groupby(keys)["cnt"].transform("sum")
    g = g.sort_values(keys + ["share", "cnt", "category"],
                      ascending=[True]*len(keys) + [False, False, True])
    top3 = g.groupby(keys, as_index=False, sort=False).head(3).copy()
    top3["category"] = top3["category"].astype(str).str.strip().str.title()
    top3["txt"] = top3.apply(lambda r: f"{r['category']}({r['share']:.0%})", axis=1)
    out = top3.groupby(keys)["txt"].agg(", ".join).reset_index(name="crime_mix")
    return out

FALLBACK_CRIME_URL = CRIME_BASE_URL

# --------------------------
# Artifact indirme yardımcıları
# --------------------------
def _gh_headers():
    if not GH_TOKEN:
        return None
    return {
        "Authorization": f"Bearer {GH_TOKEN}",
        "Accept": "application/vnd.github+json",
    }

def fetch_file_from_latest_artifact(pick_names: List[str], artifact_name: str = ARTIFACT_NAME) -> bytes | None:
    """
    Son başarılı Actions run’ının artifact’ından pick_names’teki ilk eşleşeni döndürür.
    """
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
                if a.get("name") == artifact_name and not a.get("expired", False):
                    dl = requests.get(a["archive_download_url"], headers=hdr, timeout=60)
                    import zipfile, io as _io
                    zf = zipfile.ZipFile(_io.BytesIO(dl.content))
                    names = zf.namelist()
                    # tam ad + crime_prediction_data/ altı için dene
                    for pick in pick_names:
                        for c in (pick, f"crime_prediction_data/{pick}"):
                            if c in names:
                                return zf.read(c)
                    # suffix eşleşmesi
                    for n in names:
                        if any(n.endswith(p) for p in pick_names):
                            return zf.read(n)
        return None
    except Exception:
        return None

# --------------------------
# Mevcut veri (artifact-first) — SFCRIME_Y → release sf_crime.csv
# --------------------------
def _is_valid_csv_bytes(b: bytes, min_bytes: int = 5_000) -> bool:
    if not b or len(b) < min_bytes:
        return False
    head = b[:300].decode("utf-8", errors="ignore")
    # LFS pointer kontrolü
    if "git-lfs.github.com/spec/v1" in head:
        return False
    return True

def _is_valid_local_csv(p: Path, min_bytes: int = 5_000) -> bool:
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

def download_url_to_file(url: str, dst: Path, timeout: int = 120) -> bool:
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        if not _is_valid_csv_bytes(r.content):
            return False
        _write_bytes_atomic(dst, r.content)
        return True
    except Exception:
        return False

def ensure_base_csv_remote_first() -> Path | None:
    """
    Sıra:
      1) GitHub Actions artifact içinden sf_crime_y.csv (GH_TOKEN şart)
      2) releases/latest/download/sf_crime.csv (CRIME_BASE_URL)
      3) repo/local: sf_crime_y.csv veya sf_crime.csv

    Remote dosyalar RUNNER_TEMP (/tmp) altına yazılır → run-bitince zaten kalıcı repo dosyasını ezmez.
    """
    # 1) Artifact → TMP_BASE_Y
    if PREFER_REMOTE_BASE and GH_TOKEN:
        blob = fetch_file_from_latest_artifact(
            pick_names=[Y_CSV_NAME, "sf_crime_y.csv", "sf_crime.csv"],
            artifact_name=ARTIFACT_NAME,
        )
        if blob and _is_valid_csv_bytes(blob):
            _write_bytes_atomic(TMP_BASE_Y, blob)
            print(f"📦 Base (artifact) indirildi → {TMP_BASE_Y}")
            return TMP_BASE_Y
        else:
            print("⚠️ Artifact base bulunamadı/uygun değil (boş/küçük/LFS).")

    # 2) Release latest → TMP_BASE_CSV veya TMP_BASE_GZ
    if PREFER_REMOTE_BASE and CRIME_BASE_URL:
        dst = TMP_BASE_GZ if CRIME_BASE_URL.endswith(".gz") else TMP_BASE_CSV
        ok = download_url_to_file(CRIME_BASE_URL, dst)
        if ok:
            print(f"⬇️ Base (release latest) indirildi → {dst}")
            return dst
        else:
            print(f"⚠️ Release latest base indirilemedi/uygun değil: {CRIME_BASE_URL}")

    # 3) Local fallback (repo içi)
    local_candidates = [
        Path("crime_prediction_data/sf_crime_y.csv"),
        Path("sf_crime_y.csv"),
        Path("sf_crime.csv"),
        Path("crime_prediction_data/sf_crime.csv"),
        Path("crime_prediction_data/sf_crime.csv.gz"),
    ]
    for p in local_candidates:
        if _is_valid_local_csv(p):
            print(f"📦 Base (local fallback) bulundu: {p}")
            return p

    print("❌ Base bulunamadı (artifact/release/local).")
    return None

def read_existing_crime_csv(p: Path) -> pd.DataFrame | None:
    if not p or not p.exists():
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
            df["GEOID"] = df["GEOID"].astype(str)
        print(f"\U0001F4C2 Mevcut veri yüklendi: {len(df)} satır (son tarih: {df['date'].max()})")
        return df
    except Exception as e:
        print(f"\u26A0\ufe0f Mevcut sf_crime okunamadı: {e}")
        return None

# --------------------------
# Başla: veri yükle & eksik günler
# --------------------------
today = datetime.now(SF_TZ).date()
start_date = today - timedelta(days=5 * 365)

base_path = ensure_base_csv_remote_first()
if base_path is None:
    raise SystemExit(1)

df_old = read_existing_crime_csv(base_path)
if df_old is None:
    raise SystemExit(1)

log_shape(df_old, "CRIME mevcut (df_old)")
log_date_range(df_old, "date", "Suç (mevcut)")
latest_date = df_old["date"].max()

date_range = pd.date_range(start=latest_date + timedelta(days=1), end=today)
missing_dates = [d.date() for d in date_range]
print(f"\U0001F4C6 Eksik tarihler: {len(missing_dates)}")
if not missing_dates:
    print("ℹ️ Eksik gün yok; artımlı indirme atlanacak.")

# Blok geojson
gdf_blocks = None
if os.path.exists(blocks_path):
    try:
        gdf_blocks = gpd.read_file(blocks_path)
        gdf_blocks["GEOID"] = gdf_blocks["GEOID"].astype(str).str.extract(r"(\d+)")[0].str[:DEFAULT_GEOID_LEN]
        if "GEOID" in df_old.columns:
            df_old["GEOID"] = df_old["GEOID"].astype(str).str.extract(r"(\d+)")[0].str[:DEFAULT_GEOID_LEN]
        log_shape(gdf_blocks, "BLOCKS geojson")
    except Exception as e:
        print(f"\u26A0\ufe0f Blok dosyası okunamadı ({blocks_path}): {e}. GEOID eşlemesi atlanacak.")
        gdf_blocks = None
else:
    print(f"ℹ️ {blocks_path} bulunamadı; GEOID eşlemesi atlanacak.")

# --------------------------
# API çekme (gün/gün veya aralık)
# --------------------------
headers = {"X-App-Token": SFCRIME_APP_TOKEN} if SFCRIME_APP_TOKEN else {}

def _try_small_crime_request(params):
    p = dict(params); p["$limit"] = 1; p["$offset"] = 0
    r = requests.get(CRIME_API_URL, headers=headers, params=p, timeout=60)
    r.raise_for_status(); return r

def fetch_crime_day_all_chunks(day: datetime.date) -> pd.DataFrame | None:
    dt_candidates = ["incident_datetime", "incident_date", "datetime"]
    chosen_dt, last_err = None, None
    for dt_col in dt_candidates:
        base_where = f"{dt_col} between '{day}T00:00:00' and '{day}T23:59:59'"
        try:
            _try_small_crime_request({"$where": base_where}); chosen_dt = dt_col; break
        except Exception as e:
            last_err = e; continue
    if chosen_dt is None:
        print(f"    ❌ {day} için datetime alanı bulunamadı. Son hata: {last_err}")
        return None
    pieces, offset, page = [], 0, 1
    while True:
        params = {"$where": f"{chosen_dt} between '{day}T00:00:00' and '{day}T23:59:59'",
                  "$limit": CHUNK_LIMIT, "$offset": offset}
        df = None
        for attempt in range(MAX_RETRIES):
            try:
                r = requests.get(CRIME_API_URL, headers=headers, params=params, timeout=60)
                r.raise_for_status(); df = pd.read_json(io.BytesIO(r.content)); break
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    print(f"    ❌ sayfa {page} (offset={offset}) hata: {e}")
                else:
                    time.sleep(1.0 + attempt * 0.5)
        if df is None or df.empty:
            if page == 1: print("    (bu günde veri yok)")
            break
        print(f"    + {len(df)} satır (sayfa={page}, offset={offset})"); pieces.append(df)
        if len(df) < CHUNK_LIMIT: break
        offset += CHUNK_LIMIT; page += 1; time.sleep(SLEEP_BETWEEN_REQS)
    return None if not pieces else pd.concat(pieces, ignore_index=True)

def fetch_crime_range_all_chunks(start_day: datetime.date, end_day: datetime.date) -> pd.DataFrame | None:
    dt_candidates = ["incident_datetime", "incident_date", "datetime"]
    rng_start, rng_end = f"{start_day}T00:00:00", f"{end_day}T23:59:59"
    chosen_dt, last_err = None, None
    for dt_col in dt_candidates:
        base_where = f"{dt_col} between '{rng_start}' and '{rng_end}'"
        try:
            _try_small_crime_request({"$where": base_where}); chosen_dt = dt_col; break
        except Exception as e:
            last_err = e; continue
    if chosen_dt is None:
        print(f"    ❌ Aralık için datetime alanı bulunamadı. Son hata: {last_err}")
        return None
    pieces, offset, page = [], 0, 1
    while True:
        params = {"$where": f"{chosen_dt} between '{rng_start}' and '{rng_end}'",
                  "$limit": CHUNK_LIMIT, "$offset": offset}
        df = None
        for attempt in range(MAX_RETRIES):
            try:
                r = requests.get(CRIME_API_URL, headers=headers, params=params, timeout=60)
                r.raise_for_status(); df = pd.read_json(io.BytesIO(r.content)); break
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    print(f"    ❌ range page {page} (offset={offset}) hata: {e}")
                else:
                    time.sleep(1.0 + attempt * 0.5)
        if df is None or df.empty:
            if page == 1: print("    (bu aralıkta veri yok)")
            break
        print(f"    + {len(df)} satır (range-page={page}, offset={offset})"); pieces.append(df)
        if len(df) < CHUNK_LIMIT: break
        offset += CHUNK_LIMIT; page += 1; time.sleep(SLEEP_BETWEEN_REQS)
    return None if not pieces else pd.concat(pieces, ignore_index=True)

# --------------------------
# İndir & temizle & GEOID eşle
# --------------------------
FORCE_FULL = os.getenv("CRIME_FORCE_FULL", "0").lower() in ("1","true","yes","on")

if missing_dates or FORCE_FULL:
    if FORCE_FULL:
        start_missing, end_missing = (today - timedelta(days=5*365)), today
    else:
        start_missing, end_missing = missing_dates[0], missing_dates[-1]
    print(f"\U0001F4E5 CRIME indirme penceresi: {start_missing} → {end_missing} (BULK={BULK_RANGE}, CHUNK={CHUNK_LIMIT})")
    if BULK_RANGE:
        raw_new = fetch_crime_range_all_chunks(start_missing, end_missing)
    else:
        pieces, cur = [], start_missing
        while cur <= end_missing:
            print(f"\U0001F4E5 {cur} indiriliyor...")
            day_df = fetch_crime_day_all_chunks(cur)
            if day_df is not None and not day_df.empty:
                pieces.append(day_df)
            cur += timedelta(days=1)
            time.sleep(SLEEP_BETWEEN_REQS)
        raw_new = pd.concat(pieces, ignore_index=True) if pieces else None
else:
    raw_new = None

if raw_new is not None and not raw_new.empty:
    df_new = raw_new.copy()

    # ---- incident_datetime üret ----
    if "incident_datetime" not in df_new.columns:
        if "incident_date" in df_new.columns and "incident_time" in df_new.columns:
            df_new["incident_datetime"] = pd.to_datetime(
                df_new["incident_date"].astype(str) + " " + df_new["incident_time"].astype(str), errors="coerce"
            )
        elif "incident_date" in df_new.columns:
            df_new["incident_datetime"] = pd.to_datetime(df_new["incident_date"], errors="coerce")
        elif "datetime" in df_new.columns:
            df_new["incident_datetime"] = pd.to_datetime(df_new["datetime"], errors="coerce")

    # ---- Koordinatları onar (point/location alanlarından çek) ----
    def _parse_point_like(s):
        try:
            # dict {"coordinates":[lon,lat]} olabiliyor
            if isinstance(s, dict) and "coordinates" in s:
                lon, lat = s["coordinates"]
                return pd.Series({"longitude": float(lon), "latitude": float(lat)})
            # "POINT (-122.41 37.77)" veya "-122.41, 37.77" gibi olabilir
            txt = str(s)
            m = re.search(r"(-?\d+\.\d+)[ ,]+(-?\d+\.\d+)", txt)
            if m:
                a, b = float(m.group(1)), float(m.group(2))
                # heüristik: |lat| genelde 37~38, |lon| 122 civarı (mutlak daha büyük olan lon)
                lon, lat = (a, b) if abs(a) > abs(b) else (b, a)
                return pd.Series({"longitude": lon, "latitude": lat})
        except Exception:
            pass
        return pd.Series({"longitude": np.nan, "latitude": np.nan})

    if ("latitude" not in df_new.columns) or ("longitude" not in df_new.columns):
        if "point" in df_new.columns:
            coords = df_new["point"].apply(_parse_point_like)
            for c in ("latitude","longitude"):
                if c not in df_new.columns:
                    df_new[c] = coords[c]
        elif "location" in df_new.columns:
            coords = df_new["location"].apply(_parse_point_like)
            for c in ("latitude","longitude"):
                if c not in df_new.columns:
                    df_new[c] = coords[c]

    # ---- Zaman türet ----
    df_new["datetime"]   = pd.to_datetime(df_new["incident_datetime"], utc=True, errors="coerce").dt.tz_convert(SF_TZ)
    df_new["date"]       = df_new["datetime"].dt.date
    df_new["time"]       = df_new["datetime"].dt.strftime("%H:%M:%S")
    df_new["event_hour"] = df_new["datetime"].dt.hour

    # ---- ID üret ----
    id_cols = [c for c in ["row_id","incident_id","incident_number","cad_number"] if c in df_new.columns]
    if id_cols:
        s = df_new[id_cols[0]].astype(str)
        for c in id_cols[1:]:
            s = s.where(s.notna() & (s.astype(str) != "nan"), df_new[c].astype(str))
        df_new["id"] = s
    else:
        df_new["id"] = np.nan

    # ID fallback — lat/lon varsa onları kullan, yoksa datetime+category ile üret
    mask = df_new["id"].isna() | (df_new["id"].astype(str) == "nan")
    if mask.any():
        if {"latitude","longitude"}.issubset(df_new.columns):
            df_new.loc[mask, "id"] = (
                df_new.loc[mask, "datetime"].astype(str) + "_" +
                df_new.loc[mask, "latitude"].round(6).astype(str) + "_" +
                df_new.loc[mask, "longitude"].round(6).astype(str)
            )
        else:
            base = df_new.loc[mask, "datetime"].astype(str)
            if "incident_number" in df_new.columns:
                base = base + "_" + df_new.loc[mask, "incident_number"].astype(str)
            elif "incident_id" in df_new.columns:
                base = base + "_" + df_new.loc[mask, "incident_id"].astype(str)
            if "category" in df_new.columns:
                base = base + "_" + df_new.loc[mask, "category"].astype(str)
            df_new.loc[mask, "id"] = base

    df_new["id"] = df_new["id"].astype(str)

    # ---- Kolon seçimi ----
    df_new = df_new.rename(columns={"incident_category":"category","incident_subcategory":"subcategory"})
    keep_cols = [c for c in ["id","date","time","event_hour","latitude","longitude","category","subcategory"] if c in df_new.columns]
    df_new = df_new[keep_cols]

    # ---- Güvenli dropna: sadece mevcut alanlar üzerinden ----
    subset_cols = [c for c in ["id","date","latitude","longitude"] if c in df_new.columns]
    if subset_cols:
        df_new = df_new.dropna(subset=subset_cols)

    # ---- BBOX filtresi (sadece lat/lon varsa) ----
    if {"latitude","longitude"}.issubset(df_new.columns):
        min_lon, min_lat, max_lon, max_lat = SF_BBOX
        df_new = df_new[df_new["latitude"].between(min_lat, max_lat)]
        df_new = df_new[df_new["longitude"].between(min_lon, max_lon)]
    else:
        print("⚠️ Koordinat sütunları yok; BBOX filtresi atlandı.")

    # ---- GEOID eşlemesi (sadece lat/lon varsa) ----
    if gdf_blocks is not None and {"latitude","longitude"}.issubset(df_new.columns):
        gdf_blocks = (gdf_blocks.set_crs("EPSG:4326") if gdf_blocks.crs is None else gdf_blocks.to_crs("EPSG:4326"))
        gdfp = gpd.GeoDataFrame(df_new, geometry=gpd.points_from_xy(df_new["longitude"], df_new["latitude"]), crs="EPSG:4326")
        gdfp = gpd.sjoin(gdfp, gdf_blocks[["GEOID","geometry"]], how="left", predicate="within")
        gdfp = gdfp.drop(columns=["geometry","index_right"], errors="ignore")
        gdfp["GEOID"] = gdfp["GEOID"].astype(str).str.extract(r"(\d+)")[0].str[:DEFAULT_GEOID_LEN]
        df_new = pd.DataFrame(gdfp)
    else:
        # lat/lon yoksa GEOID eşlemesi mümkün değil; sonraki adımlar grid/centroid ile dolduracak
        if "GEOID" not in df_new.columns:
            df_new["GEOID"] = np.nan
else:
    df_new = pd.DataFrame()

log_shape(df_new, "CRIME yeni (indirilen)")
log_date_range(df_new, "date", "Suç (yeni)")

# --------------------------
# Birleştir & zamanda özellikler
# --------------------------
if "time" not in df_old.columns:
    df_old["time"] = "00:00:00"
_before_merge = df_old.shape
if "date" in df_old.columns:
    df_old["date"] = pd.to_datetime(df_old["date"], errors="coerce").dt.date

if FORCE_FULL and (raw_new is not None) and (not raw_new.empty):
    df_all = df_new.copy()
else:
    df_all = pd.concat([df_old, df_new], ignore_index=True)

# normalize
df_all["id"] = df_all["id"].astype(str)
if "GEOID" in df_all.columns:
    df_all["GEOID"] = df_all["GEOID"].astype(str).str.extract(r"(\d+)")[0].str[:DEFAULT_GEOID_LEN]

# 5y pencere + datetime
start_date_5y = today - timedelta(days=5*365)
df_all = df_all[df_all["date"] >= start_date_5y]

df_all["date"] = pd.to_datetime(df_all["date"], errors="coerce")
df_all["time"] = df_all["time"].astype(str).fillna("00:00:00")
df_all["datetime"] = pd.to_datetime(df_all["date"].dt.strftime("%Y-%m-%d") + " " + df_all["time"], errors="coerce")
df_all = df_all.dropna(subset=["datetime"]).copy()
df_all["datetime"] = df_all["datetime"].dt.floor("h")

# TZ sabitle
try:
    df_all["datetime"] = df_all["datetime"].dt.tz_localize(SF_TZ)
except Exception:
    try:
        df_all["datetime"] = df_all["datetime"].dt.tz_convert(SF_TZ)
    except Exception:
        pass

# türev alanlar
df_all["event_hour"]  = df_all["datetime"].dt.hour
df_all["day_of_week"] = df_all["datetime"].dt.weekday
df_all["month"]       = df_all["datetime"].dt.month

# tatil
df_all["date_only"] = df_all["date"].dt.normalize()
if df_all["date_only"].notna().any():
    min_year = int(df_all.loc[df_all["date_only"].notna(), "date_only"].dt.year.min())
    max_year = int(df_all.loc[df_all["date_only"].notna(), "date_only"].dt.year.max())
    us_hol = holidays.US(years=range(min_year, max_year + 1))
    hol_idx = pd.DatetimeIndex(pd.to_datetime(list(us_hol.keys()))).normalize()
    df_all["is_holiday"] = df_all["date_only"].isin(hol_idx).astype(int)
else:
    df_all["is_holiday"] = 0

# bayraklar
df_all["is_weekend"]       = (df_all["day_of_week"] >= 5).astype(int)
df_all["is_night"]         = ((df_all["event_hour"] >= 22) | (df_all["event_hour"] <= 5)).astype(int)
df_all["is_school_hour"]   = df_all["event_hour"].between(8, 15).astype(int)
df_all["is_business_hour"] = (df_all["event_hour"].between(9, 17) & (df_all["day_of_week"] < 5)).astype(int)

season_map = {12:"Winter",1:"Winter",2:"Winter", 3:"Spring",4:"Spring",5:"Spring", 6:"Summer",7:"Summer",8:"Summer", 9:"Fall",10:"Fall",11:"Fall"}
df_all["season"] = df_all["month"].map(season_map)

# snapshot info
try:
    dmin = pd.to_datetime(df_all["date_only"], errors="coerce").min()
    dmax = pd.to_datetime(df_all["date_only"], errors="coerce").max()
    print("\U0001F9ED Suç tarihi aralığı:", dmin.date() if pd.notna(dmin) else None, "→", dmax.date() if pd.notna(dmax) else None)
    print("\U0001F9EE Toplam satır:", len(df_all))
except Exception:
    pass

# Çıkış hedefini seç: cache modda sadece Y dosyasına yaz
# Her koşulda event-level cache üret (run çıktısı)
event_out = Path(y_csv_path)  # default sf_crime_y.csv
safe_save(df_all.drop(columns=["date_only"], errors="ignore"), str(event_out))
print(f"💾 Event-level cache yazıldı → {event_out}")

try:
    Path("crime_prediction_data").mkdir(exist_ok=True)

    # Artifact / çıktı klasörüne her zaman koy (workflow upload-artifact için)
    shutil.copy2(event_out, "crime_prediction_data/sf_crime_y.csv")

    # İstersen statik sf_crime.csv'yi de güncelle (default KAPALI)
    if WRITE_BASE_TO_REPO:
        shutil.copy2(event_out, "crime_prediction_data/sf_crime.csv")
        shutil.copy2(event_out, "sf_crime.csv")
        print("📝 WRITE_BASE_TO_REPO=1 → sf_crime.csv güncellendi (repo workspace).")
    else:
        print("ℹ️ WRITE_BASE_TO_REPO=0 → repo sf_crime.csv EZİLMEDİ (sadece sf_crime_y artifact çıktı).")

except Exception as e:
    print("Kopya uyarısı:", e)

# --------------------------
# Takvim grid + Y_label
# --------------------------
group_cols = ["GEOID","season","day_of_week","event_hour"]
agg_dict = {"latitude":"mean", "longitude":"mean", "is_holiday":"mean", "id":"count"}

df_all_valid = df_all.dropna(subset=["GEOID"]).copy()
try:
    cats_api = fetch_all_categories_from_api()
    cats_local = []
    if "category" in df_all_valid.columns:
        cats_local = (
            df_all_valid["category"]
            .dropna().astype(str).str.strip().str.title()
            .unique().tolist()
        )
    all_cats = sorted(set(cats_api) | set(cats_local))
    catmap = {}
    if CATMAP_PATH.exists():
        with open(CATMAP_PATH, "r", encoding="utf-8") as f:
            catmap = json.load(f)
    next_id = (max(catmap.values()) + 1) if catmap else 1
    for c in all_cats:
        if c and c not in catmap:
            catmap[c] = next_id
            next_id += 1
    CATMAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CATMAP_PATH, "w", encoding="utf-8") as f:
        json.dump(catmap, f, ensure_ascii=False, indent=2)
except Exception as _e:
    print(f"⚠️ CATEGORY_MAP yazılamadı: {_e}")

grouped = (
    df_all_valid.groupby(group_cols, dropna=False)
    .agg(agg_dict).reset_index().rename(columns={"id":"crime_count"})
)
log_shape(df_all_valid, "CRIME geçerli (GEOID ≠ NaN)")
log_shape(grouped, "GROUPED (crime_count)")

if "category" in df_all_valid.columns:
    tmp = df_all_valid.dropna(subset=["category"]).copy()
    tmp["category"]    = tmp["category"].astype(str).str.strip().str.title()
    tmp["hr_key"]      = (tmp["event_hour"] // 3) * 3
    tmp["day_of_week"] = pd.to_numeric(tmp["day_of_week"], errors="coerce").astype("Int64")
    tmp["event_hour"]  = pd.to_numeric(tmp["event_hour"],  errors="coerce").astype("Int64")
    mix_full = _mix(tmp, ["GEOID","season","day_of_week","event_hour"])
    grouped = grouped.merge(mix_full, on=["GEOID","season","day_of_week","event_hour"], how="left")
    grouped["crime_mix"] = grouped["crime_mix"].fillna("").astype(str)
else:
    grouped["crime_mix"] = ""

# Grid evreni
geoid_lens = df_all_valid["GEOID"].dropna().astype(str).str.len()
trg_len = int(geoid_lens.mode().iat[0]) if not geoid_lens.empty else DEFAULT_GEOID_LEN
geoids = normalize_geoid(df_all_valid["GEOID"].dropna(), trg_len).unique()
seasons = ["Winter","Spring","Summer","Fall"]
days    = list(range(7))
hours   = list(range(24))
full_grid = pd.DataFrame(itertools.product(geoids, seasons, days, hours), columns=group_cols)
log_shape(full_grid, "FULL GRID evreni")

# Merge grid + özet
_before_grid_merge = full_grid.shape
df_final = full_grid.merge(grouped, on=group_cols, how="left")
log_delta(_before_grid_merge, df_final.shape, "FULL GRID ⨯ GROUPED")
log_shape(df_final, "GRID (merge sonrası)")

# hour_range & sezon bayrakları
df_final["hr_key"]     = (df_final["event_hour"] // 3) * 3
df_final["hour_range"] = df_final["hr_key"].map(lambda h: f"{h:02d}-{min(h+3,24):02d}")

df_final["sf_wet_season"] = df_final["season"].map({"Winter":1,"Spring":1,"Summer":0,"Fall":0}).fillna(0).astype(int)
df_final["sf_dry_season"] = df_final["season"].map({"Winter":0,"Spring":0,"Summer":1,"Fall":1}).fillna(0).astype(int)
df_final["sf_fog_season"] = df_final["season"].map({"Winter":0,"Spring":0,"Summer":1,"Fall":0}).fillna(0).astype(int)

# --------------------------
# Son 3g/7g + Komşu GEOID toplamları
# --------------------------
d_series = pd.to_datetime(df_all_valid["date"], errors="coerce").dt.normalize()
ref_max  = d_series.max()

def recent_counts(df, ref_date, days):
    if pd.isna(ref_date):
        return pd.DataFrame({"GEOID":[], f"crime_last_{days}d":[]})
    start = ref_date - pd.Timedelta(days=days-1)
    m = (pd.to_datetime(df["date"], errors="coerce").dt.normalize().between(start, ref_date))
    tmp = (df.loc[m].groupby("GEOID").size().reset_index(name=f"crime_last_{days}d"))
    return tmp

rc3 = recent_counts(df_all_valid, ref_max, 3)
rc7 = recent_counts(df_all_valid, ref_max, 7)

neighbors = None
if gdf_blocks is not None:
    try:
        tracts = gdf_blocks[["GEOID","geometry"]].copy()
        tracts = tracts.drop_duplicates("GEOID").reset_index(drop=True)
        tracts = tracts.to_crs("EPSG:26910")
        if NEIGHBOR_METHOD == "touches":
            left  = tracts.rename(columns={"GEOID":"G1"})
            right = tracts.rename(columns={"GEOID":"G2"})
            sj = gpd.sjoin(left, right, how="left", predicate="intersects")
            sj = sj[sj["G1"] != sj["G2"]].copy()
            neighbors = sj[["G1","G2"]].drop_duplicates()
        else:
            tracts["centroid"] = tracts.geometry.centroid
            buffers = gpd.GeoDataFrame(tracts[["GEOID"]], geometry=tracts["centroid"].buffer(NEIGHBOR_RADIUS_M), crs=tracts.crs)
            nb = gpd.sjoin(buffers.rename(columns={"GEOID":"G1"}), tracts.rename(columns={"GEOID":"G2"}), how="left", predicate="intersects")
            nb = nb[nb["G1"] != nb["G2"]]
            neighbors = nb[["G1","G2"]].drop_duplicates()
    except Exception as e:
        print(f"\u26A0\ufe0f Komşuluk hesaplanamadı: {e}")

def neighbor_sum(rc_df, name):
    if neighbors is None or rc_df is None or rc_df.empty:
        return pd.DataFrame({"GEOID":[], name:[]})
    m = neighbors.merge(rc_df.rename(columns={"GEOID":"G2"}), on="G2", how="left")
    out = m.groupby("G1")[rc_df.columns[-1]].sum(min_count=1).reset_index()
    out = out.rename(columns={"G1":"GEOID", rc_df.columns[-1]: name})
    out[name] = out[name].fillna(0).astype(int)
    return out

nb3 = neighbor_sum(rc3, "neigh_crime_last_3d")
nb7 = neighbor_sum(rc7, "neigh_crime_last_7d")

for feat in [rc3, rc7, nb3, nb7]:
    if feat is not None and not feat.empty:
        df_final = df_final.merge(feat, on="GEOID", how="left")

for col in ["crime_last_3d","crime_last_7d","neigh_crime_last_3d","neigh_crime_last_7d"]:
    if col in df_final.columns:
        df_final[col] = pd.to_numeric(df_final[col], errors="coerce").fillna(0).astype(int)

# --------------------------
# Centroid ile koordinat doldurma
# --------------------------
if gdf_blocks is not None:
    try:
        tracts = gdf_blocks[["GEOID","geometry"]].drop_duplicates("GEOID")
        tr_utm  = tracts.to_crs("EPSG:26910")
        cent    = tr_utm.geometry.centroid
        cent_wgs = gpd.GeoSeries(cent, crs=tr_utm.crs).to_crs("EPSG:4326")
        centroids = gpd.GeoDataFrame(tracts[["GEOID"]].copy(), geometry=cent_wgs, crs="EPSG:4326") \
                        .assign(lon_center=lambda d: d.geometry.x, lat_center=lambda d: d.geometry.y) \
                        [["GEOID","lat_center","lon_center"]]
        _before_cent = df_final.shape
        df_final = df_final.merge(centroids, on="GEOID", how="left")
        log_delta(_before_cent, df_final.shape, "GRID ⨯ GEOID centroid")
        for c in ("latitude","longitude"):
            if c not in df_final.columns: df_final[c] = np.nan
        df_final["latitude"]  = pd.to_numeric(df_final["latitude"], errors="coerce").fillna(df_final["lat_center"])
        df_final["longitude"] = pd.to_numeric(df_final["longitude"], errors="coerce").fillna(df_final["lon_center"])
        df_final = df_final.drop(columns=["lat_center","lon_center"], errors="ignore")
    except Exception as e:
        print(f"\u26A0\ufe0f GEOID centroid fallback başarısız: {e}")
else:
    df_final["latitude"]  = pd.to_numeric(df_final.get("latitude", np.nan), errors="coerce").fillna(0)
    df_final["longitude"] = pd.to_numeric(df_final.get("longitude", np.nan), errors="coerce").fillna(0)

# diğer bayraklar
df_final["is_weekend"]       = (df_final["day_of_week"] >= 5).astype(int)
df_final["is_night"]         = ((df_final["event_hour"] >= 22) | (df_final["event_hour"] <= 5)).astype(int)
df_final["is_school_hour"]   = df_final["event_hour"].between(7, 16).astype(int)
df_final["is_business_hour"] = (df_final["event_hour"].between(9, 17) & (df_final["day_of_week"] < 5)).astype(int)

log_date_range(df_all, "date", "Suç (grid girdi teyit)")
log_shape(df_final, "GRID (kayıt öncesi)")

# Kaydet
safe_save(grouped, sum_path)
safe_save(df_final, full_path)
print(f"\U0001F4BE Kaydedildi: {full_path}")
try:
    Path("crime_prediction_data").mkdir(exist_ok=True)
    shutil.copy2(full_path, "crime_prediction_data/sf_crime_grid_full_labeled.csv")
    print("\U0001F4E6 crime_prediction_data/ klasörüne GRID kopyalandı.")
except Exception as e:
    print(f"\u26A0\ufe0f GRID kopyalama uyarısı: {e}")

# Dosyadan da doğrula
try:
    _df_preview = pd.read_csv(full_path)
    print("sf_crime_grid_full_labeled.csv — ilk 5 satır (disk)")
    cols_show = [c for c in ["GEOID","season","day_of_week","event_hour","crime_count","crime_mix"] if c in _df_preview.columns]
    print(_df_preview.head(5)[cols_show].to_string(index=False))
except Exception as e:
    print("GRID ön izleme okunamadı:", e)

print(f"🔢 Toplam satır (GRID): {len(df_final):,}")
print(f"🧮 crime_count>0 satır: {(df_final['crime_count']>0).sum():,}")

# --------------------------
# Günlük grid (opsiyonel)
# --------------------------
if WRITE_DAILY_ARCHIVE:
    d_all = pd.to_datetime(df_all_valid["date"], errors="coerce").dt.normalize()
    end_ts   = d_all.max()
    start_ts = max(d_all.min(), end_ts - pd.Timedelta(days=DAILY_WINDOW_DAYS-1)) if pd.notna(end_ts) else None
    if start_ts is not None:
        def month_iter(a: pd.Timestamp, b: pd.Timestamp):
            y, m = int(a.year), int(a.month)
            ye, me = int(b.year), int(b.month)
            while (y < ye) or (y == ye and m <= me):
                yield y, m
                y, m = (y+1, 1) if m == 12 else (y, m+1)

        trg_len = int(df_all_valid["GEOID"].dropna().astype(str).str.len().mode().iat[0]) if not df_all_valid["GEOID"].dropna().empty else DEFAULT_GEOID_LEN
        geoids = normalize_geoid(df_all_valid["GEOID"].dropna(), trg_len).unique()

        centroids_df = None
        if gdf_blocks is not None:
            try:
                tracts = gdf_blocks[["GEOID","geometry"]].drop_duplicates("GEOID")
                tr_utm = tracts.to_crs("EPSG:26910")
                cent = tr_utm.geometry.centroid
                cent_wgs = gpd.GeoSeries(cent, crs=tr_utm.crs).to_crs("EPSG:4326")
                centroids_df = gpd.GeoDataFrame(tracts[["GEOID"]].copy(), geometry=cent_wgs, crs="EPSG:4326").assign(
                    lon_center=lambda d: d.geometry.x, lat_center=lambda d: d.geometry.y
                )[["GEOID","lat_center","lon_center"]]
            except Exception:
                centroids_df = None

        def daily_labels_chunk(df_all_valid: pd.DataFrame, geoids: np.ndarray, chunk_start: pd.Timestamp, chunk_end: pd.Timestamp) -> pd.DataFrame:
            msk = (pd.to_datetime(df_all_valid["date"]).dt.normalize().between(chunk_start.normalize(), chunk_end.normalize()))
            obs = (
                df_all_valid.loc[msk]
                .groupby(["date","GEOID","event_hour"], dropna=False)
                .size().rename("crime_count").reset_index()
            )
            days = pd.date_range(chunk_start.normalize(), chunk_end.normalize(), freq="D")
            grid = pd.MultiIndex.from_product([days, geoids, range(24)], names=["date","GEOID","event_hour"]).to_frame(index=False)
            out = grid.merge(obs, on=["date","GEOID","event_hour"], how="left")
            out["crime_count"] = pd.to_numeric(out["crime_count"], errors="coerce").fillna(0).astype("int32")
            out["Y_label"] = (out["crime_count"] >= DAILY_Y_THRESHOLD).astype("int8")
            out["day_of_week"] = pd.to_datetime(out["date"]).dt.weekday.astype("int8")
            out["month"]       = pd.to_datetime(out["date"]).dt.month.astype("int8")
            season_map_local = {12:"Winter",1:"Winter",2:"Winter",3:"Spring",4:"Spring",5:"Spring",6:"Summer",7:"Summer",8:"Summer",9:"Fall",10:"Fall",11:"Fall"}
            out["season"] = out["month"].map(season_map_local).astype("category")
            hr_key = (out["event_hour"] // 3) * 3
            out["hr_key"]     = hr_key.astype("int8")
            out["hour_range"] = out["hr_key"].map(lambda h: f"{h:02d}-{min(h+3,24):02d}").astype("category")
            out["is_weekend"]       = (out["day_of_week"] >= 5).astype("int8")
            out["is_night"]         = ((out["event_hour"] >= 22) | (out["event_hour"] <= 5)).astype("int8")
            out["is_school_hour"]   = out["event_hour"].between(8, 15).astype("int8")
            out["is_business_hour"] = (out["event_hour"].between(9, 17) & (out["day_of_week"] < 5)).astype("int8")
            out["range_start"] = chunk_start.normalize().date(); out["range_end"] = chunk_end.normalize().date()
            if centroids_df is not None:
                out = out.merge(centroids_df, on="GEOID", how="left")
                for c in ("latitude","longitude"):
                    if c not in out.columns: out[c] = np.nan
                out["latitude"]  = pd.to_numeric(out.get("latitude"),  errors="coerce").fillna(out["lat_center"])
                out["longitude"] = pd.to_numeric(out.get("longitude"), errors="coerce").fillna(out["lon_center"])
                out = out.drop(columns=["lat_center","lon_center"], errors="ignore")
            return out

        if USE_PARQUET:
            try:
                import pyarrow  # noqa
                import pyarrow.parquet as pq  # noqa
            except Exception:
                print("ℹ️ pyarrow yok → CSV append kullanılacak.")
                USE_PARQUET = False

        written_rows = 0
        for yy, mm in month_iter(start_ts, end_ts):
            first_day = pd.Timestamp(year=yy, month=mm, day=1)
            last_day  = first_day + pd.offsets.MonthEnd(0)
            chunk_start = max(first_day, start_ts)
            chunk_end   = min(last_day, end_ts)
            if chunk_start > chunk_end: continue
            print(f"\U0001F9E9 Aylık parça: {chunk_start.date()} → {chunk_end.date()}")
            part = daily_labels_chunk(df_all_valid, geoids, chunk_start, chunk_end)
            if USE_PARQUET:
                ym = f"{yy:04d}-{mm:02d}"
                base = Path(DAILY_OUT_BASE)
                out_dir = base.parent / base.stem
                out_dir.mkdir(parents=True, exist_ok=True)
                file_path = out_dir / f"year_month={ym}" / "part.parquet"
                if file_path.exists() and not OVERWRITE_DAILY:
                    print(f"  ↪️ atlandı (mevcut): {file_path}")
                else:
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    part.to_parquet(file_path, index=False)
                    print(f"    + {len(part):,} satır yazıldı → {file_path}")
                    written_rows += len(part)
            else:
                base = Path(DAILY_OUT_BASE)
                csv_dir = base.parent / base.stem
                csv_dir.mkdir(parents=True, exist_ok=True)
                ym = f"{yy:04d}-{mm:02d}"
                file_path = csv_dir / f"daily_{ym}.csv"
                mode = "w" if (OVERWRITE_DAILY or not file_path.exists()) else "a"
                header = (mode == "w")
                part.to_csv(file_path, index=False, mode=mode, header=header)
                print(f"    + {len(part):,} satır yazıldı → {file_path}")
                written_rows += len(part)
        print(f"✅ Günlük arşiv üretimi tamam. Toplam yazılan satır: {written_rows:,}")
        try:
            Path("crime_prediction_data").mkdir(exist_ok=True)
            base = Path(DAILY_OUT_BASE)
            src_dir = base.parent / base.stem
            if src_dir.exists():
                dst_dir = Path("crime_prediction_data/daily_parquet" if (USE_PARQUET and DAILY_PARTITION) else "crime_prediction_data/daily_csv")
                shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
                print("\U0001F4E6 crime_prediction_data/ klasörüne günlük arşiv kopyalandı.")
        except Exception as e:
            print(f"\u26A0\ufe0f Günlük arşiv kopyalama uyarısı: {e}")

# Blok dosyasını kopyala
try:
    Path("crime_prediction_data").mkdir(exist_ok=True)
    src_blocks = Path(blocks_path)
    if src_blocks.exists(): shutil.copy2(src_blocks, Path("crime_prediction_data") / src_blocks.name)
    print("\U0001F4E6 crime_prediction_data/ klasörüne gerekli kopyalar bırakıldı.")
except Exception as e:
    print(f"\u26A0\ufe0f crime_prediction_data kopyalama uyarısı: {e}")

# Son kontrol
try:
    _df_chk = pd.read_csv(full_path)
    print(_df_chk.shape)
    if "crime_count" in _df_chk.columns:
        print("crime_count>0 oranı (%):", round((_df_chk["crime_count"] > 0).mean() * 100, 2))
except Exception as e:
    print("\u26A0\ufe0f Grid tekrar okuma kontrolü başarısız:", e)

print("\n✅ Tüm işlem tamamlandı. Dosyalar güncellendi.")
print(df_final["crime_count"].isna().sum(), "— crime_count NaN sayısı")
print(df_final["crime_mix"].isna().sum(), "— crime_mix NaN sayısı")
