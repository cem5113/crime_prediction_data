# update_crime.py
from __future__ import annotations
import os
import time
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

# ENV / Config
SF_TZ = zoneinfo.ZoneInfo("America/Los_Angeles")
SF_TZ_NAME = "America/Los_Angeles"

DEFAULT_GEOID_LEN  = int(os.getenv("GEOID_LEN", "11"))          
SF_BBOX            = (-123.2, 37.6, -122.3, 37.9)

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

# Kaynak URL/Token
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
blocks_path = os.path.join(save_dir, "sf_census_blocks.geojson")

# ---- CACHE/Y-only çıkış ----
Y_CSV_NAME = os.getenv("Y_CSV_NAME", "sf_crime_y.csv")
y_csv_path = os.path.join(save_dir, Y_CSV_NAME)

# ---- GitHub Actions artifact (sf_crime_y.csv) ayarları ----
GITHUB_REPO = os.getenv("GITHUB_REPO", "cem5113/crime_prediction_data")   # owner/repo
GH_TOKEN = os.getenv("GH_TOKEN", "")
ARTIFACT_NAME = os.getenv("ARTIFACT_NAME", "sf-crime-pipeline-output")

# Helpers
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

def safe_save(df: pd.DataFrame, path: str) -> None:
    try:
        Path(os.path.dirname(path) or ".").mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    except Exception as e:
        print(f"❌ Kaydedilemedi: {path}\n{e}")
        backup_path = path + ".bak"
        df.to_csv(backup_path, index=False)
        print(f"\U0001F4C1 Yedek dosya: {backup_path}")

def is_lfs_pointer(p: Path) -> bool:
    try:
        head = p.read_text(errors="ignore")[:200]
        return "git-lfs.github.com/spec/v1" in head
    except Exception:
        return False

# Artifact indirme yardımcıları
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

# Mevcut veri (artifact-first) — SFCRIME_Y → release sf_crime.csv
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
        print(f"⬇️ Release HTTP {r.status_code} → {url}")
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

# Başla: veri yükle & eksik günler

today = datetime.now(SF_TZ).date()

base_path = ensure_base_csv_remote_first()
if base_path is None:
    raise SystemExit(1)

print(f"📦 Base path seçildi: {base_path}")
df_old = read_existing_crime_csv(base_path)
if df_old is None:
    raise SystemExit(1)

if "date" not in df_old.columns:
    raise SystemExit("❌ Base veri içinde 'date' kolonu yok.")

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


# API çekme (gün/gün veya aralık)

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


# İndir & temizle & GEOID eşle
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
        # # lat/lon yoksa GEOID eşlemesi mümkün değil; GEOID boş kalabilir.
        if "GEOID" not in df_new.columns:
            df_new["GEOID"] = np.nan
else:
    df_new = pd.DataFrame()

log_shape(df_new, "CRIME yeni (indirilen)")
log_date_range(df_new, "date", "Suç (yeni)")

# Birleştir & zamanda özellikler
if "time" not in df_old.columns:
    df_old["time"] = "00:00:00"
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

df_all["date"] = pd.to_datetime(df_all["date"], errors="coerce").dt.date
start_date_5y = today - timedelta(days=5*365)
df_all = df_all[df_all["date"].notna() & (df_all["date"] >= start_date_5y)]
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

print(f"🧾 Y output hedefi: {y_csv_path}")
print("\n🧩 [QC] sf_crime_y (df_all) güncel özet")
print(f"🧮 Shape: {df_all.shape[0]} satır × {df_all.shape[1]} sütun")

nan_counts = df_all.isna().sum().sort_values(ascending=False)
print("\n🕳️ [QC] Sütun bazında NaN sayıları (azalan):")
for col, cnt in nan_counts.items():
    print(f"  - {col}: {int(cnt)}")

# Rastgele 5 satır (tekrarlanabilir olsun diye random_state verdim)
print("\n🎲 [QC] Rastgele 5 satır örneği (df_all):")
with pd.option_context("display.max_columns", 200, "display.width", 200):
    print(df_all.sample(n=min(5, len(df_all)), random_state=42))
    
event_out = Path(y_csv_path)  # default sf_crime_y.csv
safe_save(df_all.drop(columns=["date_only"], errors="ignore"), str(event_out))
print(f"💾 Event-level cache yazıldı → {event_out}")

try:
    _tmp = pd.read_csv(event_out, dtype={"GEOID": str}, low_memory=False)
    print("\n📄 [QC] Dosyadan okunan sf_crime_y.csv özeti")
    print(f"🧮 Shape(file): {_tmp.shape[0]} satır × {_tmp.shape[1]} sütun")

    nan_counts_file = _tmp.isna().sum().sort_values(ascending=False)
    print("\n🕳️ [QC] Dosya sütun bazında NaN sayıları (azalan):")
    for col, cnt in nan_counts_file.items():
        print(f"  - {col}: {int(cnt)}")

    print("\n🎲 [QC] Dosyadan rastgele 5 satır:")
    with pd.option_context("display.max_columns", 200, "display.width", 200):
        print(_tmp.sample(n=min(5, len(_tmp)), random_state=42))
except Exception as e:
    print("⚠️ [QC] Dosyadan okuma kontrolü başarısız:", e)
    
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

print("\n✅ Tüm işlem tamamlandı. (event-level cache)")
