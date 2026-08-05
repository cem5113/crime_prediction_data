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
TMP_BASE_EVENT = RUN_TMP_DIR / "sf_crime_x.csv"
TMP_BASE_CSV = RUN_TMP_DIR / "sf_crime_x.csv"
TMP_BASE_GZ = RUN_TMP_DIR / "sf_crime_x.csv.gz"

# Kaynak URL/Token
# ➜ İstediğin akış: 1) Artifact'tan sf_crime_y.csv, 2) releases/latest sf_crime.csv
CRIME_BASE_URL = os.getenv(
    "CRIME_CSV_URL",
    "https://github.com/cem5113/crime_prediction_data/releases/latest/download/sf_crime.csv"
)
CRIME_API_URL = os.getenv("CRIME_API_URL", "https://data.sfgov.org/api/v3/views/wg3w-h783/query.json")
SFCRIME_APP_TOKEN = os.getenv("SFCRIME_API_TOKEN", "")

CHUNK_LIMIT         = int(os.getenv("SFCRIME_CHUNK_LIMIT", "50000"))
MAX_RETRIES         = int(os.getenv("SFCRIME_MAX_RETRIES", "4"))
SLEEP_BETWEEN_REQS  = float(os.getenv("SFCRIME_SLEEP", "0.2"))
BULK_RANGE          = os.getenv("SFCRIME_BULK_RANGE", "1").lower() in ("1","true","yes","on")
CRIME_REINGEST_DAYS = int(os.getenv("SFCRIME_REINGEST_DAYS", "14"))  

# Yol/çıktılar
save_dir   = "."
blocks_path = os.path.join(save_dir, "sf_census_blocks.geojson")

# ---- OUTPUTS ----
EVENT_CSV_NAME = os.getenv("EVENT_CSV_NAME", "sf_crime_x.csv")
PANEL_CSV_NAME = os.getenv("PANEL_CSV_NAME", "sf_crime_y.csv")

event_csv_path = os.path.join(save_dir, EVENT_CSV_NAME)
panel_csv_path = os.path.join(save_dir, PANEL_CSV_NAME)

# ---- GitHub Actions artifact (sf_crime_y.csv) ayarları ----
GITHUB_REPO = os.getenv("GITHUB_REPO", "cem5113/crime_prediction_data")   # owner/repo
GH_TOKEN = os.getenv("GH_TOKEN", "")
ARTIFACT_NAME = os.getenv("ARTIFACT_NAME", "sf-crime-pipeline-output")

# Helpers
def _safe_zfill_geoid(x, width=DEFAULT_GEOID_LEN):
    try:
        s = str(x)
        s = re.sub(r"\.0$", "", s)   
        s = re.sub(r"\D", "", s)    
        return s.zfill(width)      
    except Exception:
        return np.nan
        
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
        df.to_csv(path, index=False, encoding="utf-8-sig")
    except Exception as e:
        print(f"❌ Kaydedilemedi: {path}\n{e}")
        backup_path = path + ".bak"
        df.to_csv(backup_path, index=False, encoding="utf-8-sig")
        print(f"📁 Yedek dosya: {backup_path}")

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

def _is_valid_local_base(p: Path, min_bytes: int = 5_000) -> bool:
    if not p.exists():
        return False
    if p.stat().st_size < min_bytes:
        return False
    if p.suffix == ".csv" and is_lfs_pointer(p):
        return False
    return p.suffix in [".parquet", ".csv", ".gz"]

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
      1) Artifact içinden EVENT base: sf_crime_x.csv (tercih) / sf_crime.csv (event)
      2) Release latest: sf_crime.csv (event)
      3) Local fallback: önce event-level, en son panel-level
    """
    # 1) Artifact → (event-level tercih)
    if PREFER_REMOTE_BASE and GH_TOKEN:
        blob = fetch_file_from_latest_artifact(
        pick_names=[
            "sf_crime_x.parquet",
            EVENT_CSV_NAME,
            "sf_crime_x.csv",
            "sf_crime.csv",
        ],
            artifact_name=ARTIFACT_NAME,
        )
        is_parquet = blob[:4] == b"PAR1"
        
        if blob and (is_parquet or _is_valid_csv_bytes(blob)):
            # indirilen dosyayı event mi panel mi ayırmadan TMP'ye yazıyoruz
            # (okuyunca schema-guard karar verecek)
            artifact_suffix = ".parquet" if blob[:4] == b"PAR1" else ".csv"
            tmp_base = RUN_TMP_DIR / f"sf_crime_x{artifact_suffix}"
            
            _write_bytes_atomic(tmp_base, blob)
            print(f"📦 Base (artifact) indirildi → {tmp_base}")
            return tmp_base
        else:
            print("⚠️ Artifact base bulunamadı/uygun değil (boş/küçük/LFS).")

    # 2) Release latest → TMP_BASE_CSV veya TMP_BASE_GZ (aynı)
    if PREFER_REMOTE_BASE and CRIME_BASE_URL:
        dst = TMP_BASE_GZ if CRIME_BASE_URL.endswith(".gz") else TMP_BASE_CSV
        ok = download_url_to_file(CRIME_BASE_URL, dst)
        if ok:
            print(f"⬇️ Base (release latest) indirildi → {dst}")
            return dst
        else:
            print(f"⚠️ Release latest base indirilemedi/uygun değil: {CRIME_BASE_URL}")

    # 3) Local fallback (repo içi) — ✅ event-level önce, panel-level en son
    local_candidates = [
        Path("crime_prediction_data/sf_crime_x.parquet"),
        Path("sf_crime_x.parquet"),
    
        # geçiş dönemi fallback
        Path("crime_prediction_data/sf_crime_x.csv"),
        Path("sf_crime_x.csv"),
        Path("crime_prediction_data/sf_crime.csv"),
        Path("sf_crime.csv"),
    ]
    for p in local_candidates:
        if _is_valid_local_base(p):
            print(f"📦 Base (local fallback) bulundu: {p}")
            return p

    print("❌ Base bulunamadı (artifact/release/local).")
    return None

def read_existing_crime_csv(p: Path) -> pd.DataFrame | None:
    if not p or not p.exists():
        return None
    try:
        if p.suffix == ".parquet":
            df = pd.read_parquet(p)
            if "GEOID" in df.columns:
                df["GEOID"] = df["GEOID"].astype(str)
        else:
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

# API çekme (gün/gün veya aralık)
headers = {"X-App-Token": SFCRIME_APP_TOKEN} if SFCRIME_APP_TOKEN else {}

def get_latest_available_date() -> datetime.date | None:
    """
    Socrata üzerinden dataset'te gerçekten var olan en son tarihi bulur.
    1–2 gün gecikme / tatil boşluğu gibi durumlarda anchor'ı otomatik doğru kurar.
    """
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

# ✅ 1–2 gün gecikmeyi otomatik çöz: API’dan gerçek latest_available al
api_latest = get_latest_available_date()
if api_latest:
    latest_available = api_latest
    print(f"🛰️ API latest available date: {latest_available}")
else:
    latest_available = today - timedelta(days=int(os.getenv("PUBLISH_LAG_FALLBACK_DAYS", "2")))
    print(f"⚠️ API latest alınamadı → fallback latest_available: {latest_available}")

# ✅ sadece son indirilen tarihten sonrası indirilsin

if latest_date >= latest_available:
    start_missing = None
    missing_dates = []
    print(f"ℹ️ Base zaten güncel görünüyor: latest_date={latest_date} ≥ latest_available={latest_available}")
else:
    # son günü tekrar indir (güvenli overlap)
    start_missing = latest_date

    # overlap istemiyorsan:
    # start_missing = latest_date + timedelta(days=1)

    date_range = pd.date_range(
        start=start_missing,
        end=latest_available
    )
    missing_dates = [d.date() for d in date_range]

print(
    f"📆 Eksik tarihler: {len(missing_dates)} "
    f"(start={start_missing}, end={latest_available})"
)

if not missing_dates:
    print("ℹ️ Eksik gün yok; artımlı indirme atlanacak.")

print(f"📆 Eksik tarihler: {len(missing_dates)} (end={latest_available})")

if not missing_dates:
    print("ℹ️ Eksik gün yok; artımlı indirme atlanacak.")

# ============================================================
# ✅ BLOK: Census blocks geojson yükle (GEOID eşlemesi için)
#   - df_new GEOID join'inden ÖNCE tanımlı olmalı
# ============================================================
gdf_blocks = None
if os.path.exists(blocks_path):
    try:
        gdf_blocks = gpd.read_file(blocks_path)
        # GEOID normalize
        if "GEOID" in gdf_blocks.columns:
            gdf_blocks["GEOID"] = (
                gdf_blocks["GEOID"].astype(str)
                .str.extract(r"(\d+)")[0]
                .str[:DEFAULT_GEOID_LEN]
            )
        else:
            print(f"⚠️ blocks dosyasında 'GEOID' kolonu yok: {blocks_path}")
            gdf_blocks = None

        # CRS garanti (join öncesi)
        if gdf_blocks is not None:
            gdf_blocks = (gdf_blocks.set_crs("EPSG:4326") if gdf_blocks.crs is None else gdf_blocks.to_crs("EPSG:4326"))
            log_shape(gdf_blocks, "BLOCKS geojson")
    except Exception as e:
        print(f"⚠️ Blok dosyası okunamadı ({blocks_path}): {e}. GEOID eşlemesi atlanacak.")
        gdf_blocks = None
else:
    print(f"ℹ️ {blocks_path} bulunamadı; GEOID eşlemesi atlanacak.")

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
    dt = pd.to_datetime(df_new["incident_datetime"], errors="coerce")
    try:
        if getattr(dt.dt, "tz", None) is None:
            dt = dt.dt.tz_localize(SF_TZ, nonexistent="shift_forward", ambiguous="NaT")
        else:
            dt = dt.dt.tz_convert(SF_TZ)
    except Exception:
        # edge-case: dt .dt erişemiyorsa
        dt = pd.to_datetime(df_new["incident_datetime"], errors="coerce")
    
    df_new["datetime"]   = dt
    df_new["date"]       = df_new["datetime"].dt.date
    df_new["time"]       = df_new["datetime"].dt.strftime("%H:%M:%S")
    df_new["event_hour"] = df_new["datetime"].dt.hour
    print("🧪 df_new date range:", df_new["date"].min(), "→", df_new["date"].max(), "| n_days:", df_new["date"].nunique())

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
    df_new = pd.DataFrame(columns=df_old.columns)

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
if "id" not in df_all.columns:
    df_all["id"] = np.nan
df_all["id"] = df_all["id"].astype(str)
if "GEOID" in df_all.columns:
    df_all["GEOID"] = df_all["GEOID"].apply(_safe_zfill_geoid)

# ============================================================
# ✅ DUPLICATE GUARD (reingest overlap güvenliği)
# ============================================================

if "id" in df_all.columns:
    before = len(df_all)
    df_all = df_all.drop_duplicates(["id"], keep="last")
    print(f"🧹 Duplicate (id) temizlendi: {before - len(df_all)} satır")
else:
    key_cols = [c for c in ["datetime","latitude","longitude"] if c in df_all.columns]
    if len(key_cols) == 3:
        before = len(df_all)
        df_all = df_all.drop_duplicates(key_cols, keep="last")
        print(f"🧹 Duplicate (datetime+lat+lon) temizlendi: {before - len(df_all)} satır")
        
# ============================================================
# ✅ NEW: category/subcategory NaN handling (Unknown + flags)
# ============================================================
if {"category", "subcategory"}.issubset(df_all.columns):
    # "nan"/""/"None" gibi stringleri gerçek NaN'e çevir
    for col in ["category", "subcategory"]:
        df_all[col] = df_all[col].astype(str).replace({"nan": np.nan, "None": np.nan, "": np.nan})

    # eksik bayrakları (istersen modele de girebilir)
    df_all["is_category_missing"] = df_all["category"].isna().astype(int)
    df_all["is_subcategory_missing"] = df_all["subcategory"].isna().astype(int)

    # ikisi birden boşsa Unknown
    both_nan = df_all["category"].isna() & df_all["subcategory"].isna()
    df_all.loc[both_nan, ["category", "subcategory"]] = "Unknown"

    # sadece subcategory boşsa Unknown (opsiyonel ama iyi)
    only_sub_nan = df_all["subcategory"].isna() & df_all["category"].notna()
    df_all.loc[only_sub_nan, "subcategory"] = "Unknown"

df_all["date"] = pd.to_datetime(df_all["date"], errors="coerce").dt.date
start_date_5y = (pd.Timestamp(today) - pd.DateOffset(years=5)).date()
df_all = df_all[df_all["date"].notna() & (df_all["date"] >= start_date_5y)]
print("🧾 5y cutoff (takvim yılı):", start_date_5y)
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

print(f"🧾 EVENT output hedefi: {event_csv_path}")
print("\n🧩 [QC] sf_crime_x (df_all) güncel özet")
print(f"🧮 Shape: {df_all.shape[0]} satır × {df_all.shape[1]} sütun")

nan_counts = df_all.isna().sum().sort_values(ascending=False)
print("\n🕳️ [QC] Sütun bazında NaN sayıları (azalan):")
for col, cnt in nan_counts.items():
    print(f"  - {col}: {int(cnt)}")

# Rastgele 5 satır (tekrarlanabilir olsun diye random_state verdim)
print("\n🎲 [QC] Rastgele 5 satır örneği (df_all):")
with pd.option_context("display.max_columns", 200, "display.width", 200):
    print(df_all.sample(n=min(5, len(df_all)), random_state=42))

def _safe_zfill_geoid(x, width=DEFAULT_GEOID_LEN):
    try:
        s = str(x)
        s = re.sub(r"\.0$", "", s)
        s = re.sub(r"\D", "", s)
        return s.zfill(width)
    except Exception:
        return np.nan

# --- GEOID standard (yeni kolon) ---
if "GEOID" in df_all.columns:
    df_all["GEOID_std"] = df_all["GEOID"].apply(_safe_zfill_geoid)

# --- CATEGORY standard (yeni kolon) ---
if "category" in df_all.columns:
    df_all["category_std"] = (
        df_all["category"].astype(str)
          .str.strip()
          .str.replace(r"[?]+$", "", regex=True)
          .str.replace(r"\s+", " ", regex=True)
          .replace({"None": np.nan, "none": np.nan, "nan": np.nan, "NaN": np.nan, "": np.nan})
    )
else:
    df_all["category_std"] = np.nan

if "subcategory" in df_all.columns:
    df_all["subcategory_std"] = (
        df_all["subcategory"].astype(str)
          .replace({"nan": np.nan, "None": np.nan, "": np.nan})
    )
else:
    df_all["subcategory_std"] = np.nan

# A1 Unknown politikası (SADECE std kolonlarda)
both_nan = df_all["category_std"].isna() & df_all["subcategory_std"].isna()
df_all.loc[both_nan, ["category_std", "subcategory_std"]] = "Unknown"

only_sub_nan = df_all["subcategory_std"].isna() & df_all["category_std"].notna()
df_all.loc[only_sub_nan, "subcategory_std"] = "Unknown"

df_all["is_category_valid"] = df_all["category_std"].notna() & (df_all["category_std"] != "Unknown")

# --- ID boş kontrol (rapor) ---
if "id" in df_all.columns:
    bad_id = df_all["id"].isna() | (df_all["id"].astype(str).str.lower().isin(["nan", "none", ""]))
    print("🧪 [QC] invalid id rows:", int(bad_id.sum()))

    dup_id = int(df_all.duplicated(["id"]).sum())
    print("🧪 [QC] id duplicate:", dup_id)

# --- duplicate (datetime+lat+lon) rapor ---
key_cols = [c for c in ["datetime", "latitude", "longitude"] if c in df_all.columns]
if len(key_cols) == 3:
    dup_geo = int(df_all.duplicated(key_cols).sum())
    print("🧪 [QC] datetime+lat+lon duplicate:", dup_geo)

# --- date range rapor ---
if "date" in df_all.columns:
    d0 = pd.to_datetime(df_all["date"], errors="coerce").dt.date
    print(f"🧪 [QC] Date range: {d0.min()} → {d0.max()} | gün={d0.nunique()}")

# --- GEOID length rapor (std üzerinden) ---
if "GEOID_std" in df_all.columns:
    bad_geoid = df_all["GEOID_std"].astype(str).str.len().ne(DEFAULT_GEOID_LEN).sum()
    print(f"🧪 [QC] GEOID_std len != {DEFAULT_GEOID_LEN} rows:", int(bad_geoid))

# --- category dağılımı (std) ---
print("\n🧾 [QC] category_std top-10:")
print(df_all["category_std"].value_counts(dropna=False).head(10))

event_out = Path(event_csv_path)
event_df_out = df_all.drop(columns=["date_only"], errors="ignore")

safe_save(event_df_out, str(event_out))
print(f"💾 Event-level cache yazıldı → {event_out}")

event_parquet_out = Path("sf_crime_x.parquet")
event_df_out.to_parquet(
    event_parquet_out,
    index=False,
    engine="pyarrow",
    compression="snappy"
)
print(f"💾 Event-level parquet yazıldı → {event_parquet_out}")

# ============================================================
# ✅ LAST-CRIME ANCHOR FEATURES
#   - last_crime_dt
#   - crime_count_last_1d_from_last_crime
#   - crime_count_last_3d_from_last_crime
#   - crime_count_last_7d_from_last_crime
#   Hesap mantığı:
#     Her panel satırı için slot_start_dt anına kadar bilinen
#     son suç zamanı (last_crime_dt) bulunur.
#     Sayaçlar bu anchor'dan geriye doğru hesaplanır.
# ============================================================
def add_last_crime_anchor_features(panel_df: pd.DataFrame, event_df: pd.DataFrame) -> pd.DataFrame:
    """
    Her panel satırı için:
      1) slot_start_dt anına kadar aynı GEOID içindeki en son suç zamanını bulur.
      2) Bu last_crime_dt anchor alınarak geriye dönük 1d / 3d / 7d olay sayılarını hesaplar.

    Not:
      - Gelecek olayları kullanmaz.
      - Aynı slot içindeki olaylar, eğer slot_start_dt'den sonra ise dikkate alınmaz.
      - Anchor bulunamazsa sayaçlar 0 kalır.
    """
    out = panel_df.copy()

    feature_cols = [
        "crime_count_last_1d_from_last_crime",
        "crime_count_last_3d_from_last_crime",
        "crime_count_last_7d_from_last_crime",
        "last_crime_dt",
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

    ev["GEOID"] = ev["GEOID"].astype(str).str.extract(r"(\d+)")[0].str[:DEFAULT_GEOID_LEN]
    ev["datetime"] = pd.to_datetime(ev["datetime"], errors="coerce")
    ev = ev.dropna(subset=["GEOID", "datetime"]).copy()

    # tz garanti
    try:
        if getattr(ev["datetime"].dt, "tz", None) is None:
            ev["datetime"] = ev["datetime"].dt.tz_localize(
                SF_TZ, nonexistent="shift_forward", ambiguous="NaT"
            )
        else:
            ev["datetime"] = ev["datetime"].dt.tz_convert(SF_TZ)
    except Exception:
        pass

    ev = ev.dropna(subset=["datetime"]).sort_values(["GEOID", "datetime"]).copy()

    # çıktı kolonları
    out["crime_count_last_1d_from_last_crime"] = 0
    out["crime_count_last_3d_from_last_crime"] = 0
    out["crime_count_last_7d_from_last_crime"] = 0
    out["last_crime_dt"] = pd.Series(
        pd.NaT,
        index=out.index,
        dtype=f"datetime64[ns, {SF_TZ_NAME}]"
    )

    out["_row_id_tmp"] = np.arange(len(out))
    out = out.sort_values(["GEOID", "slot_start_dt"]).copy()

    # event ns map
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

        # tz hizala
        try:
            if getattr(slot_vals.dt, "tz", None) is None:
                slot_vals = slot_vals.dt.tz_localize(
                    SF_TZ, nonexistent="shift_forward", ambiguous="NaT"
                )
            else:
                slot_vals = slot_vals.dt.tz_convert(SF_TZ)
        except Exception:
            pass

        slot_ns = slot_vals.astype("int64").to_numpy()
        ev_ns = ev_groups.get(str(geoid))

        if ev_ns is None or len(ev_ns) == 0:
            continue

        # her slot için slot_start_dt anına kadar olan son suç index'i
        # last event <= slot_start_dt için side="right" - 1
        pos = np.searchsorted(ev_ns, slot_ns, side="right") - 1

        valid = pos >= 0
        if not valid.any():
            continue

        last_ns = np.full(len(slot_ns), np.nan, dtype="float64")
        last_ns[valid] = ev_ns[pos[valid]]

        # datetime olarak yaz
        last_dt = pd.to_datetime(last_ns, errors="coerce", utc=True).tz_convert(SF_TZ)
        out.loc[idx, "last_crime_dt"] = pd.Series(last_dt, index=idx)

        # pencere sayımları: [last_crime_dt - win, last_crime_dt]
        for col, win_ns in windows.items():
            counts = np.zeros(len(slot_ns), dtype=np.int32)

            valid_pos = np.where(valid)[0]
            if len(valid_pos) > 0:
                anchors = ev_ns[pos[valid]]
                left = np.searchsorted(ev_ns, anchors - win_ns, side="left")
                right = np.searchsorted(ev_ns, anchors, side="right")
                counts[valid] = (right - left).astype(np.int32)

            out.loc[idx, col] = counts

    out = out.sort_values("_row_id_tmp").drop(columns=["_row_id_tmp"])
    return out
    
# ============================================================
# ✅ PANEL (GRID) ÜRET — sf_crime_y.csv
#   - unit: GEOID × date × hour_range (3-hour)
#   - Y_label: o slotta en az 1 event varsa 1, yoksa 0
# ============================================================

# 1) Event’ten slot anahtarları
panel_evt = df_all.copy()
panel_evt["date"] = pd.to_datetime(panel_evt["date"], errors="coerce").dt.date

eh = pd.to_numeric(panel_evt["event_hour"], errors="coerce").fillna(0).astype(int) % 24
start = (eh // 3) * 3
panel_evt["hour_range"] = start.map(lambda s: f"{int(s):02d}-{int(min(s+3,24)):02d}")

# slot başlangıç saati ve datetime (event-level anchor için)
panel_evt["slot_start_hour"] = start.astype(int)
panel_evt["slot_start_dt"] = (
    pd.to_datetime(panel_evt["date"], errors="coerce") +
    pd.to_timedelta(panel_evt["slot_start_hour"], unit="h")
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
    
# 2) Slot bazında y_count + y_event/Y_label
slot_y = (
    panel_evt.dropna(subset=["GEOID","date","hour_range"])
             .groupby(["GEOID","date","hour_range"], as_index=False, observed=True)
             .size()
             .rename(columns={"size": "y_count"})
)

slot_y["y_count"] = pd.to_numeric(slot_y["y_count"], errors="coerce").fillna(0).astype("int16")

# İstersen BLOK-0/BLOK-1 terminolojisiyle:
slot_y["y_event"] = (slot_y["y_count"] > 0).astype("int8")

# Geriye dönük uyumluluk (mevcut pipeline Y_label bekliyorsa):
slot_y["Y_label"] = slot_y["y_event"].astype("int8")

# 3) FULL GRID = tüm GEOID × tüm date × 8 hour_range

if gdf_blocks is not None and "GEOID" in gdf_blocks.columns:

    all_geoids = (
        gdf_blocks["GEOID"]
        .dropna()
        .astype(str)
        .str.extract(r"(\d+)")[0]
        .str[:DEFAULT_GEOID_LEN]
        .dropna()
        .unique()
    )

    all_geoids = sorted(all_geoids)

    print(f"🧭 GEOID evreni boundary dosyasından alındı: {len(all_geoids)} GEOID")

else:

    # fallback
    all_geoids = (
        panel_evt["GEOID"]
        .dropna()
        .astype(str)
        .str.extract(r"(\d+)")[0]
        .str[:DEFAULT_GEOID_LEN]
        .dropna()
        .unique()
    )

    all_geoids = sorted(all_geoids)

    print(f"⚠️ Boundary GEOID bulunamadı → event tabanlı GEOID kullanıldı: {len(all_geoids)}")

dmin = panel_evt["date"].min()
dmax = panel_evt["date"].max()
all_dates = pd.date_range(dmin, dmax, freq="D").date
hour_starts = list(range(0, 24, 3))
hour_ranges = [f"{h:02d}-{h+3:02d}" for h in hour_starts]

grid = pd.MultiIndex.from_product(
    [all_geoids, all_dates, hour_starts],
    names=["GEOID", "date", "slot_start_hour"]
).to_frame(index=False)

grid["hour_range"] = grid["slot_start_hour"].map(lambda h: f"{int(h):02d}-{int(h+3):02d}")
grid["slot_start_dt"] = (
    pd.to_datetime(grid["date"], errors="coerce") +
    pd.to_timedelta(grid["slot_start_hour"], unit="h")
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

# ✅ Veri yayın gecikmesi nedeniyle grid'i son yayınlanan slot başlangıcına kadar kes
latest_published_dt = pd.to_datetime(panel_evt["datetime"], errors="coerce").max()
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
    print(f"🕒 Grid son yayınlanan slot'a göre kesildi: {latest_anchor_slot}")

slot_y = slot_y[["GEOID", "date", "hour_range", "y_count", "y_event", "Y_label"]].copy()

panel = grid.merge(slot_y, on=["GEOID", "date", "hour_range"], how="left")
panel = add_last_crime_anchor_features(panel, df_all)

panel["y_count"] = panel["y_count"].fillna(0).astype("int16")
panel["y_event"] = panel["y_event"].fillna(0).astype("int8")
panel["Y_label"] = panel["Y_label"].fillna(0).astype("int8")

panel["day_of_week"] = pd.to_datetime(panel["date"]).dt.weekday.astype("int8")
panel["month"] = pd.to_datetime(panel["date"]).dt.month.astype("int8")
panel["is_weekend"] = (panel["day_of_week"] >= 5).astype("int8")

# ============================================================
# ✅ CRIME-ONLY TEMPORAL / SURGE FEATURES
#    Amaç:
#    - gereksiz kolon şişirmeden
#    - stacking için ek sinyal + biraz diversity üretmek
# ============================================================
panel = panel.sort_values(["GEOID", "slot_start_dt"]).copy()

# ------------------------------------------------------------
# 1) Son suçtan beri geçen 3 saatlik slot sayısı
# ------------------------------------------------------------
delta_hours = (
    (
        pd.to_datetime(panel["slot_start_dt"], errors="coerce") -
        pd.to_datetime(panel["last_crime_dt"], errors="coerce")
    ).dt.total_seconds() / 3600.0
)

panel["slots_since_last_crime"] = np.floor(delta_hours / 3.0)
panel.loc[panel["last_crime_dt"].isna(), "slots_since_last_crime"] = np.nan

panel["slots_since_last_crime"] = (
    panel["slots_since_last_crime"]
    .replace([np.inf, -np.inf], np.nan)
    .clip(lower=0)
    .astype("float32")
)

panel["slots_since_last_crime_range"] = pd.cut(
    panel["slots_since_last_crime"],
    bins=[-1, 0, 1, 3, 7, 15, np.inf],
    labels=["0", "1", "2_3", "4_7", "8_15", "16_plus"]
).astype("object")

panel["slots_since_last_crime_range"] = (
    panel["slots_since_last_crime_range"]
    .fillna("no_prior")
)

# ------------------------------------------------------------
# 2) Aynı GEOID için kısa dönem lag sayıları
# ------------------------------------------------------------
panel["crime_count_prev_slot"] = (
    panel.groupby("GEOID", observed=True)["y_count"]
         .shift(1)
         .fillna(0)
         .astype("float32")
)

panel["crime_count_prev_2slots"] = (
    panel.groupby("GEOID", observed=True)["y_count"]
         .rolling(2, min_periods=1)
         .sum()
         .shift(1)
         .reset_index(level=0, drop=True)
         .fillna(0)
         .astype("float32")
)

panel["crime_count_prev_8slots"] = (
    panel.groupby("GEOID", observed=True)["y_count"]
         .rolling(8, min_periods=1)
         .sum()
         .shift(1)
         .reset_index(level=0, drop=True)
         .fillna(0)
         .astype("float32")
)

# ------------------------------------------------------------
# 3) Mevcut anchor feature'larından surge / trend
# ------------------------------------------------------------
c1 = panel["crime_count_last_1d_from_last_crime"].astype("float32")
c3 = panel["crime_count_last_3d_from_last_crime"].astype("float32")
c7 = panel["crime_count_last_7d_from_last_crime"].astype("float32")

panel["surge_ratio_1d_7d"] = ((c1 + 1.0) / ((c7 / 7.0) + 1.0)).astype("float32")
panel["surge_ratio_3d_7d"] = (((c3 / 3.0) + 1.0) / ((c7 / 7.0) + 1.0)).astype("float32")
panel["surge_diff_1d_7d"]  = (c1 - (c7 / 7.0)).astype("float32")

# 🔹 YENİ: trend direction
panel["trend_1d_vs_3d"] = (c1 - (c3 / 3.0)).astype("float32")
panel["trend_3d_vs_7d"] = ((c3 / 3.0) - (c7 / 7.0)).astype("float32")

# ------------------------------------------------------------
# 4) Aynı GEOID + aynı slot için tarihsel ortalama
# ------------------------------------------------------------
panel["slot_roll_mean_7d"] = (
    panel.groupby(["GEOID", "hour_range"], observed=True)["y_count"]
         .rolling(7, min_periods=1)
         .mean()
         .shift(1)
         .reset_index(level=[0, 1], drop=True)
         .fillna(0)
         .astype("float32")
)

panel["slot_roll_mean_28d"] = (
    panel.groupby(["GEOID", "hour_range"], observed=True)["y_count"]
         .rolling(28, min_periods=1)
         .mean()
         .shift(1)
         .reset_index(level=[0, 1], drop=True)
         .fillna(0)
         .astype("float32")
)

# ------------------------------------------------------------
# 5) Aynı hafta günü + aynı slot pattern'i
# ------------------------------------------------------------
panel["same_dow_slot_rate_8w"] = (
    panel.groupby(["GEOID", "day_of_week", "hour_range"], observed=True)["y_event"]
         .rolling(8, min_periods=1)
         .mean()
         .shift(1)
         .reset_index(level=[0, 1, 2], drop=True)
         .fillna(0)
         .astype("float32")
)

# ------------------------------------------------------------
# 6) Log dönüşümlü varyantlar
# ------------------------------------------------------------
panel["log_prev_8slots"] = np.log1p(panel["crime_count_prev_8slots"]).astype("float32")
panel["log_last_7d_from_last_crime"] = np.log1p(c7).astype("float32")

# ------------------------------------------------------------
# 7) YENİ: relative risk (local vs global)
#    Amaç: sadece lokal seviye değil, gün içindeki göreli risk de görülsün
# ------------------------------------------------------------
daily_global_mean = (
    panel.groupby("date", observed=True)["y_count"]
         .transform("mean")
         .astype("float32")
)

daily_global_event_rate = (
    panel.groupby("date", observed=True)["y_event"]
         .transform("mean")
         .astype("float32")
)

panel["global_daily_mean_ycount"] = daily_global_mean
panel["global_daily_event_rate"] = daily_global_event_rate

panel["relative_risk_7d_vs_global"] = (
    c7 / (daily_global_mean + 1.0)
).astype("float32")

panel["relative_event_rate_vs_global"] = (
    panel["same_dow_slot_rate_8w"].astype("float32") / (daily_global_event_rate + 1e-6)
).astype("float32")

# ------------------------------------------------------------
# 8) YENİ: volatility / stability
#    Amaç: stabil riskli bölge mi, spike yapan bölge mi?
# ------------------------------------------------------------
geo_roll_mean_7 = (
    panel.groupby("GEOID", observed=True)["y_count"]
         .rolling(7, min_periods=2)
         .mean()
         .shift(1)
         .reset_index(level=0, drop=True)
)

geo_roll_std_7 = (
    panel.groupby("GEOID", observed=True)["y_count"]
         .rolling(7, min_periods=2)
         .std()
         .shift(1)
         .reset_index(level=0, drop=True)
)

panel["geo_roll_mean_7"] = geo_roll_mean_7.fillna(0).astype("float32")
panel["geo_roll_std_7"]  = geo_roll_std_7.fillna(0).astype("float32")

panel["volatility_7d"] = (
    panel["geo_roll_std_7"] / (panel["geo_roll_mean_7"] + 1.0)
).astype("float32")

# İstersen kategoriye çevrilebilir ama şimdilik sayısal kalsın
panel["stability_score_7d"] = (
    1.0 / (1.0 + panel["volatility_7d"])
).astype("float32")

season_map = {12:"Winter",1:"Winter",2:"Winter",3:"Spring",4:"Spring",5:"Spring",6:"Summer",7:"Summer",8:"Summer",9:"Fall",10:"Fall",11:"Fall"}
panel["season"] = panel["month"].map(season_map).astype("category")

# 4) Yaz
panel = panel.drop(columns=["slot_start_hour"], errors="ignore")
print(f"🧾 PANEL output hedefi: {panel_csv_path}")
safe_save(panel, panel_csv_path)
print(f"💾 Panel (sf_crime_y) yazıldı → {panel_csv_path} | rows={len(panel):,}")

try:
    _tmp = pd.read_csv(panel_csv_path, dtype={"GEOID": str}, low_memory=False)
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

    # Artifact çıktıları (ikisi de)
    shutil.copy2(event_parquet_out, "crime_prediction_data/sf_crime_x.parquet")
    
    # CSV'leri istersen debug için bırakabilirsin, ama artifact'e koymayacağız.
    shutil.copy2(event_out, f"crime_prediction_data/{Path(event_csv_path).name}")
    shutil.copy2(panel_csv_path, f"crime_prediction_data/{Path(panel_csv_path).name}")
    
    print("✅ artifact outputs: sf_crime_x.parquet + csv debug outputs")

    # İstersen statik sf_crime.csv'yi de güncelle (default KAPALI) — DİKKAT:
    # Eğer bunu açarsan, HANGİSİNİ base sayacağını seçmelisin.
    # Ben güvenli tarafta kalıp paneli/base olarak ezmeyi önermiyorum.
    if WRITE_BASE_TO_REPO:
        shutil.copy2(event_out, "crime_prediction_data/sf_crime_x.csv")
        shutil.copy2(event_out, "sf_crime_x.csv")
    
        print("📝 WRITE_BASE_TO_REPO=1 → sf_crime_x.csv event-level base olarak güncellendi.")
    else:
        print("ℹ️ WRITE_BASE_TO_REPO=0 → repo base dosyası güncellenmedi; sadece artifact çıktıları yazıldı.")

except Exception as e:
    print("Kopya uyarısı:", e)

print("\n✅ Tüm işlem tamamlandı. (event-level cache)")

print(df_all["GEOID"].head(10))
print(df_all["GEOID"].str.len().value_counts())
