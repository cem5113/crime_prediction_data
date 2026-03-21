# scripts/update_311.py
from __future__ import annotations

import os
import re
import time
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import geopandas as gpd

# ---- TZ ---------------------------------------------------------
try:
    import zoneinfo
    SF_TZ = zoneinfo.ZoneInfo("America/Los_Angeles")
except Exception:
    SF_TZ = None

# ================== AYARLAR ==================
SAVE_DIR = os.getenv("CRIME_DATA_DIR", "crime_prediction_data")
os.makedirs(SAVE_DIR, exist_ok=True)

DEFAULT_GEOID_LEN = int(os.getenv("GEOID_LEN", "11"))
GEOID_LEN = DEFAULT_GEOID_LEN  # backward compatibility

# -----------------------------------------------------------------
# DOSYA SÖZLEŞMESİ (REVIZE)
# -----------------------------------------------------------------
# 1) GitHub'da var olan ana raw kaynak:
BASE_RAW_311_NAME = os.getenv("BASE_RAW_311_NAME", "sf_311_last_5_years.csv")

# 2) Güncellenmiş/rolling raw çalışma dosyası:
RAW_311_NAME_Y = os.getenv("RAW_311_NAME_Y", "sf_311_last_5_years_y.csv")

# 3) 3 saatlik aggregate çıktı:
AGG_BASENAME = os.getenv("AGG_311_NAME", "sf_311_last_5_years_3h.csv")
AGG_ALIAS    = os.getenv("AGG_311_ALIAS", "sf_311_last_5_years_3h_alias.csv")

LEGACY_311_Y = os.getenv("LEGACY_311_Y", "sf_311_last_5_year_y.csv")
LEGACY_311   = os.getenv("LEGACY_311",   "sf_311_last_5_year.csv")

RAW_311_PARQUET = os.getenv("RAW_311_PARQUET", "sf_311_last_5_years_y.parquet")
AGG_311_PARQUET = os.getenv("AGG_311_PARQUET", "sf_311_last_5_years_3h.parquet")

DATASET_BASE = os.getenv("SF311_DATASET", "https://data.sfgov.org/resource/vw6y-z8j6.json")
SOCRATA_APP_TOKEN = os.getenv("SOCS_APP_TOKEN", "").strip()

PAGE_LIMIT   = int(os.getenv("SF_SODA_PAGE_LIMIT", "50000"))
SLEEP_SEC    = float(os.getenv("SF_SODA_THROTTLE_SEC", "0.25"))
SODA_TIMEOUT = int(os.getenv("SF_SODA_TIMEOUT", "90"))
SODA_RETRIES = int(os.getenv("SF_SODA_RETRIES", "5"))

CHUNK_DAYS              = int(os.getenv("SF311_CHUNK_DAYS", "31"))
MAX_PAGES_PER_CHUNK     = int(os.getenv("SF311_MAX_PAGES_PER_CHUNK", "40"))
MAX_CONSEC_EMPTY_CHUNKS = int(os.getenv("SF311_MAX_EMPTY_CHUNKS", "8"))

TODAY         = datetime.utcnow().date()
DEFAULT_START = TODAY - timedelta(days=5 * 365)
BACKFILL_DAYS = int(os.getenv("BACKFILL_DAYS", "0"))
REINGEST_DAYS = int(os.getenv("SF311_REINGEST_DAYS", "14"))

GEOJSON_NAME = os.getenv("SF_BLOCKS_GEOJSON", "sf_census_blocks.geojson")
GEOJSON_CANDIDATES = [
    os.path.join(SAVE_DIR, GEOJSON_NAME),
    os.path.join("crime_prediction_data", GEOJSON_NAME),
    os.path.join(".", GEOJSON_NAME),
]

# ================== YARDIMCILAR ==================
def log_shape(df, label):
    r, c = df.shape
    print(f"📊 {label}: {r} satır × {c} sütun")

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [
        str(c).replace("\ufeff", "").strip()
        for c in out.columns
    ]
    return out

def log_merge_delta(before_shape, after_shape, label):
    br, bc = before_shape
    ar, ac = after_shape
    print(f"🔗 {label}: {br}×{bc} → {ar}×{ac} (Δr={ar-br}, Δc={ac-bc})")

def normalize_geoid(series, target_len: int | None = None):
    L = int(target_len or DEFAULT_GEOID_LEN)
    s = series.astype("string")
    s = s.str.extract(r"(\d+)", expand=False)
    s = s.str.slice(0, L)
    return s.str.zfill(L)

def normalize_geoid_11(x):
    if pd.isna(x):
        return pd.NA
    digits = re.sub(r"\D", "", str(x))
    if digits == "":
        return pd.NA
    return digits[:11].zfill(11)

def save_atomic(df, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False, encoding="utf-8")
    os.replace(tmp, path)

def save_parquet_atomic(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp.parquet"

    df2 = df.copy()
    for c in df2.select_dtypes(include=["float64"]).columns:
        df2[c] = pd.to_numeric(df2[c], downcast="float")
    for c in df2.select_dtypes(include=["int64", "Int64"]).columns:
        df2[c] = pd.to_numeric(df2[c], downcast="integer")

    df2.to_parquet(tmp, index=False, engine="pyarrow", compression="snappy")
    os.replace(tmp, path)

def is_lfs_pointer_file(p: Path) -> bool:
    try:
        return "git-lfs.github.com/spec/v1" in p.read_text(errors="ignore")[:200]
    except Exception:
        return False

def make_hour_range_from_datetime(dt_series: pd.Series) -> pd.Series:
    h = pd.to_datetime(dt_series, errors="coerce").dt.hour.fillna(0).astype(int)
    start_h = (h // 3) * 3
    end_h = start_h + 3
    end_h = end_h.where(end_h < 24, 24)
    return (
        start_h.astype(int).astype(str).str.zfill(2)
        + "-"
        + end_h.astype(int).astype(str).str.zfill(2)
    )

def is_valid_311_aggregate_df(df: pd.DataFrame) -> bool:
    cols = {str(c).replace("\ufeff", "").strip() for c in df.columns}
    required = {"GEOID", "date", "hour_range", "311_request_count"}
    return required.issubset(cols)

def is_valid_311_raw_df(df: pd.DataFrame) -> bool:
    cols = {str(c).replace("\ufeff", "").strip() for c in df.columns}

    raw_like_strict = {"id", "datetime", "date", "GEOID"}
    raw_like_soft_1 = {"category", "subcategory", "service_details"}
    raw_like_soft_2 = {"latitude", "longitude"}

    # raw event-level için daha esnek doğrulama
    if raw_like_strict.issubset(cols):
        return True
    if {"datetime", "date"}.issubset(cols) and len(raw_like_soft_1.intersection(cols)) > 0:
        return True
    if {"datetime", "date"}.issubset(cols) and len(raw_like_soft_2.intersection(cols)) > 0:
        return True
    return False

def safe_read_csv(path: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False, **kwargs)

def safe_read_any_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = safe_read_csv(path, dtype={"GEOID": str})
    return normalize_columns(df)

# ================== 311 KATEGORİ HARİTASI ==================
def classify_311_bucket(service_name: str, service_subtype: str, service_details: str) -> str:
    txt = " ".join([
        "" if pd.isna(service_name) else str(service_name),
        "" if pd.isna(service_subtype) else str(service_subtype),
        "" if pd.isna(service_details) else str(service_details),
    ]).strip().lower()

    if any(k in txt for k in ["encamp", "homeless", "tent"]):
        return "encampment"
    if any(k in txt for k in ["graffiti", "tagging", "vandal"]):
        return "graffiti"
    if any(k in txt for k in ["abandoned vehicle", "vehicle"]):
        return "abandoned_vehicle"
    if any(k in txt for k in ["noise", "loud", "music", "party"]):
        return "noise"
    if any(k in txt for k in ["street cleaning", "sidewalk cleaning", "debris", "trash", "garbage", "waste"]):
        return "street_cleaning"
    if any(k in txt for k in ["parking", "blocked driveway", "double park", "traffic"]):
        return "parking_traffic"
    return "other"

def disorder_weight(bucket: str) -> float:
    w = {
        "encampment": 1.8,
        "graffiti": 1.4,
        "abandoned_vehicle": 1.3,
        "noise": 1.2,
        "street_cleaning": 0.8,
        "parking_traffic": 0.9,
        "other": 0.6,
    }
    return w.get(bucket, 0.6)

# ================== SOCRATA ==================
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
            print(f"⚠️ Socrata retry {i+1}/{SODA_RETRIES} ({e}); {sleep_s:.1f}s bekleme…")
            time.sleep(sleep_s)
    raise last_err

# ================== GEO ==================
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
            gdf = gdf[["TRACT11", "geometry"]].dropna(subset=["TRACT11"])
            if gdf.crs is None:
                gdf.set_crs("EPSG:4326", inplace=True)
            elif str(gdf.crs).lower() not in ("epsg:4326", "wgs84", "wgs 84"):
                gdf = gdf.to_crs(epsg=4326)
            print(f"🧭 GEOJSON kullanılıyor: {os.path.abspath(cand)}")
            return gdf
    print("⚠️ GEOJSON bulunamadı; GEOID eşleme yapılamayacak.")
    return None

def geotag_to_geoid11(df_new):
    df_new = df_new.copy()
    if "latitude" not in df_new.columns and "lat" in df_new.columns:
        df_new["latitude"] = pd.to_numeric(df_new["lat"], errors="coerce")
    if "longitude" not in df_new.columns and "long" in df_new.columns:
        df_new["longitude"] = pd.to_numeric(df_new["long"], errors="coerce")

    df_new = df_new.dropna(subset=["latitude", "longitude"])
    if df_new.empty:
        df_new["GEOID"] = pd.NA
        return df_new

    gdf_blocks = ensure_blocks_gdf()
    if gdf_blocks is None:
        df_new["GEOID"] = pd.NA
        return df_new

    gdf_pts = gpd.GeoDataFrame(
        df_new,
        geometry=gpd.points_from_xy(df_new["longitude"], df_new["latitude"]),
        crs="EPSG:4326",
    )
    try:
        gdf_join = gpd.sjoin(gdf_pts, gdf_blocks, how="left", predicate="within")
    except Exception:
        try:
            gdf_join = gpd.sjoin_nearest(gdf_pts, gdf_blocks, how="left", max_distance=5)
        except Exception:
            gdf_pts["GEOID"] = pd.NA
            return pd.DataFrame(gdf_pts.drop(columns=["geometry"]))
    out = pd.DataFrame(gdf_join.drop(columns=["geometry"]))
    out.rename(columns={"TRACT11": "GEOID"}, inplace=True)
    out["GEOID"] = out["GEOID"].apply(normalize_geoid_11)
    return out

# ================== DOSYA YOLLARI ==================
def candidate_paths(names: list[str], roots: list[Path]) -> list[Path]:
    out = []
    for nm in names:
        for rt in roots:
            out.extend([
                rt / nm,
                rt / "crime_prediction_data" / nm,
                rt / "outputs" / nm,
            ])
    uniq = []
    seen = set()
    for p in out:
        k = str(p.resolve()) if p.exists() else str(p)
        if k not in seen:
            uniq.append(p)
            seen.add(k)
    return uniq

def resolve_existing_raw_path():
    """
    Yeni mantık:
    1) Önce repo'daki mevcut BASE raw dosyayı ara: sf_311_last_5_years.csv
    2) Sonra working raw: sf_311_last_5_years_y.csv / parquet
    3) En son gerçekten aggregate fallback bak
    """
    roots = [Path(SAVE_DIR), Path.cwd(), Path(".")]

    def _ok(p: Path) -> bool:
        if not p.exists() or p.is_dir():
            return False
        if p.suffix.lower() not in (".csv", ".parquet"):
            return False
        if p.suffix.lower() == ".csv" and is_lfs_pointer_file(p):
            return False
        try:
            if p.stat().st_size < 200:
                return False
        except Exception:
            return False
        return True

    # 1) Öncelik: GitHub'daki mevcut base raw kaynak
    base_raw_names = [
        "sf_311_last_5_years.parquet",
    ]
    for cand in candidate_paths(base_raw_names, roots):
        if _ok(cand):
            try:
                df_probe = safe_read_any_table(str(cand))
                if is_valid_311_raw_df(df_probe):
                    print(f"🔎 Mevcut BASE 311 raw bulundu: {cand.resolve()}")
                    return str(cand), "base_raw"
                else:
                    print(f"⚠️ BASE raw adayı bulundu ama raw yapıda değil, atlandı: {cand.resolve()}")
            except Exception as e:
                print(f"⚠️ BASE raw aday okunamadı, atlandı: {cand} | {e}")

    # 2) Working raw
    raw_names = [RAW_311_PARQUET, RAW_311_NAME_Y, LEGACY_311_Y]
    for cand in candidate_paths(raw_names, roots):
        if _ok(cand):
            try:
                df_probe = safe_read_any_table(str(cand))
                if is_valid_311_raw_df(df_probe):
                    print(f"🔎 Mevcut WORKING 311 raw bulundu: {cand.resolve()}")
                    return str(cand), "raw_y"
                else:
                    print(f"⚠️ Raw adayı bulundu ama raw yapıda değil, atlandı: {cand.resolve()}")
            except Exception as e:
                print(f"⚠️ Raw aday okunamadı, atlandı: {cand} | {e}")

    # 3) Aggregate fallback (son çare)
    agg_names = [AGG_311_PARQUET, AGG_BASENAME, AGG_ALIAS, LEGACY_311]
    for cand in candidate_paths(agg_names, roots):
        if _ok(cand):
            try:
                df_probe = safe_read_any_table(str(cand))
                if is_valid_311_aggregate_df(df_probe):
                    print(f"🔎 311 raw yok; doğrulanmış aggregate fallback kullanılacak: {cand.resolve()}")
                    return str(cand), "agg_fallback"
                else:
                    print(f"⚠️ Aggregate adayı bulundu ama aggregate yapıda değil, atlandı: {cand.resolve()}")
            except Exception as e:
                print(f"⚠️ Aggregate aday okunamadı, atlandı: {cand} | {e}")

    preferred = Path(SAVE_DIR) / RAW_311_NAME_Y
    print(f"ℹ️ Mevcut 311 ham dosyası yok; yeni oluşturulacak: {preferred.resolve()}")
    return str(preferred), "new_raw"

def load_existing_raw(path, path_kind="raw_y"):
    if not os.path.exists(path):
        return pd.DataFrame()

    df = safe_read_any_table(path)

    if path_kind == "agg_fallback":
        if not is_valid_311_aggregate_df(df):
            raise ValueError(f"❌ agg_fallback olarak seçilen dosya aggregate değil: {path}")

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        df["datetime"] = pd.NaT

        for c in [
            "id", "lat", "long", "category", "subcategory", "service_details",
            "agency_responsible", "status_description", "source",
            "latitude", "longitude", "GEOID", "time"
        ]:
            if c not in df.columns:
                df[c] = pd.NA

        mx_date = pd.to_datetime(df["date"], errors="coerce").max()
        print(f"📁 Doğrulanmış aggregate fallback satır: {len(df):,} | max date={mx_date}")
        return df

    # base_raw / raw_y ortak işlenir
    if not is_valid_311_raw_df(df):
        raise ValueError(f"❌ raw olarak seçilen dosya raw yapıda değil: {path}")

    if "index_right" in df.columns:
        df = df.drop(columns=["index_right"])

    # datetime
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    elif "requested_datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["requested_datetime"], errors="coerce", utc=True)
    elif "date" in df.columns and "time" in df.columns:
        df["datetime"] = pd.to_datetime(
            df["date"].astype(str) + " " + df["time"].astype(str),
            errors="coerce",
            utc=True,
        )
    else:
        df["datetime"] = pd.NaT

    # category/subcategory isim düzelt
    rename_map = {}
    if "service_name" in df.columns and "category" not in df.columns:
        rename_map["service_name"] = "category"
    if "service_subtype" in df.columns and "subcategory" not in df.columns:
        rename_map["service_subtype"] = "subcategory"
    if rename_map:
        df = df.rename(columns=rename_map)

    # lat/long -> latitude/longitude
    if "latitude" not in df.columns and "lat" in df.columns:
        df["latitude"] = pd.to_numeric(df["lat"], errors="coerce")
    if "longitude" not in df.columns and "long" in df.columns:
        df["longitude"] = pd.to_numeric(df["long"], errors="coerce")

    if "date" not in df.columns:
        dt_local = df["datetime"].dt.tz_convert(SF_TZ) if SF_TZ is not None else df["datetime"]
        df["date"] = pd.to_datetime(dt_local, errors="coerce").dt.date
    else:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date

    if "time" not in df.columns:
        dt_local = df["datetime"].dt.tz_convert(SF_TZ) if SF_TZ is not None else df["datetime"]
        df["time"] = dt_local.dt.strftime("%H:%M:%S")

    for c in [
        "id", "lat", "long", "category", "subcategory", "service_details",
        "agency_responsible", "status_description", "source",
        "latitude", "longitude", "GEOID", "time"
    ]:
        if c not in df.columns:
            df[c] = pd.NA

    # id yoksa üret
    if "id" not in df.columns or df["id"].isna().all():
        df["id"] = (
            df["datetime"].astype(str).fillna("")
            + "_"
            + df["latitude"].astype(str).fillna("")
            + "_"
            + df["longitude"].astype(str).fillna("")
            + "_"
            + df["category"].astype(str).fillna("")
        )

    mx = pd.to_datetime(df["datetime"], errors="coerce", utc=True).max()
    print(f"📁 Mevcut raw satır: {len(df):,} | max datetime={mx}")
    return df

def decide_start_date(df_existing, source_kind="raw_y"):
    if BACKFILL_DAYS > 0:
        start = TODAY - timedelta(days=BACKFILL_DAYS)
        print(f"📌 Mod: backfill | start={start}")
        return start, "backfill"

    if df_existing.empty:
        print(f"📌 Mod: full-5y | window ≥ {DEFAULT_START}")
        return DEFAULT_START, "full-5y"

    if source_kind in {"raw_y", "base_raw"} and "datetime" in df_existing.columns and df_existing["datetime"].notna().any():
        last_dt = pd.to_datetime(df_existing["datetime"], errors="coerce", utc=True).max()
        if pd.notna(last_dt):
            last_date = last_dt.date()
            start = last_date - timedelta(days=max(1, REINGEST_DAYS))
            if start < DEFAULT_START:
                start = DEFAULT_START
            print(f"📌 Mod: incremental({source_kind})+overlap | start={start} | last={last_date} | reingest={REINGEST_DAYS}d")
            return start, f"incremental-{source_kind}"

    if source_kind == "agg_fallback" and "date" in df_existing.columns and df_existing["date"].notna().any():
        last_date = pd.to_datetime(df_existing["date"], errors="coerce").dt.date.max()
        if pd.notna(last_date):
            start = last_date - timedelta(days=max(1, REINGEST_DAYS))
            if start < DEFAULT_START:
                start = DEFAULT_START
            print(f"📌 Mod: incremental(agg_fallback)+overlap | start={start} | last={last_date} | reingest={REINGEST_DAYS}d")
            return start, "incremental-agg_fallback"

    print(f"📌 Mod: full-5y | window ≥ {DEFAULT_START}")
    return DEFAULT_START, "full-5y"

# ================== İNDİRME ==================
def download_by_date_chunks(start_date):
    print(f"🧩 İndirme modu: DATE-CHUNKS ({CHUNK_DAYS}gün) + paging")
    session = requests.Session()

    cols = ",".join([
        "service_request_id",
        "requested_datetime",
        "closed_date",
        "updated_datetime",
        "status_description",
        "agency_responsible",
        "service_name",
        "service_subtype",
        "service_details",
        "lat",
        "long",
        "point",
        "source",
    ])

    all_chunks = []
    consec_empty = 0
    cur = start_date
    end = TODAY

    while cur <= end:
        chunk_end = min(cur + timedelta(days=CHUNK_DAYS - 1), end)
        start_iso = f"{cur.isoformat()}T00:00:00.000"
        end_iso   = f"{chunk_end.isoformat()}T23:59:59.999"
        print(f"⛏️  {cur} → {chunk_end} aralığı çekiliyor…")

        offset = 0
        pages = 0
        chunk_rows = []

        while True:
            params = {
                "$select": cols,
                "$where": f"requested_datetime between '{start_iso}' and '{end_iso}'",
                "$order": "requested_datetime ASC",
                "$limit": PAGE_LIMIT,
                "$offset": offset,
            }

            try:
                data = socrata_get(session, DATASET_BASE, params)
            except Exception as e:
                print(f"❌ Chunk hata ({cur}→{chunk_end}, offset={offset}): {e} → chunk geçiliyor.")
                break

            df = pd.DataFrame(data)
            if df.empty:
                break

            if pages == 0:
                print("   • kolonlar:", list(df.columns))

            chunk_rows.append(df)
            offset += len(df)
            pages += 1
            print(f"   + {offset} kayıt (sayfa={pages})")

            if len(df) < PAGE_LIMIT or pages >= MAX_PAGES_PER_CHUNK:
                if pages >= MAX_PAGES_PER_CHUNK:
                    print(f"   ↪️ MAX_PAGES_PER_CHUNK={MAX_PAGES_PER_CHUNK} doldu, chunk kesildi.")
                break

            time.sleep(SLEEP_SEC)

        if chunk_rows:
            consec_empty = 0
            all_chunks.append(pd.concat(chunk_rows, ignore_index=True))
            print(f"✅ Chunk bitti: satır={sum(len(x) for x in chunk_rows)}")
        else:
            consec_empty += 1
            print(f"ℹ️ Chunk boş döndü (ardışık boş={consec_empty}).")
            if consec_empty >= MAX_CONSEC_EMPTY_CHUNKS and cur > start_date:
                print("⏹️ Çok sayıda ardışık boş chunk; erken durdurma.")
                break

        cur = chunk_end + timedelta(days=1)
        time.sleep(SLEEP_SEC)

    return pd.concat(all_chunks, ignore_index=True) if all_chunks else pd.DataFrame()

# ================== FEATURE ENGINEERING ==================
def build_311_aggregate(df_raw: pd.DataFrame) -> pd.DataFrame:
    empty_cols = [
        "GEOID", "date", "hour_range",
        "311_request_count",
        "311_noise_count",
        "311_encampment_count",
        "311_graffiti_count",
        "311_abandoned_vehicle_count",
        "311_street_cleaning_count",
        "311_parking_traffic_count",
        "311_other_count",
        "311_disorder_count",
        "311_disorder_score",
        "311_noise_ratio",
        "311_disorder_ratio",
    ]

    if df_raw.empty:
        return pd.DataFrame(columns=empty_cols)

    df_ok = df_raw.dropna(subset=["date", "GEOID"]).copy()
    if df_ok.empty:
        return pd.DataFrame(columns=empty_cols)

    df_ok["hour_range"] = make_hour_range_from_datetime(df_ok["datetime"])

    df_ok["bucket_311"] = df_ok.apply(
        lambda r: classify_311_bucket(
            r.get("category", pd.NA),
            r.get("subcategory", pd.NA),
            r.get("service_details", pd.NA)
        ),
        axis=1
    )
    df_ok["bucket_weight"] = df_ok["bucket_311"].apply(disorder_weight)

    grp_keys = ["GEOID", "date", "hour_range"]

    total = (
        df_ok.groupby(grp_keys, as_index=False)
        .size()
        .rename(columns={"size": "311_request_count"})
    )

    pivot = (
        df_ok.groupby(grp_keys + ["bucket_311"], as_index=False)
        .size()
        .pivot_table(index=grp_keys, columns="bucket_311", values="size", fill_value=0)
        .reset_index()
    )
    pivot.columns.name = None
    pivot = pivot.rename(columns={
        "noise": "311_noise_count",
        "encampment": "311_encampment_count",
        "graffiti": "311_graffiti_count",
        "abandoned_vehicle": "311_abandoned_vehicle_count",
        "street_cleaning": "311_street_cleaning_count",
        "parking_traffic": "311_parking_traffic_count",
        "other": "311_other_count",
    })

    score = (
        df_ok.groupby(grp_keys, as_index=False)["bucket_weight"]
        .sum()
        .rename(columns={"bucket_weight": "311_disorder_score"})
    )

    out = total.merge(pivot, on=grp_keys, how="left")
    out = out.merge(score, on=grp_keys, how="left")

    count_cols = [
        "311_noise_count",
        "311_encampment_count",
        "311_graffiti_count",
        "311_abandoned_vehicle_count",
        "311_street_cleaning_count",
        "311_parking_traffic_count",
        "311_other_count",
    ]
    for c in count_cols:
        if c not in out.columns:
            out[c] = 0

    for c in ["311_request_count", "311_disorder_score"] + count_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)

    out["311_disorder_count"] = (
        out["311_encampment_count"]
        + out["311_graffiti_count"]
        + out["311_abandoned_vehicle_count"]
        + out["311_noise_count"]
    )

    out["311_noise_ratio"] = out["311_noise_count"] / (out["311_request_count"] + 1e-6)
    out["311_disorder_ratio"] = out["311_disorder_count"] / (out["311_request_count"] + 1e-6)

    out["GEOID"] = normalize_geoid(out["GEOID"], DEFAULT_GEOID_LEN)
    return out

# ================== ANA ==================
def main():
    print("🔎 CWD:", os.getcwd())
    print("🔎 SAVE_DIR:", os.path.abspath(SAVE_DIR))

    raw_path, source_kind = resolve_existing_raw_path()

    base_raw_csv = os.path.join(SAVE_DIR, BASE_RAW_311_NAME)
    canonical_raw_csv = os.path.join(SAVE_DIR, RAW_311_NAME_Y)
    canonical_raw_parquet = os.path.join(SAVE_DIR, RAW_311_PARQUET)

    canonical_agg_csv = os.path.join(SAVE_DIR, AGG_BASENAME)
    canonical_agg_parquet = os.path.join(SAVE_DIR, AGG_311_PARQUET)

    df_existing = load_existing_raw(raw_path, source_kind)
    start_date, _mode = decide_start_date(df_existing, source_kind)

    df_new = download_by_date_chunks(start_date)
    if df_new.empty:
        print("ℹ️ Yeni 311 kaydı bulunamadı.")
    else:
        print(f"➕ Yeni indirilen: {len(df_new):,}")

        df_new = df_new.rename(columns={
            "service_request_id": "id",
            "requested_datetime": "datetime",
            "service_name": "category",
            "service_subtype": "subcategory",
        })

        df_new["datetime"] = pd.to_datetime(df_new["datetime"], errors="coerce", utc=True)

        if SF_TZ is not None:
            _dt_sf = df_new["datetime"].dt.tz_convert(SF_TZ)
        else:
            _dt_sf = df_new["datetime"]

        df_new["date"] = _dt_sf.dt.date
        df_new["time"] = _dt_sf.dt.strftime("%H:%M:%S")

        if "lat" in df_new.columns:
            df_new["latitude"] = pd.to_numeric(df_new["lat"], errors="coerce")
        if "long" in df_new.columns:
            df_new["longitude"] = pd.to_numeric(df_new["long"], errors="coerce")

        df_new_geo = geotag_to_geoid11(df_new)

        keep = [
            "id", "datetime", "date", "time",
            "latitude", "longitude",
            "category", "subcategory", "service_details",
            "agency_responsible", "status_description", "source",
            "GEOID"
        ]
        for c in keep:
            if c not in df_new_geo.columns:
                df_new_geo[c] = pd.NA

        df_new_geo = df_new_geo[keep].copy()
        df_new_geo["GEOID"] = normalize_geoid(df_new_geo["GEOID"], DEFAULT_GEOID_LEN)

        if source_kind == "agg_fallback":
            df_raw = df_new_geo.copy()
        elif df_existing.empty:
            df_raw = df_new_geo.copy()
        else:
            df_raw = pd.concat([df_existing, df_new_geo], ignore_index=True)

    if df_new.empty:
        if source_kind in {"raw_y", "base_raw"}:
            df_raw = df_existing.copy()
        else:
            df_raw = pd.DataFrame()

    # ---- Ham kaydet (rolling 5y)
    if not df_raw.empty:
        df_raw["GEOID"] = normalize_geoid(df_raw["GEOID"], DEFAULT_GEOID_LEN)
        df_raw["id"] = df_raw["id"].astype(str)

        # ID yoksa fallback üret
        bad_id_mask = df_raw["id"].isin(["<NA>", "nan", "None", ""])
        if bad_id_mask.any():
            df_raw.loc[bad_id_mask, "id"] = (
                df_raw.loc[bad_id_mask, "datetime"].astype(str).fillna("")
                + "_"
                + df_raw.loc[bad_id_mask, "latitude"].astype(str).fillna("")
                + "_"
                + df_raw.loc[bad_id_mask, "longitude"].astype(str).fillna("")
                + "_"
                + df_raw.loc[bad_id_mask, "category"].astype(str).fillna("")
            )

        df_raw = df_raw.drop_duplicates(subset=["id"], keep="last")

        df_raw["date"] = pd.to_datetime(df_raw["date"], errors="coerce").dt.date
        min_date = start_date if BACKFILL_DAYS > 0 else DEFAULT_START
        df_raw = df_raw[df_raw["date"] >= min_date]

        df_raw["datetime"] = pd.to_datetime(df_raw["datetime"], errors="coerce", utc=True)
        df_raw = df_raw.sort_values("datetime")

        # asıl rolling raw çalışma dosyası
        save_atomic(df_raw, canonical_raw_csv)
        save_parquet_atomic(df_raw, canonical_raw_parquet)
        save_atomic(df_raw, os.path.join(SAVE_DIR, LEGACY_311_Y))

        # istersen repo ana raw dosyasını da güncel tut
        save_atomic(df_raw, base_raw_csv)

        print(f"✅ Ham 311 kaydedildi: {os.path.abspath(canonical_raw_csv)}")
        print(f"✅ Base raw 311 güncellendi: {os.path.abspath(base_raw_csv)}")
        log_shape(df_raw, "311 raw")
    else:
        print("⚠️ Ham 311 boş.")
        empty_raw_cols = [
            "id", "datetime", "date", "time", "lat", "long", "latitude", "longitude",
            "category", "subcategory", "service_details",
            "agency_responsible", "status_description", "source", "GEOID"
        ]
        save_atomic(pd.DataFrame(columns=empty_raw_cols), canonical_raw_csv)
        save_atomic(pd.DataFrame(columns=empty_raw_cols), os.path.join(SAVE_DIR, LEGACY_311_Y))

    # ---- Aggregate üret ve kaydet
    if source_kind == "agg_fallback" and df_raw.empty:
        grouped = safe_read_any_table(raw_path).copy()
        if not is_valid_311_aggregate_df(grouped):
            raise ValueError(f"❌ Aggregate fallback doğrulaması son aşamada başarısız: {raw_path}")
        grouped["GEOID"] = normalize_geoid(grouped["GEOID"], DEFAULT_GEOID_LEN)
        grouped["date"] = pd.to_datetime(grouped["date"], errors="coerce").dt.date
        grouped["hour_range"] = grouped["hour_range"].astype(str)
        print("ℹ️ Yeni raw event yok; mevcut doğrulanmış aggregate fallback korunuyor.")
    else:
        grouped = build_311_aggregate(df_raw)

    save_atomic(grouped, canonical_agg_csv)
    save_parquet_atomic(grouped, canonical_agg_parquet)

    if AGG_ALIAS and AGG_ALIAS != AGG_BASENAME:
        save_atomic(grouped, os.path.join(SAVE_DIR, AGG_ALIAS))

    print(f"📁 311 özet yazıldı: {os.path.abspath(canonical_agg_csv)}")
    log_shape(grouped, "311 aggregate")

    # ---- Crime merge: sf_crime_01 -> sf_crime_02
    try:
        crime_01_path = os.path.join(SAVE_DIR, "sf_crime_01.parquet")
        crime_02_path = os.path.join(SAVE_DIR, "sf_crime_02.parquet")

        if not os.path.exists(crime_01_path):
            print(f"ℹ️ {crime_01_path} yok. 311 merge atlandı.")
            return

        crime = pd.read_parquet(crime_01_path)
        if "GEOID" in crime.columns:
            crime["GEOID"] = crime["GEOID"].astype(str)
        before = crime.shape

        if "hour_range" not in crime.columns:
            raise ValueError("❌ sf_crime_01.parquet içinde hour_range yok. 311 merge panel anahtarıyla yapılmalı.")

        crime["GEOID"] = normalize_geoid(crime["GEOID"], DEFAULT_GEOID_LEN)
        crime["date"] = pd.to_datetime(crime["date"], errors="coerce").dt.date

        grouped["GEOID"] = normalize_geoid(grouped["GEOID"], DEFAULT_GEOID_LEN)
        grouped["date"] = pd.to_datetime(grouped["date"], errors="coerce").dt.date
        grouped["hour_range"] = grouped["hour_range"].astype(str)

        keys = ["GEOID", "date", "hour_range"]
        merged = crime.merge(grouped, on=keys, how="left")

        feat_cols = [c for c in grouped.columns if c not in keys]
        for c in feat_cols:
            merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0)

        log_merge_delta(before, merged.shape, "crime ⨯ full311")
        log_shape(merged, "sf_crime_02")

        save_parquet_atomic(merged, crime_02_path)
        print("✅ Suç + full 311 birleştirmesi tamamlandı.")
    except Exception as e:
        print(f"⚠️ 311 merge hatası: {e}\n↪️ PASSTHROUGH uygulanıyor…")
        try:
            crime_01_path = os.path.join(SAVE_DIR, "sf_crime_01.parquet")
            crime_02_path = os.path.join(SAVE_DIR, "sf_crime_02.parquet")
            if os.path.exists(crime_01_path):
                crime = pd.read_parquet(crime_01_path)
                if "GEOID" in crime.columns:
                    crime["GEOID"] = crime["GEOID"].astype(str)
                fallback_cols = [
                    "311_request_count",
                    "311_noise_count",
                    "311_encampment_count",
                    "311_graffiti_count",
                    "311_abandoned_vehicle_count",
                    "311_street_cleaning_count",
                    "311_parking_traffic_count",
                    "311_other_count",
                    "311_disorder_count",
                    "311_disorder_score",
                    "311_noise_ratio",
                    "311_disorder_ratio",
                ]
                for c in fallback_cols:
                    crime[c] = 0
                save_parquet_atomic(crime, crime_02_path)
                print("✅ Passthrough yazıldı.")
        except Exception as ee:
            print(f"❌ Passthrough da başarısız: {ee}")

if __name__ == "__main__":
    main()
