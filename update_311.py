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

RAW_311_NAME_Y = os.getenv("RAW_311_NAME_Y", "sf_311_last_5_years_y.csv")   # event-level
AGG_BASENAME   = os.getenv("AGG_311_NAME", "sf_311_last_5_years.csv")        # 3h aggregate
AGG_ALIAS      = os.getenv("AGG_311_ALIAS", "sf_311_last_5_years_3h.csv")

LEGACY_311_Y = os.getenv("LEGACY_311_Y", "sf_311_last_5_year_y.csv")
LEGACY_311   = os.getenv("LEGACY_311",   "sf_311_last_5_year.csv")

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
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
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
def resolve_existing_raw_path():
    y_names = [RAW_311_NAME_Y, LEGACY_311_Y]
    fallback_names = [AGG_BASENAME, AGG_ALIAS, LEGACY_311]
    roots = [Path(SAVE_DIR), Path.cwd(), Path(".")]

    def _ok(p: Path) -> bool:
        if not p.exists() or p.is_dir():
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

    # 1) Önce event-level _y ara
    for nm in y_names:
        for rt in roots:
            for cand in [rt / nm, rt / "crime_prediction_data" / nm, rt / "outputs" / nm]:
                if _ok(cand):
                    print(f"🔎 Mevcut 311 _y CSV bulundu: {cand.resolve()}")
                    return str(cand), "raw_y"

    # 2) _y yoksa aggregate fallback kullan
    for nm in fallback_names:
        for rt in roots:
            for cand in [rt / nm, rt / "crime_prediction_data" / nm, rt / "outputs" / nm]:
                if _ok(cand):
                    print(f"🔎 311 _y bulunamadı; fallback aggregate CSV kullanılacak: {cand.resolve()}")
                    return str(cand), "agg_fallback"

    preferred = Path(SAVE_DIR) / RAW_311_NAME_Y
    print(f"ℹ️ Mevcut 311 ham CSV yok; oluşturulacak: {preferred.resolve()}")
    return str(preferred), "new_raw"

def load_existing_raw(path, path_kind="raw_y"):
    if not os.path.exists(path):
        return pd.DataFrame()

    df = pd.read_csv(path, dtype={"GEOID": str}, low_memory=False)

    if path_kind == "agg_fallback":
        # aggregate dosyayı raw gibi değil, sadece date bilgisini taşıyan fallback kaynak gibi ele al
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        else:
            df["date"] = pd.NaT
        df["datetime"] = pd.NaT

        for c in [
            "id","lat","long","category","subcategory","service_details",
            "agency_responsible","status_description","source",
            "latitude","longitude","GEOID","time"
        ]:
            if c not in df.columns:
                df[c] = pd.NA

        mx_date = pd.to_datetime(df["date"], errors="coerce").max()
        print(f"📁 Fallback aggregate satır: {len(df):,} | max date={mx_date}")
        return df

    if "index_right" in df.columns:
        df = df.drop(columns=["index_right"])

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    elif "date" in df.columns and "time" in df.columns:
        df["datetime"] = pd.to_datetime(
            df["date"].astype(str) + " " + df["time"].astype(str),
            errors="coerce",
            utc=True
        )
    else:
        df["datetime"] = pd.NaT

    if "date" not in df.columns:
        dt_local = df["datetime"].dt.tz_convert(SF_TZ) if SF_TZ is not None else df["datetime"]
        df["date"] = pd.to_datetime(dt_local, errors="coerce").dt.date

    for c in [
        "id","lat","long","category","subcategory","service_details",
        "agency_responsible","status_description","source",
        "latitude","longitude","GEOID","time"
    ]:
        if c not in df.columns:
            df[c] = pd.NA

    mx = pd.to_datetime(df["datetime"], errors="coerce").max()
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

    # 1) Normal raw_y dosyası varsa datetime bazlı ilerle
    if source_kind == "raw_y" and "datetime" in df_existing.columns and df_existing["datetime"].notna().any():
        last_dt = pd.to_datetime(df_existing["datetime"], errors="coerce", utc=True).max()
        if pd.notna(last_dt):
            last_date = last_dt.date()
            start = last_date - timedelta(days=max(1, REINGEST_DAYS))
            if start < DEFAULT_START:
                start = DEFAULT_START
            print(f"📌 Mod: incremental(raw_y)+overlap | start={start} | last={last_date} | reingest={REINGEST_DAYS}d")
            return start, "incremental-raw_y"

    # 2) _y yoksa aggregate fallback date bazlı ilerle
    if "date" in df_existing.columns and df_existing["date"].notna().any():
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
    if df_raw.empty:
        return pd.DataFrame(columns=[
            "GEOID","date","hour_range",
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
        ])

    df_ok = df_raw.dropna(subset=["date", "GEOID"]).copy()
    if df_ok.empty:
        return pd.DataFrame(columns=[
            "GEOID","date","hour_range",
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
        ])

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
             .pivot_table(
                 index=grp_keys,
                 columns="bucket_311",
                 values="size",
                 fill_value=0
             )
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
    
    # aggregate yolu her durumda SAVE_DIR altında dursun
    agg_path = os.path.join(SAVE_DIR, AGG_BASENAME)
    
    df_raw = load_existing_raw(raw_path, source_kind)
    start_date, _mode = decide_start_date(df_raw, source_kind)
    
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
            "id", "datetime", "date",
            "latitude", "longitude",
            "category", "subcategory", "service_details",
            "GEOID"
        ]
        
        for c in keep:
            if c not in df_new_geo.columns:
                df_new_geo[c] = pd.NA

        df_new_geo = df_new_geo[keep].copy()
        df_new_geo["GEOID"] = normalize_geoid(df_new_geo["GEOID"], DEFAULT_GEOID_LEN)
        
        if source_kind == "agg_fallback":
            # fallback kaynak aggregate idi; raw event-level ile concat edilmez
            df_raw = df_new_geo.copy()
        elif df_raw.empty:
            df_raw = df_new_geo
        else:
            df_raw = pd.concat([df_raw, df_new_geo], ignore_index=True)

    # ---- Ham kaydet
    if not df_raw.empty:
        df_raw["GEOID"] = normalize_geoid(df_raw["GEOID"], DEFAULT_GEOID_LEN)
        df_raw["id"] = df_raw["id"].astype(str)
        df_raw = df_raw.drop_duplicates(subset=["id"], keep="last")

        df_raw["date"] = pd.to_datetime(df_raw["date"], errors="coerce").dt.date
        min_date = start_date if BACKFILL_DAYS > 0 else DEFAULT_START
        df_raw = df_raw[df_raw["date"] >= min_date]

        df_raw["datetime"] = pd.to_datetime(df_raw["datetime"], errors="coerce", utc=True)
        df_raw = df_raw.sort_values("datetime")

        save_atomic(df_raw, raw_path)
        save_parquet_atomic(df_raw, os.path.join(SAVE_DIR, "sf_311_last_5_years_y.parquet"))
        save_atomic(df_raw, os.path.join(SAVE_DIR, RAW_311_NAME_Y))
        save_atomic(df_raw, os.path.join(SAVE_DIR, LEGACY_311_Y))
        save_atomic(df_raw, os.path.join(SAVE_DIR, LEGACY_311))

        print(f"✅ Ham 311 kaydedildi: {os.path.abspath(raw_path)}")
        log_shape(df_raw, "311 raw")
    else:
        print("⚠️ Ham 311 boş.")
        empty_raw_cols = [
            "id","datetime","date","time","lat","long","latitude","longitude",
            "category","subcategory","service_details",
            "agency_responsible","status_description","source","GEOID"
        ]
        for p in [RAW_311_NAME_Y, LEGACY_311_Y, LEGACY_311]:
            save_atomic(pd.DataFrame(columns=empty_raw_cols), os.path.join(SAVE_DIR, p))

    # ---- Aggregate kaydet
    grouped = build_311_aggregate(df_raw)
    
    save_atomic(grouped, agg_path)
    save_parquet_atomic(grouped, os.path.join(SAVE_DIR, "sf_311_last_5_years.parquet"))
    save_atomic(grouped, os.path.join(SAVE_DIR, AGG_BASENAME))
    if AGG_ALIAS and AGG_ALIAS != AGG_BASENAME:
        save_atomic(grouped, os.path.join(SAVE_DIR, AGG_ALIAS))

    print(f"📁 311 özet yazıldı: {os.path.abspath(agg_path)}")
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
            raise ValueError("❌ sf_crime_01.csv içinde hour_range yok. 311 merge panel anahtarıyla yapılmalı.")

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
                save_atomic(crime, crime_02_path)
                print("✅ Passthrough yazıldı.")
        except Exception as ee:
            print(f"❌ Passthrough da başarısız: {ee}")

if __name__ == "__main__":
    main()
