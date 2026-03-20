# =========================================================
# update_sf_features_incremental.py
#
# AMAÇ
# - 4 veri setini cache/incremental mantıkla yönetmek
# - Her çalışmada hepsini baştan indirmemek
# - Son indirilen tarih / son max kayıt tarihine göre güncellemek
#
# ÇIKTILAR
#   sf_business_landuse.csv
#   sf_building_permits_vacancy.csv
#   sf_traffic_transport.csv
#   sf_street_environment.csv
#
# NOT
# - Socrata dataset id ve tarih kolonları ENV ile override edilebilir
# - GEOID eşleme için sf_census_blocks.geojson gerekir
# =========================================================

import os
import json
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import geopandas as gpd
import requests

# =========================================================
# küçük yardımcılar
# =========================================================
def log(msg: str):
    print(msg, flush=True)

def log_shape(df: pd.DataFrame, label: str):
    r, c = df.shape
    print(f"📊 {label}: {r} satır × {c} sütun")

def ensure_parent(path):
    Path(str(path)).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

def sanitize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    obj_cols = df.select_dtypes(include=["object"]).columns
    repl = {
        "–": "-", "−": "-", "≤": "<=", "≥": ">=",
        "â€“": "-", "â€": "-", "â‰¤": "<=", "â‰¥": ">=",
    }
    for c in obj_cols:
        df[c] = df[c].replace(repl, regex=False)
    return df

def read_table(parquet_path: str, csv_path: str) -> pd.DataFrame:
    if os.path.exists(parquet_path):
        log(f"📥 Parquet bulundu: {parquet_path}")
        return pd.read_parquet(parquet_path)
    if os.path.exists(csv_path):
        log(f"📥 CSV bulundu: {csv_path}")
        return pd.read_csv(csv_path, low_memory=False)
    return pd.DataFrame()
    
def safe_save_csv(df: pd.DataFrame, path: str):
    ensure_parent(path)
    tmp = str(path) + ".tmp"
    df2 = sanitize_text_columns(df)
    with open(tmp, "w", encoding="utf-8-sig", errors="replace", newline="") as f:
        df2.to_csv(f, index=False)
    os.replace(tmp, path)

DEFAULT_GEOID_LEN = int(os.getenv("GEOID_LEN", "11"))

def safe_save_parquet(df: pd.DataFrame, path: str):
    ensure_parent(path)
    tmp = str(path) + ".tmp.parquet"

    df2 = sanitize_text_columns(df)

    for c in df2.select_dtypes(include=["float64"]).columns:
        df2[c] = pd.to_numeric(df2[c], downcast="float")
    for c in df2.select_dtypes(include=["int64", "Int64"]).columns:
        df2[c] = pd.to_numeric(df2[c], downcast="integer")

    df2.to_parquet(tmp, index=False, engine="pyarrow", compression="snappy")
    os.replace(tmp, path)
    
def normalize_geoid(s: pd.Series, target_len: int = DEFAULT_GEOID_LEN) -> pd.Series:
    s = s.astype(str).str.extract(r"(\d+)", expand=False)
    return s.str[:target_len].str.zfill(target_len)

def load_json(path: str, default=None):
    if not os.path.exists(path):
        return {} if default is None else default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj, path: str):
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def utc_now_iso():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def parse_dt(x):
    try:
        return pd.to_datetime(x, errors="coerce", utc=True)
    except Exception:
        return pd.NaT

def days_since_iso(iso_str: str):
    if not iso_str:
        return 10**9
    try:
        then = pd.to_datetime(iso_str, utc=True)
        now = pd.Timestamp.now(tz="UTC")
        return (now - then).days
    except Exception:
        return 10**9

def extract_lat_lon(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    candidates = [
        ("latitude", "longitude"),
        ("lat", "lon"),
        ("lat", "long"),
        ("y", "x"),
        ("stop_lat", "stop_lon"),
        ("point_y", "point_x"),
    ]
    for la, lo in candidates:
        if la in df.columns and lo in df.columns:
            df["lat_"] = pd.to_numeric(df[la], errors="coerce")
            df["lon_"] = pd.to_numeric(df[lo], errors="coerce")
            return df

    if "location" in df.columns:
        def _pull(o, key):
            if isinstance(o, dict):
                return o.get(key)
            if isinstance(o, str):
                try:
                    j = json.loads(o)
                    return j.get(key)
                except Exception:
                    return None
            return None

        df["lat_"] = pd.to_numeric(df["location"].apply(lambda o: _pull(o, "latitude")), errors="coerce")
        df["lon_"] = pd.to_numeric(df["location"].apply(lambda o: _pull(o, "longitude")), errors="coerce")
        return df

    if "the_geom" in df.columns:
        def _coords(o):
            if isinstance(o, dict) and "coordinates" in o and len(o["coordinates"]) >= 2:
                lon, lat = o["coordinates"][:2]
                return lat, lon
            if isinstance(o, str):
                try:
                    j = json.loads(o)
                    if "coordinates" in j and len(j["coordinates"]) >= 2:
                        lon, lat = j["coordinates"][:2]
                        return lat, lon
                except Exception:
                    pass
            return None, None

        latlon = df["the_geom"].apply(_coords)
        df["lat_"] = pd.to_numeric(latlon.apply(lambda t: t[0]), errors="coerce")
        df["lon_"] = pd.to_numeric(latlon.apply(lambda t: t[1]), errors="coerce")
        return df

    df["lat_"] = np.nan
    df["lon_"] = np.nan
    return df

# =========================================================
# Socrata downloader
# =========================================================
def socrata_download_with_retry(base_url: str, headers: dict, where_clause: str | None = None,
                                select_clause: str | None = None, order_clause: str | None = None,
                                limit: int = 50000, max_retries: int = 5, backoff_base: float = 1.7):
    rows = []
    offset = 0

    while True:
        params = {"$limit": limit, "$offset": offset}
        if where_clause:
            params["$where"] = where_clause
        if select_clause:
            params["$select"] = select_clause
        if order_clause:
            params["$order"] = order_clause

        attempt = 0
        while True:
            try:
                r = requests.get(base_url, params=params, headers=headers, timeout=90)
                if r.status_code in (429,) or 500 <= r.status_code < 600:
                    attempt += 1
                    if attempt > max_retries:
                        r.raise_for_status()
                    sleep_s = backoff_base ** attempt
                    log(f"⚠️ Geçici hata status={r.status_code} offset={offset} → {attempt}. deneme, {sleep_s:.1f}s")
                    time.sleep(sleep_s)
                    continue

                r.raise_for_status()
                data = r.json()
                chunk = pd.DataFrame(data)
                break

            except requests.HTTPError as e:
                log(f"❌ HTTP hatası offset={offset}: {e}")
                return None
            except Exception as e:
                attempt += 1
                if attempt > max_retries:
                    log(f"❌ Ağ/parse hatası offset={offset}: {e}")
                    return None
                sleep_s = backoff_base ** attempt
                log(f"⚠️ Ağ/parse hatası offset={offset} → {attempt}. deneme, {sleep_s:.1f}s ({e})")
                time.sleep(sleep_s)

        if chunk is None or chunk.empty:
            break

        if offset == 0:
            log(f"🔎 İlk chunk kolonları: {list(chunk.columns)}")

        rows.append(chunk)
        offset += len(chunk)
        log(f"  + {offset} kayıt indirildi...")

        if len(chunk) < limit:
            break

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)

# =========================================================
# GEOID eşleme
# =========================================================
BASE_DIR = os.getenv("CRIME_DATA_DIR", "crime_prediction_data")
Path(BASE_DIR).mkdir(parents=True, exist_ok=True)
print(f"[INFO] BASE_DIR: {BASE_DIR}")

CENSUS_CANDIDATES = [
    os.path.join(BASE_DIR, "sf_census_blocks.geojson"),
    os.path.join(".", "sf_census_blocks.geojson"),
]

census_path = next((p for p in CENSUS_CANDIDATES if os.path.exists(p)), None)
if census_path is None:
    raise FileNotFoundError("❌ sf_census_blocks.geojson bulunamadı.")

gdf_blocks = gpd.read_file(census_path)
if gdf_blocks.crs is None:
    gdf_blocks.set_crs("EPSG:4326", inplace=True, allow_override=True)
else:
    epsg = gdf_blocks.crs.to_epsg() if hasattr(gdf_blocks.crs, "to_epsg") else None
    if epsg != 4326:
        gdf_blocks = gdf_blocks.to_crs(epsg=4326)

gcol = "GEOID" if "GEOID" in gdf_blocks.columns else next(
    (c for c in gdf_blocks.columns if str(c).upper().startswith("GEOID")), None
)
if not gcol:
    raise KeyError("❌ sf_census_blocks.geojson içinde GEOID yok.")

gdf_blocks["GEOID"] = normalize_geoid(gdf_blocks[gcol], DEFAULT_GEOID_LEN)

def assign_geoid(df: pd.DataFrame) -> pd.DataFrame:
    df = extract_lat_lon(df)
    df = df.dropna(subset=["lat_", "lon_"]).copy()
    if df.empty:
        df["GEOID"] = pd.NA
        return df

    gdf_pts = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon_"], df["lat_"]),
        crs="EPSG:4326"
    )

    try:
        gdf_pts = gpd.sjoin(gdf_pts, gdf_blocks[["GEOID", "geometry"]], how="left", predicate="within")
    except Exception as e:
        log(f"⚠️ sjoin(within) başarısız ({e}), nearest deneniyor...")
        gdf_pts = gpd.sjoin_nearest(gdf_pts, gdf_blocks[["GEOID", "geometry"]], how="left", max_distance=0.001)

    gdf_pts = gdf_pts.drop(columns=["geometry", "index_right"], errors="ignore")
    gdf_pts["GEOID"] = normalize_geoid(gdf_pts["GEOID"], DEFAULT_GEOID_LEN)
    return pd.DataFrame(gdf_pts)

# =========================================================
# Config
# =========================================================
SOCS_APP_TOKEN = os.getenv("SOCS_APP_TOKEN", "").strip()
HEADERS = {"Accept": "application/json"}
if SOCS_APP_TOKEN:
    HEADERS["X-App-Token"] = SOCS_APP_TOKEN

FORCE_ALL = os.getenv("FORCE_ALL_REFRESH", "0").strip().lower() in ("1", "true", "yes")

# yollar
BUSINESS_OUT = os.path.join(BASE_DIR, "sf_business_landuse.parquet")
BUILDING_OUT = os.path.join(BASE_DIR, "sf_building_permits_vacancy.parquet")
TRAFFIC_OUT  = os.path.join(BASE_DIR, "sf_traffic_transport.parquet")
STREET_OUT   = os.path.join(BASE_DIR, "sf_street_environment.parquet")

WRITE_CSV = os.getenv("WRITE_CSV", "0").strip().lower() in ("1", "true", "yes", "on")

BUSINESS_OUT_CSV = BUSINESS_OUT.replace(".parquet", ".csv")
BUILDING_OUT_CSV = BUILDING_OUT.replace(".parquet", ".csv")
TRAFFIC_OUT_CSV  = TRAFFIC_OUT.replace(".parquet", ".csv")
STREET_OUT_CSV   = STREET_OUT.replace(".parquet", ".csv")

BUSINESS_META = os.path.join(BASE_DIR, "sf_business_landuse_meta.json")
BUILDING_META = os.path.join(BASE_DIR, "sf_building_permits_vacancy_meta.json")
TRAFFIC_META  = os.path.join(BASE_DIR, "sf_traffic_transport_meta.json")
STREET_META   = os.path.join(BASE_DIR, "sf_street_environment_meta.json")

# =========================================================
# Dataset config
# Burada ID ve date kolonlarını kendi datasetlerinle eşleştirebilirsin.
# =========================================================
CFG = {
    "business": {
        "rid": os.getenv("BUSINESS_DATASET_ID", "rqzj-sfat"),
        "out": BUSINESS_OUT,
        "meta": BUSINESS_META,
        "mode": "static_periodic",   # statik/yavaş değişen
        "refresh_days": int(os.getenv("BUSINESS_REFRESH_DAYS", "30")),
        "date_col": os.getenv("BUSINESS_DATE_COL", ""),  # çoğu zaman yok; boş olabilir
    },
    "building": {
        "rid": os.getenv("BUILDING_DATASET_ID", "i98e-djp9"),
        "out": BUILDING_OUT,
        "meta": BUILDING_META,
        "mode": "incremental",
        "refresh_days": 0,
        "date_col": os.getenv("BUILDING_DATE_COL", "filed_date"),
    },
    "traffic": {
        "rid": os.getenv("TRAFFIC_DATASET_ID", "w969-5mn4"),
        "out": TRAFFIC_OUT,
        "meta": TRAFFIC_META,
        "mode": "incremental",
        "refresh_days": 0,
        "date_col": os.getenv("TRAFFIC_DATE_COL", "date"),
    },
    "street": {
        "rid": os.getenv("STREET_DATASET_ID", "tgmn-chn8"),
        "out": STREET_OUT,
        "meta": STREET_META,
        "mode": "static_periodic",
        "refresh_days": int(os.getenv("STREET_REFRESH_DAYS", "30")),
        "date_col": os.getenv("STREET_DATE_COL", ""),
    },
}

# =========================================================
# Temizleme / standardizasyon
# =========================================================
def prep_business(df: pd.DataFrame) -> pd.DataFrame:
    df = assign_geoid(df)

    # örnek standardizasyon
    rename_map = {}
    if "location_id" in df.columns:
        rename_map["location_id"] = "business_id"
    if "naics_code_description" in df.columns:
        rename_map["naics_code_description"] = "business_type"
    if rename_map:
        df = df.rename(columns=rename_map)

    keep = [c for c in ["business_id", "business_type", "GEOID", "lat_", "lon_"] if c in df.columns]
    if not keep:
        keep = ["GEOID"]
    df = df[keep].copy()

    # mümkünse tekrarları azalt
    subset = [c for c in ["business_id", "GEOID", "lat_", "lon_"] if c in df.columns]
    if subset:
        df = df.drop_duplicates(subset=subset, keep="last")

    df = df.sort_values(["GEOID"]).reset_index(drop=True)
    return df

def prep_building(df: pd.DataFrame) -> pd.DataFrame:
    df = assign_geoid(df)

    date_col = CFG["building"]["date_col"]
    if date_col in df.columns:
        df["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    else:
        df["date"] = pd.NaT

    # örnek izin tipleri
    type_col = None
    for c in ["permit_type", "permit_type_definition", "description", "current_status"]:
        if c in df.columns:
            type_col = c
            break

    if type_col is None:
        df["permit_type_std"] = "unknown"
    else:
        df["permit_type_std"] = df[type_col].astype(str).str.lower()

    # kaba vacancy proxy
    txt = df["permit_type_std"].fillna("")
    df["permit_count"] = 1
    df["new_construction_count"] = txt.str.contains("new|construction", regex=True).astype(int)
    df["demolition_count"] = txt.str.contains("demolition|demo", regex=True).astype(int)
    df["renovation_count"] = txt.str.contains("alter|repair|renov|addition", regex=True).astype(int)
    df["vacant_building_count"] = txt.str.contains("vacant", regex=True).astype(int)
    df["unsafe_building_count"] = txt.str.contains("unsafe|dangerous|hazard", regex=True).astype(int)

    out = (
        df.dropna(subset=["GEOID", "date"])
          .groupby(["GEOID", "date"], as_index=False)[
              ["permit_count", "new_construction_count", "demolition_count",
               "renovation_count", "vacant_building_count", "unsafe_building_count"]
          ].sum()
    )

    out = out.sort_values(["GEOID", "date"]).reset_index(drop=True)
    return out

def hour_to_hour_range(hour):
    if pd.isna(hour):
        return np.nan
    try:
        h = int(hour)
    except Exception:
        return np.nan
    start = (h // 3) * 3
    end = start + 3
    return f"{start:02d}:00-{end:02d}:00"

def prep_traffic(df: pd.DataFrame) -> pd.DataFrame:
    df = assign_geoid(df)

    date_col = CFG["traffic"]["date_col"]
    if date_col in df.columns:
        dt = pd.to_datetime(df[date_col], errors="coerce")
    else:
        dt = pd.to_datetime(pd.Series([pd.NaT] * len(df)), errors="coerce")

    df["date"] = dt.dt.normalize()

    if "hour" in df.columns:
        df["hour_num"] = pd.to_numeric(df["hour"], errors="coerce")
    else:
        df["hour_num"] = dt.dt.hour

    df["hour_range"] = df["hour_num"].apply(hour_to_hour_range)

    # ölçüm kolonu yoksa count tabanlı proxy
    measure_col = None
    for c in ["count", "traffic_count", "volume", "boardings", "pedestrian_count"]:
        if c in df.columns:
            measure_col = c
            break

    if measure_col is None:
        df["traffic_count"] = 1.0
    else:
        df["traffic_count"] = pd.to_numeric(df[measure_col], errors="coerce").fillna(0)

    # ek proxy kolonlar
    df["transit_boardings"] = df["traffic_count"] if "boardings" in str(measure_col or "") else 0.0
    df["bus_activity"] = 0.0
    df["train_activity"] = 0.0
    df["congestion_index"] = 0.0
    df["avg_speed"] = 0.0
    df["pedestrian_count"] = 0.0

    out = (
        df.dropna(subset=["GEOID", "date", "hour_range"])
          .groupby(["GEOID", "date", "hour_range"], as_index=False)[
              ["traffic_count", "transit_boardings", "bus_activity",
               "train_activity", "congestion_index", "avg_speed", "pedestrian_count"]
          ].mean()
    )

    out = out.sort_values(["GEOID", "date", "hour_range"]).reset_index(drop=True)
    return out

def prep_street(df: pd.DataFrame) -> pd.DataFrame:
    df = assign_geoid(df)

    # kaba environment proxy
    out = df.groupby("GEOID", as_index=False).size().rename(columns={"size": "street_light_count"})
    out["tree_count"] = 0
    out["sidewalk_score"] = 0
    out["road_quality_score"] = 0
    out["intersection_density"] = 0
    out["walkability_score"] = 0
    out["abandoned_vehicle_count"] = 0

    out = out.sort_values(["GEOID"]).reset_index(drop=True)
    return out

PREP_FN = {
    "business": prep_business,
    "building": prep_building,
    "traffic": prep_traffic,
    "street": prep_street,
}

# =========================================================
# Append / merge helpers
# =========================================================
def append_incremental(existing: pd.DataFrame, new_df: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    if existing is None or existing.empty:
        return new_df.copy()

    both = pd.concat([existing, new_df], ignore_index=True)
    key_cols = [c for c in key_cols if c in both.columns]
    if key_cols:
        both = both.drop_duplicates(subset=key_cols, keep="last")
    else:
        both = both.drop_duplicates(keep="last")
    return both.reset_index(drop=True)

def get_max_date_for_meta(df: pd.DataFrame, date_col: str):
    if date_col and date_col in df.columns:
        mx = pd.to_datetime(df[date_col], errors="coerce").max()
        if pd.notna(mx):
            return pd.Timestamp(mx).isoformat()
    if "date" in df.columns:
        mx = pd.to_datetime(df["date"], errors="coerce").max()
        if pd.notna(mx):
            return pd.Timestamp(mx).isoformat()
    return ""

# =========================================================
# İndirme kararı
# =========================================================
def should_refresh_static(meta_path: str, out_path: str, refresh_days: int, force: bool = False):
    if force:
        return True
    if not os.path.exists(out_path):
        return True
    meta = load_json(meta_path, {})
    last_download_utc = meta.get("last_download_utc", "")
    return days_since_iso(last_download_utc) >= refresh_days

def build_incremental_where(date_col: str, meta_path: str):
    meta = load_json(meta_path, {})
    last_max_date = meta.get("last_max_date", "")
    if not date_col or not last_max_date:
        return None
    # Socrata date literal
    return f"{date_col} > '{last_max_date}'"

# =========================================================
# Ana update fonksiyonu
# =========================================================
def run_dataset(name: str):
    cfg = CFG[name]
    rid = cfg["rid"]
    out_path = cfg["out"]
    meta_path = cfg["meta"]
    mode = cfg["mode"]
    refresh_days = cfg["refresh_days"]
    date_col = cfg["date_col"]

    base_url = f"https://data.sfgov.org/resource/{rid}.json"

    log("\n" + "=" * 70)
    log(f"🚀 DATASET: {name.upper()} | rid={rid}")

    # ---------------------
    # STATIC / PERIODIC
    # ---------------------
    if mode == "static_periodic":
        refresh = should_refresh_static(meta_path, out_path, refresh_days, force=FORCE_ALL)

        if not refresh:
            log(f"✅ {name}: cache geçerli, refresh gerekmiyor.")
            csv_fallback = out_path.replace(".parquet", ".csv")
            df = read_table(out_path, csv_fallback)
            if not df.empty:
                log_shape(df, f"{name} (cache)")
            return

        log(f"📥 {name}: periyodik full refresh başlıyor...")
        raw = socrata_download_with_retry(base_url, HEADERS, order_clause=None)
        if raw is None:
            log(f"❌ {name}: indirme başarısız.")
            return

        log_shape(raw, f"{name} raw")
        out_df = PREP_FN[name](raw)
        log_shape(out_df, f"{name} prepared")

        safe_save_parquet(out_df, out_path)
        if WRITE_CSV:
            safe_save_csv(out_df, out_path.replace(".parquet", ".csv"))
        save_json({
            "dataset": name,
            "rid": rid,
            "mode": mode,
            "last_download_utc": utc_now_iso(),
            "last_max_date": get_max_date_for_meta(out_df, date_col),
            "rows": int(len(out_df)),
            "columns": list(out_df.columns),
        }, meta_path)

        log(f"✅ {name}: yazıldı → {out_path}")
        return

    # ---------------------
    # INCREMENTAL
    # ---------------------
    if mode == "incremental":
        if FORCE_ALL or (not os.path.exists(out_path)):
            log(f"📥 {name}: full initial refresh...")
            raw = socrata_download_with_retry(base_url, HEADERS, order_clause=f"{date_col} ASC" if date_col else None)
            if raw is None:
                log(f"❌ {name}: full indirme başarısız.")
                return

            log_shape(raw, f"{name} raw")
            out_df = PREP_FN[name](raw)
            log_shape(out_df, f"{name} prepared")

            safe_save_parquet(out_df, out_path)
            if WRITE_CSV:
                safe_save_csv(out_df, out_path.replace(".parquet", ".csv"))
            save_json({
                "dataset": name,
                "rid": rid,
                "mode": mode,
                "last_download_utc": utc_now_iso(),
                "last_max_date": get_max_date_for_meta(out_df, date_col),
                "rows": int(len(out_df)),
                "columns": list(out_df.columns),
            }, meta_path)

            log(f"✅ {name}: full yazıldı → {out_path}")
            return

        # incremental path
        existing = read_table(out_path, out_path.replace(".parquet", ".csv"))
        log_shape(existing, f"{name} existing")

        where_clause = build_incremental_where(date_col, meta_path)
        if not where_clause:
            log(f"⚠️ {name}: last_max_date yok, full refresh fallback.")
            raw = socrata_download_with_retry(base_url, HEADERS, order_clause=f"{date_col} ASC" if date_col else None)
        else:
            log(f"🧠 {name}: incremental where = {where_clause}")
            raw = socrata_download_with_retry(
                base_url,
                HEADERS,
                where_clause=where_clause,
                order_clause=f"{date_col} ASC" if date_col else None
            )

        if raw is None:
            log(f"❌ {name}: incremental indirme başarısız.")
            return

        if raw.empty:
            log(f"✅ {name}: yeni kayıt yok.")
            meta = load_json(meta_path, {})
            meta["last_download_utc"] = utc_now_iso()
            save_json(meta, meta_path)
            return

        log_shape(raw, f"{name} new raw")
        new_df = PREP_FN[name](raw)
        log_shape(new_df, f"{name} new prepared")

        if name == "building":
            key_cols = ["GEOID", "date"]
        elif name == "traffic":
            key_cols = ["GEOID", "date", "hour_range"]
        else:
            key_cols = ["GEOID"]

        merged = append_incremental(existing, new_df, key_cols)
        log_shape(merged, f"{name} merged")

        safe_save_parquet(merged, out_path)
        if WRITE_CSV:
            safe_save_csv(merged, out_path.replace(".parquet", ".csv"))
        save_json({
            "dataset": name,
            "rid": rid,
            "mode": mode,
            "last_download_utc": utc_now_iso(),
            "last_max_date": get_max_date_for_meta(merged, date_col),
            "rows": int(len(merged)),
            "columns": list(merged.columns),
        }, meta_path)

        log(f"✅ {name}: incremental güncellendi → {out_path}")
        return

# =========================================================
# Çalıştır
# =========================================================
if __name__ == "__main__":
    for ds in ["business", "building", "traffic", "street"]:
        try:
            run_dataset(ds)
        except Exception as e:
            log(f"❌ {ds} hata: {e}")

    log("\n🎉 Tüm update işlemleri tamamlandı.")
