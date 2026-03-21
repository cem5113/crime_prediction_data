# =========================================================
# ✅ INCREMENTAL FEATURE UPDATE + ONLY-NEW-CRIME ENRICH
# business  -> GEOID
# building  -> GEOID + date
# traffic   -> ÇIKARILDI
# street    -> ÇIKARILDI
# =========================================================

!pip -q install geopandas pyarrow requests shapely fiona

import os
import re
import json
import time
import requests
import numpy as np
import pandas as pd
import geopandas as gpd

from pathlib import Path
from datetime import datetime, timezone

# =========================================================
# CONFIG
# =========================================================
BASE_DIR = "/content/crime_prediction_data"
os.makedirs(BASE_DIR, exist_ok=True)

CENSUS_PATH = f"{BASE_DIR}/sf_census_blocks.geojson"

CRIME_INPUT_PATH = f"{BASE_DIR}/sf_crime_09.parquet"
CRIME_OUTPUT_PATH = f"{BASE_DIR}/sf_crime_10.parquet"

WRITE_CSV = True
HEADERS = {}
GEOID_LEN = 11

try:
    import zoneinfo
    SF_TZ = zoneinfo.ZoneInfo("America/Los_Angeles")
except Exception:
    SF_TZ = None

# =========================================================
# GEOID
# =========================================================
if not os.path.exists(CENSUS_PATH):
    raise FileNotFoundError(f"❌ Census dosyası yok: {CENSUS_PATH}")

gdf_blocks = gpd.read_file(CENSUS_PATH)
if gdf_blocks.crs is None:
    gdf_blocks.set_crs("EPSG:4326", inplace=True, allow_override=True)
else:
    epsg = gdf_blocks.crs.to_epsg() if hasattr(gdf_blocks.crs, "to_epsg") else None
    if epsg != 4326:
        gdf_blocks = gdf_blocks.to_crs(epsg=4326)

gcol_candidates = [c for c in gdf_blocks.columns if "GEOID" in str(c).upper()]
if "GEOID" in gdf_blocks.columns:
    gcol = "GEOID"
elif gcol_candidates:
    gcol = gcol_candidates[0]
else:
    raise ValueError("❌ Census dosyasında GEOID kolonu bulunamadı.")

gdf_blocks[gcol] = (
    gdf_blocks[gcol]
    .astype(str)
    .str.extract(r"(\d+)", expand=False)
    .str.zfill(GEOID_LEN)
)
gdf_blocks = gdf_blocks[[gcol, "geometry"]].rename(columns={gcol: "GEOID"}).copy()

# =========================================================
# HELPERS
# =========================================================
def log(msg):
    print(msg, flush=True)

def ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def safe_save_parquet(df, path):
    ensure_parent(path)
    df.to_parquet(path, index=False)
    log(f"💾 parquet yazıldı: {path}")

def safe_save_csv(df, path):
    ensure_parent(path)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    log(f"💾 csv yazıldı: {path}")

def save_json(obj, path):
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path, default=None):
    if not os.path.exists(path):
        return {} if default is None else default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()

def normalize_geoid_series(s):
    return (
        s.astype(str)
         .str.extract(r"(\d+)", expand=False)
         .str.zfill(GEOID_LEN)
    )

def to_sf_datetime(series):
    s = pd.to_datetime(series, errors="coerce", utc=True)
    if SF_TZ is not None:
        s = s.dt.tz_convert(SF_TZ)
    return s

def to_sf_date(series):
    return to_sf_datetime(series).dt.date

def normalize_hour_range(hr):
    m = re.match(r"^\s*(\d{1,2})\s*[-:]\s*(\d{1,2})\s*$", str(hr))
    if not m:
        return np.nan
    a = int(m.group(1)) % 24
    b = int(m.group(2))
    if b <= a:
        b = min(a + 3, 24)
    return f"{a:02d}-{b:02d}"

def to_sf_hour_range(series):
    s = to_sf_datetime(series)
    h = s.dt.hour.fillna(0).astype(int)
    st = (h // 3) * 3
    return st.map(lambda x: f"{x:02d}-{min(x+3,24):02d}")

def read_table(preferred_parquet, fallback_csv=None):
    if os.path.exists(preferred_parquet):
        return pd.read_parquet(preferred_parquet)
    if fallback_csv and os.path.exists(fallback_csv):
        return pd.read_csv(fallback_csv, low_memory=False)
    return pd.DataFrame()

def geocode_to_geoid(df, lon_col="longitude", lat_col="latitude"):
    if df.empty:
        return df.copy()

    tmp = df.copy()
    tmp[lon_col] = pd.to_numeric(tmp[lon_col], errors="coerce")
    tmp[lat_col] = pd.to_numeric(tmp[lat_col], errors="coerce")
    tmp = tmp.dropna(subset=[lon_col, lat_col]).copy()
    if tmp.empty:
        return tmp

    gdf = gpd.GeoDataFrame(
        tmp,
        geometry=gpd.points_from_xy(tmp[lon_col], tmp[lat_col]),
        crs="EPSG:4326"
    )
    joined = gpd.sjoin(
        gdf,
        gdf_blocks[["GEOID", "geometry"]],
        how="left",
        predicate="within"
    )
    joined = joined.drop(columns=["geometry", "index_right"], errors="ignore")
    joined["GEOID"] = normalize_geoid_series(joined["GEOID"])
    joined = joined.dropna(subset=["GEOID"]).copy()
    return pd.DataFrame(joined)

def socrata_download_with_retry(base_url, headers=None, where=None, select=None, order=None,
                                page_limit=50000, max_retries=5, sleep_sec=0.3):
    headers = headers or {}
    pieces = []
    offset = 0

    while True:
        params = {
            "$limit": page_limit,
            "$offset": offset,
        }
        if where:
            params["$where"] = where
        if select:
            params["$select"] = select
        if order:
            params["$order"] = order

        ok = False
        last_err = None

        for k in range(max_retries):
            try:
                r = requests.get(base_url, params=params, headers=headers, timeout=90)
                r.raise_for_status()
                data = r.json()
                df = pd.DataFrame(data)
                ok = True
                break
            except Exception as e:
                last_err = e
                time.sleep(sleep_sec * (k + 1))

        if not ok:
            log(f"❌ indirme hatası: {last_err}")
            return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()

        if df.empty:
            break

        pieces.append(df)
        log(f"  + {offset + len(df)} kayıt indirildi...")

        if len(df) < page_limit:
            break

        offset += page_limit
        time.sleep(sleep_sec)

    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()

def append_dedup(old_df, new_df, subset_cols, sort_cols=None):
    if old_df is None or old_df.empty:
        out = new_df.copy()
    elif new_df is None or new_df.empty:
        out = old_df.copy()
    else:
        out = pd.concat([old_df, new_df], ignore_index=True)

    if sort_cols:
        sort_cols = [c for c in sort_cols if c in out.columns]
        if sort_cols:
            out = out.sort_values(sort_cols)

    subset_cols = [c for c in subset_cols if c in out.columns]
    if subset_cols:
        out = out.drop_duplicates(subset=subset_cols, keep="last")

    return out.reset_index(drop=True)

# =========================================================
# FEATURE PREP
# =========================================================
def prep_business(raw):
    cols = ["GEOID", "business_count", "landuse_mix_score"]
    if raw.empty:
        return pd.DataFrame(columns=cols)

    df = raw.copy()

    # sadece gerekli kolonlar varsayımı:
    # facilitytype, status, latitude, longitude, approved, expirationdate, received
    if "status" in df.columns:
        df = df[df["status"].astype(str).str.lower().isin(["approved", "requested"])].copy()

    if "longitude" not in df.columns and "x" in df.columns:
        df["longitude"] = df["x"]
    if "latitude" not in df.columns and "y" in df.columns:
        df["latitude"] = df["y"]

    df = geocode_to_geoid(df, "longitude", "latitude")
    if df.empty:
        return pd.DataFrame(columns=cols)

    if "facilitytype" not in df.columns:
        df["facilitytype"] = "unknown"

    out = (
        df.groupby("GEOID", as_index=False)
          .agg(
              business_count=("GEOID", "size"),
              landuse_mix_score=("facilitytype", lambda s: s.astype(str).nunique())
          )
    )
    return out

def prep_building(raw):
    cols = ["GEOID", "date", "building_permit_count", "building_completed_count", "building_estimated_cost_sum"]
    if raw.empty:
        return pd.DataFrame(columns=cols)

    df = raw.copy()

    if "location" in df.columns and ("longitude" not in df.columns or "latitude" not in df.columns):
        try:
            loc = df["location"].astype(str).str.extract(r"POINT \(([-\d\.]+) ([-\d\.]+)\)")
            df["longitude"] = loc[0]
            df["latitude"] = loc[1]
        except Exception:
            pass

    df = geocode_to_geoid(df, "longitude", "latitude")
    if df.empty:
        return pd.DataFrame(columns=cols)

    date_col = "issued_date" if "issued_date" in df.columns else ("filed_date" if "filed_date" in df.columns else None)
    if date_col is None:
        return pd.DataFrame(columns=cols)

    df["date"] = to_sf_date(df[date_col])

    if "completed_date" in df.columns:
        df["is_completed"] = pd.to_datetime(df["completed_date"], errors="coerce").notna().astype(int)
    else:
        df["is_completed"] = 0

    if "estimated_cost" in df.columns:
        df["estimated_cost"] = pd.to_numeric(df["estimated_cost"], errors="coerce").fillna(0)
    else:
        df["estimated_cost"] = 0

    df = df.dropna(subset=["date"]).copy()

    out = (
        df.groupby(["GEOID", "date"], as_index=False)
          .agg(
              building_permit_count=("GEOID", "size"),
              building_completed_count=("is_completed", "sum"),
              building_estimated_cost_sum=("estimated_cost", "sum")
          )
    )
    return out

# =========================================================
# PATHS / META
# =========================================================
BUSINESS_OUT = f"{BASE_DIR}/sf_business_landuse.parquet"
BUSINESS_META = f"{BASE_DIR}/sf_business_landuse.meta.json"

BUILDING_OUT = f"{BASE_DIR}/sf_building_permits_vacancy.parquet"
BUILDING_META = f"{BASE_DIR}/sf_building_permits_vacancy.meta.json"

CFG = {
    "business": {
        "rid": "rqzj-sfat",
        "out": BUSINESS_OUT,
        "meta": BUSINESS_META,
        "mode": "incremental",
        "date_col": None,
        "select": ",".join([
            "objectid",
            "facilitytype",
            "status",
            "latitude",
            "longitude",
            "approved",
            "expirationdate",
            "received"
        ]),
        "prep": prep_business,
        "dedup_keys": ["GEOID"],
        "sort_cols": ["GEOID"],
    },
    "building": {
        "rid": "i98e-djp9",
        "out": BUILDING_OUT,
        "meta": BUILDING_META,
        "mode": "incremental",
        "date_col": "issued_date",
        "select": ",".join([
            "permit_number",
            "status",
            "filed_date",
            "issued_date",
            "completed_date",
            "estimated_cost",
            "proposed_units",
            "location",
            "data_as_of",
            "data_loaded_at"
        ]),
        "prep": prep_building,
        "dedup_keys": ["GEOID", "date"],
        "sort_cols": ["GEOID", "date"],
    },
}

# =========================================================
# FEATURE UPDATE LOGIC
# =========================================================
def build_incremental_where(date_col, last_max_date):
    if not date_col or not last_max_date:
        return None
    return f"{date_col} >= '{last_max_date}T00:00:00'"

def run_dataset_incremental(name, cfg):
    rid = cfg["rid"]
    out_path = cfg["out"]
    meta_path = cfg["meta"]
    prep_fn = cfg["prep"]
    date_col = cfg["date_col"]
    select_clause = cfg["select"]
    dedup_keys = cfg["dedup_keys"]
    sort_cols = cfg["sort_cols"]

    base_url = f"https://data.sfgov.org/resource/{rid}.json"

    old_df = read_table(out_path, out_path.replace(".parquet", ".csv"))
    meta = load_json(meta_path, default={})

    last_max_date = meta.get("last_max_date", None)

    first_load = old_df.empty
    if first_load:
        log("=" * 70)
        log(f"🚀 DATASET: {name.upper()} | first full load")
        where = None
    else:
        log("=" * 70)
        log(f"🚀 DATASET: {name.upper()} | incremental update")
        where = build_incremental_where(date_col, last_max_date)

    raw = socrata_download_with_retry(
        base_url,
        headers=HEADERS,
        where=where,
        select=select_clause,
        order=f"{date_col} ASC" if date_col else None
    )

    if raw is None or raw.empty:
        log(f"ℹ️ {name}: yeni kayıt yok. Eski parquet korunacak.")
        return old_df, False

    log(f"📊 {name} raw: {raw.shape[0]} satır × {raw.shape[1]} sütun")

    new_df = prep_fn(raw)
    log(f"📊 {name} prepared(new): {new_df.shape[0]} satır × {new_df.shape[1]} sütun")

    if new_df.empty:
        log(f"ℹ️ {name}: prep sonrası yeni kayıt kalmadı.")
        return old_df, False

    merged_df = append_dedup(old_df, new_df, subset_cols=dedup_keys, sort_cols=sort_cols)

    safe_save_parquet(merged_df, out_path)
    if WRITE_CSV:
        safe_save_csv(merged_df, out_path.replace(".parquet", ".csv"))

    new_last_max_date = last_max_date
    if date_col and date_col in raw.columns:
        tmp_dates = to_sf_date(raw[date_col])
        if not tmp_dates.dropna().empty:
            new_last_max_date = str(tmp_dates.dropna().max())

    meta_new = {
        "dataset": name,
        "rid": rid,
        "mode": cfg["mode"],
        "last_download_utc": utc_now_iso(),
        "last_max_date": new_last_max_date,
        "rows": int(merged_df.shape[0]),
        "columns": list(merged_df.columns),
    }
    save_json(meta_new, meta_path)

    log(f"✅ {name}: update tamamlandı → {out_path}")
    return merged_df, True

# =========================================================
# NEW CRIME ROW DETECTION
# =========================================================
def prepare_crime(df):
    if df.empty:
        return df

    out = df.copy()

    if "GEOID" in out.columns:
        out["GEOID"] = normalize_geoid_series(out["GEOID"])

    if "date" not in out.columns:
        if "datetime" in out.columns:
            out["date"] = to_sf_date(out["datetime"])

    if "hour_range" not in out.columns:
        if "datetime" in out.columns:
            out["hour_range"] = to_sf_hour_range(out["datetime"])
        elif "event_hour" in out.columns:
            eh = pd.to_numeric(out["event_hour"], errors="coerce").fillna(0).astype(int) % 24
            st = (eh // 3) * 3
            out["hour_range"] = st.map(lambda x: f"{x:02d}-{min(x+3,24):02d}")
        else:
            out["hour_range"] = np.nan

    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
    if "hour_range" in out.columns:
        out["hour_range"] = out["hour_range"].apply(normalize_hour_range)

    return out

def get_crime_key_cols(df):
    for cand in [
        ["id"],
        ["GEOID", "datetime"],
        ["GEOID", "date", "hour_range"],
    ]:
        if all(c in df.columns for c in cand):
            return cand
    raise ValueError("❌ Crime için uygun benzersiz anahtar bulunamadı.")

def find_new_crime_rows(crime_all, crime_old_enriched):
    crime_all = prepare_crime(crime_all)

    if crime_old_enriched is None or crime_old_enriched.empty:
        return crime_all.copy()

    crime_old_enriched = prepare_crime(crime_old_enriched)

    key_cols = get_crime_key_cols(crime_all)
    old_keys = crime_old_enriched[key_cols].drop_duplicates().copy()
    old_keys["_seen_"] = 1

    merged = crime_all.merge(old_keys, on=key_cols, how="left")
    new_rows = merged[merged["_seen_"].isna()].drop(columns=["_seen_"]).copy()

    return new_rows.reset_index(drop=True)

# =========================================================
# ENRICH ONLY NEW CRIME
# =========================================================
def enrich_new_crime_rows(new_crime, business_df, building_df):
    if new_crime.empty:
        return new_crime.copy()

    out = prepare_crime(new_crime)

    if "GEOID" not in out.columns:
        raise ValueError("❌ Crime dataframe içinde GEOID yok.")

    # business -> GEOID
    if business_df is not None and not business_df.empty:
        business_df = business_df.copy()
        business_df["GEOID"] = normalize_geoid_series(business_df["GEOID"])
        overlap = [c for c in business_df.columns if c in out.columns and c != "GEOID"]
        if overlap:
            business_df = business_df.drop(columns=overlap)
        out = out.merge(business_df, on="GEOID", how="left")

    # building -> GEOID + date
    if building_df is not None and not building_df.empty:
        building_df = building_df.copy()
        building_df["GEOID"] = normalize_geoid_series(building_df["GEOID"])
        building_df["date"] = pd.to_datetime(building_df["date"], errors="coerce").dt.date
        overlap = [c for c in building_df.columns if c in out.columns and c not in ["GEOID", "date"]]
        if overlap:
            building_df = building_df.drop(columns=overlap)
        out = out.merge(building_df, on=["GEOID", "date"], how="left")

    fill_zero_cols = [
        "business_count",
        "landuse_mix_score",
        "building_permit_count",
        "building_completed_count",
        "building_estimated_cost_sum",
    ]
    for c in fill_zero_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)

    return out

# =========================================================
# MAIN
# =========================================================
if not os.path.exists(CRIME_INPUT_PATH):
    raise FileNotFoundError(f"❌ Crime input yok: {CRIME_INPUT_PATH}")

log("🏙️ Feature parquet update başlıyor...")

updated_tables = {}
update_flags = {}

for ds_name, ds_cfg in CFG.items():
    tbl, changed = run_dataset_incremental(ds_name, ds_cfg)
    updated_tables[ds_name] = tbl
    update_flags[ds_name] = changed

log("=" * 70)
log(f"📌 update flags: {update_flags}")

crime_all = pd.read_parquet(CRIME_INPUT_PATH)
log(f"📥 crime_all: {crime_all.shape}")

crime_old_enriched = read_table(CRIME_OUTPUT_PATH, CRIME_OUTPUT_PATH.replace(".parquet", ".csv"))
if not crime_old_enriched.empty:
    log(f"📥 old enriched crime: {crime_old_enriched.shape}")
else:
    log("ℹ️ Eski enriched crime yok. İlk enrich yapılacak.")

new_crime = find_new_crime_rows(crime_all, crime_old_enriched)
log(f"🆕 new crime rows: {new_crime.shape}")

if new_crime.empty:
    log("✅ Yeni suç satırı yok. Eski enriched çıktı korunuyor.")
else:
    new_enriched = enrich_new_crime_rows(
        new_crime=new_crime,
        business_df=updated_tables["business"],
        building_df=updated_tables["building"],
    )
    log(f"📊 new enriched: {new_enriched.shape}")

    crime_all_prepared = prepare_crime(crime_all)

    final_df = append_dedup(
        crime_old_enriched,
        new_enriched,
        subset_cols=get_crime_key_cols(crime_all_prepared),
        sort_cols=[c for c in ["date", "datetime", "GEOID", "hour_range"] if c in crime_all_prepared.columns]
    )

    safe_save_parquet(final_df, CRIME_OUTPUT_PATH)
    if WRITE_CSV:
        safe_save_csv(final_df, CRIME_OUTPUT_PATH.replace(".parquet", ".csv"))

    log(f"✅ Final enriched crime kaydedildi: {CRIME_OUTPUT_PATH}")
    log(f"📦 final shape: {final_df.shape}")

log("=" * 70)
log("🎉 Incremental feature update + only-new-crime enrich tamamlandı.")
