# scripts/update_pois.py
# =============================================================================
# POI ENRICH — MODEL-ORIENTED FINAL REVIZE
#
# Amaç:
#   sf_crime_05.(csv/parquet) -> POI feature'ları ekleyip sf_crime_06.(csv/parquet) üretmek
#
# Bu sürümün farkı:
#   1) Varsayılan olarak FULL RECOMPUTE (eğitim için daha doğru)
#   2) GEOID bazlı POI özetini üretir
#   3) Zaman etkileşimli feature'lar ekler
#   4) Modele doğrudan girecek düşük-gürültülü, açıklanabilir kolonlar üretir
#
# Not:
#   - POI dosyası varsa onu kullanır
#   - Yoksa geojson + census block ile GEOID atayıp üretir
#   - CSV ve Parquet ikisini de yazabilir
# =============================================================================

from __future__ import annotations

import os
import ast
import json
import time
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import geopandas as gpd
import requests

try:
    from shapely.strtree import STRtree
except Exception:
    STRtree = None


# =============================================================================
# ENV / PATH
# =============================================================================
BASE_DIR = os.getenv("CRIME_DATA_DIR", "crime_prediction_data")
Path(BASE_DIR).mkdir(parents=True, exist_ok=True)

CRIME_IN = os.getenv("CRIME_IN", os.path.join(BASE_DIR, "sf_crime_05.parquet"))
CRIME_OUT = os.getenv("CRIME_OUT", os.path.join(BASE_DIR, "sf_crime_06.parquet"))

POI_GEOJSON = os.getenv("POI_GEOJSON", os.path.join(BASE_DIR, "sf_pois.geojson"))
BLOCKS_GEOJSON = os.getenv(
    "BLOCKS_GEOJSON",
    os.path.join(BASE_DIR, "sf_census_blocks_with_population.geojson")
)
POI_CLEAN_CSV = os.getenv("POI_CLEAN_CSV", os.path.join(BASE_DIR, "sf_pois_cleaned_with_geoid.csv"))
POI_RISK_JSON = os.getenv("POI_RISK_JSON", os.path.join(BASE_DIR, "risky_pois_dynamic.json"))
POI_SUMMARY_CSV = os.getenv("POI_SUMMARY_CSV", os.path.join(BASE_DIR, "poi_geoid_summary.csv"))

FORCE_POI_REFRESH = os.getenv("FORCE_POI_REFRESH", "0").strip().lower() in ("1", "true", "yes")
WRITE_CSV_ALSO = os.getenv("WRITE_CSV_ALSO", "1").strip().lower() in ("1", "true", "yes")
INCLUDE_OFFICE_CRAFT = os.getenv("INCLUDE_OFFICE_CRAFT", "1").strip().lower() not in ("0", "false", "no")
POI_RISK_RADIUS_M = float(os.getenv("POI_RISK_RADIUS_M", "300"))
POI_RISK_LOOKBACK_YEARS = int(os.getenv("POI_RISK_LOOKBACK_YEARS", "5"))
GEOID_LEN = int(os.getenv("GEOID_LEN", "11"))


# =============================================================================
# IO HELPERS
# =============================================================================
def log(msg: str):
    print(msg, flush=True)


def log_shape(df: pd.DataFrame, label: str):
    log(f"📊 {label}: {df.shape[0]:,} satır × {df.shape[1]} sütun")


def ensure_parent(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def read_table(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dosya yok: {path}")
    ext = Path(path).suffix.lower()
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext in (".csv", ".txt"):
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"Desteklenmeyen uzantı: {path}")


def write_table(df: pd.DataFrame, path: str):
    ensure_parent(path)
    ext = Path(path).suffix.lower()
    if ext == ".parquet":
        df.to_parquet(path, index=False)
    elif ext == ".csv":
        df.to_csv(path, index=False, encoding="utf-8-sig")
    else:
        raise ValueError(f"Desteklenmeyen çıktı uzantısı: {path}")
    log(f"💾 Yazıldı: {path}")


# =============================================================================
# BASIC HELPERS
# =============================================================================
def normalize_geoid(series: pd.Series, target_len: int = 11) -> pd.Series:
    s = pd.Series(series).astype(str).str.extract(r"(\d+)", expand=False).fillna("")
    s = s.str[:target_len].str.zfill(target_len)
    s = s.mask(s.eq("0" * target_len))
    return s


def ensure_date_col(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    out = df.copy()
    out[col] = pd.to_datetime(out[col], errors="coerce").dt.normalize()
    return out


def ensure_crs(gdf, target="EPSG:4326"):
    if gdf.crs is None:
        return gdf.set_crs(target, allow_override=True)
    try:
        s = gdf.crs.to_string().upper()
    except Exception:
        s = str(gdf.crs).upper()
    if s.endswith("CRS84"):
        return gdf.set_crs("EPSG:4326", allow_override=True)
    if s != target:
        return gdf.to_crs(target)
    return gdf


def read_geojson_robust(path: str) -> gpd.GeoDataFrame:
    try:
        gdf = gpd.read_file(path)
        return ensure_crs(gdf, "EPSG:4326")
    except Exception:
        txt = Path(path).read_text(encoding="utf-8", errors="ignore").strip()
        gj = json.loads(txt)
        if "features" not in gj:
            raise ValueError("features alanı yok")
        gdf = gpd.GeoDataFrame.from_features(gj["features"], crs="EPSG:4326")
        return ensure_crs(gdf, "EPSG:4326")


def parse_tags(val):
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        for loader in (json.loads, ast.literal_eval):
            try:
                x = loader(val)
                if isinstance(x, dict):
                    return x
            except Exception:
                pass
    return {}


def bbox_ok_sf(features, min_lon=-123.5, max_lon=-121.5, min_lat=37.0, max_lat=38.5):
    lons, lats = [], []
    for f in features[: min(200, len(features))]:
        try:
            lon, lat = f["geometry"]["coordinates"]
            lons.append(float(lon))
            lats.append(float(lat))
        except Exception:
            pass
    if not lons or not lats:
        return False
    return (
        min(lons) > min_lon and max(lons) < max_lon
        and min(lats) > min_lat and max(lats) < max_lat
    )


# =============================================================================
# POI CATEGORY MAPPING
# =============================================================================
def extract_cat_sub_name(tags: dict):
    name = tags.get("name")
    for key in ("amenity", "shop", "leisure", "tourism", "office", "craft"):
        if key in tags and tags[key]:
            return key, str(tags[key]).strip().lower(), name
    return None, None, name


def map_poi_group(cat: str | None, sub: str | None) -> str:
    cat = (cat or "").strip().lower()
    sub = (sub or "").strip().lower()

    food_drink = {
        "restaurant", "cafe", "fast_food", "bar", "pub", "food_court",
        "ice_cream", "biergarten"
    }
    nightlife = {"bar", "pub", "nightclub", "casino"}
    retail = {
        "supermarket", "convenience", "mall", "clothes", "department_store",
        "beauty", "shoes", "bakery", "butcher", "kiosk", "electronics",
        "mobile_phone", "jewelry", "alcohol", "beverages", "gift"
    }
    transport = {
        "bus_station", "ferry_terminal", "taxi", "parking", "bicycle_parking",
        "car_rental", "fuel", "charging_station"
    }
    public_service = {
        "bank", "atm", "post_office", "police", "fire_station", "hospital",
        "clinic", "pharmacy", "courthouse", "townhall", "library", "school",
        "college", "university"
    }
    tourism_leisure = {
        "hotel", "hostel", "museum", "gallery", "cinema", "theatre", "park",
        "playground", "sports_centre", "stadium", "pitch", "fitness_centre",
        "swimming_pool", "attraction"
    }
    office_craft = {"office", "craft"}

    if sub in nightlife:
        return "nightlife"
    if sub in food_drink:
        return "food_drink"
    if sub in retail:
        return "retail"
    if sub in transport:
        return "transport"
    if sub in public_service:
        return "public_service"
    if sub in tourism_leisure:
        return "tourism_leisure"
    if cat == "office" or cat == "craft" or sub in office_craft:
        return "office_craft"
    return "other"


# =============================================================================
# POI DOWNLOAD / CLEAN
# =============================================================================
def ensure_sf_pois_geojson(out_path: str):
    if os.path.exists(out_path) and not FORCE_POI_REFRESH:
        try:
            gj = json.loads(Path(out_path).read_text(encoding="utf-8", errors="ignore"))
            feats = gj.get("features", [])
            if feats and bbox_ok_sf(feats):
                log("✅ sf_pois.geojson mevcut, cache kullanılacak.")
                return out_path
        except Exception:
            pass

    extra = ""
    if INCLUDE_OFFICE_CRAFT:
        extra = """
          nwr(area.sf)["office"];
          nwr(area.sf)["craft"];
        """

    query = f"""
    [out:json][timeout:600];
    area(3600111968)->.sf;
    (
      nwr(area.sf)["amenity"];
      nwr(area.sf)["shop"];
      nwr(area.sf)["leisure"];
      nwr(area.sf)["tourism"];
      {extra}
    );
    out center tags;
    """

    endpoints = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://overpass.openstreetmap.ru/api/interpreter",
    ]

    last_err = None
    for ep in endpoints:
        try:
            log(f"🌐 Overpass deneniyor: {ep}")
            r = requests.post(ep, data={"data": query}, timeout=900)
            r.raise_for_status()
            js = r.json()

            feats = []
            for el in js.get("elements", []):
                tags = el.get("tags", {}) or {}

                if el.get("type") == "node":
                    lat, lon = el.get("lat"), el.get("lon")
                else:
                    c = el.get("center") or {}
                    lat, lon = c.get("lat"), c.get("lon")

                if lat is None or lon is None:
                    continue

                feats.append({
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]},
                    "properties": {
                        "id": f"{el.get('type')}/{el.get('id')}",
                        "osm_type": el.get("type"),
                        "osm_id": el.get("id"),
                        "tags": tags,
                    }
                })

            if not feats or not bbox_ok_sf(feats):
                raise RuntimeError("İndirilen veri SF bbox sanity testini geçmedi.")

            gj = {"type": "FeatureCollection", "features": feats}
            ensure_parent(out_path)
            Path(out_path).write_text(json.dumps(gj, ensure_ascii=False), encoding="utf-8")
            log(f"✅ SF POI indirildi: {out_path} | n={len(feats):,}")
            return out_path

        except Exception as e:
            last_err = e
            log(f"⚠️ Endpoint başarısız: {ep} | {e}")

    raise RuntimeError(f"Tüm Overpass endpointleri başarısız oldu: {last_err}")


def build_poi_clean_with_geoid(blocks_path: str, poi_geojson_path: str) -> pd.DataFrame:
    log("📍 POI okunuyor ve GEOID atanıyor...")

    poi = read_geojson_robust(poi_geojson_path)
    if "tags" not in poi.columns:
        poi["tags"] = [{}] * len(poi)
    poi["tags"] = poi["tags"].apply(parse_tags)

    triples = poi["tags"].apply(extract_cat_sub_name)
    poi[["poi_category", "poi_subcategory", "poi_name"]] = pd.DataFrame(triples.tolist(), index=poi.index)

    if "geometry" not in poi.columns:
        raise ValueError("POI geometry bulunamadı")

    poi = ensure_crs(poi, "EPSG:4326")
    poi["lon"] = poi.geometry.x
    poi["lat"] = poi.geometry.y
    poi["poi_group"] = [
        map_poi_group(c, s) for c, s in zip(poi["poi_category"], poi["poi_subcategory"])
    ]

    blocks = read_geojson_robust(blocks_path)
    if "GEOID" not in blocks.columns:
        raise ValueError("Block dosyasında GEOID yok")

    blocks["GEOID"] = normalize_geoid(blocks["GEOID"], GEOID_LEN)

    try:
        joined = gpd.sjoin(
            poi,
            blocks[["GEOID", "geometry"]],
            how="left",
            predicate="within"
        )
    except Exception as e:
        log(f"⚠️ gpd.sjoin başarısız, STRtree fallback: {e}")
        if STRtree is None:
            raise RuntimeError("STRtree yok. shapely>=2 gerekli.")

        geoms = list(blocks.geometry.values)
        tree = STRtree(geoms)
        geom_id_to_geoid = {id(g): geoid for g, geoid in zip(geoms, blocks["GEOID"])}

        geoids = []
        for pt in poi.geometry.values:
            try:
                cands = tree.query(pt, predicate="contains")
            except TypeError:
                cands = [g for g in tree.query(pt) if g.contains(pt)]
            geoids.append(geom_id_to_geoid[id(cands[0])] if len(cands) else None)

        joined = poi.copy()
        joined["GEOID"] = geoids

    keep = [
        c for c in [
            "id", "lat", "lon", "poi_category", "poi_subcategory",
            "poi_name", "poi_group", "GEOID"
        ] if c in joined.columns
    ]
    out = joined[keep].copy()
    out["GEOID"] = normalize_geoid(out["GEOID"], GEOID_LEN)
    out = out.dropna(subset=["GEOID", "lat", "lon"]).copy()

    if "id" not in out.columns:
        out["id"] = np.arange(len(out))

    out = out.drop_duplicates(subset=["id"], keep="first").reset_index(drop=True)
    out.to_csv(POI_CLEAN_CSV, index=False, encoding="utf-8-sig")
    log(f"💾 POI clean yazıldı: {POI_CLEAN_CSV}")
    log_shape(out, "POI CLEAN")
    return out


# =============================================================================
# DYNAMIC POI RISK
# =============================================================================
def compute_dynamic_poi_risk(df_crime: pd.DataFrame, df_poi: pd.DataFrame, radius_m: float = 300.0) -> dict:
    dfp = df_poi.copy()
    dfp["lat"] = pd.to_numeric(dfp["lat"], errors="coerce")
    dfp["lon"] = pd.to_numeric(dfp["lon"], errors="coerce")
    dfp = dfp.dropna(subset=["lat", "lon"]).copy()

    lat_candidates = ["latitude", "lat", "y"]
    lon_candidates = ["longitude", "lon", "long", "x"]

    lat_col = next((c for c in lat_candidates if c in df_crime.columns), None)
    lon_col = next((c for c in lon_candidates if c in df_crime.columns), None)

    if lat_col is None or lon_col is None:
        log("⚠️ Crime verisinde lat/lon yok. POI risk sözlüğü sıfır dönecek.")
        with open(POI_RISK_JSON, "w", encoding="utf-8") as f:
            json.dump({}, f, ensure_ascii=False, indent=2)
        return {}

    dfc = df_crime.copy()
    dfc["latitude"] = pd.to_numeric(dfc[lat_col], errors="coerce")
    dfc["longitude"] = pd.to_numeric(dfc[lon_col], errors="coerce")
    dfc = dfc.dropna(subset=["latitude", "longitude"]).copy()

    if dfc.empty or dfp.empty:
        with open(POI_RISK_JSON, "w", encoding="utf-8") as f:
            json.dump({}, f, ensure_ascii=False, indent=2)
        return {}

    try:
        from sklearn.neighbors import BallTree
    except Exception as e:
        raise RuntimeError(f"scikit-learn gerekli: {e}")

    crime_rad = np.radians(dfc[["latitude", "longitude"]].values)
    poi_rad = np.radians(dfp[["lat", "lon"]].values)

    tree = BallTree(crime_rad, metric="haversine")
    r = radius_m / 6371000.0

    counts = []
    for pt, sub in zip(poi_rad, dfp["poi_subcategory"].fillna("").astype(str)):
        if not sub:
            continue
        idx = tree.query_radius([pt], r=r)[0]
        counts.append((sub, len(idx)))

    agg = defaultdict(list)
    for sub, cnt in counts:
        agg[sub].append(cnt)

    if not agg:
        with open(POI_RISK_JSON, "w", encoding="utf-8") as f:
            json.dump({}, f, ensure_ascii=False, indent=2)
        return {}

    avg = {k: float(np.mean(v)) for k, v in agg.items()}
    vals = list(avg.values())
    vmin, vmax = min(vals), max(vals)

    if abs(vmax - vmin) < 1e-12:
        risk = {k: 1.0 for k in avg}
    else:
        risk = {k: round(3.0 * (v - vmin) / (vmax - vmin), 4) for k, v in avg.items()}

    with open(POI_RISK_JSON, "w", encoding="utf-8") as f:
        json.dump(risk, f, ensure_ascii=False, indent=2)

    log(f"✅ POI risk sözlüğü yazıldı: {POI_RISK_JSON} | n={len(risk)}")
    return risk


# =============================================================================
# GEOID SUMMARY
# =============================================================================
def build_geoid_poi_summary(df_poi: pd.DataFrame, risk_dict: dict) -> pd.DataFrame:
    df = df_poi.copy()
    df["GEOID"] = normalize_geoid(df["GEOID"], GEOID_LEN)
    df["poi_subcategory"] = df["poi_subcategory"].fillna("").astype(str)
    df["poi_group"] = df["poi_group"].fillna("other").astype(str)
    df["risk_w"] = df["poi_subcategory"].map(risk_dict).fillna(0.0).astype(float)

    grp = df.groupby("GEOID", dropna=False)

    out = grp.size().rename("poi_total_count").reset_index()

    uniq_sub = grp["poi_subcategory"].nunique().rename("poi_diversity").reset_index()
    out = out.merge(uniq_sub, on="GEOID", how="left")

    risk_sum = grp["risk_w"].sum().rename("poi_risk_score").reset_index()
    out = out.merge(risk_sum, on="GEOID", how="left")

    # dominant type
    def mode_safe(arr):
        arr = [x for x in arr if pd.notna(x) and str(x).strip() != ""]
        if not arr:
            return "No_POI"
        return Counter(arr).most_common(1)[0][0]

    dominant = grp["poi_subcategory"].agg(mode_safe).rename("poi_dominant_type").reset_index()
    out = out.merge(dominant, on="GEOID", how="left")

    # group counts
    pivot = (
        df.assign(v=1)
          .pivot_table(index="GEOID", columns="poi_group", values="v", aggfunc="sum", fill_value=0)
          .reset_index()
    )

    wanted_groups = [
        "food_drink",
        "nightlife",
        "retail",
        "transport",
        "public_service",
        "tourism_leisure",
        "office_craft",
        "other",
    ]
    for g in wanted_groups:
        if g not in pivot.columns:
            pivot[g] = 0

    pivot = pivot.rename(columns={
        "food_drink": "poi_food_drink_count",
        "nightlife": "poi_nightlife_count",
        "retail": "poi_retail_count",
        "transport": "poi_transport_count",
        "public_service": "poi_public_service_count",
        "tourism_leisure": "poi_tourism_leisure_count",
        "office_craft": "poi_office_craft_count",
        "other": "poi_other_count",
    })

    out = out.merge(pivot, on="GEOID", how="left")

    fill_zero_cols = [c for c in out.columns if c.startswith("poi_") and c.endswith("_count")]
    fill_zero_cols += ["poi_total_count", "poi_diversity", "poi_risk_score"]
    for c in fill_zero_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)

    # simple concentration measures
    out["poi_food_share"] = np.where(out["poi_total_count"] > 0, out["poi_food_drink_count"] / out["poi_total_count"], 0.0)
    out["poi_nightlife_share"] = np.where(out["poi_total_count"] > 0, out["poi_nightlife_count"] / out["poi_total_count"], 0.0)
    out["poi_retail_share"] = np.where(out["poi_total_count"] > 0, out["poi_retail_count"] / out["poi_total_count"], 0.0)
    out["poi_transport_share"] = np.where(out["poi_total_count"] > 0, out["poi_transport_count"] / out["poi_total_count"], 0.0)

    out = out.sort_values("GEOID").drop_duplicates(subset=["GEOID"], keep="first").reset_index(drop=True)
    out.to_csv(POI_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    log(f"💾 POI summary yazıldı: {POI_SUMMARY_CSV}")
    log_shape(out, "POI SUMMARY")
    return out


# =============================================================================
# TIME FLAGS
# =============================================================================
def ensure_time_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "event_hour" not in out.columns:
        if "hour_range" in out.columns:
            # "00:00-02:59" -> 0, "03:00-05:59" -> 3 vb.
            hr = out["hour_range"].astype(str).str.extract(r"^(\d{1,2})", expand=False)
            out["event_hour"] = pd.to_numeric(hr, errors="coerce")
        else:
            out["event_hour"] = np.nan

    if "day_of_week" not in out.columns and "date" in out.columns:
        out["day_of_week"] = pd.to_datetime(out["date"], errors="coerce").dt.dayofweek

    if "is_weekend" not in out.columns:
        if "day_of_week" in out.columns:
            out["is_weekend"] = out["day_of_week"].isin([5, 6]).astype(int)
        else:
            out["is_weekend"] = 0

    if "is_night" not in out.columns:
        eh = pd.to_numeric(out.get("event_hour"), errors="coerce")
        out["is_night"] = eh.isin([0, 1, 2, 3, 4, 5, 22, 23]).astype(int)

    if "is_business_hour" not in out.columns:
        eh = pd.to_numeric(out.get("event_hour"), errors="coerce")
        out["is_business_hour"] = eh.between(8, 17, inclusive="both").fillna(False).astype(int)

    if "is_school_hour" not in out.columns:
        eh = pd.to_numeric(out.get("event_hour"), errors="coerce")
        out["is_school_hour"] = eh.between(8, 15, inclusive="both").fillna(False).astype(int)

    return out


# =============================================================================
# MERGE + INTERACTIONS
# =============================================================================
def enrich_crime_with_poi(df_crime: pd.DataFrame, geoid_poi: pd.DataFrame) -> pd.DataFrame:
    out = df_crime.copy()
    out["GEOID"] = normalize_geoid(out["GEOID"], GEOID_LEN)
    out = ensure_date_col(out, "date")
    out = ensure_time_flags(out)

    # varsa eski poi kolonlarını sil
    old_poi_cols = [c for c in out.columns if c.startswith("poi_")]
    old_poi_cols += ["poi_dominant_type"]
    old_poi_cols = sorted(set([c for c in old_poi_cols if c in out.columns]))
    if old_poi_cols:
        out = out.drop(columns=old_poi_cols, errors="ignore")

    geoid_poi = geoid_poi.copy()
    geoid_poi["GEOID"] = normalize_geoid(geoid_poi["GEOID"], GEOID_LEN)

    out = out.merge(geoid_poi, on="GEOID", how="left")

    # fill
    num_cols = [c for c in out.columns if c.startswith("poi_") and c != "poi_dominant_type"]
    for c in num_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)

    if "poi_dominant_type" in out.columns:
        out["poi_dominant_type"] = out["poi_dominant_type"].fillna("No_POI").astype(str)

    # interaction features
    out["poi_risk_night"] = out["poi_risk_score"] * out["is_night"]
    out["poi_risk_weekend"] = out["poi_risk_score"] * out["is_weekend"]
    out["poi_risk_business_hour"] = out["poi_risk_score"] * out["is_business_hour"]

    out["poi_food_night"] = out["poi_food_drink_count"] * out["is_night"]
    out["poi_food_weekend"] = out["poi_food_drink_count"] * out["is_weekend"]

    out["poi_nightlife_night"] = out["poi_nightlife_count"] * out["is_night"]
    out["poi_nightlife_weekend"] = out["poi_nightlife_count"] * out["is_weekend"]

    out["poi_retail_business_hour"] = out["poi_retail_count"] * out["is_business_hour"]
    out["poi_transport_business_hour"] = out["poi_transport_count"] * out["is_business_hour"]
    out["poi_public_service_business_hour"] = out["poi_public_service_count"] * out["is_business_hour"]

    out["poi_leisure_weekend"] = out["poi_tourism_leisure_count"] * out["is_weekend"]
    out["poi_office_business_hour"] = out["poi_office_craft_count"] * out["is_business_hour"]

    # nonlinear stabilizer
    for c in [
        "poi_total_count", "poi_diversity", "poi_risk_score",
        "poi_food_drink_count", "poi_nightlife_count", "poi_retail_count",
        "poi_transport_count", "poi_public_service_count",
        "poi_tourism_leisure_count", "poi_office_craft_count"
    ]:
        if c in out.columns:
            out[f"log1p_{c}"] = np.log1p(pd.to_numeric(out[c], errors="coerce").fillna(0))

    # güvenlik
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    return out


# =============================================================================
# MAIN
# =============================================================================
def main():
    log("🚀 POI enrich FINAL başlıyor...")

    # -------------------------------------------------------------------------
    # 1) crime input
    # -------------------------------------------------------------------------
    if not os.path.exists(CRIME_IN):
        # fallback
        for cand in [
            os.path.join(BASE_DIR, "sf_crime_05.csv"),
        ]:
             if os.path.exists(cand):
                        crime_in_path = cand
                        break
            
    df_crime = read_table(crime_in_path)
    
    if "GEOID" not in df_crime.columns:
        raise KeyError("Crime input içinde GEOID yok")
    if "date" not in df_crime.columns:
        raise KeyError("Crime input içinde date yok")

    df_crime["GEOID"] = normalize_geoid(df_crime["GEOID"], GEOID_LEN)
    df_crime = ensure_date_col(df_crime, "date")
    log_shape(df_crime, "CRIME INPUT")

    # -------------------------------------------------------------------------
    # 2) poi source
    # -------------------------------------------------------------------------
    use_clean = os.path.exists(POI_CLEAN_CSV) and not FORCE_POI_REFRESH
    if use_clean:
        log(f"✅ Mevcut temiz POI kullanılacak: {POI_CLEAN_CSV}")
        df_poi = pd.read_csv(POI_CLEAN_CSV, low_memory=False)
        if "poi_group" not in df_poi.columns:
            df_poi["poi_group"] = [
                map_poi_group(c, s)
                for c, s in zip(df_poi.get("poi_category"), df_poi.get("poi_subcategory"))
            ]
        df_poi["GEOID"] = normalize_geoid(df_poi["GEOID"], GEOID_LEN)
    else:
        poi_geojson = ensure_sf_pois_geojson(POI_GEOJSON)
        if not os.path.exists(BLOCKS_GEOJSON):
            raise FileNotFoundError(f"Block GEOJSON yok: {BLOCKS_GEOJSON}")
        df_poi = build_poi_clean_with_geoid(BLOCKS_GEOJSON, poi_geojson)

    # -------------------------------------------------------------------------
    # 3) risk universe
    # -------------------------------------------------------------------------
    risk_df = df_crime.copy()
    max_date = risk_df["date"].max()
    if pd.notna(max_date):
        min_date = max_date - pd.DateOffset(years=POI_RISK_LOOKBACK_YEARS)
        risk_df = risk_df[(risk_df["date"] >= min_date) & (risk_df["date"] <= max_date)].copy()
        log(f"🕒 Risk lookback: {min_date.date()} → {max_date.date()}")

    # -------------------------------------------------------------------------
    # 4) dynamic risk
    # -------------------------------------------------------------------------
    risk_dict = compute_dynamic_poi_risk(risk_df, df_poi, radius_m=POI_RISK_RADIUS_M)

    # -------------------------------------------------------------------------
    # 5) geoid summary
    # -------------------------------------------------------------------------
    geoid_poi = build_geoid_poi_summary(df_poi, risk_dict)

    # -------------------------------------------------------------------------
    # 6) merge + interactions
    # -------------------------------------------------------------------------
    final_df = enrich_crime_with_poi(df_crime, geoid_poi)
    log_shape(final_df, "FINAL sf_crime_06")

    # -------------------------------------------------------------------------
    # 7) save
    # -------------------------------------------------------------------------
    write_table(final_df, CRIME_OUT)

    if WRITE_CSV_ALSO:
        csv_out = str(Path(CRIME_OUT).with_suffix(".csv"))
        final_df.to_csv(csv_out, index=False, encoding="utf-8-sig")
        log(f"💾 CSV de yazıldı: {csv_out}")

    # quick preview
    preview_cols = [c for c in [
        "GEOID", "date", "hour_range", "event_hour",
        "poi_total_count", "poi_diversity", "poi_risk_score",
        "poi_food_drink_count", "poi_nightlife_count",
        "poi_retail_count", "poi_transport_count",
        "poi_risk_night", "poi_food_night", "poi_leisure_weekend"
    ] if c in final_df.columns]

    log("📌 Önizleme:")
    print(final_df[preview_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
