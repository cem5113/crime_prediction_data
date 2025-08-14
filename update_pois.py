import os
import ast
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from collections import defaultdict
from sklearn.neighbors import BallTree

BASE_DIR = "crime_data"
POI_GEOJSON = os.path.join(BASE_DIR, "sf_pois.geojson")
BLOCK_PATH = os.path.join(BASE_DIR, "sf_census_blocks_with_population.geojson")
POI_CLEANED_CSV = os.path.join(BASE_DIR, "sf_pois_cleaned_with_geoid.csv")
POI_RISK_JSON = os.path.join(BASE_DIR, "risky_pois_dynamic.json")
CRIME_INPUT = os.path.join(BASE_DIR, "sf_crime_05.csv")
CRIME_OUTPUT = os.path.join(BASE_DIR, "sf_crime_06.csv")

# --- Yardımcı Fonksiyonlar ---
def _parse_tags(val):
    if isinstance(val, dict): return val
    if isinstance(val, str):
        for loader in (json.loads, ast.literal_eval):
            try:
                out = loader(val)
                return out if isinstance(out, dict) else {}
            except: pass
    return {}

def _ensure_crs(gdf, target="EPSG:4326"):
    if gdf.crs is None: return gdf.set_crs(target, allow_override=True)
    if gdf.crs.to_string().upper().endswith("CRS84"): return gdf.set_crs("EPSG:4326", allow_override=True)
    if gdf.crs.to_string() != target: return gdf.to_crs(target)
    return gdf

def _extract_cat_sub_name_from_tags(tags: dict):
    name = tags.get("name")
    for key in ("amenity", "shop", "leisure"):
        if key in tags and tags[key]:
            return key, tags[key], name
    return None, None, name

# --- 1. POI Temizleme + GEOID Ekleme ---
def clean_and_assign_geoid_to_pois():
    gdf = gpd.read_file(POI_GEOJSON)
    gdf = _ensure_crs(gdf)

    if "tags" not in gdf.columns:
        gdf["tags"] = [{}] * len(gdf)
    gdf["tags"] = gdf["tags"].apply(_parse_tags)

    cat_sub_name = gdf["tags"].apply(_extract_cat_sub_name_from_tags)
    gdf[["poi_category", "poi_subcategory", "poi_name"]] = pd.DataFrame(cat_sub_name.tolist(), index=gdf.index)

    if "geometry" not in gdf.columns:
        if {"lon", "lat"}.issubset(gdf.columns):
            gdf["geometry"] = gpd.points_from_xy(gdf["lon"], gdf["lat"])
        else:
            raise ValueError("GeoJSON içinde 'geometry' yok")
    gdf = _ensure_crs(gdf)
    gdf["lon"] = gdf.geometry.x
    gdf["lat"] = gdf.geometry.y

    gdf_blocks = gpd.read_file(BLOCK_PATH)
    gdf_blocks = _ensure_crs(gdf_blocks)
    gdf_blocks["GEOID"] = gdf_blocks["GEOID"].astype(str).str.zfill(11)

    gdf_joined = gpd.sjoin(gdf, gdf_blocks[["GEOID", "geometry"]], how="left", predicate="within")
    keep_cols = ["id", "lat", "lon", "poi_category", "poi_subcategory", "poi_name", "GEOID"]
    df_cleaned = gdf_joined[[c for c in keep_cols if c in gdf_joined.columns]].copy()

    if "id" not in df_cleaned.columns:
        df_cleaned["id"] = np.arange(len(df_cleaned))

    df_cleaned = df_cleaned.dropna(subset=["lat", "lon"])
    df_cleaned.to_csv(POI_CLEANED_CSV, index=False)
    return df_cleaned

# --- 2. Dinamik Risk Skoru ---
def calculate_dynamic_risk(df_crime, df_poi):
    dfc = df_crime.dropna(subset=["latitude", "longitude"]).copy()
    dfc[["latitude", "longitude"]] = dfc[["latitude", "longitude"]].apply(pd.to_numeric, errors="coerce").dropna()
    dfp = df_poi.dropna(subset=["lat", "lon"]).copy()
    dfp[["lat", "lon"]] = dfp[["lat", "lon"]].apply(pd.to_numeric, errors="coerce").dropna()
    dfp = dfp[~dfp["poi_subcategory"].isin(["police", "ranger_station"])]

    crime_rad = np.radians(dfc[["latitude", "longitude"]].values)
    poi_rad = np.radians(dfp[["lat", "lon"]].values)
    tree = BallTree(crime_rad, metric="haversine")
    radius = 300 / 6371000.0

    poi_crime_counts = []
    for pt, subtype in zip(poi_rad, dfp["poi_subcategory"].fillna("")):
        if not subtype: continue
        idx = tree.query_radius([pt], r=radius)[0]
        poi_crime_counts.append((subtype, len(idx)))

    poi_risk_raw = defaultdict(list)
    for subtype, count in poi_crime_counts:
        poi_risk_raw[subtype].append(count)
    poi_risk_avg = {k: np.mean(v) for k, v in poi_risk_raw.items()}

    vals = list(poi_risk_avg.values())
    vmin, vmax = min(vals), max(vals)
    poi_risk_normalized = {k: round(3.0 * (v - vmin) / (vmax - vmin), 2) if vmax != vmin else 1.5 for k, v in poi_risk_avg.items()}

    with open(POI_RISK_JSON, "w") as f:
        json.dump(poi_risk_normalized, f, indent=2)

    return poi_risk_normalized

# --- 3. Dinamik Kategorileme ---
def make_dynamic_range_func(series):
    vals = pd.to_numeric(series, errors="coerce").dropna().values
    if len(vals) == 0:
        return lambda _: "Q1 (0-0)"
    qs = np.quantile(vals, [i / 5 for i in range(6)])
    def label(x):
        if pd.isna(x): return f"Q1 ({qs[0]:.1f}-{qs[1]:.1f})"
        for i in range(5):
            if x <= qs[i+1]: return f"Q{i+1} ({qs[i]:.1f}-{qs[i+1]:.1f})"
        return f"Q5 ({qs[-2]:.1f}-{qs[-1]:.1f})"
    return label

# --- 4. Suç Verisini POI ile Zenginleştirme ---
def enrich_crime_with_poi(df_crime, df_poi, poi_risk_scores):
    df_poi["risk_score"] = df_poi["poi_subcategory"].map(poi_risk_scores).fillna(0.0)
    poi_summary = df_poi.groupby("GEOID").agg(
        poi_total_count=("poi_subcategory", "count"),
        poi_dominant_type=("poi_subcategory", lambda x: x.mode()[0] if not x.mode().empty else "No_POI"),
        poi_risk_score=("risk_score", "sum")
    ).reset_index()

    poi_summary["poi_total_count_range"] = poi_summary["poi_total_count"].apply(make_dynamic_range_func(poi_summary["poi_total_count"]))
    poi_summary["poi_risk_score_range"] = poi_summary["poi_risk_score"].apply(make_dynamic_range_func(poi_summary["poi_risk_score"]))

    df_crime["GEOID"] = df_crime["GEOID"].astype(str).str.zfill(11)
    df_out = df_crime.merge(poi_summary, on="GEOID", how="left")
    df_out.to_csv(CRIME_OUTPUT, index=False)
    return df_out

# --- Ana Akış ---
if __name__ == "__main__":
    df_crime = pd.read_csv(CRIME_INPUT, dtype={"GEOID": str})
    df_poi = clean_and_assign_geoid_to_pois()
    poi_risk_scores = calculate_dynamic_risk(df_crime, df_poi)
    enrich_crime_with_poi(df_crime, df_poi, poi_risk_scores)
