# update_pois.py

import os
import pandas as pd
import numpy as np
import geopandas as gpd
import ast
import json
from sklearn.neighbors import BallTree
from collections import defaultdict

# === Dosya Yolları ===
BASE_PATH = "/content/drive/MyDrive/crime_data"
GEOJSON_POI_PATH = "https://raw.githubusercontent.com/cem5113/crime_prediction_data/main/sf_pois.geojson"
BLOCK_PATH = f"{BASE_PATH}/sf_census_blocks_with_population.geojson"
CLEANED_POI_PATH = f"{BASE_PATH}/sf_pois_cleaned_with_geoid.csv"
CRIME_PATH = f"{BASE_PATH}/sf_crime.csv"
DYNAMIC_JSON_PATH = f"{BASE_PATH}/risky_pois_dynamic.json"
DYNAMIC_CRIME_OUT = f"{BASE_PATH}/sf_poi_risk_crime.csv"
DYNAMIC_POI_OUT = f"{BASE_PATH}/sf_pois_with_risk_score.csv"
MEDIAN_RISK_OUT = f"{BASE_PATH}/sf_poi_geoid_based_risk.csv"
FINAL_MERGE_OUT = f"{BASE_PATH}/sf_crime_08.csv"

# === 1. POI Ayrıştır ve GEOID Ekle ===
def process_pois():
    gdf = gpd.read_file(GEOJSON_POI_PATH)

    def extract_poi_fields(tags):
        if isinstance(tags, str):
            try:
                tags = ast.literal_eval(tags)
            except Exception:
                return pd.Series([None, None, None])
        if isinstance(tags, dict):
            for key in ['amenity', 'shop', 'leisure']:
                if key in tags:
                    return pd.Series([key, tags.get(key), tags.get('name')])
            return pd.Series([None, None, tags.get('name')])
        return pd.Series([None, None, None])

    gdf[['poi_category', 'poi_subcategory', 'poi_name']] = gdf['tags'].apply(extract_poi_fields)

    gdf_blocks = gpd.read_file(BLOCK_PATH)
    gdf_blocks["GEOID"] = gdf_blocks["GEOID"].astype(str).str.zfill(12)

    if 'geometry' not in gdf.columns:
        gdf['geometry'] = gpd.points_from_xy(gdf['lon'], gdf['lat'])
    gdf = gdf.set_geometry('geometry').set_crs("EPSG:4326")

    gdf_joined = gpd.sjoin(gdf, gdf_blocks[['GEOID', 'geometry']], how='left', predicate='within')
    df_cleaned = gdf_joined[['id', 'lat', 'lon', 'poi_category', 'poi_subcategory', 'poi_name', 'GEOID']].copy()
    df_cleaned.to_csv(CLEANED_POI_PATH, index=False)
    print(f"✅ POI verisi ayrıştırıldı ve kaydedildi → {CLEANED_POI_PATH}")

# === 2. Dinamik POI Risk Skoru Hesaplama ===
def calculate_dynamic_risk():
    gdf_crime = pd.read_csv(CRIME_PATH)
    gdf_crime = gpd.GeoDataFrame(
        gdf_crime,
        geometry=gpd.points_from_xy(gdf_crime["longitude"], gdf_crime["latitude"]),
        crs="EPSG:4326"
    )
    gdf_poi = pd.read_csv(CLEANED_POI_PATH)
    gdf_poi = gpd.GeoDataFrame(
        gdf_poi,
        geometry=gpd.points_from_xy(gdf_poi["lon"], gdf_poi["lat"]),
        crs="EPSG:4326"
    )

    gdf_poi = gdf_poi[~gdf_poi["poi_subcategory"].isin(["police", "ranger_station"])]

    poi_rad = np.radians(gdf_poi[["lat", "lon"]].values)
    crime_rad = np.radians(gdf_crime[["latitude", "longitude"]].values)
    tree = BallTree(crime_rad, metric="haversine")
    radius = 300 / 6371000

    poi_types = gdf_poi["poi_subcategory"].fillna("")
    poi_crime_counts = []
    for i, pt in enumerate(poi_rad):
        subtype = poi_types.iloc[i]
        if subtype == "":
            continue
        crime_ids = tree.query_radius([pt], r=radius)[0]
        poi_crime_counts.append((subtype, len(crime_ids)))

    poi_crime_sum = defaultdict(int)
    poi_type_count = defaultdict(int)
    for subtype, count in poi_crime_counts:
        poi_crime_sum[subtype] += count
        poi_type_count[subtype] += 1

    poi_risk_raw = {
        subtype: round(poi_crime_sum[subtype] / poi_type_count[subtype], 4)
        for subtype in poi_crime_sum
    }

    min_val = min(poi_risk_raw.values())
    max_val = max(poi_risk_raw.values())
    poi_risk_normalized = {
        subtype: round(3 * (score - min_val) / (max_val - min_val + 1e-6), 2)
        for subtype, score in poi_risk_raw.items()
    }

    with open(DYNAMIC_JSON_PATH, "w") as f:
        json.dump(poi_risk_normalized, f, indent=2)
    print(f"✅ Dinamik risk skorları kaydedildi → {DYNAMIC_JSON_PATH}")

# === MAIN ===
if __name__ == "__main__":
    print("🔄 POI verisi işleniyor...")
    process_pois()
    calculate_dynamic_risk()
    print("🎯 Tüm POI risk hesaplamaları tamamlandı. Diğer adımlar ayrı scriptlerle devam edecek.")

    # Ek kontroller
    check_output_files()
    
    try:
        df_preview = pd.read_csv(CLEANED_POI_PATH)
        print("📌 İlk 3 POI:")
        print(df_preview[["poi_category", "poi_subcategory", "GEOID"]].dropna().head(3))
    except Exception as e:
        print(f"⚠️ POI örnek verisi gösterilemedi: {e}")

    
