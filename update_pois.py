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
BASE_PATH = os.getcwd()
GEOJSON_POI_PATH = "https://github.com/cem5113/crime_prediction_data/raw/main/sf_pois.geojson"
BLOCK_PATH = os.path.join(BASE_PATH, "sf_census_blocks_with_population.geojson")
CLEANED_POI_PATH = os.path.join(BASE_PATH, "sf_pois_cleaned_with_geoid.csv")
CRIME_PATH = os.path.join(BASE_PATH, "sf_crime.csv")
DYNAMIC_JSON_PATH = os.path.join(BASE_PATH, "risky_pois_dynamic.json")

# === 1. POI Verisini Ayrıştır ve GEOID Ekle ===
def process_pois():
    print("🔹 POI verisi indiriliyor ve işleniyor...")
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
    print(f"✅ POI verisi kaydedildi: {CLEANED_POI_PATH}")

# === 2. Risk Skoru Hesapla ===
def calculate_dynamic_risk():
    print("🔹 Risk skoru hesaplanıyor...")
    df_crime = pd.read_csv(CRIME_PATH)
    gdf_crime = gpd.GeoDataFrame(
        df_crime,
        geometry=gpd.points_from_xy(df_crime["longitude"], df_crime["latitude"]),
        crs="EPSG:4326"
    )

    df_poi = pd.read_csv(CLEANED_POI_PATH)
    gdf_poi = gpd.GeoDataFrame(
        df_poi,
        geometry=gpd.points_from_xy(df_poi["lon"], df_poi["lat"]),
        crs="EPSG:4326"
    )

    # Polis istasyonlarını hariç tut
    gdf_poi = gdf_poi[~gdf_poi["poi_subcategory"].isin(["police", "ranger_station"])]

    poi_rad = np.radians(gdf_poi[["lat", "lon"]].values)
    crime_rad = np.radians(gdf_crime[["latitude", "longitude"]].values)
    tree = BallTree(crime_rad, metric="haversine")
    radius = 300 / 6371000  # 300 metre

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
    print(f"✅ Dinamik risk skorları kaydedildi: {DYNAMIC_JSON_PATH}")

# === MAIN ===
if __name__ == "__main__":
    print("📦 POI güncelleme başlatıldı...")
    process_pois()
    calculate_dynamic_risk()

    print("\n📁 Kontrol ediliyor...")
    if os.path.exists(CLEANED_POI_PATH):
        print(f"✅ {CLEANED_POI_PATH}")
    if os.path.exists(DYNAMIC_JSON_PATH):
        print(f"✅ {DYNAMIC_JSON_PATH}")

    try:
        df_preview = pd.read_csv(CLEANED_POI_PATH)
        print("\n📌 Örnek POI kayıtları:")
        print(df_preview[["poi_category", "poi_subcategory", "GEOID"]].dropna().head(3))
    except Exception as e:
        print(f"⚠️ Örnek POI verisi gösterilemedi: {e}")
