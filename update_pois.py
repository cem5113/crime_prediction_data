# update_pois.py

import os
import ast
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from collections import defaultdict
from shapely.geometry import Point
from sklearn.neighbors import BallTree

# === DOSYA YOLLARI ===
BASE_DIR = "/content/drive/MyDrive/crime_data"
POI_GEOJSON = os.path.join(BASE_DIR, "sf_pois.geojson")
BLOCK_PATH = os.path.join(BASE_DIR, "sf_census_blocks_with_population.geojson")
POI_CLEANED_CSV = os.path.join(BASE_DIR, "sf_pois_cleaned_with_geoid.csv")
POI_RISK_JSON = os.path.join(BASE_DIR, "risky_pois_dynamic.json")
CRIME_INPUT = os.path.join(BASE_DIR, "sf_crime_05.csv")
CRIME_OUTPUT = os.path.join(BASE_DIR, "sf_crime_06.csv")

# === 1. POI VERİSİNİ TEMİZLE + GEOID EKLE ===
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

def clean_and_assign_geoid_to_pois():
    print("📍 POI verisi indiriliyor ve GEOID atanıyor...")
    gdf = gpd.read_file(POI_GEOJSON)
    gdf[['poi_category', 'poi_subcategory', 'poi_name']] = gdf['tags'].apply(extract_poi_fields)
    if 'geometry' not in gdf.columns:
        gdf['geometry'] = gpd.points_from_xy(gdf['lon'], gdf['lat'])
    gdf = gdf.set_geometry('geometry').set_crs("EPSG:4326")

    gdf_blocks = gpd.read_file(BLOCK_PATH)
    gdf_blocks["GEOID"] = gdf_blocks["GEOID"].astype(str).str.zfill(12)
    gdf_joined = gpd.sjoin(gdf, gdf_blocks[['GEOID', 'geometry']], how='left', predicate='within')

    df_cleaned = gdf_joined[['id', 'lat', 'lon', 'poi_category', 'poi_subcategory', 'poi_name', 'GEOID']].copy()
    df_cleaned.to_csv(POI_CLEANED_CSV, index=False)
    print(f"✅ Temizlenmiş POI verisi kaydedildi: {POI_CLEANED_CSV}")
    return df_cleaned


# === 2. POI RISK SKORU HESAPLA ===
def calculate_dynamic_risk(df_crime, df_poi):
    print("📊 Dinamik POI risk skorları hesaplanıyor...")
    gdf_crime = gpd.GeoDataFrame(
        df_crime, geometry=gpd.points_from_xy(df_crime["longitude"], df_crime["latitude"]), crs="EPSG:4326"
    )
    gdf_poi = gpd.GeoDataFrame(
        df_poi, geometry=gpd.points_from_xy(df_poi["lon"], df_poi["lat"]), crs="EPSG:4326"
    )
    gdf_poi = gdf_poi[~gdf_poi["poi_subcategory"].isin(["police", "ranger_station"])]

    poi_rad = np.radians(gdf_poi[["lat", "lon"]].astype(float).values)
    crime_rad = np.radians(gdf_crime[["latitude", "longitude"]].astype(float).values)
    tree = BallTree(crime_rad, metric="haversine")
    radius = 300 / 6371000  # 300m

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

    with open(POI_RISK_JSON, "w") as f:
        json.dump(poi_risk_normalized, f, indent=2)

    print("📊 Dynamically calculated POI risk scores (normalized 0–3 scale):\n")
    for subtype, score in sorted(poi_risk_normalized.items(), key=lambda x: x[1], reverse=True):
        raw = poi_risk_raw[subtype]
        print(f"{subtype:<25} → raw avg crime: {raw:.2f} | normalized score: {score:.2f}")
    return poi_risk_normalized


# === 3. DİNAMİK KATEGORİLEME (Q1–Q4) ===
def make_dynamic_range_func(values, strategy="auto", max_bins=5):
    values = np.array(values.dropna())
    n = len(values)
    std = np.std(values)
    iqr = np.percentile(values, 75) - np.percentile(values, 25)

    if strategy == "auto":
        if n < 500:
            bin_count = 3
        elif std < 1 or iqr < 1:
            bin_count = 4
        elif std > 20:
            bin_count = min(10, max_bins)
        else:
            bin_count = 5
    else:
        bin_count = int(strategy)

    quantiles = np.quantile(values, [i / bin_count for i in range(bin_count + 1)])

    def label(x):
        for i in range(bin_count):
            if x <= quantiles[i + 1]:
                return f"Q{i+1} ({quantiles[i]:.1f}-{quantiles[i+1]:.1f})"
        return f"Q{bin_count+1} (>{quantiles[-1]:.1f})"
    return label


# === 4. SUÇ VERİSİNE POI ÖZELLİKLERİNİ ATA ===
def enrich_crime_with_poi(df_crime, df_poi, poi_risk_scores):
    print("🔗 POI verileri suç verisiyle eşleştiriliyor...")
    df_poi["risk_score"] = df_poi["poi_subcategory"].map(poi_risk_scores).fillna(0)

    gdf_crime = gpd.GeoDataFrame(
        df_crime, geometry=gpd.points_from_xy(df_crime["longitude"], df_crime["latitude"]), crs="EPSG:4326"
    )
    gdf_poi = gpd.GeoDataFrame(
        df_poi, geometry=gpd.points_from_xy(df_poi["lon"], df_poi["lat"]), crs="EPSG:4326"
    )

    crime_rad = np.radians(gdf_crime[["latitude", "longitude"]].values)
    poi_rad = np.radians(gdf_poi[["lat", "lon"]].values)
    tree = BallTree(poi_rad, metric="haversine")
    radius = 300 / 6371000

    indices = tree.query_radius(crime_rad, r=radius)

    poi_total_count = []
    poi_risk_score = []
    poi_dominant_type = []

    poi_types = gdf_poi["poi_subcategory"].fillna("")

    for idx_list in indices:
        subtypes = poi_types.iloc[idx_list]
        subtype_counts = subtypes.value_counts()
        poi_total_count.append(len(idx_list))
        risk_vals = subtypes.map(poi_risk_scores).fillna(0)
        poi_risk_score.append(risk_vals.sum())
        poi_dominant_type.append(subtype_counts.idxmax() if not subtype_counts.empty else "No_POI")

    gdf_crime["poi_total_count"] = poi_total_count
    gdf_crime["poi_risk_score"] = poi_risk_score
    gdf_crime["poi_dominant_type"] = poi_dominant_type

    # Range kategorileri
    count_label_func = make_dynamic_range_func(gdf_crime["poi_total_count"])
    score_label_func = make_dynamic_range_func(gdf_crime["poi_risk_score"])

    gdf_crime["poi_total_count_range"] = gdf_crime["poi_total_count"].apply(count_label_func)
    gdf_crime["poi_risk_score_range"] = gdf_crime["poi_risk_score"].apply(score_label_func)

    df_result = gdf_crime.drop(columns="geometry")
    df_result.to_csv(CRIME_OUTPUT, index=False)
    print(f"✅ Zenginleştirilmiş suç verisi kaydedildi: {CRIME_OUTPUT}")
    print("📄 Columns added to crime data: ['poi_total_count', 'poi_risk_score', 'poi_dominant_type', 'poi_total_count_range', 'poi_risk_score_range']")
    return df_result


# === ANA AKIŞ ===
if __name__ == "__main__":
    print("🚀 POI güncelleme ve risk analizi başlatılıyor...")
    df_crime = pd.read_csv(CRIME_INPUT)
    df_poi = clean_and_assign_geoid_to_pois()
    poi_risk_scores = calculate_dynamic_risk(df_crime, df_poi)
    enrich_crime_with_poi(df_crime, df_poi, poi_risk_scores)
