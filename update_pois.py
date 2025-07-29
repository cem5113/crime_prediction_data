import os
import ast
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from collections import defaultdict
from shapely.geometry import Point
from sklearn.neighbors import BallTree

# === 0. DOSYA YOLLARI ===
BASE_DIR = "crime_data"
POI_GEOJSON = os.path.join(BASE_DIR, "sf_pois.geojson")
BLOCK_PATH = os.path.join(BASE_DIR, "sf_census_blocks_with_population.geojson")
POI_CLEANED_CSV = os.path.join(BASE_DIR, "sf_pois_cleaned_with_geoid.csv")
POI_RISK_JSON = os.path.join(BASE_DIR, "risky_pois_dynamic.json")
CRIME_INPUT = os.path.join(BASE_DIR, "sf_crime_05.csv")
CRIME_OUTPUT = os.path.join(BASE_DIR, "sf_crime_06.csv")


# === 1. POI VERİSİ TEMİZLEME VE GEOID EKLEME ===
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
    print("📍 POI verisi okunuyor ve GEOID atanıyor...")
    gdf = gpd.read_file(POI_GEOJSON)
    gdf[['poi_category', 'poi_subcategory', 'poi_name']] = gdf['tags'].apply(extract_poi_fields)

    if 'geometry' not in gdf.columns:
        gdf['geometry'] = gpd.points_from_xy(gdf['lon'], gdf['lat'])
    gdf = gdf.set_geometry('geometry').set_crs("EPSG:4326")

    gdf_blocks = gpd.read_file(BLOCK_PATH)
    gdf_blocks["GEOID"] = gdf_blocks["GEOID"].astype(str).str.zfill(11)

    gdf_joined = gpd.sjoin(gdf, gdf_blocks[['GEOID', 'geometry']], how='left', predicate='within')

    df_cleaned = gdf_joined[['id', 'lat', 'lon', 'poi_category', 'poi_subcategory', 'poi_name', 'GEOID']].copy()
    df_cleaned.to_csv(POI_CLEANED_CSV, index=False)
    print(f"✅ Temizlenmiş POI verisi kaydedildi: {POI_CLEANED_CSV}")
    return df_cleaned


# === 2. DİNAMİK RİSK SKORU HESAPLAMA ===
def calculate_dynamic_risk(df_crime, df_poi):
    print("📊 POI risk skorları hesaplanıyor (300m yarıçap)...")
    gdf_crime = gpd.GeoDataFrame(df_crime, geometry=gpd.points_from_xy(df_crime["longitude"], df_crime["latitude"]), crs="EPSG:4326")
    gdf_poi = gpd.GeoDataFrame(df_poi, geometry=gpd.points_from_xy(df_poi["lon"], df_poi["lat"]), crs="EPSG:4326")
    gdf_poi = gdf_poi[~gdf_poi["poi_subcategory"].isin(["police", "ranger_station"])]

    poi_rad = np.radians(gdf_poi[["lat", "lon"]].astype(float).values)
    crime_rad = np.radians(gdf_crime[["latitude", "longitude"]].astype(float).values)
    tree = BallTree(crime_rad, metric="haversine")
    radius = 300 / 6371000  # 300m in radians

    poi_types = gdf_poi["poi_subcategory"].fillna("")
    poi_crime_counts = [
        (subtype, len(tree.query_radius([pt], r=radius)[0]))
        for pt, subtype in zip(poi_rad, poi_types) if subtype
    ]

    poi_risk_raw = defaultdict(list)
    for subtype, count in poi_crime_counts:
        poi_risk_raw[subtype].append(count)

    poi_risk_avg = {k: round(np.mean(v), 4) for k, v in poi_risk_raw.items()}
    min_val, max_val = min(poi_risk_avg.values()), max(poi_risk_avg.values())

    poi_risk_normalized = {
        k: round(3 * (v - min_val) / (max_val - min_val + 1e-6), 2)
        for k, v in poi_risk_avg.items()
    }

    with open(POI_RISK_JSON, "w") as f:
        json.dump(poi_risk_normalized, f, indent=2)

    print("📈 POI risk skorları (normalize edildi 0–3):")
    for k, score in sorted(poi_risk_normalized.items(), key=lambda x: -x[1]):
        print(f"{k:<25} → score: {score:.2f}")
    return poi_risk_normalized


# === 3. DİNAMİK KATEGORİLEME (Q1–Q4 vb.) ===
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


# === 4. SUÇ VERİSİNE POI ÖZELLİĞİ EKLE ===
def enrich_crime_with_poi(df_crime, df_poi, poi_risk_scores):
    print("🔗 POI özellikleri suç verisine ekleniyor...")
    df_poi["risk_score"] = df_poi["poi_subcategory"].map(poi_risk_scores).fillna(0)

    gdf_crime = gpd.GeoDataFrame(df_crime, geometry=gpd.points_from_xy(df_crime["longitude"], df_crime["latitude"]), crs="EPSG:4326")
    gdf_poi = gpd.GeoDataFrame(df_poi, geometry=gpd.points_from_xy(df_poi["lon"], df_poi["lat"]), crs="EPSG:4326")

    crime_rad = np.radians(gdf_crime[["latitude", "longitude"]].values)
    poi_rad = np.radians(gdf_poi[["lat", "lon"]].values)
    tree = BallTree(poi_rad, metric="haversine")
    radius = 300 / 6371000  # 300m

    indices = tree.query_radius(crime_rad, r=radius)
    poi_types = gdf_poi["poi_subcategory"].fillna("")

    total_count, risk_score_sum, dominant_type = [], [], []

    for idx_list in indices:
        subtypes = poi_types.iloc[idx_list]
        risk_vals = subtypes.map(poi_risk_scores).fillna(0)
        total_count.append(len(idx_list))
        risk_score_sum.append(risk_vals.sum())
        dominant_type.append(subtypes.value_counts().idxmax() if not subtypes.empty else "No_POI")

    gdf_crime["poi_total_count"] = total_count
    gdf_crime["poi_risk_score"] = risk_score_sum
    gdf_crime["poi_dominant_type"] = dominant_type

    # Aralık etiketleri
    gdf_crime["poi_total_count_range"] = gdf_crime["poi_total_count"].apply(make_dynamic_range_func(gdf_crime["poi_total_count"]))
    gdf_crime["poi_risk_score_range"] = gdf_crime["poi_risk_score"].apply(make_dynamic_range_func(gdf_crime["poi_risk_score"]))

    df_result = gdf_crime.drop(columns="geometry")
    df_result.to_csv(CRIME_OUTPUT, index=False)
    print(f"✅ Enriched crime verisi kaydedildi: {CRIME_OUTPUT}")
    return df_result


# === ANA FONKSİYON AKIŞI ===
if __name__ == "__main__":
    print("🚀 POI güncelleme işlemi başlıyor...")
    df_crime = pd.read_csv(CRIME_INPUT)
    df_poi = clean_and_assign_geoid_to_pois()
    poi_risk_scores = calculate_dynamic_risk(df_crime, df_poi)
    enrich_crime_with_poi(df_crime, df_poi, poi_risk_scores)
