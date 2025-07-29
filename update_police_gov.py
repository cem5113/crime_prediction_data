import pandas as pd
import geopandas as gpd
import numpy as np
import os
from shapely.geometry import Point
from sklearn.neighbors import BallTree

# === Dosya Yolları ===
BASE_DIR = "/content/drive/MyDrive/crime_data"
CRIME_INPUT = os.path.join(BASE_DIR, "sf_crime_06.csv")
CRIME_OUTPUT = os.path.join(BASE_DIR, "sf_crime_07.csv")
POLICE_PATH = os.path.join(BASE_DIR, "sf_police_stations.csv")
GOV_PATH = os.path.join(BASE_DIR, "sf_government_buildings.csv")

# === Yardımcı: Mesafe hesapla ve range ver ===
def haversine_features(df_crime, df_target, prefix, radius=300):
    # GeoDataFrame dönüşümü
    gdf_crime = gpd.GeoDataFrame(df_crime, geometry=gpd.points_from_xy(df_crime["longitude"], df_crime["latitude"]), crs="EPSG:4326")
    gdf_target = gpd.GeoDataFrame(df_target, geometry=gpd.points_from_xy(df_target["longitude"], df_target["latitude"]), crs="EPSG:4326")

    # Koordinatları radian'a çevir
    crime_rad = np.radians(gdf_crime[["latitude", "longitude"]].values)
    target_rad = np.radians(gdf_target[["latitude", "longitude"]].values)

    # BallTree ile mesafe hesapla
    tree = BallTree(target_rad, metric='haversine')
    dist, _ = tree.query(crime_rad, k=1)
    dist_meters = dist[:, 0] * 6371000  # metre cinsine çevir

    df_crime[f"distance_to_{prefix}"] = dist_meters.round(1)
    df_crime[f"is_near_{prefix}"] = (dist_meters <= radius).astype(int)

    # Dinamik binleme (range sütunu)
    def make_bins(col):
        q = pd.qcut(col, 4, labels=[f"Q{i+1}" for i in range(4)], duplicates='drop')
        return q.astype(str)

    df_crime[f"distance_to_{prefix}_range"] = make_bins(df_crime[f"distance_to_{prefix}"])
    return df_crime

# === Ana Fonksiyon ===
def enrich_with_police_and_government():
    print("📥 Veri yükleniyor...")
    df_crime = pd.read_csv(CRIME_INPUT)
    df_police = pd.read_csv(POLICE_PATH)
    df_gov = pd.read_csv(GOV_PATH)

    print("🚓 Polis istasyonları entegre ediliyor...")
    df_crime = haversine_features(df_crime, df_police, prefix="police", radius=300)

    print("🏛️ Hükümet binaları entegre ediliyor...")
    df_crime = haversine_features(df_crime, df_gov, prefix="government", radius=300)

    print("💾 Zenginleştirilmiş veri kaydediliyor...")
    df_crime.to_csv(CRIME_OUTPUT, index=False)

    print(f"✅ Zenginleştirme tamamlandı → {CRIME_OUTPUT}")
    print("📄 Eklenen sütunlar:")
    print([
        "distance_to_police", "is_near_police", "distance_to_police_range",
        "distance_to_government", "is_near_government", "distance_to_government_range"
    ])

# === Çalıştır ===
if __name__ == "__main__":
    enrich_with_police_and_government()
