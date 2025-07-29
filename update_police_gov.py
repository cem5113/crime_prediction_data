import pandas as pd
import geopandas as gpd
import numpy as np
import os
from shapely.geometry import Point
from sklearn.neighbors import BallTree

# === Dosya Yolları ===
BASE_DIR = "crime_data"
CRIME_INPUT = os.path.join(BASE_DIR, "sf_crime_06.csv")
CRIME_OUTPUT = os.path.join(BASE_DIR, "sf_crime_07.csv")
POLICE_PATH = os.path.join(BASE_DIR, "sf_police_stations.csv")
GOV_PATH = os.path.join(BASE_DIR, "sf_government_buildings.csv")


# === Yardımcı Fonksiyon: En yakın nesneye mesafe hesapla ve sınıflandır ===
def haversine_features(df_crime, df_target, prefix, radius=300):
    if df_target.empty:
        print(f"⚠️ Uyarı: {prefix} veri kümesi boş. İşlem atlandı.")
        return df_crime

    # GeoDataFrame oluştur
    gdf_crime = gpd.GeoDataFrame(
        df_crime,
        geometry=gpd.points_from_xy(df_crime["longitude"], df_crime["latitude"]),
        crs="EPSG:4326"
    )
    gdf_target = gpd.GeoDataFrame(
        df_target,
        geometry=gpd.points_from_xy(df_target["longitude"], df_target["latitude"]),
        crs="EPSG:4326"
    )

    # Koordinatları radyana çevir
    crime_rad = np.radians(gdf_crime[["latitude", "longitude"]].values)
    target_rad = np.radians(gdf_target[["latitude", "longitude"]].values)

    # BallTree kullanarak en yakın mesafeyi hesapla
    tree = BallTree(target_rad, metric='haversine')
    dist, _ = tree.query(crime_rad, k=1)
    dist_meters = dist[:, 0] * 6371000  # km → metre

    # Sütunlar ekle
    df_crime[f"distance_to_{prefix}"] = dist_meters.round(1)
    df_crime[f"is_near_{prefix}"] = (dist_meters <= radius).astype(int)

    # Range binleme (qcut ile otomatik Q1-Q4)
    try:
        df_crime[f"distance_to_{prefix}_range"] = pd.qcut(
            df_crime[f"distance_to_{prefix}"],
            q=4,
            labels=[f"Q{i+1}" for i in range(4)],
            duplicates="drop"
        ).astype(str)
    except ValueError:
        # Tüm mesafeler aynıysa qcut hata verir
        df_crime[f"distance_to_{prefix}_range"] = "Q1"

    return df_crime


# === Ana Zenginleştirme Fonksiyonu ===
def enrich_with_police_and_government():
    print("📥 Veriler yükleniyor...")
    if not os.path.exists(CRIME_INPUT):
        print(f"❌ Girdi dosyası bulunamadı: {CRIME_INPUT}")
        return

    df_crime = pd.read_csv(CRIME_INPUT)
    df_police = pd.read_csv(POLICE_PATH)
    df_gov = pd.read_csv(GOV_PATH)

    # Gerekli sütunlar var mı?
    for name, df in [("suç verisi", df_crime), ("polis", df_police), ("hükümet", df_gov)]:
        for col in ["latitude", "longitude"]:
            if col not in df.columns:
                raise ValueError(f"❌ {name} veri kümesinde '{col}' sütunu eksik.")

    print("🚓 Polis istasyonları entegre ediliyor...")
    df_crime = haversine_features(df_crime, df_police, prefix="police", radius=300)

    print("🏛️ Hükümet binaları entegre ediliyor...")
    df_crime = haversine_features(df_crime, df_gov, prefix="government", radius=300)

    print("💾 Zenginleştirilmiş veri kaydediliyor...")
    df_crime.to_csv(CRIME_OUTPUT, index=False)

    print(f"✅ Zenginleştirme tamamlandı → {CRIME_OUTPUT}")
    print("📄 Eklenen sütunlar:")
    for col in [
        "distance_to_police", "is_near_police", "distance_to_police_range",
        "distance_to_government", "is_near_government", "distance_to_government_range"
    ]:
        print(f"   - {col}")


# === Ana Çalıştırıcı ===
if __name__ == "__main__":
    enrich_with_police_and_government()
