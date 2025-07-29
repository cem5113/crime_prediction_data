import pandas as pd
import geopandas as gpd
import zipfile
import urllib.request
import os
import numpy as np
from shapely.geometry import Point
from scipy.spatial import cKDTree

# === 1. Dosya yolları ===
CRIME_INPUT = "/content/drive/MyDrive/crime_data/sf_crime_04.csv"
CRIME_OUTPUT = "/content/drive/MyDrive/crime_data/sf_crime_05.csv"
CENSUS_PATH = "/content/drive/MyDrive/crime_data/sf_census_blocks_with_population.geojson"
TRAIN_OUTPUT = "/content/drive/MyDrive/crime_data/sf_train_stops_with_geoid.csv"
GTFS_URL = "https://transitfeeds.com/p/bart/58/latest/download"
GTFS_ZIP = "/content/bart_gtfs.zip"
GTFS_TXT = "/content/stops.txt"

# === 2. GTFS verisini indir ve aç ===
print("🚉 BART tren verisi indiriliyor...")
urllib.request.urlretrieve(GTFS_URL, GTFS_ZIP)
with zipfile.ZipFile(GTFS_ZIP, 'r') as zip_ref:
    zip_ref.extract("stops.txt", "/content/")
bart_stops = pd.read_csv(GTFS_TXT).dropna(subset=["stop_lat", "stop_lon"])

# === 3. GEOID eşlemesi ===
gdf_stops = gpd.GeoDataFrame(
    bart_stops,
    geometry=gpd.points_from_xy(bart_stops["stop_lon"], bart_stops["stop_lat"]),
    crs="EPSG:4326"
)
gdf_blocks = gpd.read_file(CENSUS_PATH).to_crs("EPSG:4326")
gdf_joined = gpd.sjoin(gdf_stops, gdf_blocks[["geometry", "GEOID"]], how="left", predicate="within")
gdf_joined = gdf_joined[~gdf_joined["GEOID"].isna()].copy()
gdf_joined["GEOID"] = gdf_joined["GEOID"].astype(str).str.zfill(11)
gdf_joined.drop(columns="geometry").to_csv(TRAIN_OUTPUT, index=False)
print(f"✅ {len(gdf_joined)} tren durağı SF içinde bulundu → {TRAIN_OUTPUT}")

# === 4. Suç verisini yükle ===
df_crime = pd.read_csv(CRIME_INPUT, dtype={"GEOID": str})
df_crime["GEOID"] = df_crime["GEOID"].astype(str).str.zfill(11)
df_train = pd.read_csv(TRAIN_OUTPUT).dropna(subset=["stop_lat", "stop_lon"])

# === 5. GeoDataFrame dönüşümleri ===
gdf_crime = gpd.GeoDataFrame(
    df_crime,
    geometry=gpd.points_from_xy(df_crime["longitude"], df_crime["latitude"]),
    crs="EPSG:4326"
).to_crs(epsg=3857)

gdf_train = gpd.GeoDataFrame(
    df_train,
    geometry=gpd.points_from_xy(df_train["stop_lon"], df_train["stop_lat"]),
    crs="EPSG:4326"
).to_crs(epsg=3857)

# === 6. En yakın tren durağına mesafe ===
crime_coords = np.vstack([gdf_crime.geometry.x, gdf_crime.geometry.y]).T
train_coords = np.vstack([gdf_train.geometry.x, gdf_train.geometry.y]).T
tree = cKDTree(train_coords)
distances, _ = tree.query(crime_coords, k=1)
gdf_crime["distance_to_train"] = distances

# === 7. Binleme fonksiyonu ===
def freedman_diaconis_bin_count(data, max_bins=10):
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    n = len(data)
    bin_width = 2 * iqr / (n ** (1/3))
    if bin_width == 0:
        return 1
    return max(2, min(max_bins, int(np.ceil((data.max() - data.min()) / bin_width))))

# === 8. Mesafe aralıkları (binleme) ===
n_dist_bins = freedman_diaconis_bin_count(gdf_crime["distance_to_train"])
_, dist_bins = pd.qcut(gdf_crime["distance_to_train"], q=n_dist_bins, retbins=True, duplicates="drop")
dist_labels = [f"{int(dist_bins[i])}–{int(dist_bins[i+1])}m" for i in range(len(dist_bins)-1)]
gdf_crime["distance_to_train_range"] = pd.cut(gdf_crime["distance_to_train"], bins=dist_bins, labels=dist_labels, include_lowest=True)

# === 9. Yakın tren durağı sayısı ===
dynamic_radius = np.percentile(gdf_crime["distance_to_train"], 75)
gdf_crime["train_stop_count"] = gdf_crime.geometry.apply(
    lambda pt: gdf_train.distance(pt).lt(dynamic_radius).sum()
)

# === 10. Durak sayısı binleme ===
n_count_bins = freedman_diaconis_bin_count(gdf_crime["train_stop_count"])
_, count_bins = pd.qcut(gdf_crime["train_stop_count"], q=n_count_bins, retbins=True, duplicates="drop")
count_labels = [f"{int(count_bins[i])}–{int(count_bins[i+1])}" for i in range(len(count_bins)-1)]
gdf_crime["train_stop_count_range"] = pd.cut(gdf_crime["train_stop_count"], bins=count_bins, labels=count_labels, include_lowest=True)

# === 11. Kaydet ===
df_final = gdf_crime.drop(columns="geometry")
df_final.to_csv(CRIME_OUTPUT, index=False)

# === 12. Özet ===
print("📦 Yeni sütunlar eklendi:")
print(df_final[[
    "GEOID", "distance_to_train", "distance_to_train_range",
    "train_stop_count", "train_stop_count_range"
]].head())

print(f"✅ Güncellenmiş veri kaydedildi: {CRIME_OUTPUT}")
print(f"📊 Satır sayısı: {df_final.shape[0]} | Sütun sayısı: {df_final.shape[1]}")
