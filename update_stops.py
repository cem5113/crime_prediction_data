import pandas as pd
import geopandas as gpd
import requests
import zipfile
import os
from shapely.geometry import Point

# === Dosya yolları ===
BUS_OUTPUT = "sf_bus_stops_with_geoid.csv"
TRAIN_ZIP_URL = "https://transitfeeds.com/p/bart/58/latest/download"
TRAIN_OUTPUT = "sf_train_stops_with_geoid.csv"
CENSUS_PATH = "sf_census_blocks_with_population.geojson"

# === 1. Bus Stops from Socrata API ===
try:
    print("🔄 Bus verisi çekiliyor...")
    bus_data = requests.get("https://data.sfgov.org/resource/i28k-bkz6.json").json()
    df_bus = pd.DataFrame(bus_data).dropna(subset=["latitude", "longitude"])
    df_bus["stop_lat"] = df_bus["latitude"].astype(float)
    df_bus["stop_lon"] = df_bus["longitude"].astype(float)
    gdf_bus = gpd.GeoDataFrame(
        df_bus,
        geometry=gpd.points_from_xy(df_bus["stop_lon"], df_bus["stop_lat"]),
        crs="EPSG:4326"
    )
    gdf_blocks = gpd.read_file(CENSUS_PATH)[["GEOID", "geometry"]].to_crs("EPSG:4326")
    joined_bus = gpd.sjoin(gdf_bus, gdf_blocks, how="left", predicate="within")
    joined_bus["GEOID"] = joined_bus["GEOID"].astype(str).str.zfill(11)
    joined_bus.drop(columns=["geometry", "index_right"], errors="ignore").to_csv(BUS_OUTPUT, index=False)
    print(f"✅ Otobüs verisi güncellendi → {BUS_OUTPUT}")
except Exception as e:
    print(f"❌ Otobüs verisi hatası: {e}")

# === 2. Train Stops from BART GTFS ===
try:
    print("🔄 Train GTFS indiriliyor...")
    r = requests.get(TRAIN_ZIP_URL)
    with open("bart_gtfs.zip", "wb") as f:
        f.write(r.content)

    with zipfile.ZipFile("bart_gtfs.zip", "r") as zip_ref:
        zip_ref.extract("stops.txt", ".")

    df_train = pd.read_csv("stops.txt").dropna(subset=["stop_lat", "stop_lon"])
    gdf_train = gpd.GeoDataFrame(
        df_train,
        geometry=gpd.points_from_xy(df_train["stop_lon"], df_train["stop_lat"]),
        crs="EPSG:4326"
    )
    joined_train = gpd.sjoin(gdf_train, gdf_blocks, how="left", predicate="within")
    joined_train["GEOID"] = joined_train["GEOID"].astype(str).str.zfill(11)
    joined_train.drop(columns=["geometry", "index_right"], errors="ignore").to_csv(TRAIN_OUTPUT, index=False)
    print(f"✅ Tren verisi güncellendi → {TRAIN_OUTPUT}")
except Exception as e:
    print(f"❌ Tren verisi hatası: {e}")
