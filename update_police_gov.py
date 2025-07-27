import requests 
import geopandas as gpd
import pandas as pd
import os
from shapely.geometry import Point

# === Ayar ===
SF_BBOX = (37.70, -123.00, 37.83, -122.35)  # (south, west, north, east)
GEOID_PATH = "sf_blockgroup_shapes.geojson"  # GEOID sınırları

OUTPUT_POLICE = "sf_police_stations.csv"
OUTPUT_GOV = "sf_government_buildings.csv"

# === Overpass Turbo Sorgu Şablonu ===
def build_query(key, value):
    return f"""
    [out:json][timeout:25];
    (
      node["{key}"="{value}"]({SF_BBOX[0]},{SF_BBOX[1]},{SF_BBOX[2]},{SF_BBOX[3]});
      way["{key}"="{value}"]({SF_BBOX[0]},{SF_BBOX[1]},{SF_BBOX[2]},{SF_BBOX[3]});
      relation["{key}"="{value}"]({SF_BBOX[0]},{SF_BBOX[1]},{SF_BBOX[2]},{SF_BBOX[3]});
    );
    out center;
    """

# === GEOID eşleme fonksiyonu ===
def add_geoid(df, geoid_path=GEOID_PATH):
    if "latitude" not in df.columns or "longitude" not in df.columns:
        return df
    gdf_points = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs="EPSG:4326"
    )
    gdf_polygons = gpd.read_file(geoid_path)[["geometry", "GEOID"]].to_crs("EPSG:4326")
    gdf_joined = gpd.sjoin(gdf_points, gdf_polygons, how="left", predicate="within")
    df["GEOID"] = gdf_joined["GEOID"].fillna("").astype(str).str.zfill(11)
    return df

# === Veriyi indir ve işle ===
def fetch_osm_data(key, value):
    query = build_query(key, value)
    response = requests.post("https://overpass-api.de/api/interpreter", data={"data": query})
    data = response.json()

    records = []
    for element in data["elements"]:
        if "lat" in element and "lon" in element:
            lat, lon = element["lat"], element["lon"]
        elif "center" in element:
            lat, lon = element["center"]["lat"], element["center"]["lon"]
        else:
            continue
        name = element.get("tags", {}).get("name", "")
        records.append({"latitude": lat, "longitude": lon, "name": name})
    return pd.DataFrame(records)

# === 1. Polis istasyonları ===
print("🚓 Polis istasyonları indiriliyor...")
df_police = fetch_osm_data("amenity", "police")
df_police = add_geoid(df_police)
df_police.to_csv(OUTPUT_POLICE, index=False)
print(f"✅ {len(df_police)} polis noktası kaydedildi → {OUTPUT_POLICE}")

# === 2. Hükümet binaları ===
print("🏛️ Hükümet binaları indiriliyor...")
df_gov = fetch_osm_data("amenity", "townhall")
df_gov = add_geoid(df_gov)
df_gov.to_csv(OUTPUT_GOV, index=False)
print(f"✅ {len(df_gov)} kamu noktası kaydedildi → {OUTPUT_GOV}")
