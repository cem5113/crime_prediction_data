import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from datetime import datetime, timedelta
import os
import requests
import time
from urllib.parse import quote

# === 1. Dosya yolları ===
BASE_DIR = "crime_data"
raw_save_path = os.path.join(BASE_DIR, "sf_311_last_5_years.csv")
agg_save_path = os.path.join(BASE_DIR, "311_requests_range.csv")
crime_01_path = os.path.join(BASE_DIR, "sf_crime_01.csv")
output_path = os.path.join(BASE_DIR, "sf_crime_02.csv")
census_path = os.path.join(BASE_DIR, "sf_census_blocks_with_population.geojson")

# === 2. Tarih aralığı: Son 5 yıl
today = datetime.today().date()
start_date = today - timedelta(days=1825)

# === 3. 311 verisini indir (SFPD/Police)
soql = (
    f"$where=requested_datetime >= '{start_date}T00:00:00.000' "
    "AND (agency_responsible like '%Police%' OR agency_responsible like '%SFPD%')"
)
url_base = "https://data.sfgov.org/resource/vw6y-z8j6.json"
limit = 1000
offset = 0
rows = []

print("📥 311 verisi indiriliyor...")
while True:
    url = f"{url_base}?{quote(soql, safe='=&')}&$limit={limit}&$offset={offset}"
    try:
        data = pd.read_json(url)
    except Exception as e:
        print("❌ Veri çekme hatası:", e)
        break
    if data.empty:
        break
    rows.append(data)
    offset += limit
    print(f"  + {offset} kayıt indirildi...")
    time.sleep(0.3)

if not rows:
    print("⚠️ Veri alınamadı, script sonlandırıldı.")
    exit()

# === 4. Veriyi temizle ve GEOID eşle
df = pd.concat(rows, ignore_index=True)
df = df.rename(columns={
    "service_request_id": "id",
    "requested_datetime": "datetime",
    "lat": "latitude",
    "long": "longitude"
})
df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
df = df.dropna(subset=["datetime", "latitude", "longitude"])
df["date"] = df["datetime"].dt.date
df["hour"] = df["datetime"].dt.hour

# === 5. GEOID eşlemesi
print("📍 GEOID eşlemesi yapılıyor...")
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs="EPSG:4326")
gdf_blocks = gpd.read_file(census_path)
gdf_blocks["GEOID"] = gdf_blocks["GEOID"].astype(str).str.zfill(11)
gdf = gpd.sjoin(gdf, gdf_blocks[["GEOID", "geometry"]], how="left", predicate="within")

df = pd.DataFrame(gdf.drop(columns="geometry"))
df["GEOID"] = df["GEOID"].astype(str).str.extract(r"(\d{11})")[0]
df = df.dropna(subset=["GEOID"])

# === 6. Saatlik özet oluştur
df["hour_range"] = (df["hour"] // 3) * 3
df["hour_range"] = df["hour_range"].astype(str) + "-" + (df["hour_range"] + 3).astype(str)
summary = df.groupby(["GEOID", "date", "hour_range"]).size().reset_index(name="311_request_count")

# === 7. Kaydet
os.makedirs(BASE_DIR, exist_ok=True)
df.to_csv(raw_save_path, index=False)
summary.to_csv(agg_save_path, index=False)
print(f"✅ Ham 311 verisi kaydedildi → {raw_save_path}")
print(f"✅ Saatlik özet kaydedildi → {agg_save_path}")

# === 8. Suç verisi ile birleştir
if os.path.exists(crime_01_path):
    print("🔗 sf_crime_01 ile birleştiriliyor...")
    df_crime = pd.read_csv(crime_01_path, dtype={"GEOID": str})
    summary["GEOID"] = summary["GEOID"].astype(str).str.zfill(11)

    if "hour_range" not in df_crime.columns:
        if "event_hour" in df_crime.columns:
            df_crime["hour_range"] = (df_crime["event_hour"] // 3) * 3
            df_crime["hour_range"] = df_crime["hour_range"].astype(str) + "-" + (df_crime["hour_range"] + 3).astype(str)
        else:
            raise ValueError("❌ sf_crime_01 içinde 'hour_range' veya 'event_hour' sütunu eksik!")

    df_merge = pd.merge(df_crime, summary, on=["GEOID", "date", "hour_range"], how="left")
    df_merge["311_request_count"] = df_merge["311_request_count"].fillna(0).astype(int)
    df_merge.to_csv(output_path, index=False)
    print(f"✅ Birleştirilmiş çıktı → {output_path}")
else:
    print("⚠️ sf_crime_01.csv bulunamadı. Birleştirme yapılamadı.")
