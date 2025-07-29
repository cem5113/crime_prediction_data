# === SUC VERISI GUNCELLEME ve OZET GRID OLUSTURMA (GitHub icin optimize) ===

import pandas as pd
import geopandas as gpd
import time
from datetime import datetime, timedelta
from urllib.parse import quote
from shapely.geometry import Point
import numpy as np
import holidays
import os
import itertools

# === Guvenli Kaydetme Fonksiyonu ===
def safe_save(df, path):
    try:
        df.to_csv(path, index=False)
    except Exception as e:
        print(f"❌ Kaydedilemedi: {path}\n{e}")
        backup_path = path + ".bak"
        df.to_csv(backup_path, index=False)
        print(f"📁 Yedek dosya olusturuldu: {backup_path}")

# === 1. Dosya yolları ===
save_dir = "."  # GitHub klasoruyle ayni dizin
csv_path = os.path.join(save_dir, "sf_crime.csv")
sum_path = os.path.join(save_dir, "sf_crime_grid_summary_labeled.csv")
full_path = os.path.join(save_dir, "sf_crime_grid_full_labeled.csv")
blocks_path = os.path.join(save_dir, "sf_census_blocks_with_population.geojson")

# === 2. Tarih araligi ===
today = datetime.today().date()
start_date = today - timedelta(days=5*365)

# === 3. Onceki veriyi oku ===
try:
    df_old = pd.read_csv(csv_path, parse_dates=["date"], dtype={"GEOID": str})
    df_old["GEOID"] = df_old["GEOID"].astype(str).str.extract(r'(\d{12})')
    df_old["id"] = df_old["id"].astype(str)
    df_old["date"] = pd.to_datetime(df_old["date"]).dt.date
    latest_date = df_old["date"].max()
    print(f"📂 Mevcut veri yüklendi: {len(df_old)} satır (son tarih: {latest_date})")
except:
    df_old = pd.DataFrame(columns=["id", "date"])
    latest_date = start_date - timedelta(days=1)
    print("🆕 Önceki veri bulunamadı. Sıfırdan başlıyor...")

# === 4. Eksik tarihleri al ===
date_range = pd.date_range(start=latest_date + timedelta(days=1), end=today)
missing_dates = [d.date() for d in date_range]
print(f"📆 Eksik tarihler: {len(missing_dates)})")

# === 5. Veriyi indir ===
def download_crime_for_date(date_obj):
    date_str = date_obj.isoformat()
    soql = f"$where=incident_datetime between '{date_str}T00:00:00' and '{date_str}T23:59:59'"
    query_encoded = quote(soql, safe="=&")
    base_url = "https://data.sfgov.org/resource/wg3w-h783.json"
    limit, offset, all_chunks = 1000, 0, []

    while True:
        url = f"{base_url}?{query_encoded}&$limit={limit}&$offset={offset}"
        try:
            chunk = pd.read_json(url)
        except Exception as e:
            print(f"❌ {date_str} indirilemedi: {e}")
            return None
        if chunk.empty:
            break
        all_chunks.append(chunk)
        offset += limit
        time.sleep(0.3)

    if all_chunks:
        df = pd.concat(all_chunks, ignore_index=True)
        df["date"] = date_obj
        return df
    return None

# === 6. Verileri indir ve birleştir ===
new_data = []
for d in missing_dates:
    print(f"📥 {d} indiriliyor...")
    df_day = download_crime_for_date(d)
    if df_day is not None:
        new_data.append(df_day)

# === 7. Temizle ve GEOID ata ===
if new_data:
    df_new = pd.concat(new_data, ignore_index=True)
    df_new["datetime"] = pd.to_datetime(df_new["incident_datetime"], errors="coerce")
    df_new["date"] = df_new["datetime"].dt.date
    df_new["time"] = df_new["datetime"].dt.time
    df_new["event_hour"] = df_new["datetime"].dt.hour
    df_new["id"] = df_new.get("row_id", df_new.index.astype(str))
    df_new = df_new.rename(columns={"incident_category": "category", "incident_subcategory": "subcategory"})
    df_new = df_new[["id", "date", "time", "event_hour", "latitude", "longitude", "category", "subcategory"]]
    df_new = df_new.dropna(subset=["latitude", "longitude", "id", "date", "category"])
    df_new = df_new[(df_new["latitude"] > 37.6) & (df_new["latitude"] < 37.9)]
    df_new = df_new[(df_new["longitude"] > -123.2) & (df_new["longitude"] < -122.3)]

    gdf = gpd.GeoDataFrame(df_new, geometry=gpd.points_from_xy(df_new["longitude"], df_new["latitude"]), crs="EPSG:4326")
    gdf_blocks = gpd.read_file(blocks_path)
    gdf_blocks['GEOID'] = gdf_blocks['GEOID'].astype(str).str.zfill(12)
    gdf = gpd.sjoin(gdf, gdf_blocks[["GEOID", "geometry"]], how="left", predicate="within")
    gdf = gdf.drop(columns=["geometry", "index_right"], errors="ignore")
    gdf["GEOID"] = gdf["GEOID"].astype(str).str.extract(r'(\d{12})')
    df_new = pd.DataFrame(gdf)
else:
    df_new = pd.DataFrame()

# === 8. Birleştir ve özellikleri ata ===
df_all = pd.concat([df_old, df_new], ignore_index=True)
df_all["id"] = df_all["id"].astype(str)
df_all = df_all.drop_duplicates(subset="id")
df_all = df_all[df_all["date"] >= start_date]
df_all["datetime"] = pd.to_datetime(df_all["date"].astype(str) + " " + df_all["time"].astype(str), errors="coerce")
df_all = df_all.dropna(subset=["datetime"])
df_all["datetime"] = df_all["datetime"].dt.floor("H")
df_all["event_hour"] = df_all["datetime"].dt.hour

df_all["day_of_week"] = df_all["datetime"].dt.dayofweek
df_all["month"] = df_all["datetime"].dt.month
us_holidays = pd.to_datetime(list(holidays.US(years=sorted(df_all["datetime"].dt.year.unique())).keys()))
df_all["is_weekend"] = df_all["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)
df_all["is_night"] = df_all["event_hour"].apply(lambda x: 1 if (x >= 20 or x < 4) else 0)
df_all["is_holiday"] = df_all["date"].isin(us_holidays).astype(int)
df_all["is_school_hour"] = df_all["event_hour"].apply(lambda x: 1 if 7 <= x <= 16 else 0)
df_all["is_business_hour"] = df_all.apply(lambda x: 1 if (9 <= x["event_hour"] < 18 and x["day_of_week"] < 5) else 0, axis=1)
df_all["season"] = df_all["month"].map({12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring", 6: "Summer", 7: "Summer", 8: "Summer", 9: "Fall", 10: "Fall", 11: "Fall"})
df_all["Y_label"] = 1

# === 9. Kaydet ===
safe_save(df_all, csv_path)

# === 10. Grid ve Label ===
group_cols = ["GEOID", "season", "day_of_week", "event_hour"]
agg_dict = {
    "latitude": "mean",
    "longitude": "mean",
    "is_weekend": "mean",
    "is_night": "mean",
    "is_holiday": "mean",
    "is_school_hour": "mean",
    "is_business_hour": "mean",
    "date": "min",
    "id": "count"
}
grouped = df_all.groupby(group_cols).agg(agg_dict).reset_index()
grouped = grouped.rename(columns={"id": "crime_count"})
grouped["Y_label"] = (grouped["crime_count"] >= 2).astype(int)

# === Kombinasyon üret ===
geoids = df_all["GEOID"].dropna().unique()
seasons = ["Winter", "Spring", "Summer", "Fall"]
days = list(range(7))
hours = list(range(24))
full_grid = pd.DataFrame(itertools.product(geoids, seasons, days, hours), columns=group_cols)

# === Birleştir ===
df_final = full_grid.merge(grouped, on=group_cols, how="left")
df_final["crime_count"] = df_final["crime_count"].fillna(0).astype(int)
df_final["Y_label"] = df_final["Y_label"].fillna(0).astype(int)

# === Kaydet ===
safe_save(df_final, sum_path)
safe_save(df_final, full_path)

print("\n✅ Tüm işlem tamamlandı. Dosyalar güncellendi.")
