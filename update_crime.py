# === SUÇ VERİSİ GÜNCELLEME, ÖZETLEME ve Y_LABEL ÜRETİMİ ===

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

# === 1. DOSYA ve ZAMAN ARALIĞI AYARLARI ===
save_dir = "data"
os.makedirs(save_dir, exist_ok=True)
csv_path = f"{save_dir}/sf_crime.csv"

# === 2. 5 YILLIK TARİH ARALIĞI ===
today = datetime.today().date()
days_back = 1825
start_date = today - timedelta(days=days_back)

# === 3. VAR OLAN VERİYİ YÜKLE ===
if os.path.exists(csv_path):
    df_old = pd.read_csv(csv_path, parse_dates=["date"], dtype={"GEOID": str})
    df_old["GEOID"] = df_old["GEOID"].astype(str).str.extract(r'(\d{12})')
    df_old["id"] = df_old["id"].astype(str)
    df_old["date"] = pd.to_datetime(df_old["date"]).dt.date
    latest_date = df_old["date"].max()
    print(f"\U0001F4C2 Existing data loaded: {len(df_old)} rows (latest date: {latest_date})")
else:
    df_old = pd.DataFrame(columns=["id", "date"])
    latest_date = start_date - timedelta(days=1)
    print("\U0001F195 No previous data found. Starting fresh.")

# === 4. EKSİK TARİHLERİ HESAPLA ===
date_range = pd.date_range(start=latest_date + timedelta(days=1), end=today)
missing_dates = [d.date() for d in date_range]
print(f"\U0001F4C6 Missing dates to fetch: {len(missing_dates)}")

# === 5. GÜNLÜK VERİYİ İNDİR ===
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
            print(f"❌ Error fetching {date_str}: {e}")
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

new_data = []
for d in missing_dates:
    print(f"\U0001F4E5 Fetching {d}...")
    df_day = download_crime_for_date(d)
    if df_day is not None:
        new_data.append(df_day)

# === 6. İNDİRİLEN VERİYİ TEMİZLE ===
if new_data:
    df_new = pd.concat(new_data, ignore_index=True)
    df_new["datetime"] = pd.to_datetime(df_new["incident_datetime"], errors="coerce")
    df_new["date"] = df_new["datetime"].dt.date
    df_new["time"] = df_new["datetime"].dt.time
    df_new["event_hour"] = df_new["datetime"].dt.hour
    df_new["id"] = df_new["row_id"] if "row_id" in df_new.columns else df_new.index.astype(str)
    df_new = df_new.rename(columns={"incident_category": "category", "incident_subcategory": "subcategory"})
    df_new = df_new[["id", "date", "time", "event_hour", "latitude", "longitude", "category", "subcategory"]]
    df_new = df_new.dropna(subset=["latitude", "longitude", "id", "date", "category"])
    df_new = df_new[(df_new["latitude"] > 37.6) & (df_new["latitude"] < 37.9)]
    df_new = df_new[(df_new["longitude"] > -123.2) & (df_new["longitude"] < -122.3)]

    # === GEOID eşlemesi ===
    gdf = gpd.GeoDataFrame(df_new, geometry=gpd.points_from_xy(df_new["longitude"], df_new["latitude"]), crs="EPSG:4326")
    gdf_blocks = gpd.read_file(f"{save_dir}/sf_census_blocks_with_population.geojson")
    gdf_blocks['GEOID'] = gdf_blocks['GEOID'].astype(str).str.zfill(12)
    gdf = gpd.sjoin(gdf, gdf_blocks[["GEOID", "geometry"]], how="left", predicate="within")
    gdf = gdf.drop(columns=["geometry", "index_right"], errors="ignore")
    gdf["GEOID"] = gdf["GEOID"].astype(str).str.extract(r'(\d{12})')
    df_new = pd.DataFrame(gdf)
else:
    df_new = pd.DataFrame()

# === 7. BİRLEŞTİR, Y_label = 1 ve ÖZELLİK EKLE ===
df_all = pd.concat([df_old, df_new], ignore_index=True)
df_all["id"] = df_all["id"].astype(str)
df_all["date"] = pd.to_datetime(df_all["date"]).dt.date
df_all["GEOID"] = df_all["GEOID"].astype(str).str.extract(r'(\d{11})')
df_all = df_all.drop_duplicates(subset="id")
df_all = df_all[df_all["date"] >= start_date]
df_all["datetime"] = pd.to_datetime(df_all["date"].astype(str) + " " + df_all["time"].astype(str), errors="coerce")
df_all = df_all.dropna(subset=["datetime"])
df_all["datetime"] = df_all["datetime"].dt.floor("H")
df_all["event_hour"] = df_all["datetime"].dt.hour

# === ZAMANSAL ÖZELLİKLER ===
us_holidays = pd.to_datetime(list(holidays.US(years=sorted(df_all["datetime"].dt.year.unique())).keys()))
df_all["day_of_week"] = df_all["datetime"].dt.dayofweek
df_all["month"] = df_all["datetime"].dt.month
df_all["is_weekend"] = df_all["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)
df_all["is_night"] = df_all["event_hour"].apply(lambda x: 1 if (x >= 20 or x < 4) else 0)
df_all["is_holiday"] = df_all["date"].isin(us_holidays).astype(int)
df_all["latlon"] = df_all["latitude"].round(5).astype(str) + "_" + df_all["longitude"].round(5).astype(str)
df_all["is_repeat_location"] = df_all.duplicated("latlon").astype(int)
df_all.drop(columns=["latlon"], inplace=True)
df_all["is_school_hour"] = df_all["event_hour"].apply(lambda x: 1 if 7 <= x <= 16 else 0)
df_all["is_business_hour"] = df_all.apply(lambda x: 1 if (9 <= x["event_hour"] < 18 and x["day_of_week"] < 5) else 0, axis=1)
df_all["season"] = df_all["month"].map({12:"Winter", 1:"Winter", 2:"Winter", 3:"Spring", 4:"Spring", 5:"Spring", 6:"Summer", 7:"Summer", 8:"Summer", 9:"Fall", 10:"Fall", 11:"Fall"})
df_all["Y_label"] = 1

# === 8. sf_crime.csv OLARAK KAYDET ===
df_all.to_csv(csv_path, index=False)
print(f"\n✅ Final dataset saved: {len(df_all)} rows")

# === 9. sf_crime_50.csv ve sf_crime_52.csv ÜRET ===
group_cols = ["GEOID", "season", "day_of_week", "event_hour"]
mean_cols = ["latitude", "longitude", "past_7d_crimes", "crime_count_past_24h", "crime_count_past_48h", "crime_trend_score", "prev_crime_1h", "prev_crime_2h", "prev_crime_3h"]
mode_cols = ["is_weekend", "is_night", "is_holiday", "is_repeat_location", "is_school_hour", "is_business_hour", "year", "month"]

def safe_mode(x):
    try: return x.mode().iloc[0]
    except: return np.nan

agg_dict = {col: "mean" for col in mean_cols}
agg_dict.update({col: safe_mode for col in mode_cols})
agg_dict.update({"date": "min", "id": "count"})

grouped = df_all.groupby(group_cols).agg(agg_dict).reset_index()
grouped = grouped.rename(columns={"id": "crime_count"})
grouped["Y_label"] = (grouped["crime_count"] >= 2).astype(int)

geoids = df_all["GEOID"].dropna().unique()
seasons = ["Winter", "Spring", "Summer", "Fall"]
days = list(range(7))
hours = list(range(24))

full_grid = pd.DataFrame(itertools.product(geoids, seasons, days, hours), columns=group_cols)
df_final = full_grid.merge(grouped, on=group_cols, how="left")
df_final["crime_count"] = df_final["crime_count"].fillna(0).astype(int)
df_final["Y_label"] = df_final["Y_label"].fillna(0).astype(int)
df_final["is_weekend"] = df_final["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)
df_final["is_night"] = df_final["event_hour"].apply(lambda x: 1 if (x >= 20 or x < 4) else 0)
df_final["is_school_hour"] = df_final.apply(lambda x: 1 if (x["day_of_week"] < 5 and 7 <= x["event_hour"] <= 16) else 0, axis=1)
df_final["is_business_hour"] = df_final.apply(lambda x: 1 if (x["day_of_week"] < 6 and 9 <= x["event_hour"] < 18) else 0, axis=1)

columns_with_nan = ["latitude", "longitude", "past_7d_crimes", "crime_count_past_24h", "crime_count_past_48h", "crime_trend_score", "prev_crime_1h", "prev_crime_2h", "prev_crime_3h", "is_holiday", "is_repeat_location", "year", "month", "date"]
df_final_clean = df_final.dropna(subset=columns_with_nan)
df_final_clean.to_csv("/content/drive/MyDrive/crime_data/sf_crime_50.csv", index=False)

expected_grid = full_grid.copy()
existing_combinations = df_final_clean[group_cols]
merged = expected_grid.merge(existing_combinations.drop_duplicates(), on=group_cols, how="left", indicator=True)
missing = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])
missing["crime_count"] = 0
missing["Y_label"] = 0

df_full_52 = pd.concat([df_final_clean, missing], ignore_index=True)
df_full_52.to_csv("/content/drive/MyDrive/crime_data/sf_crime_52.csv", index=False)

print(f"✅ sf_crime_50: {df_final_clean.shape[0]} satır")
print(f"✅ sf_crime_52: {df_full_52.shape[0]} satır (eksikler dahil)")
