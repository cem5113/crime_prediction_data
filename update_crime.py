# === SUÇ VERİSİ GÜNCELLEME ve ÖZET GRID OLUŞTURMA (GitHub için optimize) ===

import os
import time
import itertools
from datetime import datetime, timedelta
from urllib.parse import quote
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import holidays

# === Güvenli Kaydetme Fonksiyonu ===
def safe_save(df, path):
    try:
        Path(os.path.dirname(path) or ".").mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    except Exception as e:
        print(f"❌ Kaydedilemedi: {path}\n{e}")
        backup_path = path + ".bak"
        df.to_csv(backup_path, index=False)
        print(f"📁 Yedek dosya oluşturuldu: {backup_path}")

def normalize_geoid(series: pd.Series, target_len: int) -> pd.Series:
    s = series.astype(str).str.extract(r"(\d+)")[0]
    return s.str.zfill(target_len)

# === 1. Dosya yolları ===
save_dir   = "."  # repo kökü
csv_path   = os.path.join(save_dir, "sf_crime.csv")
sum_path   = os.path.join(save_dir, "sf_crime_grid_summary_labeled.csv")
full_path  = os.path.join(save_dir, "sf_crime_grid_full_labeled.csv")
blocks_path = os.path.join(save_dir, "sf_census_blocks_with_population.geojson")

# === 2. Tarih aralığı ===
today = datetime.today().date()
start_date = today - timedelta(days=5 * 365)

# === 3. Önceki veriyi oku ===
try:
    df_old = pd.read_csv(csv_path, parse_dates=["date"], dtype={"GEOID": str})
    df_old["GEOID"] = df_old["GEOID"].astype(str)  # uzunluğu sonra normalize edeceğiz
    df_old["id"] = df_old["id"].astype(str)
    df_old["date"] = pd.to_datetime(df_old["date"]).dt.date
    latest_date = df_old["date"].max()
    print(f"📂 Mevcut veri yüklendi: {len(df_old)} satır (son tarih: {latest_date})")
except Exception:
    df_old = pd.DataFrame(columns=["id", "date"])
    latest_date = start_date - timedelta(days=1)
    print("🆕 Önceki veri bulunamadı. Sıfırdan başlıyor...")

# === 4. Eksik tarihleri al ===
date_range = pd.date_range(start=latest_date + timedelta(days=1), end=today)
missing_dates = [d.date() for d in date_range]
print(f"📆 Eksik tarihler: {len(missing_dates)}")

# === 4.1 Blok dosyasını (varsa) hazırla & GEOID hedef uzunluğu tespit et ===
gdf_blocks = None
target_len = 12  # emniyetli varsayılan
if os.path.exists(blocks_path):
    try:
        gdf_blocks = gpd.read_file(blocks_path)
        target_len = gdf_blocks["GEOID"].astype(str).str.len().mode().iat[0]
        gdf_blocks["GEOID"] = normalize_geoid(gdf_blocks["GEOID"], target_len)
    except Exception as e:
        print(f"⚠️ Blok dosyası okunamadı ({blocks_path}): {e}. GEOID eşlemesi atlanacak.")
        gdf_blocks = None
else:
    print(f"⚠️ {blocks_path} bulunamadı; GEOID eşlemesi atlandı.")

# === 5. Veriyi indir ===
def download_crime_for_date(date_obj):
    date_str = date_obj.isoformat()
    soql = f"$where=incident_datetime between '{date_str}T00:00:00' and '{date_str}T23:59:59'"
    query_encoded = quote(soql, safe="=&")
    base_url = "https://data.sfgov.org/resource/wg3w-h783.json"
    limit, offset, all_chunks = 1000, 0, []

    while True:
        url = f"{base_url}?{query_encoded}&$limit={limit}&$offset={offset}"
        # basit retry
        chunk = None
        for attempt in range(4):
            try:
                chunk = pd.read_json(url)
                break
            except Exception as e:
                if attempt == 3:
                    print(f"❌ {date_str} indirilemedi: {e}")
                    return None
                time.sleep(1.5 * (attempt + 1))
        if chunk is None or chunk.empty:
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

    # datetime & temel sütunlar
    df_new["datetime"] = pd.to_datetime(df_new["incident_datetime"], errors="coerce")
    df_new["date"] = df_new["datetime"].dt.date
    df_new["time"] = df_new["datetime"].dt.time
    df_new["event_hour"] = df_new["datetime"].dt.hour

    # sağlam ID üretimi
    id_cols = [c for c in ["row_id", "incident_id", "incident_number", "cad_number"] if c in df_new.columns]
    if id_cols:
        s = df_new[id_cols[0]].astype(str)
        for c in id_cols[1:]:
            s = s.where(s.notna() & (s.astype(str) != "nan"), df_new[c].astype(str))
        df_new["id"] = s
    else:
        df_new["id"] = np.nan
    mask = df_new["id"].isna() | (df_new["id"].astype(str) == "nan")
    if mask.any():
        df_new.loc[mask, "id"] = (
            df_new.loc[mask, "datetime"].astype(str)
            + "_"
            + df_new.loc[mask, "latitude"].round(6).astype(str)
            + "_"
            + df_new.loc[mask, "longitude"].round(6).astype(str)
        )
    df_new["id"] = df_new["id"].astype(str)

    # isimlendirme & filtreler
    df_new = df_new.rename(columns={"incident_category": "category", "incident_subcategory": "subcategory"})
    df_new = df_new[["id", "date", "time", "event_hour", "latitude", "longitude", "category", "subcategory"]]
    df_new = df_new.dropna(subset=["latitude", "longitude", "id", "date", "category"])
    df_new = df_new[(df_new["latitude"] > 37.6) & (df_new["latitude"] < 37.9)]
    df_new = df_new[(df_new["longitude"] > -123.2) & (df_new["longitude"] < -122.3)]

    # GEOID eşlemesi (opsiyonel)
    gdf = gpd.GeoDataFrame(df_new, geometry=gpd.points_from_xy(df_new["longitude"], df_new["latitude"]), crs="EPSG:4326")
    if gdf_blocks is not None:
        gdf = gpd.sjoin(gdf, gdf_blocks[["GEOID", "geometry"]], how="left", predicate="within")
        gdf = gdf.drop(columns=["geometry", "index_right"], errors="ignore")
        gdf["GEOID"] = normalize_geoid(gdf["GEOID"], target_len)
    else:
        gdf["GEOID"] = np.nan
        gdf = gdf.drop(columns=["geometry"], errors="ignore")
    df_new = pd.DataFrame(gdf)

    # df_old GEOID'lerini de aynı hedef uzunluğa çek
    if "GEOID" in df_old.columns:
        df_old["GEOID"] = normalize_geoid(df_old["GEOID"], target_len)
else:
    df_new = pd.DataFrame()

# === 8. Birleştir ve özellikleri ata ===
# Tip hizalama (df_old'ta time olmayabilir)
if "time" not in df_old.columns:
    df_old["time"] = "00:00:00"
if "date" in df_old.columns:
    df_old["date"] = pd.to_datetime(df_old["date"]).dt.date

df_all = pd.concat([df_old, df_new], ignore_index=True)
df_all["id"] = df_all["id"].astype(str)
df_all = df_all.drop_duplicates(subset="id")
df_all = df_all[df_all["date"] >= start_date]

df_all["date"] = pd.to_datetime(df_all["date"], errors="coerce")
df_all["time"] = df_all["time"].astype(str).fillna("00:00:00")
df_all["datetime"] = pd.to_datetime(df_all["date"].dt.strftime("%Y-%m-%d") + " " + df_all["time"], errors="coerce")
df_all = df_all.dropna(subset=["datetime"]).copy()
df_all["datetime"] = df_all["datetime"].dt.floor("H")
df_all["event_hour"] = df_all["datetime"].dt.hour

df_all["day_of_week"] = df_all["datetime"].dt.dayofweek
df_all["month"] = df_all["datetime"].dt.month
years = sorted(df_all["datetime"].dt.year.dropna().unique().tolist())
us_holidays = pd.to_datetime(list(holidays.US(years=years).keys()))
df_all["is_weekend"] = (df_all["day_of_week"] >= 5).astype(int)
df_all["is_night"] = ((df_all["event_hour"] >= 20) | (df_all["event_hour"] < 4)).astype(int)
df_all["is_holiday"] = df_all["date"].isin(us_holidays.normalize()).astype(int)
df_all["is_school_hour"] = df_all["event_hour"].between(7, 16).astype(int)
df_all["is_business_hour"] = ((df_all["event_hour"].between(9, 17)) & (df_all["day_of_week"] < 5)).astype(int)
df_all["season"] = df_all["month"].map({
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Fall", 10: "Fall", 11: "Fall"
})
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
    "id": "count",
}

# GEOID NaN'ları gruba sokmayalım
df_all_valid = df_all.dropna(subset=["GEOID"]).copy()
grouped = df_all_valid.groupby(group_cols).agg(agg_dict).reset_index()
grouped = grouped.rename(columns={"id": "crime_count"})
grouped["Y_label"] = (grouped["crime_count"] >= 2).astype(int)

# Kombinasyon üret
geoids = df_all_valid["GEOID"].dropna().unique()
seasons = ["Winter", "Spring", "Summer", "Fall"]
days = list(range(7))
hours = list(range(24))
full_grid = pd.DataFrame(itertools.product(geoids, seasons, days, hours), columns=group_cols)

# Birleştir
df_final = full_grid.merge(grouped, on=group_cols, how="left")
df_final["crime_count"] = df_final["crime_count"].fillna(0).astype(int)
df_final["Y_label"] = df_final["Y_label"].fillna(0).astype(int)

# Kaydet
safe_save(df_final, sum_path)
safe_save(df_final, full_path)

# 2. adımın (911) ve sonraki adımların beklediği yerlere kopyalar
try:
    Path("crime_data").mkdir(exist_ok=True)
    src_grid = Path(full_path)
    if src_grid.exists():
        shutil.copy2(src_grid, Path("crime_data") / src_grid.name)
    src_blocks = Path(blocks_path)
    if src_blocks.exists():
        shutil.copy2(src_blocks, Path("crime_data") / src_blocks.name)
    print("📦 crime_data/ klasörüne gerekli kopyalar bırakıldı.")
except Exception as e:
    print(f"⚠️ crime_data kopyalama uyarısı: {e}")

print("\n✅ Tüm işlem tamamlandı. Dosyalar güncellendi.")
