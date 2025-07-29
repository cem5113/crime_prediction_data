import pandas as pd
import os

# === Dosya Yolları ===
BASE_DIR = "/content/drive/MyDrive/crime_data"
CRIME_INPUT = os.path.join(BASE_DIR, "sf_crime_07.csv")
WEATHER_CSV = os.path.join(BASE_DIR, "sf_weather_5years.csv")
CRIME_OUTPUT = os.path.join(BASE_DIR, "sf_crime_08.csv")

# === 1. Verileri oku ===
print("📥 Veriler yükleniyor...")
if not os.path.exists(CRIME_INPUT) or not os.path.exists(WEATHER_CSV):
    raise FileNotFoundError("❌ Gerekli dosyalardan biri bulunamadı.")

df_crime = pd.read_csv(CRIME_INPUT)
df_weather = pd.read_csv(WEATHER_CSV)

# === 2. Tarih sütunlarını datetime formatına çevir ===
df_crime["date"] = pd.to_datetime(df_crime["date"], errors="coerce")
df_weather["DATE"] = pd.to_datetime(df_weather["DATE"], errors="coerce")

# === 3. NOAA sıcaklık/yağmur dönüşümleri ===
if "TMAX" in df_weather.columns:
    df_weather["temp_max"] = df_weather["TMAX"] / 10  # Celsius
else:
    df_weather["temp_max"] = None

if "TMIN" in df_weather.columns:
    df_weather["temp_min"] = df_weather["TMIN"] / 10
else:
    df_weather["temp_min"] = None

if "PRCP" in df_weather.columns:
    df_weather["precipitation_mm"] = df_weather["PRCP"] / 10  # mm
else:
    df_weather["precipitation_mm"] = None

# === 4. Günlük sıcaklık aralığı (range) ===
df_weather["temp_range"] = (df_weather["temp_max"] - df_weather["temp_min"]).round(1)

# === 5. Gerekli sütunları seç ve yeniden adlandır ===
weather_cols = ["DATE", "temp_max", "temp_min", "temp_range", "precipitation_mm"]
df_weather = df_weather[weather_cols].rename(columns={"DATE": "date"})

# === 6. Suç verisi ile birleştir ===
df_merged = pd.merge(df_crime, df_weather, on="date", how="left")

# === 7. Sonucu kaydet ===
df_merged.to_csv(CRIME_OUTPUT, index=False)

# === 8. Özet bilgi ===
print(f"✅ Hava durumu eklendi → {CRIME_OUTPUT}")
print("📄 Eklenen sütunlar:", ["temp_max", "temp_min", "temp_range", "precipitation_mm"])
print(f"📊 Satır sayısı: {df_merged.shape[0]}, Sütun sayısı: {df_merged.shape[1]}")
