# enrich_with_weather.py

import pandas as pd

# === Dosya Yolları ===
CRIME_INPUT = "sf_crime_07.csv"
WEATHER_CSV = "sf_weather_5years.csv"
CRIME_OUTPUT = "sf_crime_08.csv"

# === 1. Verileri oku ===
df_crime = pd.read_csv(CRIME_INPUT)
df_weather = pd.read_csv(WEATHER_CSV)

# === 2. Tarih formatlarını kontrol et ===
df_crime["date"] = pd.to_datetime(df_crime["date"])
df_weather["DATE"] = pd.to_datetime(df_weather["DATE"])

# === 3. Sıcaklıkları Celcius'a çevir (NOAA verisi 1/10 °C birimindedir) ===
if "TMAX" in df_weather.columns:
    df_weather["temp_max"] = df_weather["TMAX"] / 10
if "TMIN" in df_weather.columns:
    df_weather["temp_min"] = df_weather["TMIN"] / 10
if "PRCP" in df_weather.columns:
    df_weather["precipitation_mm"] = df_weather["PRCP"] / 10  # mm cinsinden

# === 4. Range hesapla (günlük sıcaklık farkı)
df_weather["temp_range"] = (df_weather["temp_max"] - df_weather["temp_min"]).round(1)

# === 5. Gerekli sütunları seç
weather_cols = ["DATE", "temp_max", "temp_min", "temp_range", "precipitation_mm"]
df_weather = df_weather[weather_cols].rename(columns={"DATE": "date"})

# === 6. Suç verisi ile birleştir
df_merged = pd.merge(df_crime, df_weather, on="date", how="left")

# === 7. Kaydet
df_merged.to_csv(CRIME_OUTPUT, index=False)

# === 8. Özet
print(f"✅ Hava durumu eklendi → {CRIME_OUTPUT}")
print("📄 Eklenen sütunlar:", ["temp_max", "temp_min", "temp_range", "precipitation_mm"])
print(f"📊 Satır sayısı: {df_merged.shape[0]}, Sütun sayısı: {df_merged.shape[1]}")
