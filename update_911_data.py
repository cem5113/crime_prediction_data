import pandas as pd
from datetime import datetime, timedelta
import os

# === 1. Güncel veri dosyasını oku (tam veri)
full_path = "/content/drive/MyDrive/crime_data/sf_911_full_raw.csv"  # Günlük biriken 911 ham verisi
df = pd.read_csv(full_path, low_memory=False, parse_dates=["datetime"])

# === 2. Son 5 yılı filtrele
today = pd.Timestamp.today().normalize()
five_years_ago = today - pd.DateOffset(years=5)
df = df[df["datetime"] >= five_years_ago]

# === 3. Gerekli zaman parçalama
df["date"] = df["datetime"].dt.date
df["hour"] = df["datetime"].dt.hour
df["hour_range"] = (df["hour"] // 3) * 3
df["hour_range"] = df["hour_range"].astype(str) + "-" + (df["hour_range"].astype(int) + 3).astype(str)

# === 4. GEOID kontrolü
df["GEOID"] = df["GEOID"].astype(str).str.extract(r'(\d+)')[0].str.zfill(11)

# === 5. Özet oluştur
summary = df.groupby(["GEOID", "date", "hour_range"]).size().reset_index(name="911_request_count_hour_range")

# Günlük toplam da eklensin
daily_summary = df.groupby(["GEOID", "date"]).size().reset_index(name="911_request_count_daily(before_24_hours)")

# === 6. Birleştir
final = pd.merge(summary, daily_summary, on=["GEOID", "date"], how="left")

# === 7. Kaydet
save_path = "/content/drive/MyDrive/crime_data/sf_911_last_5_year.csv"
final.to_csv(save_path, index=False)
print(f"✅ Güncel 911 verisi kaydedildi: {save_path}")
