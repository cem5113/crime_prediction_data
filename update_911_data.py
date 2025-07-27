import pandas as pd
from datetime import datetime
import os

# === 1. Dosya yolları ===
full_path = "/content/drive/MyDrive/crime_data/sf_911_full_raw.csv"  # Günlük biriken tüm 911 ham verisi
save_path = "/content/drive/MyDrive/crime_data/sf_911_last_5_year.csv"

# === 2. Veriyi oku (tarih dönüşümü yapılır)
df = pd.read_csv(full_path, low_memory=False, parse_dates=["datetime"])

# === 3. Son 5 yılı filtrele
today = pd.Timestamp.today().normalize()
five_years_ago = today - pd.DateOffset(years=5)
df = df[df["datetime"] >= five_years_ago].copy()

# === 4. Zaman sütunlarını oluştur
df["date"] = df["datetime"].dt.date
df["hour"] = df["datetime"].dt.hour

# 3 saatlik aralıklar (örnek: 0-3, 3-6)
df["hour_range"] = (df["hour"] // 3) * 3
df["hour_range"] = df["hour_range"].astype(str) + "-" + (df["hour_range"] + 3).astype(str)

# === 5. GEOID düzenlemesi (11 karaktere tamamla)
df["GEOID"] = df["GEOID"].astype(str).str.extract(r'(\d+)')[0].str.zfill(11)

# === 6. Saatlik 911 sayısı (3 saatlik periyotta)
hourly_summary = df.groupby(["GEOID", "date", "hour_range"]).size().reset_index(name="911_request_count_hour_range")

# === 7. Günlük 911 toplamı (24 saatlik)
daily_summary = df.groupby(["GEOID", "date"]).size().reset_index(name="911_request_count_daily(before_24_hours)")

# === 8. İki özet tabloyu birleştir
final = pd.merge(hourly_summary, daily_summary, on=["GEOID", "date"], how="left")

# === 9. Kaydet
final.to_csv(save_path, index=False)
print(f"✅ Güncel 911 özeti kaydedildi: {save_path}")
