import pandas as pd
import os
from datetime import datetime
from pathlib import Path

# === 1. Dosya yolları ===
BASE_DIR = "crime_data"
raw_911_path = os.path.join(BASE_DIR, "sf_911_full_raw.csv")
summary_911_path = os.path.join(BASE_DIR, "sf_911_last_5_year.csv")
crime_grid_path = os.path.join(BASE_DIR, "sf_crime_grid_full_labeled.csv")
output_merge_path = os.path.join(BASE_DIR, "sf_crime_01.csv")

# === 2. 911 Verisini Yükle ===
df = pd.read_csv(raw_911_path, low_memory=False, parse_dates=["datetime"])

# === 3. Son 5 Yılı Filtrele ===
today = pd.Timestamp.today().normalize()
five_years_ago = today - pd.DateOffset(years=5)
df = df[df["datetime"] >= five_years_ago].copy()

# === 4. Zaman Özellikleri Ekle ===
df["date"] = df["datetime"].dt.date
df["hour"] = df["datetime"].dt.hour
df["hour_range"] = (df["hour"] // 3) * 3
df["hour_range"] = df["hour_range"].astype(str) + "-" + (df["hour_range"] + 3).astype(str)

# === 5. GEOID Temizliği ===
df["GEOID"] = df["GEOID"].astype(str).str.extract(r'(\d+)')[0].str.zfill(11)

# === 6. 911 Özet Tablo Oluştur ===
hourly_summary = df.groupby(["GEOID", "date", "hour_range"]).size().reset_index(name="911_request_count_hour_range")
daily_summary = df.groupby(["GEOID", "date"]).size().reset_index(name="911_request_count_daily(before_24_hours)")
final_911 = pd.merge(hourly_summary, daily_summary, on=["GEOID", "date"], how="left")

# === 7. Kaydet (5 yıllık özet) ===
final_911.to_csv(summary_911_path, index=False)
print(f"✅ 911 özeti kaydedildi → {summary_911_path}")

# === 8. Suç Grid Verisi ile Birleştir ===
if os.path.exists(crime_grid_path):
    crime = pd.read_csv(crime_grid_path, dtype={"GEOID": str})

    if "event_hour" not in crime.columns:
        raise ValueError("❌ 'event_hour' sütunu eksik!")

    # Saat aralığı hesapla
    crime["hour_range"] = (crime["event_hour"] // 3) * 3
    crime["hour_range"] = crime["hour_range"].astype(str) + "-" + (crime["hour_range"] + 3).astype(str)

    # Tarih kontrolü
    if "date" not in crime.columns:
        if "datetime" in crime.columns:
            crime["date"] = pd.to_datetime(crime["datetime"]).dt.date
        else:
            raise ValueError("❌ 'date' veya 'datetime' sütunu bulunamadı!")

    # GEOID format standardı
    crime["GEOID"] = crime["GEOID"].astype(str).str.zfill(11)
    final_911["GEOID"] = final_911["GEOID"].astype(str).str.zfill(11)

    # Birleştir
    merged = pd.merge(crime, final_911, on=["GEOID", "date", "hour_range"], how="left")
    merged["911_request_count_hour_range"] = merged["911_request_count_hour_range"].fillna(0).astype(int)
    merged["911_request_count_daily(before_24_hours)"] = merged["911_request_count_daily(before_24_hours)"].fillna(0).astype(int)

    # Kaydet
    merged.to_csv(output_merge_path, index=False)
    print(f"✅ Suç + 911 birleştirmesi tamamlandı → {output_merge_path}")
else:
    print("⚠️ Suç grid dosyası bulunamadı. Birleştirme yapılamadı.")
