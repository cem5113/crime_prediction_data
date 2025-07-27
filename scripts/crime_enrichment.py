import subprocess
import os

def install_requirements():
    if os.path.exists("requirements.txt"):
        subprocess.check_call(["pip", "install", "-r", "requirements.txt"])

# Önce gereklilikleri yükle
install_requirements()

# Sonra import işlemleri
import pandas as pd
import os

# === 1. Dosya yolları ===
grid_path = "data/sf_crime_grid_full_labeled.csv"
calls_911_path = "sf_911_last_5_year.csv"
output_path = "data/sf_crime_01.csv"

# === 2. Verileri yükle ===
df_grid = pd.read_csv(grid_path, dtype={"GEOID": str})
df_911 = pd.read_csv(calls_911_path, dtype={"GEOID": str})

# === 3. Tarih ve saat bilgilerini kontrol et ===
if "event_hour" not in df_grid.columns:
    if "time" in df_grid.columns:
        df_grid["event_hour"] = pd.to_datetime(df_grid["time"], errors="coerce").dt.hour
    else:
        raise ValueError("event_hour veya time bilgisi bulunamadı.")

df_grid["hour_range"] = (df_grid["event_hour"] // 3) * 3
df_grid["hour_range"] = df_grid["hour_range"].astype(str) + "-" + (df_grid["hour_range"].astype(int) + 3).astype(str)

if "date" not in df_grid.columns:
    if "datetime" in df_grid.columns:
        df_grid["date"] = pd.to_datetime(df_grid["datetime"]).dt.date
    else:
        raise ValueError("date bilgisi eksik.")

# === 4. 911 verisini birleştir ===
df_merge = pd.merge(
    df_grid,
    df_911,
    on=["GEOID", "date", "hour_range"],
    how="left"
)

# === 5. Eksik verileri 0 ile doldur ===
df_merge["911_request_count_hour_range"] = df_merge["911_request_count_hour_range"].fillna(0).astype(int)
df_merge["911_request_count_daily(before_24_hours)"] = df_merge["911_request_count_daily(before_24_hours)"].fillna(0).astype(int)

# === 6. Kaydet ===
os.makedirs("data", exist_ok=True)
df_merge.to_csv(output_path, index=False)

# === 7. Bilgi mesajı ve ilk 5 satır ===
print("✅ Birleştirme tamamlandı →", output_path)
print("📌 Yeni sütunlar: 911_request_count_hour_range, 911_request_count_daily(before_24_hours)")
print("📋 İlk 5 satır:")
print(df_merge.head())
