# === GEREKLİ MODÜLLERİ YÜKLE VE PATH SORUNUNU GİDER ===
import sys
import site
import os
import requests

# .local/site-packages yolunu ekle
user_site = site.getusersitepackages()
if user_site not in sys.path:
    sys.path.append(user_site)

# pip ile eksik modül varsa yükle
def ensure_package(package_name):
    try:
        __import__(package_name)
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", package_name])

for package in ["pandas"]:
    ensure_package(package)

# === MODÜLLERİ İÇE AKTAR ===
import pandas as pd

# === 1. GITHUB'DAN VERİLERİ OKU ===
grid_path = "https://raw.githubusercontent.com/cem5113/crime_prediction_data/main/sf_crime_grid_full_labeled.csv"
calls_911_path = "https://github.com/cem5113/crime_prediction_data/releases/download/v1.0.1/sf_911_last_5_year.csv"
output_path = "sf_crime_01.csv"  # main klasöre yazılacak

# === 2. VERİLERİ YÜKLE ===
print("📥 Veriler yükleniyor...")
df_grid = pd.read_csv(grid_path, dtype={"GEOID": str})
df_911 = pd.read_csv(calls_911_path, dtype={"GEOID": str})
print("✅ Veriler yüklendi.")

# === 3. ZAMAN TEMELLİ ÖZELLİKLERİ ÜRET ===
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

# === 4. 911 VERİSİNİ BİRLEŞTİR ===
df_merge = pd.merge(
    df_grid,
    df_911,
    on=["GEOID", "date", "hour_range"],
    how="left"
)

# === 5. EKSİK DEĞERLERİ DOLDUR ===
df_merge["911_request_count_hour_range"] = df_merge["911_request_count_hour_range"].fillna(0).astype(int)
df_merge["911_request_count_daily(before_24_hours)"] = df_merge["911_request_count_daily(before_24_hours)"].fillna(0).astype(int)

# === 6. CSV OLARAK KAYDET ===
df_merge.to_csv(output_path, index=False)

# === 7. ÖZET BİLGİ ===
print("✅ Birleştirme tamamlandı →", output_path)
print("📌 Yeni sütunlar: 911_request_count_hour_range, 911_request_count_daily(before_24_hours)")
print("📋 İlk 5 satır:")
print(df_merge.head())
