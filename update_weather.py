# scripts/update_weather.py
import os
from pathlib import Path
import numpy as np
import pandas as pd

# ============== Yardımcılar ==============
def ensure_parent(path: str):
    Path(os.path.dirname(path) or ".").mkdir(parents=True, exist_ok=True)

def safe_save_csv(df: pd.DataFrame, path: str):
    try:
        ensure_parent(path)
        df.to_csv(path, index=False)
    except Exception as e:
        print(f"❌ Kaydetme hatası: {path}\n{e}")
        df.to_csv(path + ".bak", index=False)
        print(f"📁 Yedek oluşturuldu: {path}.bak")

def pick_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

def find_col(cols, candidates):
    m = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in m:
            return m[c.lower()]
    return None

# ============== 1) Dosya yolları ==============
BASE_DIR = "crime_data"
Path(BASE_DIR).mkdir(exist_ok=True)

CRIME_INPUT_CANDS = [
    os.path.join(BASE_DIR, "sf_crime_07.csv"),
    os.path.join(".",       "sf_crime_07.csv"),
]
WEATHER_CANDS = [
    os.path.join(BASE_DIR, "sf_weather_5years.csv"),
    os.path.join(".",       "sf_weather_5years.csv"),
]
CRIME_OUTPUT = os.path.join(BASE_DIR, "sf_crime_08.csv")

crime_path   = pick_existing(CRIME_INPUT_CANDS)
weather_path = pick_existing(WEATHER_CANDS)

if not crime_path or not weather_path:
    raise FileNotFoundError(f"❌ Gerekli dosyalardan biri yok. crime={crime_path}, weather={weather_path}")

print("📥 Veriler yükleniyor...")
# Tarihi daha rahat hizalamak için dtype'ları esnek alalım
df_crime   = pd.read_csv(crime_path, low_memory=False)
df_weather = pd.read_csv(weather_path, low_memory=False)

# ============== 2) Tarih sütunlarını normalize et ==============
# crime: date yoksa datetime'tan türet
if "date" in df_crime.columns:
    df_crime["date"] = pd.to_datetime(df_crime["date"], errors="coerce").dt.date
elif "datetime" in df_crime.columns:
    df_crime["date"] = pd.to_datetime(df_crime["datetime"], errors="coerce").dt.date
else:
    raise KeyError("❌ Suç verisinde 'date' veya 'datetime' sütunu bulunamadı.")

# weather: tarih kolonu adını bul
date_col = find_col(df_weather.columns, ["DATE", "date", "obs_date"])
if date_col is None:
    raise KeyError("❌ Hava durumu verisinde tarih kolonu (DATE/date) bulunamadı.")
df_weather[date_col] = pd.to_datetime(df_weather[date_col], errors="coerce").dt.date

# Geçersiz tarihleri temizle
df_crime  = df_crime.dropna(subset=["date"]).copy()
df_weather = df_weather.dropna(subset=[date_col]).copy()

# ============== 3) NOAA dönüşümleri (birim güvenli) ==============
# Olası kolon adlarını bul
tmax_col = find_col(df_weather.columns, ["TMAX", "tmax"])
tmin_col = find_col(df_weather.columns, ["TMIN", "tmin"])
prcp_col = find_col(df_weather.columns, ["PRCP", "prcp"])

# NOAA genelde: TMAX/TMIN = 0.1 °C, PRCP = 0.1 mm
def to_celsius(series):
    s = pd.to_numeric(series, errors="coerce")
    # bazı kaynaklar zaten °C olabilir; heuristik: tipik aralık [-50, 60]
    if s.abs().median() > 80:  # 0.1°C ölçeğinde gibi görünüyor
        s = s / 10.0
    return s

def to_mm(series):
    s = pd.to_numeric(series, errors="coerce")
    # heuristik: değerler genelde 0-2000 bandındaysa 0.1mm olabilir
    if s.max(skipna=True) and s.max(skipna=True) > 200:  # 0.1 mm ölçeği
        s = s / 10.0
    return s

df_weather["temp_max"] = to_celsius(df_weather[tmax_col]) if tmax_col else np.nan
df_weather["temp_min"] = to_celsius(df_weather[tmin_col]) if tmin_col else np.nan
df_weather["precipitation_mm"] = to_mm(df_weather[prcp_col]) if prcp_col else np.nan
df_weather["temp_range"] = (df_weather["temp_max"] - df_weather["temp_min"]).round(1)

# ============== 4) Çok istasyonlu dosya ise: Güne göre tekilleştir ==============
# Aynı tarihte birden fazla satır varsa (farklı istasyonlar), rasyonel bir şekilde özetle:
agg = (
    df_weather
    .groupby([date_col], as_index=False)
    .agg({
        "temp_max": "max",               # günün en yüksek sıcaklığı
        "temp_min": "min",               # günün en düşük sıcaklığı
        "temp_range": "max",             # range yeniden hesaplamaya gerek yok; max makul
        "precipitation_mm": "sum"        # toplam yağış (mm)
    })
    .rename(columns={date_col: "date"})
)

# ============== 5) Suç verisiyle birleştir ==============
df_merged = pd.merge(df_crime, agg, on="date", how="left")

# ============== 6) Kaydet & Özet ==============
safe_save_csv(df_merged, CRIME_OUTPUT)

print(f"✅ Hava durumu eklendi → {CRIME_OUTPUT}")
print("📄 Eklenen sütunlar:", ["temp_max", "temp_min", "temp_range", "precipitation_mm"])
print(f"📊 Satır sayısı: {df_merged.shape[0]}, Sütun sayısı: {df_merged.shape[1]}")
try:
    print(df_merged[["date", "temp_max", "temp_min", "temp_range", "precipitation_mm"]].head().to_string(index=False))
except Exception:
    pass
