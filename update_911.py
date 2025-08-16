import os
from datetime import datetime
from pathlib import Path

import pandas as pd

# === 0) Yardımcılar ===
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

def find_col(ci_names, candidates):
    """Case-insensitive sütun bulucu. Eşleşirse gerçek adını döner, yoksa None."""
    lower_map = {c.lower(): c for c in ci_names}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None

def normalize_geoid(s: pd.Series, target_len: int) -> pd.Series:
    s = s.astype(str).str.extract(r"(\d+)")[0]
    return s.str.zfill(target_len)

# === 1) Dosya yolları ===
BASE_DIR = "crime_data"
Path(BASE_DIR).mkdir(exist_ok=True)

raw_911_path       = os.path.join(BASE_DIR, "sf_911_full_raw.csv")
summary_911_path   = os.path.join(BASE_DIR, "sf_911_last_5_year.csv")
crime_grid_path_1  = os.path.join(BASE_DIR, "sf_crime_grid_full_labeled.csv")
crime_grid_path_2  = os.path.join(".",       "sf_crime_grid_full_labeled.csv")  # fallback
output_merge_path  = os.path.join(BASE_DIR, "sf_crime_01.csv")

# === 2) 911 verisini yükle ===
if not os.path.exists(raw_911_path):
    raise FileNotFoundError(f"❌ 911 ham dosyası bulunamadı: {raw_911_path}")

# Büyük dosyalarda stabil olsun diye low_memory=False
df = pd.read_csv(raw_911_path, low_memory=False)
print(f"📥 911 ham veri yüklendi: {len(df)} satır")

# === 2.1) Datetime kolonu tespiti ===
dt_col = find_col(df.columns, ["datetime", "incident_datetime", "call_datetime", "created_at",
                               "call_date", "received dttm", "received_dt", "received_dt_tm"])
if dt_col is None:
    raise ValueError("❌ 911 verisinde datetime kolonu bulunamadı (ör. 'datetime').")

df["datetime"] = pd.to_datetime(df[dt_col], errors="coerce")
df = df.dropna(subset=["datetime"]).copy()

# === 2.2) GEOID kolonu tespiti ===
geoid_col = find_col(df.columns, ["GEOID", "geoid", "geoid10", "block_geoid", "tract_geoid"])
if geoid_col is None:
    raise ValueError("❌ 911 verisinde GEOID kolonu bulunamadı (ör. 'GEOID').")
df["GEOID"] = df[geoid_col].astype(str)

# === 3) Son 5 yılı filtrele ===
today = pd.Timestamp.today().normalize()
five_years_ago = today - pd.DateOffset(years=5)
df = df[df["datetime"] >= five_years_ago].copy()
print(f"🗓️ 5 yıllık filtre sonrası: {len(df)} satır (>= {five_years_ago.date()})")

# === 4) Zaman özellikleri ===
df["date"] = df["datetime"].dt.date
df["hour"] = df["datetime"].dt.hour
hr = (df["hour"] // 3) * 3
df["hour_range"] = hr.astype(str) + "-" + (hr + 3).astype(str)

# === 5) Suç grid dosyasını yükle (hem target GEOID uzunluğunu öğrenmek hem merge için) ===
crime_grid_path = crime_grid_path_1 if os.path.exists(crime_grid_path_1) else crime_grid_path_2
if not os.path.exists(crime_grid_path):
    raise FileNotFoundError("❌ Suç grid dosyası bulunamadı: "
                            f"{crime_grid_path_1} veya {crime_grid_path_2}")

crime = pd.read_csv(crime_grid_path, dtype={"GEOID": str}, low_memory=False)
print(f"📥 Suç grid yüklendi: {len(crime)} satır ({crime_grid_path})")

if "event_hour" not in crime.columns:
    raise ValueError("❌ Suç grid dosyasında 'event_hour' sütunu eksik!")

# GEOID hedef uzunluğu (grid’e göre otomatik)
target_len = crime["GEOID"].dropna().astype(str).str.len().mode().iat[0]
df["GEOID"]     = normalize_geoid(df["GEOID"], target_len)
crime["GEOID"]  = normalize_geoid(crime["GEOID"], target_len)

# === 6) 911 özet tablo (5 yıl) ===
hourly_summary = (
    df.groupby(["GEOID", "date", "hour_range"]).size()
      .reset_index(name="911_request_count_hour_range")
)
daily_summary = (
    df.groupby(["GEOID", "date"]).size()
      .reset_index(name="911_request_count_daily(before_24_hours)")
)
final_911 = pd.merge(hourly_summary, daily_summary, on=["GEOID", "date"], how="left")

# Kaydet
safe_save_csv(final_911, summary_911_path)
print(f"✅ 911 özeti kaydedildi → {summary_911_path}")

# === 7) Suç grid ile birleştir ===
# event_hour → hour_range
crime["hour_range"] = ((crime["event_hour"] // 3) * 3).astype(int)
crime["hour_range"] = crime["hour_range"].astype(str) + "-" + (crime["hour_range"].astype(int) + 3).astype(str)

# tarih tipini hizala
if "date" not in crime.columns:
    if "datetime" in crime.columns:
        crime["date"] = pd.to_datetime(crime["datetime"], errors="coerce").dt.date
    else:
        raise ValueError("❌ Suç grid dosyasında 'date' veya 'datetime' sütunu bulunamadı!")
else:
    crime["date"] = pd.to_datetime(crime["date"], errors="coerce").dt.date

# Merge
merged = pd.merge(crime, final_911, on=["GEOID", "date", "hour_range"], how="left")
merged["911_request_count_hour_range"] = merged["911_request_count_hour_range"].fillna(0).astype(int)
merged["911_request_count_daily(before_24_hours)"] = merged["911_request_count_daily(before_24_hours)"].fillna(0).astype(int)

# Kaydet
safe_save_csv(merged, output_merge_path)
print(f"✅ Suç + 911 birleştirmesi tamamlandı → {output_merge_path}")
