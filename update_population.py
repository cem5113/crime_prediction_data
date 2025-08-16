# scripts/update_population.py
import os
from pathlib import Path
import pandas as pd
import numpy as np

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

def find_col(ci_names, candidates):
    m = {c.lower(): c for c in ci_names}
    for cand in candidates:
        if cand.lower() in m:
            return m[cand.lower()]
    return None

def normalize_geoid(s: pd.Series, target_len: int) -> pd.Series:
    s = s.astype(str).str.extract(r"(\d+)")[0]
    return s.str.zfill(target_len)

def choose_geoid_len(*series, default_len: int = 12) -> int:
    lens = []
    for ser in series:
        if ser is None:
            continue
        v = ser.astype(str).str.extract(r"(\d+)")[0].dropna()
        if not v.empty:
            lens.extend(v.str.len().tolist())
    if not lens:
        return default_len
    return int(pd.Series(lens).mode().iat[0])

# ============== 1) Dosya yolları ==============
BASE_DIR = "crime_data"
Path(BASE_DIR).mkdir(exist_ok=True)

CRIME_INPUT_CANDIDATES = [
    os.path.join(BASE_DIR, "sf_crime_02.csv"),
    os.path.join(".",       "sf_crime_02.csv"),
]
POPULATION_PATH_CANDIDATES = [
    os.path.join(BASE_DIR, "sf_population.csv"),
    os.path.join(".",       "sf_population.csv"),
]
CRIME_OUTPUT = os.path.join(BASE_DIR, "sf_crime_03.csv")

def pick_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

crime_input_path = pick_existing(CRIME_INPUT_CANDIDATES)
population_path  = pick_existing(POPULATION_PATH_CANDIDATES)

if not crime_input_path or not population_path:
    raise FileNotFoundError("❌ Gerekli dosyalardan biri eksik! "
                            f"(crime: {crime_input_path}, population: {population_path})")

print("📥 Veriler yükleniyor...")
# GEOID’leri güvenli okumak için dtype=str
df_crime = pd.read_csv(crime_input_path, dtype={"GEOID": str}, low_memory=False)
df_pop   = pd.read_csv(population_path, dtype=str, low_memory=False)

# ============== 2) Kolon adlarını tespit et ==============
crime_geoid_col = find_col(df_crime.columns, ["GEOID", "geoid", "geoid10", "block_geoid", "tract_geoid"])
if crime_geoid_col is None:
    raise KeyError("❌ Suç verisinde GEOID kolonu bulunamadı.")

pop_geoid_col = find_col(df_pop.columns, ["GEOID", "geoid", "GEOID10", "geoid10", "block_geoid", "TRACTCE", "BLOCKID"])
if pop_geoid_col is None:
    raise KeyError("❌ Nüfus verisinde GEOID kolonu bulunamadı.")

pop_val_col = find_col(
    df_pop.columns,
    ["population", "total_population", "pop_total", "POP", "POPTOTL", "B01003e1", "B01003_001E"]
)
if pop_val_col is None:
    raise KeyError("❌ Nüfus verisinde population değeri için bir kolon bulunamadı.")

# ============== 3) GEOID uzunluğunu belirle & normalize et ==============
target_len = choose_geoid_len(df_crime[crime_geoid_col], df_pop[pop_geoid_col], default_len=12)
df_crime["GEOID"] = normalize_geoid(df_crime[crime_geoid_col], target_len)
df_pop["GEOID"]   = normalize_geoid(df_pop[pop_geoid_col], target_len)

# Nüfus sayısını sayısal yap
df_pop["population"] = pd.to_numeric(df_pop[pop_val_col], errors="coerce")

# ============== 4) Birleştir ==============
to_merge = df_pop[["GEOID", "population"]].copy()
df_merged = pd.merge(df_crime, to_merge, on="GEOID", how="left")

# ============== 5) Eksikleri doldur & tipler ==============
if df_merged["population"].isna().any():
    df_merged["population"] = df_merged["population"].fillna(0)

# mümkünse tam sayı tut
if np.isclose(df_merged["population"] % 1, 0, atol=1e-9).all():
    df_merged["population"] = df_merged["population"].round().astype("Int64")
else:
    df_merged["population"] = df_merged["population"].astype(float)

# ============== 6) Özet ==============
print("🔍 İlk 5 satır:")
try:
    print(df_merged[["GEOID", "population"]].head().to_string(index=False))
except Exception:
    print(df_merged.head())

print(f"\n📊 Satır sayısı: {df_merged.shape[0]}")
print(f"📊 Sütun sayısı: {df_merged.shape[1]}")

# ============== 7) Kaydet ==============
safe_save_csv(df_merged, CRIME_OUTPUT)
print(f"\n✅ Birleştirilmiş çıktı kaydedildi → {CRIME_OUTPUT}")
