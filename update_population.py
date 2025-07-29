import os
import pandas as pd

# === 1. Dosya yolları (GitHub ve Streamlit uyumlu hale getirildi) ===
BASE_DIR = "crime_data"
os.makedirs(BASE_DIR, exist_ok=True)

CRIME_INPUT = os.path.join(BASE_DIR, "sf_crime_02.csv")
POPULATION_PATH = os.path.join(BASE_DIR, "sf_population.csv")
CRIME_OUTPUT = os.path.join(BASE_DIR, "sf_crime_03.csv")

# === 2. Verileri yükle ===
print("📥 Veriler yükleniyor...")
if not os.path.exists(CRIME_INPUT) or not os.path.exists(POPULATION_PATH):
    raise FileNotFoundError("❌ Gerekli dosyalardan biri eksik!")

df_crime = pd.read_csv(CRIME_INPUT)
df_population = pd.read_csv(POPULATION_PATH)

# === 3. GEOID formatlarını düzelt ===
df_crime["GEOID"] = df_crime["GEOID"].apply(lambda x: str(int(x)).zfill(11) if pd.notna(x) else pd.NA)
df_population["GEOID"] = df_population["GEOID"].astype(str).str.zfill(11)

# === 4. Nüfus verisini birleştir ===
df_merged = pd.merge(df_crime, df_population, on="GEOID", how="left")

# === 5. Eksik nüfusları 0 ile doldur ===
df_merged["population"] = df_merged["population"].fillna(0).astype("Int64")

# === 6. Özet bilgi ver ===
print("🔍 İlk 5 satır:")
print(df_merged[["GEOID", "population"]].head())

print(f"\n📊 Satır sayısı: {df_merged.shape[0]}")
print(f"📊 Sütun sayısı: {df_merged.shape[1]}")

# === 7. Kaydet ===
df_merged.to_csv(CRIME_OUTPUT, index=False)
print(f"\n✅ Birleştirilmiş çıktı kaydedildi → {CRIME_OUTPUT}")
