# scripts/update_sf_crime.py
import pandas as pd
import requests
from datetime import datetime, timedelta

url = "https://raw.githubusercontent.com/cem5113/crime_prediction_data/main/sf_crime.csv"
response = requests.get(url)

if response.status_code == 200:
    with open("sf_crime.csv", "wb") as f:
        f.write(response.content)
    print("✅ sf_crime.csv başarıyla indirildi.")
else:
    print("❌ İndirme başarısız.")

# GEOID temizliği ve 5 yıl filtresi
df = pd.read_csv("sf_crime.csv")
df["GEOID"] = df["GEOID"].astype(str).str.extract(r"(\d+)")[0].str.zfill(11)

if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    five_years_ago = datetime.now() - timedelta(days=5*365)
    df = df[df["date"] >= five_years_ago]

df = df.dropna()
df.to_csv("sf_crime.csv", index=False)
print(f"📁 sf_crime.csv dosyası güncellendi. Kayıt sayısı: {len(df):,}")
