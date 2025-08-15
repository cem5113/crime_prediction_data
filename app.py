import streamlit as st
import pandas as pd
import requests
import os
import geopandas as gpd
import json
import subprocess
from datetime import datetime

# === Streamlit sayfa ayarları ===
st.set_page_config(page_title="Veri Güncelleme", layout="wide")
st.title("📦 Günlük Suç Tahmin Grid'i ve Zenginleştirme Paneli")


# === Gereklilikleri yükleme fonksiyonu ===
def install_requirements():
    if os.path.exists("requirements.txt"):
        try:
            subprocess.check_call(["pip", "install", "-r", "requirements.txt"])
            st.success("✅ requirements.txt başarıyla yüklendi.")
        except Exception as e:
            st.error(f"❌ Gereklilikler yüklenemedi: {e}")
    else:
        st.error("❌ requirements.txt dosyası bulunamadı.")


# === Dosya URL ve yolları ===
DOWNLOADS = {
    "Tahmin Grid Verisi (GEOID × Zaman + Y_label)": {
        "url": "https://raw.githubusercontent.com/cem5113/crime_prediction_data/main/sf_crime_grid_full_labeled.csv",
        "path": "sf_crime_grid_full_labeled.csv"
    },
    "911 Çağrıları": {
        "url": "https://github.com/cem5113/crime_prediction_data/releases/download/v1.0.1/sf_911_last_5_year.csv",
        "path": "sf_911_last_5_year.csv"
    },
    "311 Çağrıları": {
        "url": "https://github.com/cem5113/crime_prediction_data/releases/download/v1.0.2/sf_311_last_5_years.csv",
        "path": "sf_311_last_5_years.csv"
    },
    "Nüfus Verisi": {
        "url": "https://github.com/cem5113/crime_prediction_data/raw/main/sf_population.csv",
        "path": "sf_population.csv"
    },
    "Otobüs Durakları": {
        "url": "https://github.com/cem5113/crime_prediction_data/raw/main/sf_bus_stops_with_geoid.csv",
        "path": "sf_bus_stops_with_geoid.csv"
    },
    "Tren Durakları": {
        "url": "https://github.com/cem5113/crime_prediction_data/raw/main/sf_train_stops_with_geoid.csv",
        "path": "sf_train_stops_with_geoid.csv"
    },
    "POI GeoJSON": {
        "url": "https://github.com/cem5113/crime_prediction_data/raw/main/sf_pois.geojson",
        "path": "sf_pois.geojson",
        "is_json": True
    },
    "POI Risk Skorları": {
        "url": "https://github.com/cem5113/crime_prediction_data/raw/main/risky_pois_dynamic.json",
        "path": "risky_pois_dynamic.json",
        "is_json": True
    },
    "Polis İstasyonları": {
        "url": "https://github.com/cem5113/crime_prediction_data/raw/main/sf_police_stations.csv",
        "path": "sf_police_stations.csv"
    },
    "Devlet Binaları": {
        "url": "https://github.com/cem5113/crime_prediction_data/raw/main/sf_government_buildings.csv",
        "path": "sf_government_buildings.csv"
    },
    "Hava Durumu": {
        "url": "https://raw.githubusercontent.com/cem5113/crime_prediction_data/main/sf_weather_5years.csv",
        "path": "sf_weather_5years.csv"
    },
}


# === Veri indirme ve önizleme fonksiyonu ===
def download_and_preview(name, url, file_path, is_json=False):
    st.markdown(f"### 🔹 {name}")
    try:
        response = requests.get(url)
        if response.status_code == 200:
            if is_json:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(response.text)
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                st.json(data if isinstance(data, dict) else data[:3])
            else:
                with open(file_path, "wb") as f:
                    f.write(response.content)
                df = pd.read_csv(file_path)
                st.dataframe(df.head(3))
                st.caption(f"📌 Sütunlar: {list(df.columns)}")
        else:
            st.error(f"❌ {name} indirilemedi. HTTP Kod: {response.status_code}")
    except Exception as e:
        st.error(f"🚨 {name} indirilemedi: {e}")


# === 1. Verileri indir ve göster ===
if st.button("📥 Verileri İndir ve Önizle (İlk 3 Satır)"):
    for name, info in DOWNLOADS.items():
        download_and_preview(name, info["url"], info["path"], is_json=info.get("is_json", False))
    st.success("✅ Tüm veriler indirildi ve önizleme tamamlandı.")


# === 2. Zenginleştirme scriptini çalıştır ===
if st.button("⚙️ Zenginleştirme Scriptini Çalıştır (crime_enrichment.py)"):
    try:
        result = subprocess.run(["python", "scripts/crime_enrichment.py"], capture_output=True, text=True)
        if result.returncode == 0:
            st.success("✅ crime_enrichment.py başarıyla çalıştırıldı.")
            st.code(result.stdout)
        else:
            st.error("❌ Script çalıştırılırken hata oluştu.")
            st.code(result.stderr)
    except Exception as e:
        st.error(f"🚨 Subprocess hatası: {e}")
