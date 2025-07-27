import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import geopandas as gpd
import json
from shapely.geometry import Point
from datetime import datetime, timedelta

st.set_page_config(page_title="Veri Güncelleme", layout="wide")
st.title("📦 Günlük Suç Tahmin Grid'i ve Zenginleştirme Paneli")

# === Dosya URL ve yolları ===
DOWNLOAD_GRID_URL = DOWNLOAD_GRID_URL = "https://raw.githubusercontent.com/cem5113/crime_prediction_data/main/crime_data/sf_crime_grid_full_labeled.csv"
DOWNLOAD_911_URL = "https://github.com/cem5113/crime_prediction_data/releases/download/v1.0.1/sf_911_last_5_year.csv"
DOWNLOAD_311_URL = "https://github.com/cem5113/crime_prediction_data/releases/download/v1.0.2/sf_311_last_5_years.csv"
DOWNLOAD_POPULATION_URL = "https://github.com/cem5113/crime_prediction_data/raw/main/sf_population.csv"
DOWNLOAD_BUS_URL = "https://github.com/cem5113/crime_prediction_data/raw/main/sf_bus_stops_with_geoid.csv"
DOWNLOAD_TRAIN_URL = "https://github.com/cem5113/crime_prediction_data/raw/main/sf_train_stops_with_geoid.csv"
DOWNLOAD_POIS_URL = "https://github.com/cem5113/crime_prediction_data/raw/main/sf_pois.geojson"
RISKY_POIS_JSON_PATH = "risky_pois_dynamic.json"
DOWNLOAD_RISKY_POIS_JSON_URL = "https://github.com/cem5113/crime_prediction_data/raw/main/risky_pois_dynamic.json"
DOWNLOAD_POLICE_URL = "https://github.com/cem5113/crime_prediction_data/raw/main/sf_police_stations.csv"
DOWNLOAD_GOV_URL = "https://github.com/cem5113/crime_prediction_data/raw/main/sf_government_buildings.csv"
DOWNLOAD_WEATHER_URL = "https://github.com/cem5113/crime_prediction_data/releases/download/latest/sf_weather_5years.csv"

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
            st.error(f"❌ {name} indirilemedi. Kod: {response.status_code}")
    except Exception as e:
        st.error(f"🚨 {name} indirilirken hata: {e}")

# === Butonla çalışan ana fonksiyon ===
if st.button("📅 Verileri İndir ve İlk 3 Satırı Göster"):
    try:
        download_and_preview("Tahmin Grid Verisi (Tüm GEOID × Zaman + Y_label)", DOWNLOAD_GRID_URL, "sf_crime_grid_full_labeled.csv")
        download_and_preview("911 Çağrıları", DOWNLOAD_911_URL, "sf_911_last_5_year.csv")
        download_and_preview("311 Çağrıları", DOWNLOAD_311_URL, "sf_311_last_5_years.csv")
        download_and_preview("Nüfus Verisi", DOWNLOAD_POPULATION_URL, "sf_population.csv")
        download_and_preview("Otobüs Durakları", DOWNLOAD_BUS_URL, "sf_bus_stops_with_geoid.csv")
        download_and_preview("Tren Durakları", DOWNLOAD_TRAIN_URL, "sf_train_stops_with_geoid.csv")
        download_and_preview("POI GeoJSON", DOWNLOAD_POIS_URL, "sf_pois.geojson", is_json=True)
        download_and_preview("POI Risk Skorları", DOWNLOAD_RISKY_POIS_JSON_URL, RISKY_POIS_JSON_PATH, is_json=True)
        download_and_preview("Polis İstasyonları", DOWNLOAD_POLICE_URL, "sf_police_stations.csv")
        download_and_preview("Devlet Binaları", DOWNLOAD_GOV_URL, "sf_government_buildings.csv")
        download_and_preview("Hava Durumu", DOWNLOAD_WEATHER_URL, "sf_weather_5years.csv")

        st.success("✅ Tüm veriler başarıyla indirildi ve ilk 3 satır gösterildi.")

    except Exception as e:
        st.error(f"❌ Genel hata oluştu: {e}")
