import streamlit as st
import pandas as pd
import requests
import os

st.title("📦 Veri Güncelleme ve Kontrol Arayüzü")

# 1. GitHub'dan veri indir
if st.button("📥 sf_crime.csv dosyasını indir"):
    url = "https://raw.githubusercontent.com/cem5113/crime_prediction_data/main/sf_crime.csv"
    response = requests.get(url)
    if response.status_code == 200:
        with open("sf_crime.csv", "wb") as f:
            f.write(response.content)
        st.success("sf_crime.csv başarıyla indirildi.")
    else:
        st.error("İndirme başarısız.")

# 2. Veriyi göster
if os.path.exists("sf_crime.csv"):
    df = pd.read_csv("sf_crime.csv")
    st.dataframe(df.head())
    st.write(f"Toplam satır: {len(df):,}")
