import streamlit as st
import pandas as pd
import requests
import os

st.set_page_config(page_title="Veri Güncelleme", layout="wide")
st.title("📦 Veri Güncelleme ve Kontrol Arayüzü")

# 1. GitHub'dan veri indir
if st.button("📥 sf_crime.csv dosyasını indir"):
    url = "https://raw.githubusercontent.com/cem5113/crime_prediction_data/main/sf_crime.csv"
    response = requests.get(url)
    if response.status_code == 200:
        with open("sf_crime.csv", "wb") as f:
            f.write(response.content)
        st.success("✅ sf_crime.csv başarıyla indirildi.")
    else:
        st.error("❌ İndirme başarısız.")

# 2. Veriyi oku ve göster
if os.path.exists("sf_crime.csv"):
    try:
        df = pd.read_csv("sf_crime.csv")

        # GEOID sütununu düzelt
        if "GEOID" in df.columns:
            df["GEOID"] = df["GEOID"].astype(str).str.extract(r"(\d+)")[0].str.zfill(11)

        st.subheader("📊 İlk 5 Satır")
        st.dataframe(df.head())

        # Genel istatistik
        st.info(f"Toplam satır sayısı: {len(df):,}")

        # NaN analiz
        nan_summary = df.isna().sum()
        nan_cols = nan_summary[nan_summary > 0]

        if not nan_cols.empty:
            st.warning("⚠️ Eksik veriler bulundu:")
            st.dataframe(nan_cols.rename("NaN Sayısı"))
            st.write(f"NaN içeren sütun sayısı: {len(nan_cols)}")
            st.write(f"NaN içeren toplam satır sayısı (en az 1 sütun): {df.isna().any(axis=1).sum()}")
        else:
            st.success("✅ Herhangi bir NaN değeri bulunamadı.")

    except Exception as e:
        st.error(f"❌ Dosya okunurken hata oluştu: {e}")
else:
    st.warning("Henüz sf_crime.csv dosyası indirilmedi.")
