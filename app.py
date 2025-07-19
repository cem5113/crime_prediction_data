import streamlit as st
import pandas as pd
import requests
import os
from fpdf import FPDF
from datetime import datetime

st.set_page_config(page_title="Veri Güncelleme", layout="wide")
st.title("📦 Veri Güncelleme ve Kontrol Arayüzü")

def create_pdf_report(file_name, row_count, nan_cols):
    now = datetime.now()
    timestamp = now.strftime("%d.%m.%Y %H:%M:%S")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt=f"Veri Güncelleme Raporu", ln=True, align='L')
    pdf.cell(200, 10, txt=f"Tarih/Saat: {timestamp}", ln=True, align='L')
    pdf.cell(200, 10, txt=f"Dosya: {file_name}", ln=True, align='L')
    pdf.cell(200, 10, txt=f"Toplam satır sayısı: {row_count:,}", ln=True, align='L')
    pdf.cell(200, 10, txt=f"NaN içeren sütun sayısı: {len(nan_cols)}", ln=True, align='L')

    if not nan_cols.empty:
        pdf.cell(200, 10, txt="NaN içeren sütunlar:", ln=True, align='L')
        for col, count in nan_cols.items():
            pdf.cell(200, 10, txt=f"- {col}: {count}", ln=True, align='L')
    else:
        pdf.cell(200, 10, txt="✅ Hiçbir sütunda eksik veri yok.", ln=True, align='L')

    output_name = f"report_{now.strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf.output(output_name)
    return output_name

# --- İndirme ---
if st.button("📥 sf_crime.csv dosyasını indir"):
    url = "https://raw.githubusercontent.com/cem5113/crime_prediction_data/main/sf_crime.csv"
    response = requests.get(url)
    if response.status_code == 200:
        with open("sf_crime.csv", "wb") as f:
            f.write(response.content)
        st.success("✅ sf_crime.csv başarıyla indirildi.")
    else:
        st.error("❌ İndirme başarısız.")

# --- Görüntüleme ve PDF ---
if os.path.exists("sf_crime.csv"):
    df = pd.read_csv("sf_crime.csv")

    # GEOID düzenle
    if "GEOID" in df.columns:
        df["GEOID"] = df["GEOID"].astype(str).str.extract(r"(\d+)")[0].str.zfill(11)

    st.subheader("📊 İlk 5 Satır")
    st.dataframe(df.head())
    st.info(f"Toplam satır sayısı: {len(df):,}")

    # NaN analiz
    nan_summary = df.isna().sum()
    nan_cols = nan_summary[nan_summary > 0]

    if not nan_cols.empty:
        st.warning("⚠️ Eksik veriler bulundu:")
        st.dataframe(nan_cols.rename("NaN Sayısı"))
        st.write(f"NaN içeren sütun sayısı: {len(nan_cols)}")
        st.write(f"NaN içeren toplam satır sayısı: {df.isna().any(axis=1).sum()}")
    else:
        st.success("✅ Hiçbir sütunda NaN yok.")

    # PDF oluştur
    if st.button("📝 PDF Rapor Oluştur"):
        report_file = create_pdf_report("sf_crime.csv", len(df), nan_cols)
        with open(report_file, "rb") as f:
            st.download_button("📄 Raporu İndir", data=f, file_name=report_file, mime="application/pdf")
