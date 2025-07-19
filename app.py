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

    # Tüm satırlarda unicode karakterlerden kaçın (ç, ğ, ü, ö, ş, ✓, 📄 vb.)
    pdf.cell(200, 10, txt="Veri Guncelleme Raporu", ln=True, align='L')
    pdf.cell(200, 10, txt=f"Tarih/Saat: {timestamp}", ln=True, align='L')
    pdf.cell(200, 10, txt=f"Dosya: {file_name}", ln=True, align='L')
    pdf.cell(200, 10, txt=f"Toplam satir sayisi: {row_count:,}", ln=True, align='L')
    pdf.cell(200, 10, txt=f"NaN iceren sutun sayisi: {len(nan_cols)}", ln=True, align='L')

    if not nan_cols.empty:
        pdf.cell(200, 10, txt="NaN iceren sutunlar:", ln=True, align='L')
        for col, count in nan_cols.items():
            # Her bir satırı latin1 uyumlu hale getir
            line = f"- {col}: {count}"
            try:
                safe_line = line.encode("latin1", "replace").decode("latin1")
                pdf.cell(200, 10, txt=safe_line, ln=True, align='L')
            except:
                pdf.cell(200, 10, txt="(Yazdirilamayan karakter bulundu)", ln=True, align='L')
    else:
        pdf.cell(200, 10, txt="Hicbir sutunda eksik veri yok.", ln=True, align='L')

    output_name = f"report_{now.strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf.output(output_name)
    return output_name

# --- İndirme ---
if st.button("📥 sf_crime.csv dosyasini indir"):
    url = "https://raw.githubusercontent.com/cem5113/crime_prediction_data/main/sf_crime.csv"
    response = requests.get(url)
    if response.status_code == 200:
        with open("sf_crime.csv", "wb") as f:
            f.write(response.content)
        st.success("sf_crime.csv basariyla indirildi.")
    else:
        st.error("Indirme basarisiz.")

# --- Görüntüleme ve PDF ---
if os.path.exists("sf_crime.csv"):
    df = pd.read_csv("sf_crime.csv")

    # GEOID düzenle
    if "GEOID" in df.columns:
        df["GEOID"] = df["GEOID"].astype(str).str.extract(r"(\d+)")[0].str.zfill(11)

    st.subheader("Ilk 5 Satir")
    st.dataframe(df.head())
    st.info(f"Toplam satir sayisi: {len(df):,}")

    # NaN analiz
    nan_summary = df.isna().sum()
    nan_cols = nan_summary[nan_summary > 0]

    if not nan_cols.empty:
        st.warning("Eksik veriler bulundu:")
        st.dataframe(nan_cols.rename("NaN Sayisi"))
        st.write(f"NaN iceren sutun sayisi: {len(nan_cols)}")
        st.write(f"NaN iceren toplam satir sayisi: {df.isna().any(axis=1).sum()}")
    else:
        st.success("Hicbir sutunda NaN yok.")

    # PDF oluştur
    if st.button("PDF Rapor Olustur"):
        report_file = create_pdf_report("sf_crime.csv", len(df), nan_cols)
        with open(report_file, "rb") as f:
            st.download_button("Raporu Indir", data=f, file_name=report_file, mime="application/pdf")
