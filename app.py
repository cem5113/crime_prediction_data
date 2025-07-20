import streamlit as st
import pandas as pd
import requests
import os
from fpdf import FPDF
from datetime import datetime, timedelta

st.set_page_config(page_title="Veri Güncelleme", layout="wide")
st.title("📦 Veri Güncelleme ve Kontrol Arayüzü")

def create_pdf_report(file_name, row_count_before, nan_cols, row_count_after, removed_rows):
    now = datetime.now()
    timestamp = now.strftime("%d.%m.%Y %H:%M:%S")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    summary = (
        f"Tarih/Saat: {timestamp}; Dosya: {file_name} ; "
        f"Toplam satir sayisi: {row_count_before:,}; "
        f"NaN iceren sutun sayisi: {len(nan_cols)}"
    )
    pdf.cell(200, 10, txt=summary.encode("latin1", "replace").decode("latin1"), ln=True, align='L')

    if not nan_cols.empty:
        pdf.cell(200, 10, txt="NaN iceren sutunlar:", ln=True, align='L')
        for col, count in nan_cols.items():
            line = f"- {col}: {count}"
            safe_line = line.encode("latin1", "replace").decode("latin1")
            pdf.cell(200, 10, txt=safe_line, ln=True, align='L')

    pdf.cell(200, 10, txt=f"Revize satir sayisi: {row_count_after:,}", ln=True, align='L')
    pdf.cell(200, 10, txt=f"Silinen eski tarihli satir sayisi: {removed_rows:,}", ln=True, align='L')

    output_name = f"report_{now.strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf.output(output_name)
    return output_name

# 📥 İndirme
if st.button("📥 sf_crime.csv dosyasini indir"):
    url = "https://raw.githubusercontent.com/cem5113/crime_prediction_data/main/sf_crime.csv"
    response = requests.get(url)
    if response.status_code == 200:
        with open("sf_crime.csv", "wb") as f:
            f.write(response.content)
        st.success("sf_crime.csv basariyla indirildi.")
    else:
        st.error("Indirme basarisiz.")

# 🧹 Temizlik ve Gösterim
if os.path.exists("sf_crime.csv"):
    df = pd.read_csv("sf_crime.csv")
    original_row_count = len(df)

    if "GEOID" in df.columns:
        df["GEOID"] = df["GEOID"].astype(str).str.extract(r"(\d+)")[0].str.zfill(11)

    st.subheader("📋 Ilk 5 Satir")
    st.dataframe(df.head())
    st.info(f"Toplam satir sayisi: {original_row_count:,}")

    nan_summary = df.isna().sum()
    nan_cols = nan_summary[nan_summary > 0]

    if not nan_cols.empty:
        st.warning("Eksik veriler bulundu:")
        st.dataframe(nan_cols.rename("NaN Sayisi"))
        st.write(f"NaN iceren sutun sayisi: {len(nan_cols)}")
        st.write(f"NaN iceren toplam satir sayisi: {df.isna().any(axis=1).sum()}")

        df = df.dropna()
    else:
        st.success("Hicbir sutunda NaN yok.")

    # 🔁 Yalnızca son 5 yıla ait veriler
    removed_rows = 0
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        five_years_ago = datetime.now() - timedelta(days=5*365)
        before_filter = len(df)
        df = df[df["date"] >= five_years_ago]
        removed_rows = before_filter - len(df)

    # 📄 PDF Oluştur
    if st.button("📄 PDF Rapor Olustur"):
        report_file = create_pdf_report("sf_crime.csv", original_row_count, nan_cols, len(df), removed_rows)
        with open(report_file, "rb") as f:
            st.download_button("📎 Raporu Indir", data=f, file_name=report_file, mime="application/pdf")
