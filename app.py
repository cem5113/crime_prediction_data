import streamlit as st
import pandas as pd
import requests
import os
from fpdf import FPDF
from datetime import datetime, timedelta

st.set_page_config(page_title="Veri Guncelleme", layout="wide")
st.title("📦 Veri Guncelleme ve Kontrol Arayuzu")

def create_pdf_report(file_name, original_count, revised_count, nan_cols):
    now = datetime.now()
    timestamp = now.strftime("%d.%m.%Y %H:%M:%S")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Tek satırda özet
    line = f"Tarih/Saat: {timestamp}; Dosya: {file_name} ; Toplam satir sayisi: {original_count:,}; NaN iceren sutun sayisi: {len(nan_cols)}"
    
    if not nan_cols.empty:
        for col, count in nan_cols.items():
            line += f"; - {col}: {count}"
    
    line += f"; Revize satir sayisi: {revised_count:,}"

    safe_line = line.encode("latin1", "replace").decode("latin1")
    pdf.multi_cell(0, 10, txt=safe_line)

    output_name = f"report_{now.strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf.output(output_name)
    return output_name

# --- Dosya indir ---
if st.button("📥 sf_crime.csv dosyasini indir"):
    url = "https://raw.githubusercontent.com/cem5113/crime_prediction_data/main/sf_crime.csv"
    response = requests.get(url)
    if response.status_code == 200:
        with open("sf_crime.csv", "wb") as f:
            f.write(response.content)
        st.success("sf_crime.csv basariyla indirildi.")
    else:
        st.error("Indirme basarisiz.")

# --- Veriyi isleme ve rapor ---
if os.path.exists("sf_crime.csv"):
    df = pd.read_csv("sf_crime.csv")
    original_count = len(df)

    # GEOID temizle
    if "GEOID" in df.columns:
        df["GEOID"] = df["GEOID"].astype(str).str.extract(r"(\\d+)")[0].str.zfill(11)

    # NaN analiz
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

    # 5 yildan eski satirlari filtrele (tarih sütunu varsa)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        cutoff = datetime.today() - timedelta(days=5*365)
        df = df[df["date"] >= cutoff]

    revised_count = len(df)

    st.subheader("Ilk 5 Satir")
    st.dataframe(df.head())
    st.info(f"Revize satir sayisi: {revised_count:,}")

    if st.button("📄 PDF Rapor Olustur"):
        report_file = create_pdf_report("sf_crime.csv", original_count, revised_count, nan_cols)
        with open(report_file, "rb") as f:
            st.download_button("📎 Raporu Indir", data=f, file_name=report_file, mime="application/pdf")
