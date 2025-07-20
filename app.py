import streamlit as st
import pandas as pd
import requests
import os
from fpdf import FPDF
from datetime import datetime, timedelta

st.set_page_config(page_title="Veri Güncelleme", layout="wide")
st.title("📦 Veri Güncelleme ve Kontrol Arayüzü")

def create_pdf_report(file_name, row_count_before, nan_cols, row_count_after, removed_rows, dropped_nan_rows):
    now = datetime.now()
    timestamp = now.strftime("%d.%m.%Y %H:%M:%S")

    # NaN sütunlarını tek stringe dönüştür
    if not nan_cols.empty:
        nan_parts = [f"- {col}: {count}" for col, count in nan_cols.items()]
        nan_text = " ".join(nan_parts)
    else:
        nan_text = "Yok"

    # PDF özet satırı
    summary = (
        f"- Tarih/Saat: {timestamp}; "
        f"Dosya: {file_name} ; "
        f"Toplam satir sayisi: {row_count_before:,}; "
        f"NaN iceren sutunlar: {nan_text}; "
        f"NaN nedeniyle silinen satir sayisi: {dropped_nan_rows}; "
        f"Revize satir sayisi: {row_count_after:,}; "
        f"Silinen eski tarihli satir sayisi: {removed_rows}"
    )

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=summary.encode("latin1", "replace").decode("latin1"))

    output_name = f"report_{now.strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf.output(output_name)
    return output_name

# 📥 GitHub'dan veri indir
if st.button("📥 sf_crime.csv dosyasini indir"):
    url = "https://raw.githubusercontent.com/cem5113/crime_prediction_data/main/sf_crime.csv"
    response = requests.get(url)
    if response.status_code == 200:
        with open("sf_crime.csv", "wb") as f:
            f.write(response.content)
        st.success("sf_crime.csv basariyla indirildi.")
    else:
        st.error("Indirme basarisiz.")

# 📋 Temizlik ve raporlama
if os.path.exists("sf_crime.csv"):
    df = pd.read_csv("sf_crime.csv")
    original_row_count = len(df)

    # GEOID düzelt
    if "GEOID" in df.columns:
        df = df.dropna(subset=["GEOID"])  # NaN GEOID varsa at
        df["GEOID"] = df["GEOID"].astype(str).str.extract(r"(\\d+)")[0].str.zfill(11)

    st.subheader("📋 Ilk 5 Satir")
    st.dataframe(df.head())
    st.info(f"Toplam satir sayisi: {original_row_count:,}")

    # NaN analizi
    nan_summary = df.isna().sum()
    nan_cols = nan_summary[nan_summary > 0]
    dropped_nan_rows = 0

    if not nan_cols.empty:
        st.warning("Eksik veriler bulundu:")
        st.dataframe(nan_cols.rename("NaN Sayisi"))
        st.write(f"NaN iceren sutun sayisi: {len(nan_cols)}")
        st.write(f"NaN iceren toplam satir sayisi: {df.isna().any(axis=1).sum()}")
        dropped_nan_rows = df.isna().any(axis=1).sum()
        df = df.dropna()
    else:
        st.success("Hicbir sutunda NaN yok.")

    # 🔁 5 yıl filtre
    removed_rows = 0
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        five_years_ago = datetime.now() - timedelta(days=5*365)
        before_filter = len(df)
        df = df[df["date"] >= five_years_ago]
        removed_rows = before_filter - len(df)
    else:
        st.warning("'date' sutunu bulunamadi, tarih filtreleme atlandi.")

    # 📄 PDF raporu oluştur
    if st.button("📄 PDF Rapor Olustur"):
        report_file = create_pdf_report(
            file_name="sf_crime.csv",
            row_count_before=original_row_count,
            nan_cols=nan_cols,
            row_count_after=len(df),
            removed_rows=removed_rows,
            dropped_nan_rows=dropped_nan_rows
        )
        with open(report_file, "rb") as f:
            st.download_button("📎 Raporu Indir", data=f, file_name=report_file, mime="application/pdf")
