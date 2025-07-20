import streamlit as st
import pandas as pd
import requests
import os
from fpdf import FPDF
from datetime import datetime, timedelta

st.set_page_config(page_title="Veri Güncelleme", layout="wide")
st.title("📦 Veri Güncelleme ve Kontrol Arayüzü")

# 📄 PDF Rapor Fonksiyonu
def create_pdf_report(file_name, row_count_before, nan_cols, row_count_after, removed_rows):
    now = datetime.now()
    timestamp = now.strftime("%d.%m.%Y %H:%M:%S")

    if not nan_cols.empty:
        nan_parts = [f"- {col}: {count}" for col, count in nan_cols.items()]
        nan_text = " ".join(nan_parts)
    else:
        nan_text = "Yok"

    summary = (
        f"- Tarih/Saat: {timestamp}; "
        f"Dosya: {file_name} ; "
        f"Toplam satir sayisi: {row_count_before:,}; "
        f"NaN iceren sutunlar: {nan_text}; "
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

# 📥 1. Dosya İndirme
if st.button("📥 sf_crime.csv dosyasini indir ve işle"):
    url = "https://raw.githubusercontent.com/cem5113/crime_prediction_data/main/sf_crime.csv"
    response = requests.get(url)
    if response.status_code == 200:
        with open("sf_crime.csv", "wb") as f:
            f.write(response.content)
        st.success("✅ sf_crime.csv başarıyla indirildi.")

        # 🧹 2. Temizlik & NaN Kontrol
        df = pd.read_csv("sf_crime.csv")
        original_row_count = len(df)

        if "GEOID" in df.columns:
            df["GEOID"] = df["GEOID"].astype(str).str.extract(r"(\d+)")[0].str.zfill(11)

        nan_summary = df.isna().sum()
        nan_cols = nan_summary[nan_summary > 0]
        df = df.dropna()

        removed_rows = 0
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            five_years_ago = datetime.now() - timedelta(days=5*365)
            before_filter = len(df)
            df = df[df["date"] >= five_years_ago]
            removed_rows = before_filter - len(df)

        st.dataframe(df.head())
        st.info(f"Toplam satır: {original_row_count:,}, Kalan satır: {len(df):,}")

        # 📄 3. PDF Rapor Oluştur
        report_file = create_pdf_report("sf_crime.csv", original_row_count, nan_cols, len(df), removed_rows)
        with open(report_file, "rb") as f:
            st.download_button("📎 PDF Raporu İndir", data=f, file_name=report_file, mime="application/pdf")

        # 🔄 4. enrich_sf_crime_49.py çalıştır
        st.subheader("🔄 Veri Zenginleştirme (sf_crime_49.csv)")
        with st.spinner("Zenginleştirme işlemi sürüyor..."):
            os.system("python scripts/enrich_sf_crime_49.py")
        st.success("✅ sf_crime_49.csv üretildi.")

        # 🧠 5. generate_sf_crime_52.py çalıştır
        st.subheader("🧠 Y_label ve Kombinasyon Hesaplama (sf_crime_52.csv)")
        with st.spinner("Kombinasyonlar oluşturuluyor..."):
            os.system("python scripts/generate_sf_crime_52.py")
        st.success("✅ sf_crime_52.csv üretildi.")

    else:
        st.error("❌ Dosya indirilemedi. Lütfen bağlantıyı kontrol edin.")
