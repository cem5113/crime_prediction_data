# app.py
import os
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(layout="wide")

# ---------- STYLE (Times 12, klasik görünüm) ----------
st.markdown("""
<style>
html, body, [class*="css"]  {
    font-family: "Times New Roman", Times, serif;
    font-size: 12pt;
}
h1 {
    font-size: 16pt;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>SUTAM – Veri Hazırlama Süreci</h1>", unsafe_allow_html=True)

# ---------- Dosya sırası ----------
FILES = [
    ("00", "sf_crime.csv", "Ham + temiz + GEOID + zaman feature"),
    ("01", "sf_crime_01.csv", "+ 911"),
    ("02", "sf_crime_02.csv", "+ 311"),
    ("03", "sf_crime_03.csv", "+ nüfus/demografi"),
    ("04", "sf_crime_04.csv", "+ otobüs mesafe/yoğunluk"),
    ("05", "sf_crime_05.csv", "+ tren mesafe/yoğunluk"),
    ("06", "sf_crime_06.csv", "+ POI risk/yoğunluk"),
    ("07", "sf_crime_07.csv", "+ police/gov mesafe/yakınlık"),
    ("08", "sf_crime_08.csv", "(senin akışında burası netleştirilecek)"),
    ("09", "sf_crime_09.csv", "+ neighbors/otokorelasyon"),
]

SEARCH_DIRS = [
    Path("."),
    Path("./crime_prediction_data"),
    Path("./data"),
    Path("./outputs")
]

# ---------- Fonksiyon ----------
def find_file(filename):
    for d in SEARCH_DIRS:
        p = d / filename
        if p.exists():
            return p
    return None


def analyze_csv(path):
    try:
        df = pd.read_csv(path, low_memory=False)
        rows, cols = df.shape

        total_cells = rows * cols
        nan_cells = df.isna().sum().sum()
        nan_pct = (nan_cells / total_cells * 100) if total_cells > 0 else 0

        empty_rows = df.isna().all(axis=1).sum()

        return rows, cols, round(nan_pct, 3), int(empty_rows)
    except Exception:
        return "-", "-", "-", "-"


# ---------- Tablo oluştur ----------
data = []

for stage, fname, note in FILES:
    path = find_file(fname)

    if path:
        rows, cols, nan_pct, empty_rows = analyze_csv(path)
    else:
        rows, cols, nan_pct, empty_rows = "-", "-", "-", "-"

    data.append([
        stage,
        fname,
        rows,
        cols,
        nan_pct,
        empty_rows,
        note
    ])

df_summary = pd.DataFrame(
    data,
    columns=[
        "Aşama",
        "Dosya",
        "Satır",
        "Sütun",
        "NaN hücre (%)",
        "Tam boş satır",
        "Not"
    ]
)

st.dataframe(df_summary, use_container_width=True, hide_index=True)
