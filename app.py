import streamlit as st
import subprocess
import os
import pandas as pd
import json
import requests

st.set_page_config(page_title="📦 Günlük Suç Verisi Pipeline", layout="wide")
st.title("📦 Günlük Suç Tahmin Zenginleştirme ve Güncelleme Paneli")

# -----------------------------
# Fallback fonksiyonu
# -----------------------------
def load_with_fallback(name, path, url, is_json=False):
    """Veri indirme - hata durumunda yereldeki son sağlam dosyayı kullanır,
    o da yoksa boş DataFrame veya default JSON döner."""
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        if is_json:
            with open(path, "w", encoding="utf-8") as f:
                f.write(resp.text)
            return json.loads(resp.text)
        else:
            with open(path, "wb") as f:
                f.write(resp.content)
            return pd.read_csv(path)
    except Exception as e:
        st.warning(f"⚠️ {name} indirilemedi, hata: {e}")
        if os.path.exists(path):
            st.info(f"📂 Yereldeki son sağlam {name} kullanılıyor.")
            return pd.read_csv(path) if not is_json else json.load(open(path))
        else:
            st.error(f"🚨 {name} için veri bulunamadı, boş veri ile devam.")
            return pd.DataFrame() if not is_json else {}

# -----------------------------
# GitHub veri kaynakları
# -----------------------------
DATASETS = {
    "crime_grid": {
        "url": "https://raw.githubusercontent.com/cem5113/crime_prediction_data/main/sf_crime_grid_full_labeled.csv",
        "path": "sf_crime_grid_full_labeled.csv"
    },
    "weather": {
        "url": "https://raw.githubusercontent.com/cem5113/crime_prediction_data/main/sf_weather_5years.csv",
        "path": "sf_weather_5years.csv"
    },
    "poi_json": {
        "url": "https://github.com/cem5113/crime_prediction_data/raw/main/risky_pois_dynamic.json",
        "path": "risky_pois_dynamic.json",
        "is_json": True
    }
}

# -----------------------------
# 1. (Opsiyonel) Verileri indir
# -----------------------------
if st.button("📥 Verileri İndir (Sunum/Test)"):
    for name, info in DATASETS.items():
        data = load_with_fallback(name, info["path"], info["url"], info.get("is_json", False))
        if isinstance(data, pd.DataFrame) and not data.empty:
            st.dataframe(data.head(3))
        elif isinstance(data, dict):
            st.json(data)
    st.success("✅ İndirme tamamlandı.")

# -----------------------------
# 2. Asıl enrichment pipeline
# -----------------------------
if st.button("⚙️ Güncelleme ve Zenginleştirme"):
    steps = [
        "update_crime.py",
        "update_911.py",
        "update_311.py",
        "update_population.py",
        "update_bus.py",
        "update_train.py",
        "update_poi.py",
        "update_police_gov.py",
        "update_weather.py"
    ]
    for step in steps:
        try:
            result = subprocess.run(["python", f"scripts/{step}"], capture_output=True, text=True)
            if result.returncode == 0:
                st.success(f"✅ {step} tamamlandı")
            else:
                st.error(f"❌ {step} hata verdi")
                st.code(result.stderr)
        except Exception as e:
            st.error(f"🚨 {step} çalıştırılamadı: {e}")
