import streamlit as st
import pandas as pd
import requests
import os, json, subprocess, sys
from pathlib import Path

# === Streamlit ===
st.set_page_config(page_title="Veri Güncelleme", layout="wide")
st.title("📦 Günlük Suç Tahmin Zenginleştirme ve Güncelleme Paneli")

ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT / "scripts"
SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

# --- Sunum/Test için indirilebilecek dosyalar ---
DOWNLOADS = {
    "Tahmin Grid Verisi (GEOID × Zaman + Y_label)": {
        "url": "https://raw.githubusercontent.com/cem5113/crime_prediction_data/main/sf_crime_grid_full_labeled.csv",
        "path": str(ROOT / "crime_data" / "sf_crime_grid_full_labeled.csv")
    },
    "911 Çağrıları": {
        "url": "https://github.com/cem5113/crime_prediction_data/releases/download/v1.0.1/sf_911_last_5_year.csv",
        "path": str(ROOT / "crime_data" / "sf_911_last_5_year.csv")
    },
    "311 Çağrıları": {
        "url": "https://github.com/cem5113/crime_prediction_data/releases/download/v1.0.2/sf_311_last_5_years.csv",
        "path": str(ROOT / "crime_data" / "sf_311_last_5_years.csv")
    },
    "Nüfus Verisi": {
        "url": "https://raw.githubusercontent.com/cem5113/crime_prediction_data/main/sf_population.csv",
        "path": str(ROOT / "crime_data" / "sf_population.csv")
    },
    "Otobüs Durakları": {
        "url": "https://raw.githubusercontent.com/cem5113/crime_prediction_data/main/sf_bus_stops_with_geoid.csv",
        "path": str(ROOT / "crime_data" / "sf_bus_stops_with_geoid.csv")
    },
    "Tren Durakları": {
        "url": "https://raw.githubusercontent.com/cem5113/crime_prediction_data/main/sf_train_stops_with_geoid.csv",
        "path": str(ROOT / "crime_data" / "sf_train_stops_with_geoid.csv")
    },
    "POI GeoJSON": {
        "url": "https://raw.githubusercontent.com/cem5113/crime_prediction_data/main/sf_pois.geojson",
        "path": str(ROOT / "crime_data" / "sf_pois.geojson"),
        "is_json": True
    },
    "POI Risk Skorları": {
        "url": "https://raw.githubusercontent.com/cem5113/crime_prediction_data/main/risky_pois_dynamic.json",
        "path": str(ROOT / "crime_data" / "risky_pois_dynamic.json"),
        "is_json": True
    },
    "Polis İstasyonları": {
        "url": "https://raw.githubusercontent.com/cem5113/crime_prediction_data/main/sf_police_stations.csv",
        "path": str(ROOT / "crime_data" / "sf_police_stations.csv")
    },
    "Devlet Binaları": {
        "url": "https://raw.githubusercontent.com/cem5113/crime_prediction_data/main/sf_government_buildings.csv",
        "path": str(ROOT / "crime_data" / "sf_government_buildings.csv")
    },
    "Hava Durumu": {
        "url": "https://raw.githubusercontent.com/cem5113/crime_prediction_data/main/sf_weather_5years.csv",
        "path": str(ROOT / "crime_data" / "sf_weather_5years.csv")
    },
}
Path(ROOT/"crime_data").mkdir(exist_ok=True)

def download_and_preview(name, url, file_path, is_json=False):
    st.markdown(f"### 🔹 {name}")
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        if is_json:
            Path(file_path).write_text(r.text, encoding="utf-8")
            data = json.loads(r.text)
            st.json(data if isinstance(data, dict) else (data[:3] if isinstance(data, list) else data))
        else:
            with open(file_path, "wb") as f:
                f.write(r.content)
            df = pd.read_csv(file_path)
            st.dataframe(df.head(3))
            st.caption(f"📌 Sütunlar: {list(df.columns)}")
    except Exception as e:
        st.error(f"❌ {name} indirilemedi: {e}")

st.markdown("### 1) (Opsiyonel) Verileri indir ve önizle")
if st.button("📥 Verileri İndir ve Önizle (İlk 3 Satır)"):
    for name, info in DOWNLOADS.items():
        download_and_preview(name, info["url"], info["path"], is_json=info.get("is_json", False))
    st.success("✅ İndirme tamamlandı.")

# -----------------------------
# Script çözümleyici: varsa çalıştırır, yoksa GitHub'dan indirir
# -----------------------------
GITHUB_SCRIPTS_BASE = "https://raw.githubusercontent.com/cem5113/crime_prediction_data/main/scripts"

# Her adım için olası dosya adları (projede farklı adlar kullanılmış olabilir diye alternatifler veriyoruz)
PIPELINE = [
    # 01
    {"name": "update_crime.py",      "alts": ["build_crime_grid.py", "crime_grid_build.py"]},
    # 02
    {"name": "update_911.py",        "alts": ["enrich_911.py"]},
    # 03
    {"name": "update_311.py",        "alts": ["enrich_311.py"]},
    # 04
    {"name": "update_population.py", "alts": ["enrich_population.py"]},
    # 05
    {"name": "update_bus.py",        "alts": ["enrich_bus.py"]},
    # 06
    {"name": "update_train.py",      "alts": ["enrich_train.py"]},
    # 07
    {"name": "update_poi.py",        "alts": ["app_poi_to_06.py", "enrich_poi.py"]},
    # 08
    {"name": "update_police_gov.py", "alts": ["enrich_police_gov.py", "enrich_police.py"]},
    # 09
    {"name": "update_weather.py",    "alts": ["enrich_weather.py"]},
]

def ensure_script(local_name: str) -> Path | None:
    """scripts/<local_name> mevcutsa döner; yoksa GitHub'dan indirir (varsa)."""
    p = SCRIPTS_DIR / local_name
    if p.exists():
        return p
    # GitHub'da aynı adla dene
    url = f"{GITHUB_SCRIPTS_BASE}/{local_name}"
    try:
        r = requests.get(url, timeout=20)
        if r.status_code == 200 and r.text.strip():
            p.write_text(r.text, encoding="utf-8")
            st.info(f"⬇️ Script indirildi: {local_name}")
            return p
    except Exception:
        pass
    return None

def resolve_script(entry: dict) -> Path | None:
    # 1) Ana ad
    p = ensure_script(entry["name"])
    if p:
        return p
    # 2) Alternatifler
    for alt in entry.get("alts", []):
        pp = ensure_script(alt)
        if pp:
            # alternatifi ana ada symlink/kopya yap
            target = SCRIPTS_DIR / entry["name"]
            try:
                target.write_text(pp.read_text(encoding="utf-8"), encoding="utf-8")
                st.info(f"🔁 {alt} → {entry['name']} olarak kopyalandı.")
                return target
            except Exception:
                return pp
    return None

def run_script(path: Path) -> bool:
    st.write(f"▶️ {path.name} çalıştırılıyor…")
    try:
        res = subprocess.run([sys.executable, str(path)], capture_output=True, text=True)
        if res.returncode == 0:
            st.success(f"✅ {path.name} tamamlandı")
            if res.stdout:
                st.code(res.stdout)
            return True
        else:
            st.error(f"❌ {path.name} hata verdi")
            st.code(res.stderr or "(stderr boş)")
            return False
    except Exception as e:
        st.error(f"🚨 {path.name} çağrılamadı: {e}")
        return False

# === 2) Güncelle & Zenginleştir ===
st.markdown("### 2) Güncelleme ve Zenginleştirme (01 → 08)")
if st.button("⚙️ Güncelleme ve Zenginleştirme (01 → 08)"):
    all_ok = True
    for entry in PIPELINE:
        script_path = resolve_script(entry)
        if not script_path:
            st.warning(f"⏭️ {entry['name']} bulunamadı/indirilemedi, atlanıyor.")
            all_ok = False  # dilersen True bırak, pipeline'ı yeşil saymak için
            continue
        ok = run_script(script_path)
        all_ok = all_ok and ok
    if all_ok:
        st.success("🎉 Pipeline bitti: Tüm adımlar başarıyla tamamlandı.")
    else:
        st.warning("ℹ️ Pipeline tamamlandı; eksik/hatalı adımlar var. Logları kontrol edin.")
