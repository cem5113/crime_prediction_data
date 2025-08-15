# app.py — Gold-safe + Fallback + Mean-Impute
import streamlit as st
import subprocess, os, shutil, json, io, requests, time
import pandas as pd

st.set_page_config(page_title="📦 Günlük Suç Verisi Pipeline", layout="wide")
st.title("📦 Günlük Suç Tahmin Zenginleştirme ve Güncelleme Paneli")

# =========================
# Genel Ayarlar
# =========================
BASE_DIR = "crime_data"   # tüm adımlar aynı kök
os.makedirs(BASE_DIR, exist_ok=True)

GITHUB_RAW = "https://raw.githubusercontent.com/cem5113/crime_prediction_data/main"  # <- gerekirse değiştir

# Adım manifesti: script adı, giriş-çıkış dosyaları, GitHub URL'leri ve doğrulama kuralları
STEPS = [
    # 01 - 911
    {
        "name": "01_911",
        "script": "update_911.py",
        "inputs": [
            {"path": os.path.join(BASE_DIR, "sf_crime_grid_full_labeled.csv"),
             "url": f"{GITHUB_RAW}/sf_crime_grid_full_labeled.csv"},
            {"path": os.path.join(BASE_DIR, "sf_911_last_5_year.csv"),
             "url": "https://github.com/cem5113/crime_prediction_data/releases/download/v1.0.1/sf_911_last_5_year.csv"}
        ],
        "output": os.path.join(BASE_DIR, "sf_crime_01.csv"),
        "required_cols": ["GEOID","date","event_hour"],  # minimum genel kontrol
        "impute_cols": []  # gerekirse ekle
    },
    # 02 - 311
    {
        "name": "02_311",
        "script": "update_311.py",
        "inputs": [
            {"path": os.path.join(BASE_DIR, "sf_crime_01.csv"),
             "url": f"{GITHUB_RAW}/sf_crime_01.csv"},
            {"path": os.path.join(BASE_DIR, "sf_311_last_5_years.csv"),
             "url": "https://github.com/cem5113/crime_prediction_data/releases/download/v1.0.2/sf_311_last_5_years.csv"},
        ],
        "output": os.path.join(BASE_DIR, "sf_crime_02.csv"),
        "required_cols": ["GEOID","date","event_hour"],
        "impute_cols": []
    },
    # 03 - Nüfus
    {
        "name": "03_population",
        "script": "update_population.py",
        "inputs": [
            {"path": os.path.join(BASE_DIR, "sf_crime_02.csv"),
             "url": f"{GITHUB_RAW}/sf_crime_02.csv"},
            {"path": os.path.join(BASE_DIR, "sf_population.csv"),
             "url": f"{GITHUB_RAW}/sf_population.csv"},
        ],
        "output": os.path.join(BASE_DIR, "sf_crime_03.csv"),
        "required_cols": ["GEOID"],
        "impute_cols": ["population"]  # yoksa ortalama ile doldurulabilir
    },
    # 04 - Otobüs
    {
        "name": "04_bus",
        "script": "update_bus.py",
        "inputs": [
            {"path": os.path.join(BASE_DIR, "sf_crime_03.csv"),
             "url": f"{GITHUB_RAW}/sf_crime_03.csv"},
            {"path": os.path.join(BASE_DIR, "sf_bus_stops_with_geoid.csv"),
             "url": f"{GITHUB_RAW}/sf_bus_stops_with_geoid.csv"},
            {"path": os.path.join(BASE_DIR, "sf_census_blocks_with_population.geojson"),
             "url": f"{GITHUB_RAW}/sf_census_blocks_with_population.geojson"},
        ],
        "output": os.path.join(BASE_DIR, "sf_crime_04.csv"),
        "required_cols": ["GEOID"],
        "impute_cols": ["bus_stop_count","bus_stop_dist_min"]
    },
    # 05 - Tren
    {
        "name": "05_train",
        "script": "update_train.py",
        "inputs": [
            {"path": os.path.join(BASE_DIR, "sf_crime_04.csv"),
             "url": f"{GITHUB_RAW}/sf_crime_04.csv"},
            {"path": os.path.join(BASE_DIR, "sf_train_stops_with_geoid.csv"),
             "url": f"{GITHUB_RAW}/sf_train_stops_with_geoid.csv"},
            {"path": os.path.join(BASE_DIR, "sf_census_blocks_with_population.geojson"),
             "url": f"{GITHUB_RAW}/sf_census_blocks_with_population.geojson"},
        ],
        "output": os.path.join(BASE_DIR, "sf_crime_05.csv"),
        "required_cols": ["GEOID"],
        "impute_cols": ["train_stop_count","train_stop_dist_min"]
    },
    # 06 - POI
    {
        "name": "06_poi",
        "script": "update_poi.py",
        "inputs": [
            {"path": os.path.join(BASE_DIR, "sf_crime_05.csv"),
             "url": f"{GITHUB_RAW}/sf_crime_05.csv"},
            {"path": os.path.join(BASE_DIR, "sf_pois.geojson"),
             "url": f"{GITHUB_RAW}/sf_pois.geojson"},
            {"path": os.path.join(BASE_DIR, "sf_census_blocks_with_population.geojson"),
             "url": f"{GITHUB_RAW}/sf_census_blocks_with_population.geojson"},
            {"path": os.path.join(BASE_DIR, "risky_pois_dynamic.json"),
             "url": f"{GITHUB_RAW}/risky_pois_dynamic.json"},
        ],
        "output": os.path.join(BASE_DIR, "sf_crime_06.csv"),
        "required_cols": ["GEOID"],
        "impute_cols": ["poi_total_count","poi_risk_score"]
    },
    # 07 - Polis & Devlet
    {
        "name": "07_police_gov",
        "script": "update_police_gov.py",
        "inputs": [
            {"path": os.path.join(BASE_DIR, "sf_crime_06.csv"),
             "url": f"{GITHUB_RAW}/sf_crime_06.csv"},
            {"path": os.path.join(BASE_DIR, "sf_police_stations.csv"),
             "url": f"{GITHUB_RAW}/sf_police_stations.csv"},
            {"path": os.path.join(BASE_DIR, "sf_government_buildings.csv"),
             "url": f"{GITHUB_RAW}/sf_government_buildings.csv"},
        ],
        "output": os.path.join(BASE_DIR, "sf_crime_07.csv"),
        "required_cols": ["GEOID","distance_to_police","distance_to_government_building"],
        "impute_cols": ["distance_to_police","distance_to_government_building","is_near_police","is_near_government"]
    },
    # 08 - Hava Durumu
    {
        "name": "08_weather",
        "script": "update_weather.py",
        "inputs": [
            {"path": os.path.join(BASE_DIR, "sf_crime_07.csv"),
             "url": f"{GITHUB_RAW}/sf_crime_07.csv"},
            {"path": os.path.join(BASE_DIR, "sf_weather_5years.csv"),
             "url": f"{GITHUB_RAW}/sf_weather_5years.csv"},
        ],
        "output": os.path.join(BASE_DIR, "sf_crime_08.csv"),
        "required_cols": ["GEOID","date"],
        "impute_cols": ["PRCP","TMAX","TMIN"]
    },
]

# =========================
# Yardımcılar
# =========================
def http_get(url, timeout=30):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r

def ensure_input_file(path, url, allow_empty=False):
    """Yoksa GitHub'dan indir; indirme başarısızsa yereldeki dosyayı kullan; o da yoksa allow_empty True ise iskelet oluştur."""
    if os.path.exists(path):
        return "local"
    try:
        resp = http_get(url)
        with open(path, "wb") as f:
            f.write(resp.content)
        return "downloaded"
    except Exception as e:
        st.warning(f"⚠️ İndirilemedi: {os.path.basename(path)} -> {e}")
        if allow_empty:
            # boş iskelet (CSV başlık yoksa bazı scriptler bozulabilir; bu yüzden mümkünse GitHub'dan iskelet tut)
            with open(path, "w", encoding="utf-8") as f:
                f.write("")  # en minimal
            return "created_empty"
        else:
            return "missing"

def backup(path):
    if os.path.exists(path):
        bak = path + ".bak"
        shutil.copy2(path, bak)
        return bak
    return None

def restore_backup(path, bak):
    if bak and os.path.exists(bak):
        shutil.move(bak, path)
    elif bak and not os.path.exists(bak):
        st.warning(f"⚠️ Yedek bulunamadı: {bak}")

def drop_backup(bak):
    if bak and os.path.exists(bak):
        os.remove(bak)

def validate_output(csv_path, required_cols):
    if not os.path.exists(csv_path):
        return False, "çıktı oluşmadı"
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return False, f"okunamadı: {e}"
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return False, f"eksik kolonlar: {missing}"
    if len(df) == 0:
        return False, "boş çıktı"
    return True, df

def mean_impute(df, cols):
    for c in cols:
        if c in df.columns:
            try:
                if df[c].dtype.kind in "biufc":
                    mean_val = pd.to_numeric(df[c], errors="coerce").mean()
                    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(mean_val)
                else:
                    # kategorik: en sık değerle doldur
                    mode = df[c].mode()
                    if not mode.empty:
                        df[c] = df[c].fillna(mode.iloc[0])
            except Exception:
                pass
    return df

def run_step(step):
    st.subheader(f"🔧 {step['name']}")

    # 1) Girdileri garanti et (GitHub → yerel fallback)
    for inp in step["inputs"]:
        status = ensure_input_file(inp["path"], inp["url"], allow_empty=True)
        st.caption(f"• {os.path.basename(inp['path'])}: {status}")

    # 2) Gold çıktıyı yedekle
    bak = backup(step["output"])

    # 3) Script'i çalıştır
    try:
        proc = subprocess.run(["python", os.path.join("scripts", step["script"])],
                              capture_output=True, text=True)
    except Exception as e:
        st.error(f"🚨 Çalıştırılamadı: {step['script']} -> {e}")
        restore_backup(step["output"], bak)
        return False

    if proc.returncode != 0:
        st.error(f"❌ Hata: {step['script']}")
        st.code(proc.stderr)
        restore_backup(step["output"], bak)
        return False
    else:
        st.success(f"✅ Tamam: {step['script']}")
        if proc.stdout:
            st.code(proc.stdout)

    # 4) Çıktıyı doğrula
    ok, res = validate_output(step["output"], step["required_cols"])
    if ok:
        drop_backup(bak)
        st.success(f"🟢 Doğrulandı: {os.path.basename(step['output'])}")
        return True
    else:
        st.warning(f"⚠️ İlk doğrulama başarısız: {res}")
        # 4a) Ortalama ile doldurmayı dene (geçici iyileştirme)
        try:
            df_bad = pd.read_csv(step["output"]) if os.path.exists(step["output"]) else pd.DataFrame()
            if not df_bad.empty:
                df_fixed = mean_impute(df_bad, step.get("impute_cols", []))
                # tekrar minimum doğrulama
                missing = [c for c in step["required_cols"] if c not in df_fixed.columns]
                if len(df_fixed) > 0 and not missing:
                    df_fixed.to_csv(step["output"], index=False)
                    st.info("🧩 Ortalama/Mod ile dolduruldu ve tekrar doğrulandı.")
                    drop_backup(bak)
                    return True
        except Exception as e:
            st.info(f"ℹ️ İmpute denemesi yapılamadı: {e}")

        # 4b) Olmadı → gold'u geri yükle
        restore_backup(step["output"], bak)
        st.error("🔴 Çıktı geri alındı (gold korundu).")
        return False

# =========================
# 1) Sunum/Test için indirme (opsiyonel)
# =========================
def load_with_fallback(name, path, url, is_json=False):
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
        st.warning(f"⚠️ {name} indirilemedi: {e}")
        if os.path.exists(path):
            st.info(f"📂 Yereldeki {name} kullanılıyor.")
            return pd.read_csv(path) if not is_json else json.load(open(path))
        else:
            st.error(f"🚨 {name} yok, boş veri gösterilecek.")
            return pd.DataFrame() if not is_json else {}

DATASETS = {
    "crime_grid": {
        "url": f"{GITHUB_RAW}/sf_crime_grid_full_labeled.csv",
        "path": os.path.join(BASE_DIR, "sf_crime_grid_full_labeled.csv")
    },
    "weather": {
        "url": f"{GITHUB_RAW}/sf_weather_5years.csv",
        "path": os.path.join(BASE_DIR, "sf_weather_5years.csv")
    },
    "poi_json": {
        "url": f"{GITHUB_RAW}/risky_pois_dynamic.json",
        "path": os.path.join(BASE_DIR, "risky_pois_dynamic.json"),
        "is_json": True
    }
}

st.markdown("### 1) (Opsiyonel) Verileri indir ve önizle")
if st.button("📥 Verileri İndir (Sunum/Test)"):
    for name, info in DATASETS.items():
        data = load_with_fallback(name, info["path"], info["url"], info.get("is_json", False))
        if isinstance(data, pd.DataFrame) and not data.empty:
            st.write(f"**{name}**")
            st.dataframe(data.head(5))
        elif isinstance(data, dict):
            st.write(f"**{name}**")
            st.json(data)
    st.success("✅ İndirme tamamlandı.")

# =========================
# 2) Asıl Güncelleme & Zenginleştirme
# =========================
st.markdown("### 2) Güncelleme ve Zenginleştirme (Gold-safe + Fallback)")
if st.button("⚙️ Pipeline'ı Çalıştır"):
    all_ok = True
    for step in STEPS:
        ok = run_step(step)
        all_ok = all_ok and ok
        # Bir adım başarısız olsa bile, zinciri kesmek istemiyorsan devam et.
        # Kesmek istersen: if not ok: break
    if all_ok:
        st.success("🎉 Tüm adımlar başarıyla tamamlandı.")
    else:
        st.warning("ℹ️ Bazı adımlar başarısız veya fallback ile tamamlandı. Logları kontrol edin.")
