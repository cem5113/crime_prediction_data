import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import geopandas as gpd

# =========================
# Yardımcılar
# =========================
def ensure_parent(path: str):
    Path(os.path.dirname(path) or ".").mkdir(parents=True, exist_ok=True)

def safe_save_csv(df: pd.DataFrame, path: str):
    try:
        ensure_parent(path)
        df.to_csv(path, index=False)
    except Exception as e:
        print(f"❌ Kaydetme hatası: {path}\n{e}")
        df.to_csv(path + ".bak", index=False)
        print(f"📁 Yedek oluşturuldu: {path}.bak")

def normalize_geoid(s: pd.Series, target_len: int) -> pd.Series:
    s = s.astype(str).str.extract(r"(\d+)")[0]
    return s.str.zfill(target_len)

def find_col(ci_names, candidates):
    m = {c.lower(): c for c in ci_names}
    for cand in candidates:
        if cand.lower() in m:
            return m[cand.lower()]
    return None

# =========================
# 1) Dosya yolları
# =========================
BASE_DIR = "crime_data"
Path(BASE_DIR).mkdir(exist_ok=True)

raw_save_path   = os.path.join(BASE_DIR, "sf_311_last_5_years.csv")
agg_save_path   = os.path.join(BASE_DIR, "311_requests_range.csv")
crime_01_path   = os.path.join(BASE_DIR, "sf_crime_01.csv")
output_path     = os.path.join(BASE_DIR, "sf_crime_02.csv")

# census geojson hem crime_data/ hem kökte olabilir
census_candidates = [
    os.path.join(BASE_DIR, "sf_census_blocks_with_population.geojson"),
    os.path.join(".", "sf_census_blocks_with_population.geojson"),
]

# =========================
# 2) Tarih aralığı (son 5 yıl)
# =========================
today = datetime.today().date()
start_date = today - timedelta(days=5 * 365)

# =========================
# 3) 311 verisini indir (SFPD/Police)
# =========================
soql = (
    f"$where=requested_datetime >= '{start_date}T00:00:00.000' "
    "AND (agency_responsible like '%Police%' OR agency_responsible like '%SFPD%')"
)
url_base = "https://data.sfgov.org/resource/vw6y-z8j6.json"
limit = 1000
offset = 0
rows = []

print("📥 311 verisi indiriliyor...")
while True:
    url = f"{url_base}?{quote(soql, safe='=&')}&$limit={limit}&$offset={offset}"
    chunk = None
    for attempt in range(4):
        try:
            chunk = pd.read_json(url)
            break
        except Exception as e:
            if attempt == 3:
                print("❌ Veri çekme hatası:", e)
                chunk = None
            time.sleep(1.2 * (attempt + 1))
    if chunk is None or chunk.empty:
        break
    rows.append(chunk)
    offset += limit
    print(f"  + {offset} kayıt indirildi...")
    time.sleep(0.25)

if not rows:
    print("⚠️ Veri alınamadı, script sonlandırıldı.")
    raise SystemExit(0)

# =========================
# 4) Temizleme & kolon adları
# =========================
df = pd.concat(rows, ignore_index=True)

dt_col  = find_col(df.columns, ["requested_datetime", "datetime", "created_date", "created_at"])
lat_col = find_col(df.columns, ["lat", "latitude", "y"])
lon_col = find_col(df.columns, ["long", "longitude", "x"])
id_col  = find_col(df.columns, ["service_request_id", "service_requestid", "id"])

if not all([dt_col, lat_col, lon_col]):
    missing = [("datetime", dt_col), ("latitude", lat_col), ("longitude", lon_col)]
    missing = ", ".join([k for k, v in missing if v is None])
    raise ValueError(f"❌ Zorunlu kolon(lar) eksik: {missing}")

df = df.rename(columns={
    dt_col: "datetime",
    lat_col: "latitude",
    lon_col: "longitude",
    **({id_col: "id"} if id_col else {}),
})

df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
df = df.dropna(subset=["datetime", "latitude", "longitude"]).copy()
df["date"] = df["datetime"].dt.date
df["hour"] = df["datetime"].dt.hour

# =========================
# 5) GEOID eşlemesi (census geojson → sjoin)
# =========================
print("📍 GEOID eşlemesi yapılıyor...")
census_path = next((p for p in census_candidates if os.path.exists(p)), None)
if census_path is None:
    raise FileNotFoundError("❌ Nüfus blokları GeoJSON bulunamadı (crime_data/ veya kök).")

gdf = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs="EPSG:4326"
)
gdf_blocks = gpd.read_file(census_path)

# Hedef GEOID uzunluğunu block dosyasından öğren
target_len = gdf_blocks["GEOID"].astype(str).str.len().mode().iat[0]
gdf_blocks["GEOID"] = normalize_geoid(gdf_blocks["GEOID"], target_len)

# sjoin
gdf = gpd.sjoin(gdf, gdf_blocks[["GEOID", "geometry"]], how="left", predicate="within")
df = pd.DataFrame(gdf.drop(columns=["geometry", "index_right"], errors="ignore"))
df["GEOID"] = normalize_geoid(df["GEOID"], target_len)
df = df.dropna(subset=["GEOID"]).copy()

# =========================
# 6) Saatlik özet
# =========================
hr = (df["hour"] // 3) * 3
df["hour_range"] = hr.astype(str) + "-" + (hr + 3).astype(str)
summary = (
    df.groupby(["GEOID", "date", "hour_range"])
      .size()
      .reset_index(name="311_request_count")
)

# =========================
# 7) Kaydet (ham + özet)
# =========================
safe_save_csv(df, raw_save_path)
safe_save_csv(summary, agg_save_path)
print(f"✅ Ham 311 verisi → {raw_save_path}")
print(f"✅ Saatlik özet  → {agg_save_path}")

# =========================
# 8) Suç verisi (sf_crime_01) ile birleştir
# =========================
if not os.path.exists(crime_01_path):
    print("⚠️ sf_crime_01.csv bulunamadı. Birleştirme yapılamadı.")
    raise SystemExit(0)

print("🔗 sf_crime_01 ile birleştiriliyor...")
crime = pd.read_csv(crime_01_path, dtype={"GEOID": str}, low_memory=False)

# GEOID uzunluğunu crime dosyasına da uydur (güvenlik)
target_len2 = crime["GEOID"].dropna().astype(str).str.len().mode().iat[0]
summary["GEOID"] = normalize_geoid(summary["GEOID"], target_len2)
crime["GEOID"]   = normalize_geoid(crime["GEOID"], target_len2)

# hour_range üret (event_hour varsa oradan)
if "hour_range" not in crime.columns:
    if "event_hour" in crime.columns:
        hr2 = (crime["event_hour"] // 3) * 3
        crime["hour_range"] = hr2.astype(str) + "-" + (hr2 + 3).astype(str)
    else:
        raise ValueError("❌ sf_crime_01 içinde 'hour_range' veya 'event_hour' sütunu eksik!")

# tarih tipini hizala
crime["date"] = pd.to_datetime(
    crime["date"] if "date" in crime.columns else crime["datetime"], errors="coerce"
).dt.date

merged = pd.merge(crime, summary, on=["GEOID", "date", "hour_range"], how="left")
merged["311_request_count"] = merged["311_request_count"].fillna(0).astype(int)

safe_save_csv(merged, output_path)
print(f"✅ Birleştirilmiş çıktı → {output_path}")
