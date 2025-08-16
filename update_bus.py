import os
from pathlib import Path
from datetime import datetime
from urllib.parse import quote

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree

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

def freedman_diaconis_bin_count(data: np.ndarray, max_bins: int = 10) -> int:
    data = np.asarray(data)
    if len(data) < 2 or np.all(data == data[0]):
        return 1
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    if iqr == 0:
        return min(max_bins, max(2, int(np.sqrt(len(data)))))
    bin_width = 2 * iqr / (len(data) ** (1 / 3))
    if bin_width <= 0:
        return min(max_bins, max(2, int(np.sqrt(len(data)))))
    return max(2, min(max_bins, int(np.ceil((data.max() - data.min()) / bin_width))))

# =========================
# 1) Dosya yolları
# =========================
BASE_DIR = "crime_data"
Path(BASE_DIR).mkdir(exist_ok=True)

CRIME_INPUT     = os.path.join(BASE_DIR, "sf_crime_03.csv")
BUS_OUTPUT      = os.path.join(BASE_DIR, "sf_bus_stops_with_geoid.csv")
CRIME_OUTPUT    = os.path.join(BASE_DIR, "sf_crime_04.csv")

# census geojson hem crime_data/ hem kökte olabilir
CENSUS_CANDIDATES = [
    os.path.join(BASE_DIR, "sf_census_blocks_with_population.geojson"),
    os.path.join(".", "sf_census_blocks_with_population.geojson"),
]

# =========================
# 2) Socrata’dan otobüs duraklarını indir (paginasyonlu)
# =========================
print("🚌 Otobüs durakları Socrata API'den indiriliyor...")
rows, limit, offset = [], 50000, 0  # büyük çekelim; Socrata limit üst sınırı 50k
base = "https://data.sfgov.org/resource/i28k-bkz6.json"
select = "$select=stop_id,stop_name,latitude,longitude"
while True:
    url = f"{base}?{select}&$limit={limit}&$offset={offset}"
    try:
        chunk = pd.read_json(url)
    except Exception as e:
        print(f"❌ İndirme hatası (offset={offset}): {e}")
        break
    if chunk is None or chunk.empty:
        break
    rows.append(chunk)
    offset += len(chunk)
    print(f"  + {offset} kayıt indirildi...")
    if len(chunk) < limit:
        break

if not rows:
    raise SystemExit("⚠️ Otobüs durakları alınamadı; çıkılıyor.")

bus = pd.concat(rows, ignore_index=True)
bus = bus.dropna(subset=["latitude", "longitude"]).copy()
bus["stop_lat"] = bus["latitude"].astype(float)
bus["stop_lon"] = bus["longitude"].astype(float)

# =========================
# 3) GEOID eşlemesi (census sjoin)
# =========================
census_path = next((p for p in CENSUS_CANDIDATES if os.path.exists(p)), None)
if census_path is None:
    raise FileNotFoundError("❌ Nüfus blokları GeoJSON bulunamadı (crime_data/ veya kök).")

gdf_bus = gpd.GeoDataFrame(
    bus, geometry=gpd.points_from_xy(bus["stop_lon"], bus["stop_lat"]), crs="EPSG:4326"
)
gdf_blocks = gpd.read_file(census_path)

# GEOID hedef uzunluğunu block dosyasından öğren
target_len = gdf_blocks["GEOID"].astype(str).str.len().mode().iat[0]
gdf_blocks["GEOID"] = normalize_geoid(gdf_blocks["GEOID"], target_len)

gdf_bus = gpd.sjoin(gdf_bus, gdf_blocks[["GEOID", "geometry"]], how="left", predicate="within")
gdf_bus = gdf_bus.drop(columns=["geometry", "index_right"], errors="ignore")
gdf_bus["GEOID"] = normalize_geoid(gdf_bus["GEOID"], target_len)

safe_save_csv(gdf_bus, BUS_OUTPUT)
print(f"✅ Otobüs durakları (GEOID ile) kaydedildi → {BUS_OUTPUT}")

# =========================
# 4) Suç verisini yükle
# =========================
if not os.path.exists(CRIME_INPUT):
    raise FileNotFoundError(f"❌ Suç girdi dosyası yok: {CRIME_INPUT}")

crime = pd.read_csv(CRIME_INPUT, dtype={"GEOID": str}, low_memory=False)
if not {"latitude", "longitude"}.issubset(crime.columns):
    raise ValueError("❌ Suç verisinde 'latitude' ve/veya 'longitude' sütunu eksik!")

# GEOID uzunluğunu suç verisine de uydur
crime["GEOID"] = normalize_geoid(crime["GEOID"], target_len)

# =========================
# 5) Projeksiyon (EPSG:3857) ve KDTree için koordinat matrisleri
# =========================
gdf_crime = gpd.GeoDataFrame(
    crime, geometry=gpd.points_from_xy(crime["longitude"], crime["latitude"]), crs="EPSG:4326"
).to_crs(epsg=3857)

gdf_bus_xy = gpd.GeoDataFrame(
    gdf_bus.copy(), geometry=gpd.points_from_xy(gdf_bus["stop_lon"], gdf_bus["stop_lat"]), crs="EPSG:4326"
).to_crs(epsg=3857)

crime_coords = np.vstack([gdf_crime.geometry.x.values, gdf_crime.geometry.y.values]).T
bus_coords   = np.vstack([gdf_bus_xy.geometry.x.values, gdf_bus_xy.geometry.y.values]).T

if len(bus_coords) == 0:
    # Güvenli geri dönüş: durak yoksa tüm metrikler 0/NaN
    gdf_crime["distance_to_bus"] = np.nan
    gdf_crime["bus_stop_count"] = 0
else:
    tree = cKDTree(bus_coords)
    # En yakın durağa mesafe (metre)
    distances, _ = tree.query(crime_coords, k=1)
    gdf_crime["distance_to_bus"] = distances

    # Dinamik yarıçap: 75. persantil (metre)
    radius = np.nanpercentile(distances, 75) if np.isfinite(distances).any() else 0.0
    radius = float(radius) if radius > 0 else 0.0

    if radius > 0:
        # Yarıçap içinde kaç durak var? (KDTree query_ball_point ile hızlı)
        neigh_lists = tree.query_ball_point(crime_coords, r=radius)
        gdf_crime["bus_stop_count"] = [len(lst) for lst in neigh_lists]
    else:
        gdf_crime["bus_stop_count"] = 0

# =========================
# 6) Binleme (distance & count)
# =========================
# Distance bins
dist = gdf_crime["distance_to_bus"].replace([np.inf, -np.inf], np.nan).dropna()
if len(dist) >= 2 and dist.max() > dist.min():
    n_bins = freedman_diaconis_bin_count(dist.to_numpy(), max_bins=10)
    # quantile bazlı kenarlar → daha dengeli
    _, dist_edges = pd.qcut(dist, q=n_bins, retbins=True, duplicates="drop")
    dist_labels = [f"{int(dist_edges[i])}–{int(dist_edges[i+1])}m" for i in range(len(dist_edges) - 1)]
    gdf_crime["distance_to_bus_range"] = pd.cut(
        gdf_crime["distance_to_bus"], bins=dist_edges, labels=dist_labels, include_lowest=True
    )
else:
    gdf_crime["distance_to_bus_range"] = pd.Series(["0–0m"] * len(gdf_crime))

# Count bins
cnt = gdf_crime["bus_stop_count"].fillna(0)
if cnt.nunique() > 1:
    n_c_bins = freedman_diaconis_bin_count(cnt.to_numpy(), max_bins=8)
    _, cnt_edges = pd.qcut(cnt, q=n_c_bins, retbins=True, duplicates="drop")
    cnt_labels = [f"{int(cnt_edges[i])}–{int(cnt_edges[i+1])}" for i in range(len(cnt_edges) - 1)]
    gdf_crime["bus_stop_count_range"] = pd.cut(
        cnt, bins=cnt_edges, labels=cnt_labels, include_lowest=True
    )
else:
    gdf_crime["bus_stop_count_range"] = pd.Series([f"{int(cnt.min())}–{int(cnt.max())}"] * len(cnt))

# =========================
# 7) Kaydet
# =========================
df_final = gdf_crime.drop(columns="geometry")
safe_save_csv(df_final, CRIME_OUTPUT)
print("✅ Otobüs verisi başarıyla entegre edildi.")
print("📁 Kayıt tamamlandı →", CRIME_OUTPUT)
