import os
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

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
    data = data[np.isfinite(data)]
    if data.size < 2 or np.allclose(data.min(), data.max()):
        return 1
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    if iqr <= 0:
        return min(max_bins, max(2, int(np.sqrt(len(data)))))
    bw = 2 * iqr / (len(data) ** (1 / 3))
    if bw <= 0:
        return min(max_bins, max(2, int(np.sqrt(len(data)))))
    return max(2, min(max_bins, int(np.ceil((data.max() - data.min()) / bw))))

# =========================
# 1) Dosya yolları
# =========================
BASE_DIR     = "crime_data"
Path(BASE_DIR).mkdir(exist_ok=True)

CRIME_INPUT  = os.path.join(BASE_DIR, "sf_crime_04.csv")
CRIME_OUTPUT = os.path.join(BASE_DIR, "sf_crime_05.csv")
TRAIN_OUTPUT = os.path.join(BASE_DIR, "sf_train_stops_with_geoid.csv")

# census geojson hem crime_data/ hem kökte olabilir
CENSUS_CANDIDATES = [
    os.path.join(BASE_DIR, "sf_census_blocks_with_population.geojson"),
    os.path.join(".", "sf_census_blocks_with_population.geojson"),
]

# GTFS kaynakları
GTFS_URL = "https://transitfeeds.com/p/bart/58/latest/download"  # BART
GTFS_ZIP = "/tmp/bart_gtfs.zip"
GTFS_TXT = "/tmp/stops.txt"

# =========================
# 2) GTFS verisini indir ve çıkar
# =========================
print("🚉 BART tren verisi indiriliyor...")
download_ok = False
for attempt in range(3):
    try:
        urlretrieve(GTFS_URL, GTFS_ZIP)
        with zipfile.ZipFile(GTFS_ZIP, "r") as zf:
            # Bazı paketlerde path farklı olabilir; güvenli çıkarma
            members = [m for m in zf.namelist() if m.lower().endswith("stops.txt")]
            if not members:
                raise FileNotFoundError("stops.txt GTFS paketinde bulunamadı.")
            zf.extract(members[0], "/tmp/")
            extracted = os.path.join("/tmp", members[0].split("/")[-1])
            os.rename(extracted, GTFS_TXT) if extracted != GTFS_TXT else None
        download_ok = True
        break
    except Exception as e:
        print(f"⚠️ İndirme/çıkarma denemesi {attempt+1} başarısız: {e}")

if not download_ok:
    raise SystemExit("❌ GTFS indirilemedi; çıkılıyor.")

bart_stops = pd.read_csv(GTFS_TXT, dtype={"stop_lat": float, "stop_lon": float})
bart_stops = bart_stops.dropna(subset=["stop_lat", "stop_lon"]).copy()
print(f"📥 GTFS stops: {len(bart_stops)} kayıt")

# =========================
# 3) GEOID eşlemesi (census sjoin)
# =========================
census_path = next((p for p in CENSUS_CANDIDATES if os.path.exists(p)), None)
if census_path is None:
    raise FileNotFoundError("❌ Nüfus blokları GeoJSON bulunamadı (crime_data/ veya kök).")

gdf_stops = gpd.GeoDataFrame(
    bart_stops,
    geometry=gpd.points_from_xy(bart_stops["stop_lon"], bart_stops["stop_lat"]),
    crs="EPSG:4326",
)

gdf_blocks = gpd.read_file(census_path)
target_len = gdf_blocks["GEOID"].astype(str).str.len().mode().iat[0]
gdf_blocks["GEOID"] = normalize_geoid(gdf_blocks["GEOID"], target_len)

gdf_joined = gpd.sjoin(gdf_stops, gdf_blocks[["geometry", "GEOID"]], how="left", predicate="within")
gdf_joined = gdf_joined.drop(columns=["index_right"], errors="ignore")
gdf_joined["GEOID"] = normalize_geoid(gdf_joined["GEOID"], target_len)
gdf_joined = gdf_joined.drop(columns=["geometry"])

safe_save_csv(gdf_joined, TRAIN_OUTPUT)
print(f"✅ {len(gdf_joined)} tren durağı SF içinde bulundu → {TRAIN_OUTPUT}")

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
# 5) Projeksiyon (EPSG:3857) ve KDTree için matrisler
# =========================
gdf_crime = gpd.GeoDataFrame(
    crime, geometry=gpd.points_from_xy(crime["longitude"], crime["latitude"]), crs="EPSG:4326"
).to_crs(epsg=3857)

gdf_train_xy = gpd.GeoDataFrame(
    gdf_joined.copy(), geometry=gpd.points_from_xy(gdf_joined["stop_lon"], gdf_joined["stop_lat"]), crs="EPSG:4326"
).to_crs(epsg=3857)

crime_coords = np.vstack([gdf_crime.geometry.x.values, gdf_crime.geometry.y.values]).T
train_coords = np.vstack([gdf_train_xy.geometry.x.values, gdf_train_xy.geometry.y.values]).T

if len(train_coords) == 0:
    gdf_crime["distance_to_train"] = np.nan
    gdf_crime["train_stop_count"] = 0
else:
    tree = cKDTree(train_coords)
    # En yakın durağa mesafe (metre)
    distances, _ = tree.query(crime_coords, k=1)
    gdf_crime["distance_to_train"] = distances

    # Dinamik yarıçap: 75. persantil
    radius = np.nanpercentile(distances, 75) if np.isfinite(distances).any() else 0.0
    radius = float(radius) if radius > 0 else 0.0

    if radius > 0:
        neigh_lists = tree.query_ball_point(crime_coords, r=radius)
        gdf_crime["train_stop_count"] = [len(lst) for lst in neigh_lists]
    else:
        gdf_crime["train_stop_count"] = 0

# =========================
# 6) Binleme (mesafe & sayı)
# =========================
# Mesafe aralığı
dist = gdf_crime["distance_to_train"].replace([np.inf, -np.inf], np.nan).dropna()
if len(dist) >= 2 and dist.max() > dist.min():
    n_bins = freedman_diaconis_bin_count(dist.to_numpy(), max_bins=10)
    _, edges = pd.qcut(dist, q=n_bins, retbins=True, duplicates="drop")
    labels = [f"{int(edges[i])}–{int(edges[i+1])}m" for i in range(len(edges) - 1)]
    gdf_crime["distance_to_train_range"] = pd.cut(
        gdf_crime["distance_to_train"], bins=edges, labels=labels, include_lowest=True
    )
else:
    gdf_crime["distance_to_train_range"] = pd.Series(["0–0m"] * len(gdf_crime))

# Sayı aralığı
cnt = gdf_crime["train_stop_count"].fillna(0)
if cnt.nunique() > 1:
    n_c_bins = freedman_diaconis_bin_count(cnt.to_numpy(), max_bins=8)
    _, c_edges = pd.qcut(cnt, q=n_c_bins, retbins=True, duplicates="drop")
    c_labels = [f"{int(c_edges[i])}–{int(c_edges[i+1])}" for i in range(len(c_edges) - 1)]
    gdf_crime["train_stop_count_range"] = pd.cut(
        cnt, bins=c_edges, labels=c_labels, include_lowest=True
    )
else:
    gdf_crime["train_stop_count_range"] = pd.Series([f"{int(cnt.min())}–{int(cnt.max())}"] * len(cnt))

# =========================
# 7) Kaydet & Özet
# =========================
df_final = gdf_crime.drop(columns="geometry")
safe_save_csv(df_final, CRIME_OUTPUT)

print("📦 Yeni sütunlar eklendi:")
print(df_final[[
    "GEOID", "distance_to_train", "distance_to_train_range",
    "train_stop_count", "train_stop_count_range"
]].head())

print(f"✅ Güncellenmiş veri kaydedildi → {CRIME_OUTPUT}")
print(f"📊 Satır sayısı: {df_final.shape[0]} | Sütun sayısı: {df_final.shape[1]}")
