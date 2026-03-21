# update_police_gov.py
from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import requests
from sklearn.neighbors import BallTree


# =============================================================================
# LOG / YARDIMCI
# =============================================================================
def log_shape(df: pd.DataFrame, label: str):
    r, c = df.shape
    print(f"📊 {label}: {r} satır × {c} sütun")


def log_delta(before_shape, after_shape, label: str):
    br, bc = before_shape
    ar, ac = after_shape
    print(f"🔗 {label}: {br}×{bc} → {ar}×{ac} (Δr={ar-br}, Δc={ac-bc})")


def ensure_parent(path: str):
    Path(os.path.dirname(path) or ".").mkdir(parents=True, exist_ok=True)


def safe_save_csv(df: pd.DataFrame, path: str):
    try:
        ensure_parent(path)
        tmp = path + ".tmp"
        df.to_csv(tmp, index=False, encoding="utf-8-sig")
        os.replace(tmp, path)
        print(f"💾 Kaydedildi: {path}")
    except Exception as e:
        print(f"❌ Kaydetme hatası: {path}\n{e}")
        try:
            df.to_csv(path + ".bak", index=False, encoding="utf-8-sig")
            print(f"📁 Yedek oluşturuldu: {path}.bak")
        except Exception as e2:
            print(f"❌ Yedek de kaydedilemedi: {e2}")


def find_col(ci_names, candidates):
    m = {str(c).lower(): c for c in ci_names}
    for cand in candidates:
        if str(cand).lower() in m:
            return m[str(cand).lower()]
    return None


def normalize_geoid(series: pd.Series, target_len: int = 11) -> pd.Series:
    s = pd.Series(series).astype(str).str.extract(r"(\d+)", expand=False).fillna("")
    s = s.str[:target_len].str.zfill(target_len)
    s = s.mask(s.eq("0" * target_len))
    return s


def make_quantile_ranges(
    series: pd.Series,
    max_bins: int = 5,
    fallback_label: str = "Unknown"
) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    mask = s.notna()
    s_valid = s[mask]

    if s_valid.nunique() <= 1 or len(s_valid) < 2:
        return pd.Series([fallback_label] * len(series), index=series.index, dtype="object")

    q = min(max_bins, max(3, int(s_valid.nunique())))
    try:
        _, edges = pd.qcut(s_valid, q=q, retbins=True, duplicates="drop")
    except Exception:
        return pd.Series([fallback_label] * len(series), index=series.index, dtype="object")

    if len(edges) < 3:
        return pd.Series([fallback_label] * len(series), index=series.index, dtype="object")

    labels = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i == 0:
            labels.append(f"Q{i+1} (≤{hi:.1f})")
        else:
            labels.append(f"Q{i+1} ({lo:.1f}-{hi:.1f})")

    out = pd.Series(fallback_label, index=series.index, dtype="object")
    out.loc[mask] = pd.cut(
        s_valid,
        bins=edges,
        labels=labels,
        include_lowest=True
    ).astype(str)
    return out


def prep_points(df_points: pd.DataFrame) -> pd.DataFrame:
    if df_points is None or df_points.empty:
        return pd.DataFrame(columns=["latitude", "longitude"])

    lat_col = find_col(df_points.columns, ["latitude", "lat", "y"])
    lon_col = find_col(df_points.columns, ["longitude", "lon", "long", "x"])

    if lat_col is None or lon_col is None:
        return pd.DataFrame(columns=["latitude", "longitude"])

    out = df_points.rename(columns={lat_col: "latitude", lon_col: "longitude"}).copy()
    out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")
    out = out.dropna(subset=["latitude", "longitude"]).copy()

    return out[["latitude", "longitude"]]


def pick_existing(paths):
    for p in paths:
        if p and Path(p).expanduser().resolve().exists():
            return str(Path(p).expanduser().resolve())
    return None


def read_table_auto(path: str) -> pd.DataFrame:
    if str(path).lower().endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def here(*p):
    return Path.cwd().joinpath(*p)


# =============================================================================
# GİRİŞ / ÇIKIŞ YOLLARI
# =============================================================================
CRIME_DATA_DIR = os.getenv("CRIME_DATA_DIR", "").strip()

BASE_CANDIDATES = [
    CRIME_DATA_DIR,
    str(here("crime_prediction_data")),
    str(here()),
]

INPUT_FILES = ["sf_crime_06.parquet", "sf_crime_06.csv"]

CRIME_INPUT_CANDIDATES = []
for base in BASE_CANDIDATES:
    if base:
        for fname in INPUT_FILES:
            CRIME_INPUT_CANDIDATES.append(str(Path(base) / fname))

for fname in INPUT_FILES:
    CRIME_INPUT_CANDIDATES.append(str(here(fname)))

CRIME_INPUT_CANDIDATES = list(dict.fromkeys(CRIME_INPUT_CANDIDATES))

print("🔎 sf_crime_06 giriş aday yolları:")
for p in CRIME_INPUT_CANDIDATES:
    ok = Path(p).expanduser().resolve().exists()
    print("  -", p, "✅" if ok else "❌")

CRIME_IN = pick_existing(CRIME_INPUT_CANDIDATES)
if CRIME_IN is None:
    raise FileNotFoundError(
        "❌ Suç girdisi bulunamadı. Şunlardan en az biri olmalı: "
        + ", ".join(CRIME_INPUT_CANDIDATES)
    )

in_dir = Path(CRIME_IN).parent
CRIME_OUT_PARQUET = str(in_dir / "sf_crime_07.parquet")
CRIME_OUT_CSV = str(in_dir / "sf_crime_07.csv")

print(f"📂 Seçilen giriş: {CRIME_IN}")
print(f"📂 Yazılacak parquet çıkış: {CRIME_OUT_PARQUET}")
print(f"📂 Yazılacak csv çıkış: {CRIME_OUT_CSV}")


# =============================================================================
# VERİ OKU
# =============================================================================
df = read_table_auto(CRIME_IN)
log_shape(df, "CRIME (yükleme)")

if "GEOID" not in df.columns:
    raise KeyError("❌ Suç verisinde 'GEOID' kolonu yok.")

df["GEOID"] = normalize_geoid(df["GEOID"], target_len=11)


# =============================================================================
# CENTROID DÜZELTMESİ
# =============================================================================
CENTROID_CANDIDATES = [
    str(in_dir / "tract_centroids_sf.csv"),
    str(Path(CRIME_DATA_DIR) / "tract_centroids_sf.csv") if CRIME_DATA_DIR else "",
    str(here("tract_centroids_sf.csv")),
    str(here("crime_prediction_data", "tract_centroids_sf.csv")),
]
CENTROID_CANDIDATES = [p for p in CENTROID_CANDIDATES if p]

cent_path = pick_existing(CENTROID_CANDIDATES)
if cent_path:
    print(f"🧭 centroid kaynağı bulundu: {cent_path}")
    cent = pd.read_csv(cent_path, low_memory=False, dtype={"GEOID": str})

    c_lat = find_col(cent.columns, ["centroid_lat", "lat", "latitude", "y"])
    c_lon = find_col(cent.columns, ["centroid_lon", "lon", "longitude", "long", "x"])

    if ("GEOID" in cent.columns) and (c_lat is not None) and (c_lon is not None):
        cent["GEOID"] = normalize_geoid(cent["GEOID"], target_len=11)
        cent = cent[["GEOID", c_lat, c_lon]].rename(
            columns={c_lat: "centroid_lat", c_lon: "centroid_lon"}
        )
        cent["centroid_lat"] = pd.to_numeric(cent["centroid_lat"], errors="coerce")
        cent["centroid_lon"] = pd.to_numeric(cent["centroid_lon"], errors="coerce")
        cent = cent.dropna(subset=["GEOID", "centroid_lat", "centroid_lon"]).drop_duplicates("GEOID")

        need_merge = ("centroid_lat" not in df.columns) or ("centroid_lon" not in df.columns)
        if need_merge:
            _b = df.shape
            df = df.merge(cent, on="GEOID", how="left")
            log_delta(_b, df.shape, "CRIME ⨯ tract_centroids_sf")
    else:
        print("⚠️ tract_centroids_sf.csv bulundu ama GEOID/lat/lon kolonları okunamadı.")
else:
    print("⚠️ tract_centroids_sf.csv bulunamadı. centroid üretimi için başka kaynak gerekir.")

lat_pref = find_col(df.columns, ["centroid_lat"])
lon_pref = find_col(df.columns, ["centroid_lon"])

if lat_pref is None or lon_pref is None:
    lat_pref = find_col(df.columns, ["latitude", "lat", "y"])
    lon_pref = find_col(df.columns, ["longitude", "lon", "long", "x"])

if lat_pref is None or lon_pref is None:
    raise KeyError(
        "❌ centroid_lat/centroid_lon veya latitude/longitude bulunamadı. "
        "(tract_centroids_sf.csv merge edilmemiş olabilir)"
    )

lat_tmp = pd.to_numeric(df[lat_pref], errors="coerce")
lon_tmp = pd.to_numeric(df[lon_pref], errors="coerce")

geo = (
    df.assign(_lat=lat_tmp, _lon=lon_tmp)
      .loc[lambda x: x["_lat"].notna() & x["_lon"].notna(), ["GEOID", "_lat", "_lon"]]
      .rename(columns={"_lat": "centroid_lat", "_lon": "centroid_lon"})
      .groupby("GEOID", as_index=False)[["centroid_lat", "centroid_lon"]]
      .mean()
)

log_shape(geo, "GEOID centroid (hazır)")


# =============================================================================
# POLICE / GOVERNMENT
# =============================================================================
POLICE_CANDIDATES = [
    str(in_dir / "sf_police_stations.csv"),
    str(Path(CRIME_DATA_DIR) / "sf_police_stations.csv") if CRIME_DATA_DIR else "",
    str(here("crime_prediction_data", "sf_police_stations.csv")),
    str(here("sf_police_stations.csv")),
]
GOV_CANDIDATES = [
    str(in_dir / "sf_government_buildings.csv"),
    str(Path(CRIME_DATA_DIR) / "sf_government_buildings.csv") if CRIME_DATA_DIR else "",
    str(here("crime_prediction_data", "sf_government_buildings.csv")),
    str(here("sf_government_buildings.csv")),
]
POLICE_CANDIDATES = [p for p in POLICE_CANDIDATES if p]
GOV_CANDIDATES = [p for p in GOV_CANDIDATES if p]

FORCE_PG_REFRESH = os.getenv("FORCE_PG_REFRESH", "0").strip().lower() in ("1", "true", "yes")

police_csv = pick_existing(POLICE_CANDIDATES)
gov_csv = pick_existing(GOV_CANDIDATES)

CACHE_OK = (police_csv is not None) and (gov_csv is not None)

if CACHE_OK and (not FORCE_PG_REFRESH):
    print("✅ POLICE/GOV cache bulundu ve FORCE_PG_REFRESH=0 → indirme atlanıyor.")
    police_path = police_csv
    gov_path = gov_csv
else:
    print("♻️ POLICE/GOV refresh (Overpass) başlıyor...")

    OVERPASS_URL = os.getenv("OVERPASS_URL", "https://overpass-api.de/api/interpreter")
    SF_BBOX = os.getenv("SF_BBOX", "-122.525,37.708,-122.355,37.833")

    def overpass(query: str):
        r = requests.post(OVERPASS_URL, data={"data": query}, timeout=180)
        r.raise_for_status()
        return r.json()

    def to_points(osm_json):
        rows = []
        for el in osm_json.get("elements", []):
            lat = el.get("lat") or (el.get("center") or {}).get("lat")
            lon = el.get("lon") or (el.get("center") or {}).get("lon")
            if lat is None or lon is None:
                continue

            tags = el.get("tags", {}) or {}
            rows.append({
                "name": tags.get("name", ""),
                "latitude": float(lat),
                "longitude": float(lon),
                "osm_id": el.get("id"),
                "osm_type": el.get("type"),
                "source": "OpenStreetMap/Overpass",
                "data_as_of": datetime.utcnow().strftime("%Y-%m-%d"),
            })
        return pd.DataFrame(rows)

    q_police = f"""
    [out:json][timeout:180];
    (
      node["amenity"="police"]({SF_BBOX});
      way["amenity"="police"]({SF_BBOX});
      relation["amenity"="police"]({SF_BBOX});
    );
    out center;
    """

    q_gov = f"""
    [out:json][timeout:180];
    (
      node["office"="government"]({SF_BBOX});
      way["office"="government"]({SF_BBOX});
      relation["office"="government"]({SF_BBOX});
      node["amenity"="townhall"]({SF_BBOX});
      way["amenity"="townhall"]({SF_BBOX});
      relation["amenity"="townhall"]({SF_BBOX});
    );
    out center;
    """

    try:
        dfp = to_points(overpass(q_police))
        dfg = to_points(overpass(q_gov))

        if dfp.empty:
            print("⚠️ Overpass police boş döndü.")
        if dfg.empty:
            print("⚠️ Overpass gov boş döndü.")

        police_out = str(in_dir / "sf_police_stations.csv")
        gov_out = str(in_dir / "sf_government_buildings.csv")

        if not dfp.empty:
            safe_save_csv(dfp, police_out)
            print("✅ Yazıldı:", police_out)
        if not dfg.empty:
            safe_save_csv(dfg, gov_out)
            print("✅ Yazıldı:", gov_out)

        police_path = police_out if os.path.exists(police_out) else pick_existing(POLICE_CANDIDATES)
        gov_path = gov_out if os.path.exists(gov_out) else pick_existing(GOV_CANDIDATES)

    except Exception as e:
        print("⚠️ Overpass indirme hatası:", e)
        police_path = pick_existing(POLICE_CANDIDATES)
        gov_path = pick_existing(GOV_CANDIDATES)

if police_path is None:
    print("⚠️ sf_police_stations.csv bulunamadı; polis mesafeleri NaN olacak.")
    df_police = pd.DataFrame(columns=["latitude", "longitude"])
else:
    df_police = pd.read_csv(police_path, low_memory=False)

if gov_path is None:
    print("⚠️ sf_government_buildings.csv bulunamadı; government mesafeleri NaN olacak.")
    df_gov = pd.DataFrame(columns=["latitude", "longitude"])
else:
    df_gov = pd.read_csv(gov_path, low_memory=False)

df_police = prep_points(df_police)
df_gov = prep_points(df_gov)


# =============================================================================
# BALLTREE (HAVERSINE)
# =============================================================================
EARTH_R = 6_371_000.0

if geo.empty:
    raise ValueError("❌ GEOID centroid tablosu boş. distance_to_police / government üretilemez.")

centroids_rad = np.radians(geo[["centroid_lat", "centroid_lon"]].to_numpy(dtype=float))

if not df_police.empty:
    police_rad = np.radians(df_police[["latitude", "longitude"]].to_numpy(dtype=float))
    police_tree = BallTree(police_rad, metric="haversine")
    dist_police, _ = police_tree.query(centroids_rad, k=1)
    geo["distance_to_police"] = (dist_police[:, 0] * EARTH_R).round(1)
else:
    geo["distance_to_police"] = np.nan

if not df_gov.empty:
    gov_rad = np.radians(df_gov[["latitude", "longitude"]].to_numpy(dtype=float))
    gov_tree = BallTree(gov_rad, metric="haversine")
    dist_gov, _ = gov_tree.query(centroids_rad, k=1)
    geo["distance_to_government_building"] = (dist_gov[:, 0] * EARTH_R).round(1)
else:
    geo["distance_to_government_building"] = np.nan

geo["is_near_police"] = pd.Series(geo["distance_to_police"] <= 300, index=geo.index).astype("Int64")
geo["is_near_government"] = pd.Series(geo["distance_to_government_building"] <= 300, index=geo.index).astype("Int64")

geo["distance_to_police_range"] = make_quantile_ranges(
    geo["distance_to_police"],
    max_bins=5,
    fallback_label="Unknown"
)
geo["distance_to_government_building_range"] = make_quantile_ranges(
    geo["distance_to_government_building"],
    max_bins=5,
    fallback_label="Unknown"
)

log_shape(geo, "GEOID metrikleri (polis+gov)")


# =============================================================================
# MERGE
# =============================================================================
keep_cols = [
    "GEOID",
    "distance_to_police",
    "distance_to_police_range",
    "distance_to_government_building",
    "distance_to_government_building_range",
    "is_near_police",
    "is_near_government",
]

to_drop = [c for c in keep_cols if c != "GEOID" and c in df.columns]
if to_drop:
    df = df.drop(columns=to_drop, errors="ignore")

_before = df.shape
df = df.merge(geo[keep_cols], on="GEOID", how="left", suffixes=("", "_pg"))
log_delta(_before, df.shape, "CRIME ⨯ GEOID(polis+gov)")


# =============================================================================
# NaN RAPORU
# =============================================================================
nan_counts = df.isna().sum()
nan_counts = nan_counts[nan_counts > 0].sort_values(ascending=False)

print("🔎 NaN sayıları (sf_crime_07 yazılmadan önce):")
if nan_counts.empty:
    print("✅ NaN yok.")
else:
    print(nan_counts.to_string())

pg_cols = [
    "distance_to_police",
    "distance_to_police_range",
    "distance_to_government_building",
    "distance_to_government_building_range",
    "is_near_police",
    "is_near_government",
]
pg_cols = [c for c in pg_cols if c in df.columns]

if pg_cols:
    print("🔎 Police/Gov kolonları NaN sayıları:")
    print(df[pg_cols].isna().sum().to_string())


# =============================================================================
# KAYDET
# =============================================================================
print("💾 sf_crime_07 kaydetme aşaması başlıyor...")

tmp_parquet = CRIME_OUT_PARQUET + ".tmp.parquet"
df.to_parquet(
    tmp_parquet,
    index=False,
    engine="pyarrow",
    compression="snappy"
)
os.replace(tmp_parquet, CRIME_OUT_PARQUET)
print(f"✅ Parquet kaydedildi: {CRIME_OUT_PARQUET} | Satır: {len(df):,} | Sütun: {df.shape[1]}")

WRITE_CSV = os.getenv("WRITE_CSV", "0").strip().lower() in ("1", "true", "yes", "on")
if WRITE_CSV:
    safe_save_csv(df, CRIME_OUT_CSV)
    print(f"✅ CSV kaydedildi: {CRIME_OUT_CSV} | Satır: {len(df):,} | Sütun: {df.shape[1]}")
else:
    print("ℹ️ CSV yazımı kapalı; yalnız parquet kaydedildi.")

try:
    preview = pd.read_parquet(CRIME_OUT_PARQUET).head(3)
    print(f"📄 {CRIME_OUT_PARQUET} — ilk 3 satır:")
    print(preview.to_string(index=False))
except Exception as e:
    print(f"⚠️ Parquet önizleme okunamadı: {e}")
