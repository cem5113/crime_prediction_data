# scripts/enrich_police_gov_06_to_07.py

import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

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

def find_col(ci_names, candidates):
    m = {c.lower(): c for c in ci_names}
    for cand in candidates:
        if cand.lower() in m:
            return m[cand.lower()]
    return None

def normalize_geoid(series: pd.Series, target_len: int) -> pd.Series:
    s = series.astype(str).str.extract(r"(\d+)")[0]
    return s.str.zfill(target_len)

def choose_geoid_len(s: pd.Series, default_len: int = 12) -> int:
    vals = s.astype(str).str.extract(r"(\d+)")[0].dropna()
    if vals.empty:
        return default_len
    lens = vals.str.len()
    mode = lens.mode()
    return int(mode.iat[0]) if not mode.empty else default_len

def make_quantile_ranges(series: pd.Series, max_bins: int = 5, fallback_label: str = "Unknown") -> pd.Series:
    """Serinin tamamı için Q-etiketleri döndürür (Q1..Qk)."""
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    mask = s.notna()
    s_valid = s[mask]
    if s_valid.nunique() <= 1 or len(s_valid) < 2:
        return pd.Series([fallback_label] * len(series), index=series.index)

    q = min(max_bins, max(3, s_valid.nunique()))
    # quantile kenarlarını al
    try:
        _, edges = pd.qcut(s_valid, q=q, retbins=True, duplicates="drop")
    except Exception:
        return pd.Series([fallback_label] * len(series), index=series.index)
    if len(edges) < 3:  # çok az kenar
        return pd.Series([fallback_label] * len(series), index=series.index)

    labels = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if i == 0:
            labels.append(f"Q{i+1} (≤{hi:.1f})")
        else:
            labels.append(f"Q{i+1} ({lo:.1f}-{hi:.1f})")

    # Tam seriyi aynı kenarlarla etiketle
    out = pd.Series(fallback_label, index=series.index, dtype="object")
    out.loc[mask] = pd.cut(s_valid, bins=edges, labels=labels, include_lowest=True).astype(str)
    return out

# =========================
# 1) Dosya yolları
# =========================
BASE_DIR  = "crime_data"
Path(BASE_DIR).mkdir(exist_ok=True)

CRIME_IN  = os.path.join(BASE_DIR, "sf_crime_06.csv")
CRIME_OUT = os.path.join(BASE_DIR, "sf_crime_07.csv")

# Polis & devlet dosyalarını hem crime_data/ hem kökte ara
POLICE_CANDIDATES = [
    os.path.join(BASE_DIR, "sf_police_stations.csv"),
    os.path.join(".",      "sf_police_stations.csv"),
]
GOV_CANDIDATES = [
    os.path.join(BASE_DIR, "sf_government_buildings.csv"),
    os.path.join(".",      "sf_government_buildings.csv"),
]

def pick_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

# =========================
# 2) Verileri yükle
# =========================
if not os.path.exists(CRIME_IN):
    raise FileNotFoundError(f"❌ Suç girdisi bulunamadı: {CRIME_IN}")
df = pd.read_csv(CRIME_IN, low_memory=False)

police_path = pick_existing(POLICE_CANDIDATES)
gov_path    = pick_existing(GOV_CANDIDATES)

if police_path is None:
    print("⚠️ sf_police_stations.csv bulunamadı; polis mesafe metrikleri NaN/0 olacak.")
    df_police = pd.DataFrame(columns=["latitude", "longitude"])
else:
    df_police = pd.read_csv(police_path, low_memory=False)

if gov_path is None:
    print("⚠️ sf_government_buildings.csv bulunamadı; devlet binası metrikleri NaN/0 olacak.")
    df_gov = pd.DataFrame(columns=["latitude", "longitude"])
else:
    df_gov = pd.read_csv(gov_path, low_memory=False)

# Suç lat/lon isimlerini normalize et
if "longitude" not in df.columns and "lon" in df.columns:
    df = df.rename(columns={"lon": "longitude"})
if "latitude" not in df.columns and "lat" in df.columns:
    df = df.rename(columns={"lat": "latitude"})

req_cols = {"latitude", "longitude"}
missing = [c for c in req_cols if c not in df.columns]
if missing:
    raise KeyError(f"❌ sf_crime_06.csv içinde eksik kolon(lar): {missing}")

# Sayısal ve temizlik
df["latitude"]  = pd.to_numeric(df["latitude"], errors="coerce")
df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
df = df.dropna(subset=["latitude", "longitude"]).copy()

# GEOID normalize (dosyadaki baskın uzunluğa göre)
if "GEOID" in df.columns:
    tgt_len = choose_geoid_len(df["GEOID"], default_len=12)
    df["GEOID"] = normalize_geoid(df["GEOID"], tgt_len)

# Polis/gov lat/lon kolonlarını bul ve normalize et
def prep_points(df_points: pd.DataFrame) -> pd.DataFrame:
    if df_points.empty:
        return df_points
    lat_col = find_col(df_points.columns, ["latitude", "lat", "y"])
    lon_col = find_col(df_points.columns, ["longitude", "lon", "x"])
    if lat_col is None or lon_col is None:
        return pd.DataFrame(columns=["latitude", "longitude"])
    out = df_points.rename(columns={lat_col: "latitude", lon_col: "longitude"})
    out["latitude"]  = pd.to_numeric(out["latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")
    out = out.dropna(subset=["latitude", "longitude"]).copy()
    return out

df_police = prep_points(df_police)
df_gov    = prep_points(df_gov)

# =========================
# 3) BallTree ile en yakın mesafeler (metre)
# =========================
EARTH_R = 6_371_000.0

crime_rad  = np.radians(df[["latitude", "longitude"]].to_numpy(dtype=float))
if not df_police.empty:
    police_rad = np.radians(df_police[["latitude", "longitude"]].to_numpy(dtype=float))
    police_tree = BallTree(police_rad, metric="haversine")
    dist_police, _ = police_tree.query(crime_rad, k=1)
    df["distance_to_police"] = (dist_police[:, 0] * EARTH_R).round(1)
else:
    df["distance_to_police"] = np.nan

if not df_gov.empty:
    gov_rad = np.radians(df_gov[["latitude", "longitude"]].to_numpy(dtype=float))
    gov_tree = BallTree(gov_rad, metric="haversine")
    dist_gov, _ = gov_tree.query(crime_rad, k=1)
    df["distance_to_government_building"] = (dist_gov[:, 0] * EARTH_R).round(1)
else:
    df["distance_to_government_building"] = np.nan

# 300m yakınlık bayrakları (NaN’lar False -> 0)
df["is_near_police"] = (df["distance_to_police"] <= 300).astype(int).where(df["distance_to_police"].notna(), 0)
df["is_near_government"] = (df["distance_to_government_building"] <= 300).astype(int).where(
    df["distance_to_government_building"].notna(), 0
)

# =========================
# 4) Dinamik aralık etiketleme
# =========================
df["distance_to_police_range"] = make_quantile_ranges(df["distance_to_police"], max_bins=5, fallback_label="Unknown")
df["distance_to_government_building_range"] = make_quantile_ranges(
    df["distance_to_government_building"], max_bins=5, fallback_label="Unknown"
)

# =========================
# 5) Kaydet & özet
# =========================
safe_save_csv(df, CRIME_OUT)
print("✅ Polis/devlet yakınlık ölçümleri eklendi.")
print(f"📁 Kaydedildi: {CRIME_OUT}")
try:
    print(
        df[[
            "GEOID",
            "distance_to_police", "distance_to_police_range",
            "distance_to_government_building", "distance_to_government_building_range",
            "is_near_police", "is_near_government"
        ]].head().to_string(index=False)
    )
except Exception:
    pass
