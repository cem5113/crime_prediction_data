# enrich_police_gov_06_to_07.py
import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
import os

# -------------------------
# 1) Dosya yolları
# -------------------------
BASE_DIR = "crime_data"
CRIME_IN  = os.path.join(BASE_DIR, "sf_crime_06.csv")
POLICE    = os.path.join(BASE_DIR, "sf_police_stations.csv")
GOV       = os.path.join(BASE_DIR, "sf_government_buildings.csv")
CRIME_OUT = os.path.join(BASE_DIR, "sf_crime_07.csv")

# -------------------------
# 2) Verileri yükle
# -------------------------
df = pd.read_csv(CRIME_IN)
df_police = pd.read_csv(POLICE).dropna(subset=["latitude", "longitude"])
df_gov    = pd.read_csv(GOV).dropna(subset=["latitude", "longitude"])

# longitude/latitude kolon isimlerini normalize et (lon/lat -> longitude/latitude)
if "longitude" not in df.columns and "lon" in df.columns:
    df = df.rename(columns={"lon": "longitude"})
if "latitude" not in df.columns and "lat" in df.columns:
    df = df.rename(columns={"lat": "latitude"})

required_cols = {"latitude", "longitude"}
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise KeyError(f"❌ sf_crime_06.csv içinde eksik kolon(lar): {missing}")

# GEOID formatını düzelt (11 hane varsayımı; veri yapına göre gerekirse 12/15 yapılabilir)
if "GEOID" in df.columns:
    df["GEOID"] = (
        df["GEOID"]
        .astype(str)
        .str.extract(r"(\d+)", expand=False)
        .fillna("")
        .apply(lambda x: x.zfill(11) if x else x)
    )

# -------------------------
# 3) BallTree ile en yakın mesafeler
# -------------------------
# Radyan dönüşümleri
crime_rad  = np.radians(df[["latitude", "longitude"]].values.astype(float))
police_rad = np.radians(df_police[["latitude", "longitude"]].values.astype(float))
gov_rad    = np.radians(df_gov[["latitude", "longitude"]].values.astype(float))

# Boş güvenlikleri
if len(police_rad) == 0:
    raise ValueError("❌ sf_police_stations.csv boş veya koordinatları eksik.")
if len(gov_rad) == 0:
    raise ValueError("❌ sf_government_buildings.csv boş veya koordinatları eksik.")

police_tree = BallTree(police_rad, metric="haversine")
gov_tree    = BallTree(gov_rad,    metric="haversine")

dist_police, _ = police_tree.query(crime_rad, k=1)
dist_gov, _    = gov_tree.query(crime_rad, k=1)

# metreye çevir
df["distance_to_police"]               = (dist_police[:, 0] * 6371000).round(1)
df["distance_to_government_building"]  = (dist_gov[:, 0] * 6371000).round(1)

# 300m yakınlık bayrakları
df["is_near_police"]     = (df["distance_to_police"] <= 300).astype(int)
df["is_near_government"] = (df["distance_to_government_building"] <= 300).astype(int)

# -------------------------
# 4) Dinamik aralık etiketleme
# -------------------------
def make_dynamic_range_func(data: pd.DataFrame, col: str, strategy: str = "auto", max_bins: int = 5):
    vals = data[col].dropna().values
    if len(vals) == 0:
        # boşsa sabit kenarlar
        edges = np.array([0, 1, 2])
        def label_empty(x):
            if pd.isna(x): return "Unknown"
            return "Q1 (≤1.0)" if x <= 1 else ("Q2 (1.0-2.0)" if x <= 2 else "Q3 (>2.0)")
        return label_empty

    std = np.std(vals)
    iqr = np.percentile(vals, 75) - np.percentile(vals, 25)
    if strategy == "auto":
        if len(vals) < 500:
            bin_count = 3
        elif std < 1 or iqr < 1:
            bin_count = 4
        elif std > 20:
            bin_count = min(10, max_bins)
        else:
            bin_count = 5
    else:
        bin_count = int(strategy)

    qs = [i / bin_count for i in range(bin_count + 1)]
    edges = data[col].quantile(qs).values
    # tüm değerler aynıysa ufak bir genişlik ver
    if np.allclose(edges.min(), edges.max()):
        v = float(edges[0])
        edges = np.array([v - 1e-6, v, v + 1e-6])

    def label(x):
        if pd.isna(x):
            return "Unknown"
        for i in range(len(edges) - 1):
            if x <= edges[i + 1]:
                if i == 0:
                    return f"Q{i+1} (≤{edges[i+1]:.1f})"
                else:
                    return f"Q{i+1} ({edges[i]:.1f}-{edges[i+1]:.1f})"
        return f"Q{len(edges)} (>{edges[-1]:.1f})"

    return label

df["distance_to_police_range"] = df["distance_to_police"].apply(
    make_dynamic_range_func(df, "distance_to_police", strategy="auto")
)
df["distance_to_government_building_range"] = df["distance_to_government_building"].apply(
    make_dynamic_range_func(df, "distance_to_government_building", strategy="auto")
)

# -------------------------
# 5) Kaydet & özet
# -------------------------
df.to_csv(CRIME_OUT, index=False)
print(f"✅ Polis/devlet yakınlık ölçümleri eklendi.")
print(f"📁 Kaydedildi: {CRIME_OUT}")
print("📋 İlk satırlar:")
print(
    df[[
        "GEOID",
        "distance_to_police", "distance_to_police_range",
        "distance_to_government_building", "distance_to_government_building_range",
        "is_near_police", "is_near_government"
    ]].head()
)
