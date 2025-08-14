import os
import ast
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from collections import defaultdict, Counter
from sklearn.neighbors import BallTree

# === 0. DOSYA YOLLARI ===
BASE_DIR = "crime_data"
POI_GEOJSON = os.path.join(BASE_DIR, "sf_pois.geojson")
BLOCK_PATH = os.path.join(BASE_DIR, "sf_census_blocks_with_population.geojson")
POI_CLEANED_CSV = os.path.join(BASE_DIR, "sf_pois_cleaned_with_geoid.csv")
POI_RISK_JSON = os.path.join(BASE_DIR, "risky_pois_dynamic.json")
CRIME_INPUT = os.path.join(BASE_DIR, "sf_crime_05.csv")
CRIME_OUTPUT = os.path.join(BASE_DIR, "sf_crime_06.csv")

# ---------- Yardımcılar ----------
def _parse_tags(val):
    if isinstance(val, dict): return val
    if isinstance(val, str):
        for loader in (json.loads, ast.literal_eval):
            try:
                out = loader(val)
                return out if isinstance(out, dict) else {}
            except Exception:
                pass
    return {}

def _ensure_crs(gdf, target="EPSG:4326"):
    if gdf.crs is None:
        gdf = gdf.set_crs(target, allow_override=True)
    elif gdf.crs.to_string().upper().endswith("CRS84"):
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    elif gdf.crs.to_string() != target:
        gdf = gdf.to_crs(target)
    return gdf

def _extract_cat_sub_name_from_tags(tags: dict):
    name = tags.get("name")
    for key in ("amenity", "shop", "leisure"):
        if key in tags and tags[key]:
            return key, tags[key], name
    return None, None, name

def _q_labels(series, q=5, labels=None):
    s = pd.to_numeric(series, errors="coerce")
    if labels is None:
        labels = [f"Q{i}" for i in range(1, q+1)]
    try:
        return pd.qcut(s, q=q, duplicates="drop", labels=labels)
    except Exception:
        # tekil değer vb. durumlarda hepsini tek etikete al
        return pd.Series([labels[-1]]*len(s), index=s.index)

# === 1) POI TEMİZLEME + GEOID EKLEME ===
def clean_and_assign_geoid_to_pois():
    print("📍 POI verisi okunuyor ve GEOID atanıyor...")
    gdf = gpd.read_file(POI_GEOJSON)
    gdf = _ensure_crs(gdf, "EPSG:4326")

    if "tags" not in gdf.columns:
        gdf["tags"] = [{}] * len(gdf)
    gdf["tags"] = gdf["tags"].apply(_parse_tags)

    cat_sub_name = gdf["tags"].apply(_extract_cat_sub_name_from_tags)
    gdf[["poi_category", "poi_subcategory", "poi_name"]] = pd.DataFrame(
        cat_sub_name.tolist(), index=gdf.index
    )

    if "geometry" not in gdf.columns:
        if {"lon","lat"}.issubset(gdf.columns):
            gdf["geometry"] = gpd.points_from_xy(gdf["lon"], gdf["lat"])
        else:
            raise ValueError("GeoJSON içinde geometry yok ve lon/lat da bulunamadı.")
    gdf = _ensure_crs(gdf, "EPSG:4326")
    gdf["lon"] = gdf.geometry.x
    gdf["lat"] = gdf.geometry.y

    gdf_blocks = gpd.read_file(BLOCK_PATH)
    gdf_blocks = _ensure_crs(gdf_blocks, "EPSG:4326")
    gdf_blocks["GEOID"] = gdf_blocks["GEOID"].astype(str).str.zfill(11)

    # Point within polygon
    gdf_joined = gpd.sjoin(
        gdf,
        gdf_blocks[["GEOID", "geometry"]],
        how="left",
        predicate="within"
    )

    keep = [c for c in ["id","lat","lon","poi_category","poi_subcategory","poi_name","GEOID"] if c in gdf_joined.columns]
    df_cleaned = gdf_joined[keep].copy()
    if "id" not in df_cleaned.columns:
        df_cleaned["id"] = np.arange(len(df_cleaned))
    df_cleaned = df_cleaned.dropna(subset=["lat","lon"])
    df_cleaned.to_csv(POI_CLEANED_CSV, index=False)
    print(f"✅ Temizlenmiş POI verisi kaydedildi: {POI_CLEANED_CSV} (satır: {len(df_cleaned)})")
    return df_cleaned

# === 2) DİNAMİK RİSK SKORU (300 m) ===
def calculate_dynamic_risk(df_crime, df_poi):
    print("📊 Risk skoru hesaplanıyor (300 m)...")
    dfc = df_crime.dropna(subset=["latitude","longitude"]).copy()
    dfc["latitude"] = pd.to_numeric(dfc["latitude"], errors="coerce")
    dfc["longitude"] = pd.to_numeric(dfc["longitude"], errors="coerce")
    dfc = dfc.dropna(subset=["latitude","longitude"])

    dfp = df_poi.dropna(subset=["lat","lon"]).copy()
    dfp["lat"] = pd.to_numeric(dfp["lat"], errors="coerce")
    dfp["lon"] = pd.to_numeric(dfp["lon"], errors="coerce")
    dfp = dfp.dropna(subset=["lat","lon"])

    # bazı kurumları hariç bırak
    if "poi_subcategory" in dfp.columns:
        dfp = dfp[~dfp["poi_subcategory"].isin(["police","ranger_station"])]

    if dfc.empty or dfp.empty:
        print("⚠️ Risk için yetersiz veri. Boş skor döndü.")
        with open(POI_RISK_JSON, "w") as f:
            json.dump({}, f, indent=2)
        return {}

    tree = BallTree(np.radians(dfc[["latitude","longitude"]].values), metric="haversine")
    radius = 300 / 6371000.0  # 300 m (radyan)

    counts = defaultdict(list)  # subtype -> [crime_count_nearby]
    poi_types = dfp["poi_subcategory"].fillna("")
    for pt, subtype in zip(np.radians(dfp[["lat","lon"]].values), poi_types):
        if not subtype:
            continue
        idx = tree.query_radius([pt], r=radius)[0]
        counts[subtype].append(len(idx))

    if not counts:
        with open(POI_RISK_JSON, "w") as f:
            json.dump({}, f, indent=2)
        return {}

    avg = {k: float(np.mean(v)) for k, v in counts.items()}
    vmin, vmax = min(avg.values()), max(avg.values())
    if vmax == vmin:
        norm = {k: 1.5 for k in avg}
    else:
        norm = {k: round(3.0 * (v - vmin) / (vmax - vmin), 2) for k, v in avg.items()}

    with open(POI_RISK_JSON, "w") as f:
        json.dump(norm, f, indent=2)
    print(f"✅ Risk skoru yazıldı: {POI_RISK_JSON} (kategori sayısı: {len(norm)})")
    return norm

# === 3) SUÇ VERİSİNE POI ÖZELLİKLERİ EKLE (GEOID + 300m) ===
def enrich_crime_with_poi(df_crime, df_poi, poi_risk_scores):
    print("🔗 Suç verisi POI ile zenginleştiriliyor...")

    # --- 3A: GEOID bazlı özetler ---
    df_poi = df_poi.copy()
    df_poi["risk_score"] = df_poi["poi_subcategory"].map(poi_risk_scores).fillna(0.0)

    geoid_sum = df_poi.groupby("GEOID").agg(
        poi_total_count_geoid=("poi_subcategory","count"),
        poi_dominant_type_geoid=("poi_subcategory", lambda x: x.mode()[0] if not x.mode().empty else "No_POI"),
        poi_risk_score_geoid=("risk_score","sum")
    ).reset_index()

    geoid_sum["poi_total_count_geoid_range"] = _q_labels(geoid_sum["poi_total_count_geoid"])
    geoid_sum["poi_risk_score_geoid_range"] = _q_labels(geoid_sum["poi_risk_score_geoid"])

    # merge GEOID
    out = df_crime.copy()
    if "GEOID" in out.columns:
        out["GEOID"] = out["GEOID"].astype(str).str.zfill(11)
        geoid_sum["GEOID"] = geoid_sum["GEOID"].astype(str).str.zfill(11)
        out = out.merge(geoid_sum, on="GEOID", how="left")
    else:
        # GEOID yoksa boş sütunları ekle
        for col in ["poi_total_count_geoid","poi_dominant_type_geoid","poi_risk_score_geoid",
                    "poi_total_count_geoid_range","poi_risk_score_geoid_range"]:
            out[col] = np.nan

    # --- 3B: 300 m yarıçap nokta bazlı özellikler ---
    valid_idx = out.index[~out[["latitude","longitude"]].isna().any(axis=1)]
    if len(valid_idx) > 0 and not df_poi.empty:
        crime_rad = np.radians(out.loc[valid_idx, ["latitude","longitude"]].astype(float).values)
        poi_rad = np.radians(df_poi[["lat","lon"]].astype(float).values)
        tree = BallTree(poi_rad, metric="haversine")
        r = 300 / 6371000.0

        subtypes = df_poi["poi_subcategory"].fillna("").to_numpy()
        risks = df_poi["risk_score"].to_numpy()

        total = np.zeros(len(valid_idx), dtype=int)
        rsum  = np.zeros(len(valid_idx), dtype=float)
        dtyp  = []

        neigh = tree.query_radius(crime_rad, r=r)
        for i, idxs in enumerate(neigh):
            if len(idxs) == 0:
                total[i] = 0
                rsum[i] = 0.0
                dtyp.append("No_POI")
            else:
                total[i] = len(idxs)
                rsum[i]  = float(risks[idxs].sum())
                # baskın alt kategori
                st = subtypes[idxs]
                st = st[st != ""]
                dtyp.append(Counter(st).most_common(1)[0][0] if len(st) else "No_POI")

        out["poi_total_count_300m"] = np.nan
        out["poi_risk_score_300m"] = np.nan
        out["poi_dominant_type_300m"] = np.nan

        out.loc[valid_idx, "poi_total_count_300m"]   = total
        out.loc[valid_idx, "poi_risk_score_300m"]    = rsum
        out.loc[valid_idx, "poi_dominant_type_300m"] = dtyp

        # aralık etiketleri sadece mevcut değerler üzerinden
        out["poi_total_count_300m_range"] = _q_labels(out["poi_total_count_300m"])
        out["poi_risk_score_300m_range"]  = _q_labels(out["poi_risk_score_300m"])
    else:
        for col in ["poi_total_count_300m","poi_risk_score_300m","poi_dominant_type_300m",
                    "poi_total_count_300m_range","poi_risk_score_300m_range"]:
            out[col] = np.nan

    # === Kaydet & özet log ===
    out.to_csv(CRIME_OUTPUT, index=False)
    print(f"✅ Zenginleştirilmiş dosya yazıldı: {CRIME_OUTPUT}")
    new_cols = [c for c in out.columns if c.startswith("poi_")]
    print(f"➕ Eklenen sütunlar: {new_cols}")
    print("\n📊 İlk 5 satır:")
    print(out.head())
    return out

# === ANA AKIŞ ===
if __name__ == "__main__":
    print("🚀 Akış başladı...")
    df_crime = pd.read_csv(CRIME_INPUT, dtype={"GEOID": str})
    df_poi   = clean_and_assign_geoid_to_pois()
    poi_risk = calculate_dynamic_risk(df_crime, df_poi)
    enrich_crime_with_poi(df_crime, df_poi, poi_risk)
