# POI HESAPLAMA
import os, ast, json
import numpy as np
import pandas as pd
import geopandas as gpd
from collections import defaultdict, Counter
from sklearn.neighbors import BallTree

# ---- Dosya yolları ----
BASE_DIR = "crime_data"
POI_GEOJSON   = os.path.join(BASE_DIR, "sf_pois.geojson")
BLOCK_PATH    = os.path.join(BASE_DIR, "sf_census_blocks_with_population.geojson")
POI_CLEANED   = os.path.join(BASE_DIR, "sf_pois_cleaned_with_geoid.csv")
POI_RISK_JSON = os.path.join(BASE_DIR, "risky_pois_dynamic.json")
CRIME_INPUT   = os.path.join(BASE_DIR, "sf_crime_05.csv")
CRIME_OUTPUT  = os.path.join(BASE_DIR, "sf_crime_06.csv")

# YARDIMCILAR
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
        return pd.Series([labels[-1]]*len(s), index=s.index)

# %% ============== 1) POI temizle + GEOID ekle ==============
def clean_and_assign_geoid_to_pois():
    print("📍 POI verisi okunuyor ve GEOID atanıyor...")
    gdf = gpd.read_file(POI_GEOJSON)
    gdf = _ensure_crs(gdf, "EPSG:4326")

    # tags → dict
    if "tags" not in gdf.columns:
        gdf["tags"] = [{}] * len(gdf)
    gdf["tags"] = gdf["tags"].apply(_parse_tags)

    # kategori/alt-kategori/isim
    cat_sub_name = gdf["tags"].apply(_extract_cat_sub_name_from_tags)
    gdf[["poi_category","poi_subcategory","poi_name"]] = pd.DataFrame(cat_sub_name.tolist(), index=gdf.index)

    # geometri & lat/lon
    if "geometry" not in gdf.columns:
        if {"lon","lat"}.issubset(gdf.columns):
            gdf["geometry"] = gpd.points_from_xy(gdf["lon"], gdf["lat"])
        else:
            raise ValueError("GeoJSON'da geometry yok ve lon/lat bulunamadı.")
    gdf = _ensure_crs(gdf, "EPSG:4326")
    gdf["lon"] = gdf.geometry.x
    gdf["lat"] = gdf.geometry.y

    # bloklar & GEOID (11 hane)
    blocks = gpd.read_file(BLOCK_PATH)
    blocks = _ensure_crs(blocks, "EPSG:4326")
    if "GEOID" not in blocks.columns:
        raise ValueError("Block dosyasında GEOID kolonu yok.")
    blocks["GEOID"] = blocks["GEOID"].astype(str).str.extract(r"(\d+)")[0].str.zfill(11)

    # point-in-polygon
    joined = gpd.sjoin(gdf, blocks[["GEOID","geometry"]], how="left", predicate="within")

    keep = [c for c in ["id","lat","lon","poi_category","poi_subcategory","poi_name","GEOID"] if c in joined.columns]
    df = joined[keep].copy()
    if "id" not in df.columns:
        df["id"] = np.arange(len(df))
    df = df.dropna(subset=["lat","lon"])

    df.to_csv(POI_CLEANED, index=False)
    print(f"✅ POI temizlendi + GEOID eklendi → {POI_CLEANED} | satır: {len(df)}")
    print(df.head())
    return df

# %% ============== 2) Dinamik risk skoru (0–3) ==============
def calculate_dynamic_risk(df_crime, df_poi):
    print("📊 Dinamik risk (300 m) hesaplanıyor...")
    dfc = df_crime.dropna(subset=["latitude","longitude"]).copy()
    dfc["latitude"]  = pd.to_numeric(dfc["latitude"], errors="coerce")
    dfc["longitude"] = pd.to_numeric(dfc["longitude"], errors="coerce")
    dfc = dfc.dropna(subset=["latitude","longitude"])

    dfp = df_poi.dropna(subset=["lat","lon"]).copy()
    dfp["lat"] = pd.to_numeric(dfp["lat"], errors="coerce")
    dfp["lon"] = pd.to_numeric(dfp["lon"], errors="coerce")
    dfp = dfp.dropna(subset=["lat","lon"])

    # katkı vermesin istediklerin
    if "poi_subcategory" in dfp.columns:
        dfp = dfp[~dfp["poi_subcategory"].isin(["police","ranger_station"])]

    if dfc.empty or dfp.empty:
        print("⚠️ Risk için yetersiz veri, boş sözlük döndü.")
        with open(POI_RISK_JSON, "w") as f: json.dump({}, f, indent=2)
        return {}

    tree   = BallTree(np.radians(dfc[["latitude","longitude"]].values), metric="haversine")
    radius = 300 / 6371000.0

    counts = defaultdict(list)
    for pt, subtype in zip(np.radians(dfp[["lat","lon"]].values), dfp["poi_subcategory"].fillna("")):
        if not subtype: continue
        idx = tree.query_radius([pt], r=radius)[0]
        counts[subtype].append(len(idx))

    if not counts:
        with open(POI_RISK_JSON, "w") as f: json.dump({}, f, indent=2)
        return {}

    avg  = {k: float(np.mean(v)) for k, v in counts.items()}
    vmin, vmax = min(avg.values()), max(avg.values())
    norm = {k: (1.5 if vmax==vmin else round(3.0*(v-vmin)/(vmax-vmin), 2)) for k, v in avg.items()}

    with open(POI_RISK_JSON, "w") as f: json.dump(norm, f, indent=2)
    print(f"✅ Risk skorları yazıldı: {POI_RISK_JSON} | kategori: {len(norm)}")
    return norm

# %% ============== 3) Suç verisini zenginleştir ==============
def enrich_crime_with_poi(df_crime, df_poi, risky):
    print("🔗 Suç verisine GEOID ve 300 m POI özellikleri ekleniyor...")

    out = df_crime.copy()
    # GEOID 11 hane
    if "GEOID" in out.columns:
        out["GEOID"] = out["GEOID"].astype(str).str.extract(r"(\d+)")[0].str.zfill(11)

    # --- POI'ye risk ata
    dfp = df_poi.copy()
    dfp["risk_score"] = dfp["poi_subcategory"].map(risky).fillna(0.0)

    # --- 3A) GEOID bazlı özet ---
    if "GEOID" in out.columns:
        geoid_sum = (
            dfp.groupby("GEOID")
               .agg(poi_total_count_geoid=("poi_subcategory","count"),
                    poi_risk_score_geoid=("risk_score","sum"),
                    poi_dominant_type_geoid=("poi_subcategory",
                        lambda x: (x.mode()[0] if not x.mode().empty else "No_POI")))
               .reset_index()
        )
        geoid_sum["poi_total_count_geoid_range"] = _q_labels(geoid_sum["poi_total_count_geoid"])
        geoid_sum["poi_risk_score_geoid_range"]  = _q_labels(geoid_sum["poi_risk_score_geoid"])

        geoid_sum["GEOID"] = geoid_sum["GEOID"].astype(str).str.zfill(11)
        out = out.merge(geoid_sum, on="GEOID", how="left")
    else:
        for c in ["poi_total_count_geoid","poi_risk_score_geoid","poi_dominant_type_geoid",
                  "poi_total_count_geoid_range","poi_risk_score_geoid_range"]:
            out[c] = np.nan

    # --- 3B) 300 m yarıçap nokta-bazlı ---
    valid = out.index[~out[["latitude","longitude"]].isna().any(axis=1)]
    if len(valid) > 0 and not dfp.empty:
        tree   = BallTree(np.radians(dfp[["lat","lon"]].astype(float).values), metric="haversine")
        radius = 300 / 6371000.0

        poi_types = dfp["poi_subcategory"].fillna("").to_numpy()
        risks     = dfp["risk_score"].to_numpy()

        crime_rad = np.radians(out.loc[valid, ["latitude","longitude"]].astype(float).values)
        neigh = tree.query_radius(crime_rad, r=radius)

        total = np.zeros(len(valid), dtype=int)
        rsum  = np.zeros(len(valid), dtype=float)
        dtyp  = []

        for i, idxs in enumerate(neigh):
            if len(idxs)==0:
                dtyp.append("No_POI")
            else:
                total[i] = len(idxs)
                rsum[i]  = float(risks[idxs].sum())
                st = poi_types[idxs]
                st = st[st!=""]
                dtyp.append(Counter(st).most_common(1)[0][0] if len(st) else "No_POI")

        out["poi_total_count_300m"]   = np.nan
        out["poi_risk_score_300m"]    = np.nan
        out["poi_dominant_type_300m"] = np.nan
        out.loc[valid, "poi_total_count_300m"]   = total
        out.loc[valid, "poi_risk_score_300m"]    = rsum
        out.loc[valid, "poi_dominant_type_300m"] = dtyp

        out["poi_total_count_300m_range"] = _q_labels(out["poi_total_count_300m"])
        out["poi_risk_score_300m_range"]  = _q_labels(out["poi_risk_score_300m"])
    else:
        for c in ["poi_total_count_300m","poi_risk_score_300m","poi_dominant_type_300m",
                  "poi_total_count_300m_range","poi_risk_score_300m_range"]:
            out[c] = np.nan

    # --- Kaydet & özet ---
    out.to_csv(CRIME_OUTPUT, index=False)
    print(f"✅ sf_crime_06 yazıldı → {CRIME_OUTPUT}")
    added = [c for c in out.columns if c.startswith("poi_")]
    print("➕ Eklenen sütunlar:", added)
    print("📄 İlk 5 satır:")
    print(out.head())
    return out

# %% ============== MAIN ==============
if __name__ == "__main__":
    print("🚀 Akış başladı")
    # 0) crime yükle
    df_crime = pd.read_csv(CRIME_INPUT)

    # 1) POI’yi güncelle & GEOID ata
    df_poi = clean_and_assign_geoid_to_pois()

    # 2) Dinamik risk skoru hesapla
    risky = calculate_dynamic_risk(df_crime, df_poi)

    # 3) Suç verisini zenginleştir ve kaydet
    enrich_crime_with_poi(df_crime, df_poi, risky)
