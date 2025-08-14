# pipeline_make_sf_crime_06.py
import os, ast, json, numpy as np, pandas as pd, geopandas as gpd
from collections import defaultdict
from sklearn.neighbors import BallTree

# ================== 0) YOLLAR ==================
BASE_DIR = "crime_data"
POI_GEOJSON   = os.path.join(BASE_DIR, "sf_pois.geojson")
BLOCK_PATH    = os.path.join(BASE_DIR, "sf_census_blocks_with_population.geojson")
POI_CLEAN_CSV = os.path.join(BASE_DIR, "sf_pois_cleaned_with_geoid.csv")
POI_RISK_JSON = os.path.join(BASE_DIR, "risky_pois_dynamic.json")
CRIME_IN      = os.path.join(BASE_DIR, "sf_crime_05.csv")
CRIME_OUT     = os.path.join(BASE_DIR, "sf_crime_06.csv")

# ================== YARDIMCI ==================
def _ensure_crs(gdf, target="EPSG:4326"):
    if gdf.crs is None:
        return gdf.set_crs(target, allow_override=True)
    s = gdf.crs.to_string().upper()
    if s.endswith("CRS84"):  # CRS84 == 4326 (lon,lat)
        return gdf.set_crs("EPSG:4326", allow_override=True)
    if s != target:
        return gdf.to_crs(target)
    return gdf

def _parse_tags(val):
    if isinstance(val, dict): return val
    if isinstance(val, str):
        for loader in (json.loads, ast.literal_eval):
            try:
                x = loader(val)
                return x if isinstance(x, dict) else {}
            except Exception:
                pass
    return {}

def _extract_cat_sub_name(tags: dict):
    name = tags.get("name")
    for key in ("amenity", "shop", "leisure"):
        if key in tags and tags[key]:
            return key, tags[key], name
    return None, None, name

def _normalize_geoid(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.extract(r"(\d+)")[0]
    s = s.str.zfill(11)              # SF verinde 11 hane (örn. 60750117002)
    return s

def _make_dynamic_labels(series: pd.Series, strategy="auto", max_bins=5):
    vals = pd.to_numeric(series, errors="coerce").dropna().values
    if vals.size == 0:
        def lab(_): return "Q1 (0-0)"
        return lab
    n   = len(vals)
    std = np.std(vals)
    iqr = np.percentile(vals, 75) - np.percentile(vals, 25)
    if strategy == "auto":
        if n < 500:              bin_count = 3
        elif std < 1 or iqr < 1: bin_count = 4
        elif std > 20:           bin_count = min(10, max_bins)
        else:                    bin_count = 5
    else:
        bin_count = int(strategy)
    qs = np.quantile(vals, [i/bin_count for i in range(bin_count+1)])
    def lab(x):
        if pd.isna(x): return f"Q1 ({qs[0]:.1f}-{qs[1]:.1f})"
        for i in range(bin_count):
            if x <= qs[i+1]:
                return f"Q{i+1} ({qs[i]:.1f}-{qs[i+1]:.1f})"
        return f"Q{bin_count} ({qs[-2]:.1f}-{qs[-1]:.1f})"
    return lab

# ================== 1) POI'yi güncelle + GEOID ata ==================
def build_poi_clean_with_geoid() -> pd.DataFrame:
    print("📍 POI okunuyor → kategoriler çıkarılıyor → GEOID atanıyor...")
    gdf = gpd.read_file(POI_GEOJSON)
    gdf = _ensure_crs(gdf, "EPSG:4326")

    if "tags" not in gdf.columns:
        gdf["tags"] = [{}]*len(gdf)
    gdf["tags"] = gdf["tags"].apply(_parse_tags)

    triples = gdf["tags"].apply(_extract_cat_sub_name)
    gdf[["poi_category","poi_subcategory","poi_name"]] = pd.DataFrame(triples.tolist(), index=gdf.index)

    # geometry → lat/lon güvence
    if "geometry" not in gdf.columns:
        if {"lon","lat"}.issubset(gdf.columns):
            gdf["geometry"] = gpd.points_from_xy(gdf["lon"], gdf["lat"])
        else:
            raise ValueError("GeoJSON 'geometry' veya 'lon/lat' içermiyor.")
    gdf = _ensure_crs(gdf, "EPSG:4326")
    gdf["lon"] = gdf.get("lon", pd.Series(index=gdf.index, dtype=float)).fillna(gdf.geometry.x)
    gdf["lat"] = gdf.get("lat", pd.Series(index=gdf.index, dtype=float)).fillna(gdf.geometry.y)

    # Nüfus bloklarına spatial join (within)
    blocks = gpd.read_file(BLOCK_PATH)
    blocks = _ensure_crs(blocks, "EPSG:4326")
    if "GEOID" not in blocks.columns:
        raise ValueError("Block dosyasında 'GEOID' yok.")
    blocks["GEOID"] = _normalize_geoid(blocks["GEOID"])

    joined = gpd.sjoin(gdf, blocks[["GEOID","geometry"]], how="left", predicate="within")

    keep = [c for c in ["id","lat","lon","poi_category","poi_subcategory","poi_name","GEOID"] if c in joined.columns]
    df = joined[keep].copy()
    if "id" not in df.columns:
        df["id"] = np.arange(len(df))
    df = df.dropna(subset=["lat","lon"])

    df["GEOID"] = _normalize_geoid(df["GEOID"])
    df.to_csv(POI_CLEAN_CSV, index=False)
    print(f"✅ Kaydedildi: {POI_CLEAN_CSV}  |  Satır: {len(df):,}")
    print(df.head(5).to_string(index=False))
    return df

# ================== 2) Dinamik risk skoru (0–3) ==================
def compute_dynamic_poi_risk(df_crime: pd.DataFrame, df_poi: pd.DataFrame, radius_m=300) -> dict:
    print("📊 Dinamik POI risk (ortalama çevre suç sayısı → 0–3 normalize)...")
    # temiz koordinatlar
    dfc = df_crime.dropna(subset=["latitude","longitude"]).copy()
    dfc["latitude"]  = pd.to_numeric(dfc["latitude"], errors="coerce")
    dfc["longitude"] = pd.to_numeric(dfc["longitude"], errors="coerce")
    dfc = dfc.dropna(subset=["latitude","longitude"])

    dfp = df_poi.dropna(subset=["lat","lon"]).copy()
    dfp["lat"] = pd.to_numeric(dfp["lat"], errors="coerce")
    dfp["lon"] = pd.to_numeric(dfp["lon"], errors="coerce")
    # polis vb. çıkar
    if "poi_subcategory" in dfp.columns:
        dfp = dfp[~dfp["poi_subcategory"].isin(["police","ranger_station"])]

    if dfc.empty or dfp.empty:
        print("⚠️ Risk için yeterli nokta yok.")
        with open(POI_RISK_JSON,"w") as f: json.dump({}, f, indent=2)
        return {}

    crime_rad = np.radians(dfc[["latitude","longitude"]].values)
    poi_rad   = np.radians(dfp[["lat","lon"]].values)
    tree = BallTree(crime_rad, metric="haversine")
    r = radius_m/6371000.0

    poi_types = dfp["poi_subcategory"].fillna("")
    counts = []
    for pt, t in zip(poi_rad, poi_types):
        if not t: continue
        idx = tree.query_radius([pt], r=r)[0]
        counts.append((t, len(idx)))

    if not counts:
        with open(POI_RISK_JSON,"w") as f: json.dump({}, f, indent=2)
        return {}

    agg = defaultdict(list)
    for t, c in counts:
        agg[t].append(c)
    avg = {t: float(np.mean(v)) for t, v in agg.items()}

    v = list(avg.values())
    vmin, vmax = min(v), max(v)
    if vmax - vmin < 1e-9:
        norm = {t: 1.5 for t in avg}
    else:
        norm = {t: round(3*(x - vmin)/(vmax - vmin), 2) for t, x in avg.items()}

    with open(POI_RISK_JSON, "w") as f:
        json.dump(norm, f, indent=2)

    print("🔝 İlk 15 alt-kategori (skora göre):")
    for k, s in sorted(norm.items(), key=lambda x: -x[1])[:15]:
        print(f"  {k:<24} → {s:.2f}")
    return norm

# ================== 3) Suçu POI ile zenginleştir (300m) ==================
def enrich_crime_with_poi(df_crime: pd.DataFrame, df_poi: pd.DataFrame, poi_risk: dict, radius_m=300) -> pd.DataFrame:
    print("🔗 Suç satırlarına POI metrikleri ekleniyor (300m yarıçap)...")
    dfc = df_crime.copy()
    dfc["latitude"]  = pd.to_numeric(dfc["latitude"], errors="coerce")
    dfc["longitude"] = pd.to_numeric(dfc["longitude"], errors="coerce")
    dfc = dfc.dropna(subset=["latitude","longitude"])

    dfp = df_poi.dropna(subset=["lat","lon"]).copy()
    dfp["lat"] = pd.to_numeric(dfp["lat"], errors="coerce")
    dfp["lon"] = pd.to_numeric(dfp["lon"], errors="coerce")
    dfp = dfp.dropna(subset=["lat","lon"])
    dfp["risk_score"] = dfp.get("poi_subcategory","").map(poi_risk).fillna(0.0)

    if dfc.empty or dfp.empty:
        print("⚠️ Eksik koordinatlar nedeniyle varsayılan 0 değerleri yazılacak.")
        out = df_crime.copy()
        out["poi_total_count"]     = 0
        out["poi_risk_score"]      = 0.0
        out["poi_dominant_type"]   = "No_POI"
        out["poi_total_count_range"] = "Q1 (0-0)"
        out["poi_risk_score_range"]  = "Q1 (0-0)"
        out.to_csv(CRIME_OUT, index=False)
        print(f"💾 {CRIME_OUT}")
        return out

    crime_rad = np.radians(dfc[["latitude","longitude"]].values)
    poi_rad   = np.radians(dfp[["lat","lon"]].values)
    tree = BallTree(poi_rad, metric="haversine")
    r = radius_m/6371000.0

    idxs = tree.query_radius(crime_rad, r=r)
    poi_types = dfp["poi_subcategory"].fillna("")
    poi_risks = dfp["risk_score"].fillna(0.0)

    tot, risk, dom = [], [], []
    for ids in idxs:
        if len(ids) == 0:
            tot.append(0); risk.append(0.0); dom.append("No_POI"); continue
        subs  = poi_types.iloc[ids]
        risks = poi_risks.iloc[ids]
        tot.append(len(ids))
        risk.append(float(risks.sum()))
        dom.append(subs.value_counts().idxmax() if not subs.empty else "No_POI")

    dfc["poi_total_count"]   = tot
    dfc["poi_risk_score"]    = risk
    dfc["poi_dominant_type"] = dom

    # Dinamik aralık etiketleri
    lab_cnt  = _make_dynamic_labels(dfc["poi_total_count"])
    lab_risk = _make_dynamic_labels(dfc["poi_risk_score"])
    dfc["poi_total_count_range"] = dfc["poi_total_count"].apply(lab_cnt)
    dfc["poi_risk_score_range"]  = dfc["poi_risk_score"].apply(lab_risk)

    # Orijinal sıralamayı korumak için index'e göre geri birleştir
    out = df_crime.copy()
    out = out.drop(columns=[c for c in ["poi_total_count","poi_risk_score","poi_dominant_type",
                                        "poi_total_count_range","poi_risk_score_range"] if c in out.columns])
    out = out.merge(
        dfc[["latitude","longitude","poi_total_count","poi_risk_score","poi_dominant_type",
             "poi_total_count_range","poi_risk_score_range"]],
        left_index=True, right_index=True, how="left"
    ).fillna({"poi_total_count":0, "poi_risk_score":0.0,
              "poi_dominant_type":"No_POI",
              "poi_total_count_range":"Q1 (0-0)", "poi_risk_score_range":"Q1 (0-0)"})

    out.to_csv(CRIME_OUT, index=False)
    print(f"✅ Yazıldı: {CRIME_OUT}  |  Satır: {len(out):,}")
    print(out.head(5)[["latitude","longitude","poi_total_count","poi_risk_score","poi_dominant_type"]].to_string(index=False))
    return out

# ================== MAIN ==================
if __name__ == "__main__":
    print("🚀 Başlıyor...")
    # 0) Suç verisi
    df_crime = pd.read_csv(CRIME_IN)
    # 1) POI’yi temizle + GEOID ekle
    df_poi = build_poi_clean_with_geoid()
    # 2) Dinamik risk sözlüğü (0–3)
    risk_dict = compute_dynamic_poi_risk(df_crime, df_poi, radius_m=300)
    # 3) Suçu POI ile zenginleştir
    _ = enrich_crime_with_poi(df_crime, df_poi, risk_dict, radius_m=300)
    print("🎉 Bitti.")

import pandas as pd
import streamlit as st
CRIME_OUTPUT = "crime_data/sf_crime_06.csv"

if os.path.exists(CRIME_OUTPUT):
    df_out = pd.read_csv(CRIME_OUTPUT)
    st.write(df_out.shape, "satır, sütun")
    st.dataframe(df_out[['poi_total_count','poi_risk_score','poi_dominant_type',
                         'poi_total_count_range','poi_risk_score_range']].head(10))
else:
    st.warning("sf_crime_06.csv henüz oluşmamış görünüyor. Pipeline'ın son adımı çalıştı mı?")

