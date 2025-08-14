import os
import ast
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from collections import defaultdict
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
    """tags string/dict gelir; güvenle dict'e çevir."""
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        for loader in (json.loads, ast.literal_eval):
            try:
                out = loader(val)
                return out if isinstance(out, dict) else {}
            except Exception:
                pass
    return {}

def _ensure_crs(gdf, target="EPSG:4326"):
    """CRS yoksa/uyuşmuyorsa 4326'ya sabitle."""
    if gdf.crs is None:
        gdf = gdf.set_crs(target, allow_override=True)
    elif gdf.crs.to_string().upper().endswith("CRS84"):
        # CRS84 → EPSG:4326 eşdeğer (lon,lat)
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    elif gdf.crs.to_string() != target:
        gdf = gdf.to_crs(target)
    return gdf

def _extract_cat_sub_name_from_tags(tags: dict):
    """Öncelik: amenity → shop → leisure; isim varsa al."""
    name = tags.get("name")
    for key in ("amenity", "shop", "leisure"):
        if key in tags and tags[key]:
            return key, tags[key], name
    return None, None, name


# === 1. POI VERİSİ TEMİZLEME VE GEOID EKLEME ===
def clean_and_assign_geoid_to_pois():
    print("📍 POI verisi okunuyor ve GEOID atanıyor...")
    gdf = gpd.read_file(POI_GEOJSON)
    gdf = _ensure_crs(gdf, "EPSG:4326")

    # tags her zaman dict olsun
    if "tags" not in gdf.columns:
        gdf["tags"] = [{}] * len(gdf)
    gdf["tags"] = gdf["tags"].apply(_parse_tags)

    # kategori/alt-kategori/isim
    cat_sub_name = gdf["tags"].apply(_extract_cat_sub_name_from_tags)
    gdf[["poi_category", "poi_subcategory", "poi_name"]] = pd.DataFrame(
        cat_sub_name.tolist(), index=gdf.index
    )

    # geometry → lat/lon üret
    if "geometry" not in gdf.columns:
        # Bazı export'larda properties.lat/lon var; son çare oradan nokta üret
        if {"lon", "lat"}.issubset(set(gdf.columns)):
            gdf["geometry"] = gpd.points_from_xy(gdf["lon"], gdf["lat"])
        else:
            raise ValueError("GeoJSON içinde 'geometry' yok ve 'lat/lon' da bulunamadı.")
    gdf = _ensure_crs(gdf, "EPSG:4326")
    gdf["lon"] = gdf.geometry.x if "lon" not in gdf.columns else gdf["lon"].fillna(gdf.geometry.x)
    gdf["lat"] = gdf.geometry.y if "lat" not in gdf.columns else gdf["lat"].fillna(gdf.geometry.y)

    # Nüfus blokları (polygon) yükle ve GEOID formatını sabitle
    gdf_blocks = gpd.read_file(BLOCK_PATH)
    gdf_blocks = _ensure_crs(gdf_blocks, "EPSG:4326")
    if "GEOID" not in gdf_blocks.columns:
        raise ValueError("Block dosyasında 'GEOID' kolonu yok.")
    gdf_blocks["GEOID"] = gdf_blocks["GEOID"].astype(str).str.zfill(11)

    # Spatial join (Point within Polygon)
    gdf_joined = gpd.sjoin(
        gdf,
        gdf_blocks[["GEOID", "geometry"]],
        how="left",
        predicate="within"
    )

    # Çıktı kolonlarını güvenle seç
    keep_cols = [c for c in ["id", "lat", "lon", "poi_category", "poi_subcategory", "poi_name", "GEOID"] if c in gdf_joined.columns]
    df_cleaned = gdf_joined[keep_cols].copy()
    
    if "id" not in df_cleaned.columns:
        df_cleaned["id"] = np.arange(len(df_cleaned))

    # Koordinatı olmayanları at
    df_cleaned = df_cleaned.dropna(subset=["lat", "lon"])

    df_cleaned.to_csv(POI_CLEANED_CSV, index=False)
    print(f"✅ Temizlenmiş POI verisi kaydedildi: {POI_CLEANED_CSV}")
    return df_cleaned


# === 2. DİNAMİK RİSK SKORU HESAPLAMA ===
def calculate_dynamic_risk(df_crime, df_poi):
    print("📊 POI risk skorları hesaplanıyor (300m yarıçap)...")

    # Suç verisi: geçerli koordinatlar
    dfc = df_crime.dropna(subset=["latitude", "longitude"]).copy()
    dfc["latitude"] = pd.to_numeric(dfc["latitude"], errors="coerce")
    dfc["longitude"] = pd.to_numeric(dfc["longitude"], errors="coerce")
    dfc = dfc.dropna(subset=["latitude", "longitude"])
    if dfc.empty:
        print("⚠️ Suç verisinde geçerli koordinat yok; risk haritası boş dönecek.")
        with open(POI_RISK_JSON, "w") as f:
            json.dump({}, f, indent=2)
        return {}

    # POI verisi: geçerli koordinatlar
    dfp = df_poi.dropna(subset=["lat", "lon"]).copy()
    dfp["lat"] = pd.to_numeric(dfp["lat"], errors="coerce")
    dfp["lon"] = pd.to_numeric(dfp["lon"], errors="coerce")
    dfp = dfp.dropna(subset=["lat", "lon"])

    # İsteğe bağlı: polis vb. hariç
    if "poi_subcategory" in dfp.columns:
        dfp = dfp[~dfp["poi_subcategory"].isin(["police", "ranger_station"])]

    if dfp.empty:
        print("⚠️ POI verisi boş; risk haritası boş dönecek.")
        with open(POI_RISK_JSON, "w") as f:
            json.dump({}, f, indent=2)
        return {}

    # Radyan ve BallTree
    crime_rad = np.radians(dfc[["latitude", "longitude"]].values)
    poi_rad = np.radians(dfp[["lat", "lon"]].values)
    tree = BallTree(crime_rad, metric="haversine")
    radius = 300 / 6371000.0  # 300m

    poi_types = dfp["poi_subcategory"].fillna("")
    poi_crime_counts = []
    for pt, subtype in zip(poi_rad, poi_types):
        if not subtype:
            continue
        idx = tree.query_radius([pt], r=radius)[0]
        poi_crime_counts.append((subtype, len(idx)))

    if not poi_crime_counts:
        print("⚠️ Hiç POI alt kategorisi etrafında suç bulunamadı.")
        with open(POI_RISK_JSON, "w") as f:
            json.dump({}, f, indent=2)
        return {}

    # Alt kategori bazında ortalama suç sayısı
    poi_risk_raw = defaultdict(list)
    for subtype, count in poi_crime_counts:
        poi_risk_raw[subtype].append(count)
    poi_risk_avg = {k: float(np.mean(v)) for k, v in poi_risk_raw.items()}

    # Normalize (0–3)
    vals = list(poi_risk_avg.values())
    vmin, vmax = min(vals), max(vals)
    if vmax - vmin < 1e-9:
        poi_risk_normalized = {k: 1.5 for k in poi_risk_avg.keys()}
    else:
        poi_risk_normalized = {k: round(3.0 * (v - vmin) / (vmax - vmin), 2) for k, v in poi_risk_avg.items()}

    with open(POI_RISK_JSON, "w") as f:
        json.dump(poi_risk_normalized, f, indent=2)

    print("📈 POI risk skorları (0–3 normalize) örnek:")
    for k, score in sorted(poi_risk_normalized.items(), key=lambda x: -x[1])[:20]:
        print(f"{k:<25} → {score:.2f}")
    return poi_risk_normalized


# === 3. DİNAMİK KATEGORİLEME (Q1–Q4 vb.) ===
def make_dynamic_range_func(series, strategy="auto", max_bins=5):
    vals = pd.to_numeric(series, errors="coerce").dropna().values
    if vals.size == 0:
        def label(_): return "Q1 (0-0)"
        return label

    n = len(vals)
    std = np.std(vals)
    iqr = np.percentile(vals, 75) - np.percentile(vals, 25)

    if strategy == "auto":
        if n < 500:
            bin_count = 3
        elif std < 1 or iqr < 1:
            bin_count = 4
        elif std > 20:
            bin_count = min(10, max_bins)
        else:
            bin_count = 5
    else:
        bin_count = int(strategy)

    qs = np.quantile(vals, [i / bin_count for i in range(bin_count + 1)])

    def label(x):
        if pd.isna(x):
            return f"Q1 ({qs[0]:.1f}-{qs[1]:.1f})"
        for i in range(bin_count):
            if x <= qs[i + 1]:
                return f"Q{i+1} ({qs[i]:.1f}-{qs[i+1]:.1f})"
        return f"Q{bin_count} ({qs[-2]:.1f}-{qs[-1]:.1f})"

    return label


# === 4. SUÇ VERİSİNE POI ÖZELLİĞİ EKLE ===
def enrich_crime_with_poi(df_crime, df_poi, poi_risk_scores):
    print("🔗 POI özellikleri suç verisine ekleniyor...")

    # Risk skoru eşle
    if "poi_subcategory" in df_poi.columns:
        df_poi["risk_score"] = df_poi["poi_subcategory"].map(poi_risk_scores).fillna(0.0)
    else:
        df_poi["risk_score"] = 0.0

    # Crime koordinatları
    dfc = df_crime.dropna(subset=["latitude", "longitude"]).copy()
    dfc["latitude"] = pd.to_numeric(dfc["latitude"], errors="coerce")
    dfc["longitude"] = pd.to_numeric(dfc["longitude"], errors="coerce")
    dfc = dfc.dropna(subset=["latitude", "longitude"])

    # POI koordinatları
    dfp = df_poi.dropna(subset=["lat", "lon"]).copy()
    dfp["lat"] = pd.to_numeric(dfp["lat"], errors="coerce")
    dfp["lon"] = pd.to_numeric(dfp["lon"], errors="coerce")
    dfp = dfp.dropna(subset=["lat", "lon"])

    if dfc.empty:
        print("⚠️ Suç verisi boş (geçerli koordinat yok). Varsayılan 0 değerler yazılacak.")
        out = df_crime.copy()
        out["poi_total_count"] = 0
        out["poi_risk_score"] = 0.0
        out["poi_dominant_type"] = "No_POI"
        out["poi_total_count_range"] = "Q1 (0-0)"
        out["poi_risk_score_range"] = "Q1 (0-0)"
        out.to_csv(CRIME_OUTPUT, index=False)
        return out

    if dfp.empty:
        print("⚠️ POI verisi boş. Varsayılan 0 değerler yazılacak.")
        out = dfc.copy()
        out["poi_total_count"] = 0
        out["poi_risk_score"] = 0.0
        out["poi_dominant_type"] = "No_POI"
        out["poi_total_count_range"] = "Q1 (0-0)"
        out["poi_risk_score_range"] = "Q1 (0-0)"
        out.to_csv(CRIME_OUTPUT, index=False)
        return out

    # KD-Tree
    crime_rad = np.radians(dfc[["latitude", "longitude"]].values)
    poi_rad = np.radians(dfp[["lat", "lon"]].values)
    tree = BallTree(poi_rad, metric="haversine")
    radius = 300 / 6371000.0  # 300m

    indices = tree.query_radius(crime_rad, r=radius)
    types = dfp["poi_subcategory"].fillna("")
    dfp_risk = dfp["risk_score"]

    total_count, risk_sum, dom_type = [], [], []
    for idx_list in indices:
        if len(idx_list) == 0:
            total_count.append(0)
            risk_sum.append(0.0)
            dom_type.append("No_POI")
            continue
        subs = types.iloc[idx_list]
        risks = dfp_risk.iloc[idx_list]
        total_count.append(len(idx_list))
        risk_sum.append(float(risks.sum()))
        dom_type.append(subs.value_counts().idxmax() if not subs.empty else "No_POI")

    dfc["poi_total_count"] = total_count
    dfc["poi_risk_score"] = risk_sum
    dfc["poi_dominant_type"] = dom_type

    # Aralık etiketleri
    count_labeller = make_dynamic_range_func(dfc["poi_total_count"])
    risk_labeller = make_dynamic_range_func(dfc["poi_risk_score"])
    dfc["poi_total_count_range"] = dfc["poi_total_count"].apply(count_labeller)
    dfc["poi_risk_score_range"] = dfc["poi_risk_score"].apply(risk_labeller)

    dfc.to_csv(CRIME_OUTPUT, index=False)
    print(f"✅ Enriched crime verisi kaydedildi: {CRIME_OUTPUT}")
    return dfc


# === ANA FONKSİYON AKIŞI ===
if __name__ == "__main__":
    print("🚀 POI güncelleme işlemi başlıyor...")

    # 0) Suç verisi
    df_crime = pd.read_csv(CRIME_INPUT)

    # 1) POI temizle + GEOID ata
    df_poi = clean_and_assign_geoid_to_pois()

    # 2) POI risk skoru hesapla
    poi_risk_scores = calculate_dynamic_risk(df_crime, df_poi)

    # 3) Suç verisini POI ile zenginleştir
    enrich_crime_with_poi(df_crime, df_poi, poi_risk_scores)
