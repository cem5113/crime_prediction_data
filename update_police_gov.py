# app_poi_to_06.py
import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import json, ast, io, os, sys, math, warnings, requests
from shapely.geometry import Point
from sklearn.neighbors import BallTree

warnings.filterwarnings("ignore")
st.set_page_config(page_title="POI → sf_crime_06", layout="wide")
st.title("🧭 POI Entegrasyonu ve Risk Skoru → sf_crime_06.csv")

# =========================
# Sidebar: Kaynak URL'leri
# =========================
st.sidebar.header("🔗 Kaynak Dosya URL’leri (GitHub)")
GITHUB_HINT = "Raw URL / Releases URL girin (CSV/GeoJSON)."
crime05_url = st.sidebar.text_input(
    "sf_crime_05.csv URL", 
    value="https://raw.githubusercontent.com/<kullanici>/<repo>/main/sf_crime_05.csv",
    help=GITHUB_HINT
)
pois_geojson_url = st.sidebar.text_input(
    "sf_pois.geojson URL",
    value="https://raw.githubusercontent.com/<kullanici>/<repo>/main/sf_pois.geojson",
    help=GITHUB_HINT
)
blocks_geojson_url = st.sidebar.text_input(
    "sf_census_blocks_with_population.geojson URL",
    value="https://raw.githubusercontent.com/<kullanici>/<repo>/main/sf_census_blocks_with_population.geojson",
    help=GITHUB_HINT
)

SAVE_DIR = st.sidebar.text_input("Çıktı klasörü", value="data")
os.makedirs(SAVE_DIR, exist_ok=True)
OUT_06 = os.path.join(SAVE_DIR, "sf_crime_06.csv")
TMP_POI_CLEAN = os.path.join(SAVE_DIR, "sf_pois_cleaned_with_geoid.csv")
TMP_POI_RISK_JSON = os.path.join(SAVE_DIR, "risky_pois_dynamic.json")
TMP_CRIME_POI = os.path.join(SAVE_DIR, "sf_poi_risk_crime.csv")
TMP_POI_WITH_RISK = os.path.join(SAVE_DIR, "sf_pois_with_risk_score.csv")

st.sidebar.caption("Kaynaklar indirilecek, işlenecek ve 06 çıktısı üretilecek.")

# =========================
# Yardımcı Fonksiyonlar
# =========================
def fetch(url: str, binary=False) -> bytes | str:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.content if binary else r.text

def read_csv_url(url: str, **kwargs) -> pd.DataFrame:
    text = fetch(url, binary=False)
    return pd.read_csv(io.StringIO(text), **kwargs)

def read_geojson_url(url: str) -> gpd.GeoDataFrame:
    # geopandas doğrudan URL okuyabiliyor ama bazı hostlarda CORS takılabiliyor
    content = fetch(url, binary=True)
    return gpd.read_file(io.BytesIO(content))

def ensure_geoid_str(s: pd.Series, fallback_len: int | None = None) -> pd.Series:
    # Rakamları çek + en uygun uzunluğa zfill uygula (11 veya 12; projede genelde 11)
    s = s.astype(str).str.extract(r"(\d+)")[0]
    s = s.fillna("")
    # Uzunluk kararı
    if fallback_len is None:
        # en çok görülen uzunluk üzerinden karar ver
        lens = s.str.len().value_counts()
        if not lens.empty and lens.index[0] in (11,12,15):
            z = lens.index[0]
        else:
            z = 11
    else:
        z = fallback_len
    return s.apply(lambda x: x.zfill(z) if x else x)

def extract_poi_fields(tags):
    # amenity/shop/leisure hiyerarşisine göre türleri çıkar
    if isinstance(tags, str):
        try:
            tags = ast.literal_eval(tags)
        except Exception:
            return pd.Series([None, None, None])
    if isinstance(tags, dict):
        for key in ["amenity", "shop", "leisure"]:
            if key in tags:
                return pd.Series([key, tags.get(key), tags.get("name")])
        # name varsa yine dönelim
        return pd.Series([None, None, tags.get("name")])
    return pd.Series([None, None, None])

def to_rad(df_latlon: np.ndarray) -> np.ndarray:
    return np.radians(df_latlon.astype(float))

def dynamic_bin_labels(values: pd.Series, strategy="auto", max_bins=10):
    vals = values.dropna().to_numpy()
    n = len(vals)
    if n == 0:
        edges = np.array([0, 1, 2, 3])
    else:
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
        elif strategy == "fd":
            bin_width = 2 * iqr / (n ** (1/3))
            if bin_width <= 0:
                bin_count = 3
            else:
                bin_count = min(int(math.ceil((vals.max() - vals.min())/bin_width)), max_bins)
        else:
            bin_count = int(strategy)
        qs = [i/bin_count for i in range(bin_count+1)]
        edges = np.quantile(vals, qs)
    def _label(x):
        if pd.isna(x):
            return "Unknown"
        for i in range(len(edges)-1):
            if x <= edges[i+1]:
                if i == 0:
                    return f"Q{i+1} (≤{edges[i+1]:.1f})"
                return f"Q{i+1} ({edges[i]:.1f}-{edges[i+1]:.1f})"
        return f"Q{len(edges)} (>{edges[-1]:.1f})"
    return _label

# =========================
# Ana İşlem
# =========================
def run_pipeline():
    st.write("### 1) Dosyaları GitHub’dan indir & oku")
    # 1.1 crime_05
    df_crime = read_csv_url(crime05_url)
    # GEOID normalize
    if "GEOID" in df_crime.columns:
        df_crime["GEOID"] = ensure_geoid_str(df_crime["GEOID"])
    # Koordinatlar
    # olası 'lon'/'lat' adları:
    if "longitude" not in df_crime.columns and "lon" in df_crime.columns:
        df_crime = df_crime.rename(columns={"lon":"longitude"})
    if "latitude" not in df_crime.columns and "lat" in df_crime.columns:
        df_crime = df_crime.rename(columns={"lat":"latitude"})
    if not {"longitude","latitude"}.issubset(df_crime.columns):
        st.error("❌ sf_crime_05.csv dosyasında 'longitude' ve 'latitude' bulunmalı.")
        return

    st.success(f"✅ crime_05 okundu: {df_crime.shape}")

    # 1.2 POI geojson
    gdf_pois = read_geojson_url(pois_geojson_url)
    st.success(f"✅ sf_pois.geojson okundu: {gdf_pois.shape}")

    # 1.3 Blocks geojson
    gdf_blocks = read_geojson_url(blocks_geojson_url)
    # GEOID sütun adı farklı olabilir (GEOID, geoid, GEOID10 vs.)
    geoid_col = None
    for cand in ["GEOID", "geoid", "GEOID10", "GEOID12", "GEOID11"]:
        if cand in gdf_blocks.columns:
            geoid_col = cand
            break
    if geoid_col is None:
        st.error("❌ Blocks dosyasında GEOID benzeri sütun bulunamadı.")
        return
    gdf_blocks["GEOID"] = ensure_geoid_str(gdf_blocks[geoid_col])
    st.success(f"✅ Blocks okundu: {gdf_blocks.shape} | GEOID sütunu: {geoid_col} → GEOID")

    # 2) POI alanlarını ayıkla
    st.write("### 2) POI alanlarını ayıkla (amenity/shop/leisure)")
    if "tags" not in gdf_pois.columns:
        # Bazı OSM export'larında tag'ler ayrı ayrı gelir: amenity, shop, leisure...
        # Bu durumda alanları doğrudan doldururuz.
        poi_cat = []
        poi_subcat = []
        poi_name = []
        for _, r in gdf_pois.iterrows():
            sub = None
            cat = None
            name = r.get("name", None)
            for k in ["amenity","shop","leisure"]:
                if k in gdf_pois.columns and pd.notna(r.get(k)):
                    cat = k
                    sub = r.get(k)
                    break
            poi_cat.append(cat)
            poi_subcat.append(sub)
            poi_name.append(name)
        gdf_pois["poi_category"] = poi_cat
        gdf_pois["poi_subcategory"] = poi_subcat
        gdf_pois["poi_name"] = poi_name
    else:
        gdf_pois[["poi_category","poi_subcategory","poi_name"]] = gdf_pois["tags"].apply(extract_poi_fields)

    # 3) POI → GEOID (spatial join)
    st.write("### 3) POI → GEOID eşleştirme (spatial join)")
    if "geometry" not in gdf_pois.columns:
        # bazen lon/lat ayrı olabilir
        # en yaygın isimler: lon/lng/longitude ve lat/latitude
        lon_col = None
        lat_col = None
        for lc in ["lon","lng","longitude","x"]:
            if lc in gdf_pois.columns: lon_col = lc; break
        for la in ["lat","latitude","y"]:
            if la in gdf_pois.columns: lat_col = la; break
        if lon_col and lat_col:
            gdf_pois = gpd.GeoDataFrame(
                gdf_pois,
                geometry=gpd.points_from_xy(gdf_pois[lon_col], gdf_pois[lat_col]),
                crs="EPSG:4326"
            )
        else:
            st.error("❌ POI dosyasında geometry veya lon/lat yok.")
            return
    # Blocks zaten polygon olmalı (geometry var varsayıyoruz)
    gdf_joined = gpd.sjoin(
        gdf_pois.to_crs(4326), 
        gdf_blocks[["GEOID","geometry"]].to_crs(4326),
        how="left",
        predicate="within"
    )

    # POI cleaned CSV
    keep_cols = []
    for c in ["id","lat","lon","longitude","latitude","poi_category","poi_subcategory","poi_name","GEOID"]:
        if c in gdf_joined.columns:
            keep_cols.append(c)
    if "lon" not in keep_cols and "longitude" in gdf_joined.columns:
        gdf_joined = gdf_joined.rename(columns={"longitude":"lon"})
        keep_cols.append("lon")
    if "lat" not in keep_cols and "latitude" in gdf_joined.columns:
        gdf_joined = gdf_joined.rename(columns={"latitude":"lat"})
        keep_cols.append("lat")
    df_poi_clean = gdf_joined[keep_cols].copy()
    df_poi_clean.to_csv(TMP_POI_CLEAN, index=False)
    st.success(f"✅ POI temizlendi ve GEOID eklendi → {TMP_POI_CLEAN} | {df_poi_clean.shape}")

    # 4) Dinamik POI risk skorları (0–3 normalize)
    st.write("### 4) Dinamik POI risk skoru hesapla (0–3)")
    # Suç verisini GeoDataFrame’e çevir
    gdf_crime = gpd.GeoDataFrame(
        df_crime.copy(),
        geometry=gpd.points_from_xy(df_crime["longitude"], df_crime["latitude"]),
        crs="EPSG:4326"
    )
    df_poi_tmp = pd.read_csv(TMP_POI_CLEAN)
    gdf_poi_tmp = gpd.GeoDataFrame(
        df_poi_tmp.copy(),
        geometry=gpd.points_from_xy(df_poi_tmp["lon"], df_poi_tmp["lat"]),
        crs="EPSG:4326"
    )
    # hariç tutmak istediklerin (police vb.)
    exclude_subtypes = {"police", "ranger_station"}
    gdf_poi_tmp = gdf_poi_tmp[~gdf_poi_tmp["poi_subcategory"].fillna("").isin(exclude_subtypes)]

    # Haversine için radyan
    poi_rad = to_rad(gdf_poi_tmp[["lat","lon"]].values)
    crime_rad = to_rad(gdf_crime[["latitude","longitude"]].values)

    tree = BallTree(crime_rad, metric="haversine")
    radius_m = 300
    radius_rad = radius_m / 6371000.0

    poi_types = gdf_poi_tmp["poi_subcategory"].fillna("")
    sums = {}
    cnts = {}
    for i, pt in enumerate(poi_rad):
        subtype = poi_types.iloc[i]
        if not subtype:
            continue
        idxs = tree.query_radius([pt], r=radius_rad)[0]
        c = len(idxs)
        sums[subtype] = sums.get(subtype, 0) + c
        cnts[subtype] = cnts.get(subtype, 0) + 1

    if not sums:
        st.warning("⚠️ POI etrafında suç bulunamadı; tüm riskler 0 kabul edilecek.")
        poi_risk_raw = {}
    else:
        poi_risk_raw = {k: round(sums[k]/cnts[k], 4) for k in sums.keys()}
    if poi_risk_raw:
        mn, mx = min(poi_risk_raw.values()), max(poi_risk_raw.values())
        poi_risk_norm = {k: round(3*(v - mn)/(mx - mn + 1e-6), 2) for k, v in poi_risk_raw.items()}
    else:
        poi_risk_norm = {}

    with open(TMP_POI_RISK_JSON, "w") as f:
        json.dump(poi_risk_norm, f, indent=2)
    st.success(f"✅ Dinamik risk JSON yazıldı → {TMP_POI_RISK_JSON} | tür sayısı: {len(poi_risk_norm)}")

    # 5) Suçlara POI metriklerini yaz (BallTree ile 300m içi)
    st.write("### 5) Suçlara POI metriklerini ekle (300 m)")
    # POI'lere risk skoru ata
    df_poi_tmp["risk_score"] = df_poi_tmp["poi_subcategory"].map(poi_risk_norm).fillna(0).round(2)
    gdf_poi_tmp = gpd.GeoDataFrame(
        df_poi_tmp,
        geometry=gpd.points_from_xy(df_poi_tmp["lon"], df_poi_tmp["lat"]),
        crs="EPSG:4326"
    )
    # Tekrar BallTree, bu kez POI ağacı
    poi_rad2 = to_rad(gdf_poi_tmp[["lat","lon"]].values)
    tree2 = BallTree(poi_rad2, metric="haversine")
    crime_rad2 = to_rad(gdf_crime[["latitude","longitude"]].values)
    neigh_idxs = tree2.query_radius(crime_rad2, r=radius_rad)

    poi_total_count = np.zeros(len(gdf_crime), dtype=int)
    poi_risk_score = np.zeros(len(gdf_crime), dtype=float)
    poi_dom_type = []

    subtype_series = gdf_poi_tmp["poi_subcategory"].fillna("")
    subtype_to_score = df_poi_tmp.set_index("poi_subcategory")["risk_score"].to_dict()

    for i, idxs in enumerate(neigh_idxs):
        subs = subtype_series.iloc[idxs]
        poi_total_count[i] = len(idxs)
        if len(subs) == 0:
            poi_risk_score[i] = 0.0
            poi_dom_type.append(None)
        else:
            scores = subs.map(subtype_to_score).fillna(0)
            poi_risk_score[i] = float(scores.sum())
            poi_dom_type.append(subs.value_counts().idxmax())

    gdf_crime["poi_total_count"] = poi_total_count
    gdf_crime["poi_risk_score"] = poi_risk_score
    gdf_crime["poi_dominant_type"] = poi_dom_type

    # Dinamik aralık etiketleri
    lbl_count = dynamic_bin_labels(gdf_crime["poi_total_count"], strategy="auto")
    lbl_risk = dynamic_bin_labels(gdf_crime["poi_risk_score"], strategy="auto")
    gdf_crime["poi_total_count_range"] = gdf_crime["poi_total_count"].apply(lbl_count)
    gdf_crime["poi_risk_score_range"] = gdf_crime["poi_risk_score"].apply(lbl_risk)

    # Ara çıktı: suç + POI ölçümleri
    gdf_crime.drop(columns=["geometry"]).to_csv(TMP_CRIME_POI, index=False)
    df_poi_tmp.to_csv(TMP_POI_WITH_RISK, index=False)
    st.success(f"✅ Ara çıktılar yazıldı → {TMP_CRIME_POI} ve {TMP_POI_WITH_RISK}")

    # 6) GEOID düzeyinde tekilleştirip crime_05 ile birleştir → crime_06
    st.write("### 6) GEOID düzeyinde tekilleştir → sf_crime_06.csv")
    df_crime05 = df_crime.copy()
    df_crime05["GEOID"] = ensure_geoid_str(df_crime05["GEOID"])

    df_poi_crime = pd.read_csv(TMP_CRIME_POI)
    if "GEOID" in df_poi_crime.columns:
        df_poi_crime["GEOID"] = ensure_geoid_str(df_poi_crime["GEOID"])

    # GEOID’e göre tekilleştir (mean/mode)
    def mode_or_nan(s: pd.Series):
        m = s.mode()
        return m.iloc[0] if not m.empty else np.nan

    poi_cols = [
        "GEOID", "poi_total_count", "poi_risk_score",
        "poi_dominant_type", "poi_total_count_range", "poi_risk_score_range"
    ]
    exist_cols = [c for c in poi_cols if c in df_poi_crime.columns]
    df_poi_grouped = (
        df_poi_crime[exist_cols]
        .groupby("GEOID", as_index=False)
        .agg({
            "poi_total_count": "mean",
            "poi_risk_score": "mean",
            "poi_dominant_type": mode_or_nan,
            "poi_total_count_range": mode_or_nan,
            "poi_risk_score_range": mode_or_nan
        })
    )

    # Eksik kategorileri doldur
    if "poi_dominant_type" in df_poi_grouped.columns:
        df_poi_grouped["poi_dominant_type"] = df_poi_grouped["poi_dominant_type"].fillna("No_POI")
    for c in ["poi_total_count_range","poi_risk_score_range"]:
        if c in df_poi_grouped.columns:
            df_poi_grouped[c] = df_poi_grouped[c].fillna("Unknown")

    # Merge (many-to-one)
    df_merged = pd.merge(df_crime05, df_poi_grouped, on="GEOID", how="left")
    # Tür dönüşümleri
    if "poi_total_count" in df_merged.columns:
        df_merged["poi_total_count"] = pd.to_numeric(df_merged["poi_total_count"], errors="coerce").fillna(0).astype(int)
    if "poi_risk_score" in df_merged.columns:
        df_merged["poi_risk_score"] = pd.to_numeric(df_merged["poi_risk_score"], errors="coerce").fillna(0.0)

    # KAYDET → sf_crime_06.csv
    df_merged.to_csv(OUT_06, index=False)
    st.success(f"🎉 Tamamdır! sf_crime_06 kaydedildi → {OUT_06}")
    st.dataframe(df_merged.head(20))

with st.expander("ℹ️ Notlar", expanded=False):
    st.markdown("""
- **GEOID uzunluğu** veri setinize göre otomatik 11/12/15 hane tespit edilip `zfill` uygulanır.  
- **Dinamik risk**: Her POI alt türü için 300 m içindeki suç sayılarının **POI başına ortalaması** alınır ve 0–3 aralığına normalize edilir.  
- Suç kayıtlarına **300 m içindeki POI’lerin toplam sayısı**, bu POI’lerin **risk skorları toplamı** ve **baskın POI türü** yazılır.  
- Son olarak, GEOID düzeyinde tekilleştirilip `sf_crime_06.csv` üretilir.
""")

if st.button("🚀 Çalıştır (05 → 06)"):
    try:
        run_pipeline()
    except Exception as e:
        st.exception(e)
        st.error("İşlem hata verdi. Yukarıdaki istisnayı kontrol edin.")
