# update_train.py  (SON REVIZE — MOBILITYDB "latest.zip" OTOMATİK)

import os, io, zipfile, time, re
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import requests
from scipy.spatial import cKDTree

# =========================
# Küçük yardımcılar
# =========================
def ensure_parent(path: str) -> None:
    Path(os.path.dirname(path) or ".").mkdir(parents=True, exist_ok=True)

def sanitize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    obj_cols = df.select_dtypes(include=["object"]).columns
    if len(obj_cols) == 0:
        return df

    repl = {
        "–": "-", "−": "-",
        "≤": "<=", "≥": ">=",
        "â€“": "-", "â€": "-",
        "â‰¤": "<=", "â‰¥": ">=",
    }
    for c in obj_cols:
        df[c] = df[c].replace(repl, regex=False)
    return df

def safe_save_csv(df: pd.DataFrame, path: str) -> None:
    ensure_parent(path)
    tmp = path + ".tmp"
    try:
        df2 = sanitize_text_columns(df)
        with open(tmp, "w", encoding="utf-8-sig", errors="replace", newline="") as f:
            df2.to_csv(f, index=False)
        os.replace(tmp, path)
        print(f"💾 Kaydedildi: {path}")
    except Exception as e:
        print(f"❌ Kaydetme hatası: {path}\n{e}")
        try:
            df.to_csv(path + ".bak", index=False, encoding="utf-8-sig")
            print(f"📁 Yedek oluşturuldu: {path}.bak")
        except Exception:
            pass

def log_shape(df: pd.DataFrame, label: str) -> None:
    r, c = df.shape
    print(f"📊 {label}: {r} satır × {c} sütun")

def log_delta(before_shape, after_shape, label: str) -> None:
    br, bc = before_shape
    ar, ac = after_shape
    print(f"🔗 {label}: {br}×{bc} → {ar}×{ac} (Δr={ar-br}, Δc={ac-bc})")

DEFAULT_GEOID_LEN = int(os.getenv("GEOID_LEN", "11"))

def normalize_geoid(series: pd.Series, target_len: int = DEFAULT_GEOID_LEN) -> pd.Series:
    """Yalnızca rakamları al, SOL’dan target_len haneyi tut, zfill."""
    s = series.astype(str).str.extract(r"(\d+)", expand=False)
    L = int(target_len)
    return s.str[:L].str.zfill(L)

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
# ENV / Yollar
# =========================
BASE_DIR = os.getenv("CRIME_DATA_DIR", "crime_prediction_data")
Path(BASE_DIR).mkdir(parents=True, exist_ok=True)

# Suç girdisi adayları
CRIME_CANDIDATES = [
    os.path.join(BASE_DIR, "sf_crime_04.csv"),
]
CRIME_INPUT = next((p for p in CRIME_CANDIDATES if os.path.exists(p)), None)
if CRIME_INPUT is None:
    raise FileNotFoundError("❌ Suç girdi dosyası bulunamadı (sf_crime_04.csv).")
print(f"📄 Train enrich girdi: {os.path.abspath(CRIME_INPUT)}")

CRIME_OUTPUT = os.path.join(BASE_DIR, "sf_crime_05.csv")

# Ara veri / çıktı adları (kanonik + uyumluluk)
TRAIN_STOPS_WITH_GEOID = os.path.join(BASE_DIR, os.getenv("TRAIN_STOPS_NAME", "sf_train_stops_with_geoid.csv"))
TRAIN_LEGACY_RAW_Y     = os.path.join(BASE_DIR, os.getenv("TRAIN_LEGACY_RAW_Y", "train_y.csv"))  # legacy ham
TRAIN_SUMMARY_NAME     = os.path.join(BASE_DIR, os.getenv("TRAIN_SUMMARY_NAME", "train.csv"))     # legacy özet/feature

# =========================
# WEEKLY CACHE POLICY (TRAIN)
# =========================
FORCE_TRAIN_REFRESH = os.getenv("FORCE_TRAIN_REFRESH", "0").strip().lower() in ("1", "true", "yes")

# TRAIN cache dosyaları mevcutsa ve force yoksa:
# - GTFS indirmeyi ve geoid-metric hesaplamayı atla
# - sadece bu cache'lerle CRIME merge yapıp sf_crime_05 yaz
TRAIN_CACHE_OK = (
    os.path.exists(TRAIN_STOPS_WITH_GEOID) and
    os.path.exists(TRAIN_SUMMARY_NAME)
)

if TRAIN_CACHE_OK and (not FORCE_TRAIN_REFRESH):
    print("✅ TRAIN cache bulundu ve FORCE_TRAIN_REFRESH=0 → indirme/hesaplama atlanıyor.")
    print("   cache stops:", os.path.abspath(TRAIN_STOPS_WITH_GEOID))
    print("   cache metrics:", os.path.abspath(TRAIN_SUMMARY_NAME))

    # 1) cache metrikleri oku
    geo_metrics = pd.read_csv(TRAIN_SUMMARY_NAME, low_memory=False)
    if "GEOID" not in geo_metrics.columns:
        raise KeyError("❌ train.csv içinde GEOID yok (cache bozuk).")
    geo_metrics["GEOID"] = normalize_geoid(geo_metrics["GEOID"], DEFAULT_GEOID_LEN)

    # 2) crime oku + merge + yaz
    crime = pd.read_csv(CRIME_INPUT, dtype={"GEOID": str}, low_memory=False)
    _before = crime.shape
    crime["GEOID"] = normalize_geoid(crime["GEOID"], DEFAULT_GEOID_LEN)

    _overlap = (set(crime.columns) & set(geo_metrics.columns)) - {"GEOID"}
    if _overlap:
        print(f"🧹 TRAIN cache merge overlap bulundu, geo_metrics'ten düşürüldü: {sorted(_overlap)}")
        geo_metrics = geo_metrics.drop(columns=list(_overlap), errors="ignore")

    crime_enriched = crime.merge(
        geo_metrics,
        on="GEOID",
        how="left",
        validate="many_to_one"
    )
    log_delta(_before, crime_enriched.shape, "CRIME ⨯ TRAIN (CACHE GEOID enrich)")
    log_shape(crime_enriched, "CRIME (train enrich sonrası - CACHE)")

    # NaN raporu
    nan_counts = crime_enriched.isna().sum()
    nan_counts = nan_counts[nan_counts > 0].sort_values(ascending=False)
    print("🔎 NaN sayıları (sf_crime_05 yazılmadan önce) [CACHE]:")
    print("✅ NaN yok." if nan_counts.empty else nan_counts.to_string())

    safe_save_csv(crime_enriched, CRIME_OUTPUT)
    print(f"✅ CACHE ile güncellendi → {CRIME_OUTPUT}")
    raise SystemExit(0)
    
# Census GeoJSON adayları
CENSUS_CANDIDATES = [
    os.path.join(BASE_DIR, "sf_census_blocks.geojson"),
    os.path.join(".",      "sf_census_blocks.geojson"),
]

# Mobility Database katalog (feeds_v2.csv) — DOĞRU kaynak: files.mobilitydatabase.org
MOBILITYDB_FEEDS_V2_URL = os.getenv("MOBILITYDB_FEEDS_V2_URL", "https://files.mobilitydatabase.org/feeds_v2.csv")

def get_mobilitydb_latest_zip(feed_id: str = "mdb-53") -> str | None:
    """
    MobilityDB feeds_v2.csv içinden <feed_id> için urls.latest alanını alır.
    Örn: https://files.mobilitydatabase.org/mdb-53/latest.zip
    """
    try:
        df = pd.read_csv(MOBILITYDB_FEEDS_V2_URL, low_memory=False)
        if "id" not in df.columns:
            print("⚠️ feeds_v2.csv beklenmeyen şema: 'id' kolonu yok.")
            return None
        row = df[df["id"].astype(str).str.strip().eq(str(feed_id))]
        if row.empty:
            print(f"⚠️ feeds_v2.csv içinde {feed_id} bulunamadı.")
            return None
        url = row.iloc[0].get("urls.latest")
        if isinstance(url, str) and url.strip().startswith("http"):
            return url.strip()
        print(f"⚠️ {feed_id} için 'urls.latest' boş/uygunsuz.")
        return None
    except Exception as e:
        print(f"⚠️ feeds_v2.csv okunamadı ({MOBILITYDB_FEEDS_V2_URL}): {e}")
        return None

# GTFS kaynakları (BART) — ÖNCE latest.zip, sonra fallback
latest_zip = get_mobilitydb_latest_zip("mdb-53")

GTFS_URLS = []
if latest_zip:
    print("✅ MobilityDB latest.zip:", latest_zip)
    GTFS_URLS.append(latest_zip)

GTFS_URLS += [
    os.getenv("BART_GTFS_URL", "https://files.mobilitydatabase.org/mdb-53/mdb-53-202512180015/mdb-53-202512180015.zip"),
    os.getenv("BART_GTFS_URL_ALT1", "").strip(),
    os.getenv("BART_GTFS_URL_ALT2", "").strip(),
]
GTFS_URLS = [u for u in GTFS_URLS if u]
print("🧭 GTFS_URLS sırası:", GTFS_URLS)

# İndirme başarısız olursa cache / stub?
ALLOW_STUB = os.getenv("ALLOW_STUB_ON_API_FAIL", "1").strip().lower() not in ("0", "false")

# =========================
# 1) GTFS stops.txt indir/çıkar (retry/backoff)
# =========================
def download_gtfs_stops(urls: list[str], max_retries: int = 4, backoff_base: float = 1.7) -> pd.DataFrame | None:
    sess = requests.Session()
    for url in urls:
        print(f"🚉 BART GTFS deneniyor: {url}")
        for attempt in range(max_retries + 1):
            try:
                r = sess.get(url, timeout=60, allow_redirects=True)
                if r.status_code in (429,) or 500 <= r.status_code < 600:
                    if attempt >= max_retries:
                        r.raise_for_status()
                    sleep_s = backoff_base ** (attempt + 1)
                    print(f"⚠️ Geçici hata (HTTP {r.status_code}) → {attempt+1}. deneme, {sleep_s:.1f}s bekleme…")
                    time.sleep(sleep_s)
                    continue

                r.raise_for_status()
                content = r.content

                # ZIP dosyaları genelde "PK\x03\x04" ile başlar
                if not content.startswith(b"PK"):
                    snippet = content[:200].decode("utf-8", errors="ignore")
                    raise ValueError(f"ZIP gelmedi (ilk 200 char): {snippet}")

                buf = io.BytesIO(content)
                with zipfile.ZipFile(buf, "r") as zf:
                    members = [m for m in zf.namelist() if m.lower().endswith("stops.txt")]
                    if not members:
                        raise FileNotFoundError("stops.txt GTFS paketinde bulunamadı.")
                    with zf.open(members[0], "r") as f:
                        stops = pd.read_csv(f, dtype={"stop_lat": float, "stop_lon": float})
                return stops

            except Exception as e:
                if attempt >= max_retries:
                    print(f"❌ GTFS indirme/çıkarma hatası (url={url}): {e}")
                else:
                    sleep_s = backoff_base ** (attempt + 1)
                    print(f"⚠️ Hata (url={url}) → tekrar denenecek ({attempt+1}/{max_retries}), {sleep_s:.1f}s bekleme. ({e})")
                    time.sleep(sleep_s)

        print(f"↪️ URL başarısız: {url}")
    return None

# =========================
# 2) Suç verisini oku (GEOID evreni)
# =========================
crime = pd.read_csv(CRIME_INPUT, low_memory=False)
log_shape(crime, "CRIME (okundu)")
if "GEOID" not in crime.columns:
    raise KeyError("❌ Suç verisinde 'GEOID' kolonu yok.")
crime["GEOID"] = normalize_geoid(crime["GEOID"], DEFAULT_GEOID_LEN)
crime_geoids = pd.Series(crime["GEOID"].unique(), name="GEOID")
print(f"🧩 CRIME farklı GEOID sayısı: {crime_geoids.size}")

# =========================
# 3) GTFS duraklarını edin / cache / stub
# =========================
stops = download_gtfs_stops(GTFS_URLS, max_retries=4, backoff_base=1.7)
if stops is None:
    # cache'ten oku
    if os.path.exists(TRAIN_STOPS_WITH_GEOID):
        print("⚠️ GTFS indirilemedi; mevcut cache kullanılacak:", os.path.abspath(TRAIN_STOPS_WITH_GEOID))
        try:
            stops = pd.read_csv(TRAIN_STOPS_WITH_GEOID, low_memory=False)
        except Exception:
            stops = pd.DataFrame(columns=["stop_lat", "stop_lon"])
    elif os.path.exists(TRAIN_LEGACY_RAW_Y):
        print("⚠️ GTFS indirilemedi; legacy cache kullanılacak:", os.path.abspath(TRAIN_LEGACY_RAW_Y))
        try:
            stops = pd.read_csv(TRAIN_LEGACY_RAW_Y, low_memory=False)
        except Exception:
            stops = pd.DataFrame(columns=["stop_lat", "stop_lon"])
    elif ALLOW_STUB:
        print("⚠️ GTFS ve yerel cache yok → STUB (0 durak, NaN metrik).")
        stops = pd.DataFrame(columns=["stop_lat", "stop_lon"])
    else:
        raise SystemExit("❌ GTFS alınamadı ve cache yok; çıkılıyor.")

# Kolon isimleri normalize
low = {c.lower(): c for c in stops.columns}
if "stop_lat" not in low or "stop_lon" not in low:
    # bazı GTFS'lerde 'stop_latitude/stop_longitude' olabilir
    for a, b in (("stop_latitude", "stop_longitude"), ("latitude", "longitude"), ("lat", "lon"), ("lat", "long")):
        if a in low and b in low:
            stops.rename(columns={low[a]: "stop_lat", low[b]: "stop_lon"}, inplace=True)
            break
else:
    # doğru isimler varsa standardize
    if low["stop_lat"] != "stop_lat":
        stops.rename(columns={low["stop_lat"]: "stop_lat"}, inplace=True)
    if low["stop_lon"] != "stop_lon":
        stops.rename(columns={low["stop_lon"]: "stop_lon"}, inplace=True)

stops["stop_lat"] = pd.to_numeric(stops.get("stop_lat"), errors="coerce")
stops["stop_lon"] = pd.to_numeric(stops.get("stop_lon"), errors="coerce")
stops = stops.dropna(subset=["stop_lat", "stop_lon"]).copy()
log_shape(stops, "GTFS stops (temiz)")

# =========================
# 4) Census blokları ve GEOID eşleme
# =========================
census_path = next((p for p in CENSUS_CANDIDATES if os.path.exists(p)), None)
blocks_ok = True
if census_path is None:
    print("⚠️ Nüfus blokları GeoJSON bulunamadı. GEOID eşleme/mesafe → stub.")
    blocks_ok = False

if blocks_ok:
    gdf_blocks = gpd.read_file(census_path)
    # CRS normalize
    if gdf_blocks.crs is None:
        gdf_blocks.set_crs("EPSG:4326", inplace=True, allow_override=True)
    else:
        try:
            epsg = gdf_blocks.crs.to_epsg()
        except Exception:
            epsg = None
        if epsg != 4326:
            gdf_blocks = gdf_blocks.to_crs(epsg=4326)

    gcol = "GEOID" if "GEOID" in gdf_blocks.columns else next(
        (c for c in gdf_blocks.columns if str(c).upper().startswith("GEOID")), None
    )
    if not gcol:
        print("⚠️ GeoJSON'da GEOID vari sütun yok. Stub’a düşülecek.")
        blocks_ok = False
    else:
        gdf_blocks["GEOID"] = normalize_geoid(gdf_blocks[gcol], DEFAULT_GEOID_LEN)
else:
    gdf_blocks = None

if blocks_ok and not stops.empty:
    gdf_stops = gpd.GeoDataFrame(
        stops, geometry=gpd.points_from_xy(stops["stop_lon"], stops["stop_lat"]), crs="EPSG:4326"
    )
    try:
        gdf_joined = gpd.sjoin(gdf_stops, gdf_blocks[["geometry", "GEOID"]], how="left", predicate="within")
    except Exception as e:
        print(f"⚠️ sjoin(within) başarısız ({e}). sjoin_nearest(max_distance=5 m) deneniyor…")

        gdf_stops_m  = gdf_stops.to_crs(epsg=3857)
        gdf_blocks_m = gdf_blocks[["geometry", "GEOID"]].to_crs(epsg=3857)
        
        gdf_joined_m = gpd.sjoin_nearest(
            gdf_stops_m,
            gdf_blocks_m,
            how="left",
            max_distance=50
        )

        gdf_joined = gdf_joined_m.to_crs(epsg=4326)

    gdf_joined = gdf_joined.drop(columns=["index_right"], errors="ignore")
    gdf_joined["GEOID"] = normalize_geoid(gdf_joined["GEOID"], DEFAULT_GEOID_LEN)
    train_stops_geo = pd.DataFrame(gdf_joined.drop(columns=["geometry"], errors="ignore")).copy()
    log_shape(train_stops_geo, "TRAIN stops ⨯ GEOID (eşleme)")
else:
    train_stops_geo = stops.copy()
    train_stops_geo["GEOID"] = pd.NA

# =========================
# 5) Ham dosyaları yaz (kanonik + legacy)
# =========================
safe_save_csv(train_stops_geo, TRAIN_STOPS_WITH_GEOID)   # kanonik
try:
    safe_save_csv(train_stops_geo, TRAIN_LEGACY_RAW_Y)   # legacy uyumluluk
    print(f"✅ TRAIN ham (kanonik): {TRAIN_STOPS_WITH_GEOID}")
    print(f"↪️ Legacy kopya (train_y.csv): {TRAIN_LEGACY_RAW_Y}")
except Exception as e:
    print(f"⚠️ Legacy train_y.csv yazılamadı: {e}")

# =========================
# 6) GEOID-level metrikler (distance & count) + binleme
# =========================
if blocks_ok:
    gdf_blocks_3857 = gdf_blocks[["GEOID", "geometry"]].copy().to_crs(epsg=3857)
    gdf_blocks_3857["cx"] = gdf_blocks_3857.geometry.centroid.x
    gdf_blocks_3857["cy"] = gdf_blocks_3857.geometry.centroid.y
    blocks_xy = np.vstack([gdf_blocks_3857["cx"].values, gdf_blocks_3857["cy"].values]).T

    bad = gdf_blocks_3857["cx"].isna() | gdf_blocks_3857["cy"].isna()
    if bad.any():
        print(f"⚠️ {bad.sum()} blok centroid NaN (geometry sorunu). Mesafe bu GEOID'lerde NaN kalacak.")
    
    tmp_pts = train_stops_geo.dropna(subset=["stop_lat", "stop_lon"]).copy()
    gdf_train_xy = gpd.GeoDataFrame(
        tmp_pts[["stop_lat", "stop_lon"]].copy(),
        geometry=gpd.points_from_xy(tmp_pts["stop_lon"], tmp_pts["stop_lat"]),
        crs="EPSG:4326"
    ).to_crs(epsg=3857)

    train_xy = np.vstack([gdf_train_xy.geometry.x.values, gdf_train_xy.geometry.y.values]).T

    geo_metrics = pd.DataFrame({"GEOID": gdf_blocks_3857["GEOID"].values})
    geo_metrics["distance_to_train"] = np.nan
    geo_metrics["train_stop_count"] = 0

    if len(train_xy) > 0 and len(blocks_xy) > 0:
        tree = cKDTree(train_xy)
        nearest_dist, _ = tree.query(blocks_xy, k=1)
        geo_metrics["distance_to_train"] = nearest_dist

        finite_d = nearest_dist[np.isfinite(nearest_dist)]
        if finite_d.size > 0 and np.nanmax(finite_d) > 0:
            radius = float(np.nanpercentile(finite_d, 75))  # p75 yarıçap
            neighbor_lists = tree.query_ball_point(blocks_xy, r=radius)
            geo_metrics["train_stop_count"] = [len(lst) for lst in neighbor_lists]
            print(f"🟢 Sayım yarıçapı (p75): ~{int(round(radius))} m")
else:
    # GeoJSON yoksa: CRIME GEOID evrenine stub
    geo_metrics = pd.DataFrame({"GEOID": crime_geoids})
    geo_metrics["distance_to_train"] = np.nan
    geo_metrics["train_stop_count"] = 0

# Tekilleştirme + doğrulama
geo_metrics["GEOID"] = normalize_geoid(geo_metrics["GEOID"], DEFAULT_GEOID_LEN)
geo_metrics = geo_metrics.sort_values("GEOID").drop_duplicates("GEOID", keep="first")
assert geo_metrics["GEOID"].is_unique, "TRAIN: GEOID eşsiz değil!"
log_shape(geo_metrics, "GEOID-bazlı metrikler (ham)")

cov = geo_metrics["distance_to_train"].notna().mean() if "distance_to_train" in geo_metrics.columns else 0.0
print(f"🧪 TRAIN mesafe coverage: {cov:.3%}")

# Binleme
dist = pd.to_numeric(geo_metrics["distance_to_train"], errors="coerce").replace([np.inf, -np.inf], np.nan)
finite_dist = dist.dropna()
if len(finite_dist) >= 2 and finite_dist.max() > finite_dist.min():
    n_bins = freedman_diaconis_bin_count(finite_dist.to_numpy(), max_bins=10)
    _, edges = pd.qcut(finite_dist, q=n_bins, retbins=True, duplicates="drop")
    labels = [f"{int(round(edges[i]))}-{int(round(edges[i+1]))}m" for i in range(len(edges) - 1)]
    geo_metrics["distance_to_train_range"] = pd.cut(dist, bins=edges, labels=labels, include_lowest=True)
else:
    geo_metrics["distance_to_train_range"] = "0-0m"

cnt = pd.to_numeric(geo_metrics["train_stop_count"], errors="coerce").fillna(0)
if cnt.nunique() > 1:
    n_c_bins = freedman_diaconis_bin_count(cnt.to_numpy(), max_bins=8)
    _, c_edges = pd.qcut(cnt, q=n_c_bins, retbins=True, duplicates="drop")
    c_labels = [f"{int(round(c_edges[i]))}-{int(round(c_edges[i+1]))}" for i in range(len(c_edges) - 1)]
    geo_metrics["train_stop_count_range"] = pd.cut(cnt, bins=c_edges, labels=c_labels, include_lowest=True)
else:
    geo_metrics["train_stop_count_range"] = f"{int(cnt.min())}-{int(cnt.max())}"

log_shape(geo_metrics, "GEOID-bazlı metrikler (binlenmiş)")

# =========================
# 7) GEOID-level özeti de yaz (legacy: train.csv)
# =========================
safe_save_csv(geo_metrics, TRAIN_SUMMARY_NAME)
print(f"✅ TRAIN özet (GEOID-level) yazıldı → {TRAIN_SUMMARY_NAME}")

# =========================
# 8) Suç verisine GEOID ile merge
# =========================
crime = pd.read_csv(CRIME_INPUT, dtype={"GEOID": str}, low_memory=False)
_before = crime.shape

crime["GEOID"] = normalize_geoid(crime["GEOID"], DEFAULT_GEOID_LEN)
geo_metrics["GEOID"] = normalize_geoid(geo_metrics["GEOID"], DEFAULT_GEOID_LEN)

# Yalnızca CRIME evrenindeki GEOID’ler
geo_metrics = geo_metrics[geo_metrics["GEOID"].isin(crime["GEOID"].unique())].copy()

_overlap = (set(crime.columns) & set(geo_metrics.columns)) - {"GEOID"}
if _overlap:
    print(f"🧹 TRAIN merge overlap bulundu, geo_metrics'ten düşürüldü: {sorted(_overlap)}")
    geo_metrics = geo_metrics.drop(columns=list(_overlap), errors="ignore")

crime_enriched = crime.merge(
    geo_metrics,
    on="GEOID",
    how="left",
    validate="many_to_one"  # satır patlaması yok
)

log_delta(_before, crime_enriched.shape, "CRIME ⨯ TRAIN (GEOID enrich)")
log_shape(crime_enriched, "CRIME (train enrich sonrası)")

# =========================
# 9) Kaydet & önizleme
# =========================

# -------- NaN raporu (kayıttan hemen önce) --------
nan_counts = crime_enriched.isna().sum()
nan_counts = nan_counts[nan_counts > 0].sort_values(ascending=False)

print("🔎 NaN sayıları (sf_crime_05 yazılmadan önce):")
if nan_counts.empty:
    print("✅ NaN yok.")
else:
    print(nan_counts.to_string())
# -----------------------------------------------

safe_save_csv(crime_enriched, CRIME_OUTPUT)
print("📦 Yeni sütunlar eklendi (örnek):", ["distance_to_train", "distance_to_train_range", "train_stop_count", "train_stop_count_range"])
print(f"✅ Güncellenmiş veri kaydedildi → {CRIME_OUTPUT}")
try:
    print("sf_crime_05.csv — ilk 5 satır")
    print(crime_enriched.head(5).to_string(index=False))
except Exception:
    pass
