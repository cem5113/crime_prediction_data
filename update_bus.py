# update_bus.py (INCREMENTAL + STABLE BINS)

import os, json, time
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import requests
from scipy.spatial import cKDTree

# =========================
# küçük yardımcılar
# =========================
def log_shape(df: pd.DataFrame, label: str):
    r, c = df.shape
    print(f"📊 {label}: {r} satır × {c} sütun")

def log_delta(before_shape, after_shape, label: str):
    br, bc = before_shape
    ar, ac = after_shape
    print(f"🔗 {label}: {br}×{bc} → {ar}×{ac} (Δr={ar-br}, Δc={ac-bc})")

def ensure_parent(path):
    Path(str(path)).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)

def sanitize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    obj_cols = df.select_dtypes(include=["object"]).columns
    if len(obj_cols) == 0:
        return df
    repl = {
        "–": "-", "−": "-", "≤": "<=", "≥": ">=",
        "â€“": "-", "â€": "-", "â‰¤": "<=", "â‰¥": ">=",
    }
    for c in obj_cols:
        df[c] = df[c].replace(repl, regex=False)
    return df

def safe_save_csv(df: pd.DataFrame, path: str):
    ensure_parent(path)
    tmp = str(path) + ".tmp"
    df2 = sanitize_text_columns(df)
    with open(tmp, "w", encoding="utf-8-sig", errors="replace", newline="") as f:
        df2.to_csv(f, index=False)
    os.replace(tmp, path)

DEFAULT_GEOID_LEN = int(os.getenv("GEOID_LEN", "11"))
def normalize_geoid(s: pd.Series, target_len: int = DEFAULT_GEOID_LEN) -> pd.Series:
    s = s.astype(str).str.extract(r"(\d+)", expand=False)
    L = int(target_len)
    return s.str[:L].str.zfill(L)

def freedman_diaconis_bin_count(data: np.ndarray, max_bins: int = 10) -> int:
    data = np.asarray(data)
    if len(data) < 2 or np.all(data == data[0]):
        return 1
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    if iqr == 0:
        return min(max_bins, max(2, int(np.sqrt(len(data)))))
    bw = 2 * iqr / (len(data) ** (1 / 3))
    if bw <= 0:
        return min(max_bins, max(2, int(np.sqrt(len(data)))))
    return max(2, min(max_bins, int(np.ceil((data.max() - data.min()) / bw))))

def extract_lat_lon(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    candidates = [
        ("latitude", "longitude"),
        ("lat", "long"),
        ("y", "x"),
        ("stop_lat", "stop_lon"),
        ("stop_latitude", "stop_longitude"),
        ("position_latitude", "position_longitude"),
    ]
    for la, lo in candidates:
        if la in df.columns and lo in df.columns:
            df["stop_lat"] = pd.to_numeric(df[la], errors="coerce")
            df["stop_lon"] = pd.to_numeric(df[lo], errors="coerce")
            return df

    if "location" in df.columns:
        def _g(o, k):
            if isinstance(o, dict):
                return o.get(k)
            if isinstance(o, str):
                try:
                    j = json.loads(o); return j.get(k)
                except Exception:
                    return None
            return None
        df["stop_lat"] = pd.to_numeric(df["location"].apply(lambda o: _g(o, "latitude")), errors="coerce")
        df["stop_lon"] = pd.to_numeric(df["location"].apply(lambda o: _g(o, "longitude")), errors="coerce")
        if df["stop_lat"].notna().any() and df["stop_lon"].notna().any():
            return df

    if "the_geom" in df.columns:
        def _coords(o):
            if isinstance(o, dict) and "coordinates" in o and isinstance(o["coordinates"], (list, tuple)) and len(o["coordinates"]) >= 2:
                lon, lat = o["coordinates"][:2]; return lat, lon
            if isinstance(o, str):
                try:
                    j = json.loads(o)
                    if "coordinates" in j and len(j["coordinates"]) >= 2:
                        lon, lat = j["coordinates"][:2]; return lat, lon
                except Exception:
                    return None, None
            return None, None
        latlon = df["the_geom"].apply(_coords)
        df["stop_lat"] = pd.to_numeric(latlon.apply(lambda t: t[0]), errors="coerce")
        df["stop_lon"] = pd.to_numeric(latlon.apply(lambda t: t[1]), errors="coerce")
        return df

    df["stop_lat"] = np.nan
    df["stop_lon"] = np.nan
    return df

def download_bus_with_retry(base_url: str, headers: dict, limit: int = 50000, max_retries: int = 5, backoff_base: float = 1.6):
    rows, offset = [], 0
    while True:
        params = {"$limit": limit, "$offset": offset}
        attempt = 0
        while True:
            try:
                r = requests.get(base_url, params=params, headers=headers, timeout=60)
                if r.status_code in (429,) or 500 <= r.status_code < 600:
                    attempt += 1
                    if attempt > max_retries:
                        r.raise_for_status()
                    sleep_s = backoff_base ** attempt
                    print(f"⚠️ Geçici hata (status={r.status_code}) offset={offset} → {attempt}. deneme, {sleep_s:.1f}s bekleme...")
                    time.sleep(sleep_s); continue
                r.raise_for_status()
                data = r.json()
                chunk = pd.DataFrame(data)
                break
            except requests.HTTPError as e:
                print(f"❌ İndirme hatası (offset={offset}): {e}")
                return None
            except Exception as e:
                attempt += 1
                if attempt > max_retries:
                    print(f"❌ Ağ/parse hatası (offset={offset}): {e}")
                    return None
                sleep_s = backoff_base ** attempt
                print(f"⚠️ Ağ/parse hatası (offset={offset}) → {attempt}. deneme, {sleep_s:.1f}s bekleme... ({e})")
                time.sleep(sleep_s)

        if chunk is None or chunk.empty:
            break
        if offset == 0:
            print("🔎 İlk chunk kolonları:", list(chunk.columns))
        rows.append(chunk)
        offset += len(chunk)
        print(f"  + {offset} kayıt indirildi...")
        if len(chunk) < limit:
            break

    if not rows:
        return None
    return pd.concat(rows, ignore_index=True)

# =========================
# STABLE BIN EDGES (range etiketleri oynamasın)
# =========================
def load_bins(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_bins(obj, path: str):
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def make_or_apply_bins_distance(series: pd.Series, bins_json: dict | None):
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)

    # varsa aynı kenarları uygula
    if bins_json and "distance_edges" in bins_json and len(bins_json["distance_edges"]) >= 2:
        edges = np.array(bins_json["distance_edges"], dtype=float)
        labels = [f"{edges[i]:.1f}-{edges[i+1]:.1f}m" for i in range(len(edges)-1)]
        return pd.cut(s, bins=edges, labels=labels, include_lowest=True), edges.tolist()

    d = s.dropna()
    if len(d) >= 2 and d.max() > d.min():
        n_bins = freedman_diaconis_bin_count(d.to_numpy(), max_bins=10)
        _, edges = pd.qcut(d, q=n_bins, retbins=True, duplicates="drop")
        edges = np.unique(edges)
        if len(edges) >= 2:
            labels = [f"{edges[i]:.1f}-{edges[i+1]:.1f}m" for i in range(len(edges)-1)]
            return pd.cut(s, bins=edges, labels=labels, include_lowest=True), edges.tolist()

    # fallback
    edges = [0.0, 0.0]
    return pd.Series(["0-0m"] * len(series)), edges

def make_or_apply_bins_count(series: pd.Series, bins_json: dict | None):
    s = pd.to_numeric(series, errors="coerce").fillna(0)

    if bins_json and "count_edges" in bins_json and len(bins_json["count_edges"]) >= 2:
        edges = np.array(bins_json["count_edges"], dtype=float)
        labels = [f"{int(edges[i])}-{int(edges[i+1])}" for i in range(len(edges)-1)]
        return pd.cut(s, bins=edges, labels=labels, include_lowest=True), edges.tolist()

    if s.nunique() > 1:
        n_bins = freedman_diaconis_bin_count(s.to_numpy(), max_bins=8)
        _, edges = pd.qcut(s, q=n_bins, retbins=True, duplicates="drop")
        edges = np.unique(edges)
        if len(edges) >= 2:
            labels = [f"{int(edges[i])}-{int(edges[i+1])}" for i in range(len(edges)-1)]
            return pd.cut(s, bins=edges, labels=labels, include_lowest=True), edges.tolist()

    edges = [float(s.min()), float(s.max())]
    return pd.Series([f"{int(s.min())}-{int(s.max())}"] * len(series)), edges

# =========================
# yollar & ENV
# =========================
BASE_DIR = os.getenv("CRIME_DATA_DIR", "crime_prediction_data")
Path(BASE_DIR).mkdir(parents=True, exist_ok=True)

CRIME_INPUT  = os.path.join(BASE_DIR, os.getenv("CRIME_INPUT_NAME", "sf_crime_03.csv"))
CRIME_OUTPUT = os.path.join(BASE_DIR, os.getenv("CRIME_OUTPUT_NAME", "sf_crime_04.csv"))

BUS_CANON_RAW    = os.path.join(BASE_DIR, os.getenv("BUS_CANON_RAW", "sf_bus_stops_with_geoid.csv"))
BUS_LEGACY_RAW_Y = os.path.join(BASE_DIR, os.getenv("BUS_LEGACY_RAW_Y", "bus_y.csv"))
BUS_SUMMARY_NAME = os.path.join(BASE_DIR, os.getenv("BUS_SUMMARY_NAME", "bus.csv"))

BUS_BINS_JSON    = os.path.join(BASE_DIR, os.getenv("BUS_BINS_JSON", "bus_bins.json"))

CENSUS_CANDIDATES = [
    os.path.join(BASE_DIR, "sf_census_blocks.geojson"),
    os.path.join(".",      "sf_census_blocks.geojson"),
]

RID    = os.getenv("BUS_DATASET_ID", "i28k-bkz6")
BASE   = f"https://data.sfgov.org/resource/{RID}.json"
TOKEN  = os.getenv("SOCS_APP_TOKEN", "").strip()
HEADERS = {"Accept": "application/json"}
if TOKEN:
    HEADERS["X-App-Token"] = TOKEN

ALLOW_STUB = os.getenv("ALLOW_STUB_ON_API_FAIL", "1").strip().lower() not in ("0", "false")

FORCE_BUS_REFRESH = os.getenv("FORCE_BUS_REFRESH", "0").strip().lower() in ("1", "true", "yes")
INCREMENTAL = os.getenv("BUS_INCREMENTAL", "1").strip().lower() not in ("0", "false")

BUS_CACHE_OK = (os.path.exists(BUS_CANON_RAW) and os.path.exists(BUS_SUMMARY_NAME))

# =========================
# 1) crime oku
# =========================
if not os.path.exists(CRIME_INPUT):
    raise FileNotFoundError(f"❌ Suç girdi dosyası yok: {CRIME_INPUT}")

crime_in = pd.read_csv(CRIME_INPUT, low_memory=False)
log_shape(crime_in, "CRIME_INPUT (okundu)")

if "GEOID" not in crime_in.columns:
    raise KeyError("❌ Suç verisinde 'GEOID' kolonu yok.")
crime_in["GEOID"] = normalize_geoid(crime_in["GEOID"], DEFAULT_GEOID_LEN)

# panel anahtarı (incremental için)
KEYS = [c for c in ["GEOID", "date", "hour_range"] if c in crime_in.columns]
if len(KEYS) < 2:
    raise KeyError(f"❌ Incremental için anahtar kolonlar yetersiz: bulundu={KEYS}. En az GEOID+date(+hour_range) olmalı.")

crime_geoids = pd.Series(crime_in["GEOID"].unique(), name="GEOID")
print(f"🧩 CRIME_INPUT farklı GEOID sayısı: {crime_geoids.size}")

# =========================
# 2) Eğer çıktı varsa: incremental mod
# =========================
crime_out_exists = os.path.exists(CRIME_OUTPUT)
bus_cols_expected = ["distance_to_bus", "bus_stop_count", "distance_to_bus_range", "bus_stop_count_range"]

crime_old = None
crime_new = None
RUN_MODE = "FULL"

if INCREMENTAL and crime_out_exists:
    print("🧠 INCREMENTAL mod açık ve sf_crime_04.csv mevcut.")
    print("🧩 Kural: eski satırlar korunur, yalnızca yeni crime satırları güncel bus snapshot ile enrich edilir.")

    crime_old = pd.read_csv(CRIME_OUTPUT, low_memory=False)
    log_shape(crime_old, "CRIME_OUTPUT (mevcut)")

    if "GEOID" not in crime_old.columns:
        raise KeyError("❌ Mevcut sf_crime_04.csv içinde GEOID yok.")
    crime_old["GEOID"] = normalize_geoid(crime_old["GEOID"], DEFAULT_GEOID_LEN)

    # mevcut çıktı bus kolonlarını tam içermiyorsa 1 defalık full backfill
    if not all(c in crime_old.columns for c in bus_cols_expected):
        print("⚠️ Mevcut çıktı bus kolonlarını tam içermiyor → FULL backfill yapılacak.")
        crime_old = None
        crime_new = None
        RUN_MODE = "FULL"

    else:
        old_keys = crime_old[KEYS].drop_duplicates()
        new_keys = crime_in[KEYS].drop_duplicates()

        marker = new_keys.merge(old_keys, on=KEYS, how="left", indicator=True)
        only_new_keys = marker.loc[marker["_merge"] == "left_only", KEYS].copy()

        n_new = len(only_new_keys)
        print(f"➕ Yeni satır anahtar sayısı: {n_new}")

        if n_new == 0:
            print("✅ Yeni crime satırı yok → bus indirme / hesaplama atlandı.")
            print("✅ Eski sf_crime_04 aynen korunuyor.")
            raise SystemExit(0)

        crime_new = crime_in.merge(only_new_keys, on=KEYS, how="inner")
        log_shape(crime_new, "CRIME_NEW (sadece yeni satırlar)")

        RUN_MODE = "INCREMENTAL"

else:
    RUN_MODE = "FULL"

# =========================
# 3) bus_feat'i hazırla (cache veya full)
# =========================
# Cache varsa ve force yoksa -> bus.csv oku (değerler stabil)
if BUS_CACHE_OK and (not FORCE_BUS_REFRESH):
    print("✅ BUS cache bulundu ve FORCE_BUS_REFRESH=0 → bus.csv kullanılacak.")
    bus_feat = pd.read_csv(BUS_SUMMARY_NAME, low_memory=False)
    if "GEOID" not in bus_feat.columns:
        raise KeyError("❌ bus.csv içinde GEOID yok (cache bozuk).")
    bus_feat["GEOID"] = normalize_geoid(bus_feat["GEOID"], DEFAULT_GEOID_LEN)

else:
    # FULL refresh: socrata indir + GEOID ata + distance/count hesapla + stable bins uygula + bus.csv yaz
    print("🚌 Otobüs durakları Socrata'dan indiriliyor…")
    bus = download_bus_with_retry(BASE, HEADERS, limit=50000, max_retries=5, backoff_base=1.7)

    if bus is None:
        if os.path.exists(BUS_CANON_RAW):
            print("⚠️ API başarısız; mevcut cache kullanılacak:", os.path.abspath(BUS_CANON_RAW))
            bus = pd.read_csv(BUS_CANON_RAW, low_memory=False)
        elif os.path.exists(BUS_LEGACY_RAW_Y):
            print("⚠️ API başarısız; legacy cache kullanılacak:", os.path.abspath(BUS_LEGACY_RAW_Y))
            bus = pd.read_csv(BUS_LEGACY_RAW_Y, low_memory=False)
        elif ALLOW_STUB:
            print("⚠️ API ve yerel cache yok → STUB (0 durak, NaN mesafe).")
            bus = pd.DataFrame(columns=["stop_lat", "stop_lon"])
        else:
            raise SystemExit("⚠️ Otobüs durakları alınamadı; cache de yok.")

    bus = extract_lat_lon(bus)
    bus = bus.dropna(subset=["stop_lat", "stop_lon"]).copy()
    log_shape(bus, "BUS (lat/lon sonrası)")

    for cand in ["stop_id", "stopid", "stop", "id"]:
        if cand in bus.columns:
            bus.rename(columns={cand: "stop_id"}, inplace=True)
            break

    census_path = next((p for p in CENSUS_CANDIDATES if os.path.exists(p)), None)
    blocks_ok = True
    if census_path is None:
        print("⚠️ Nüfus blokları GeoJSON bulunamadı. GEOID eşleme/mesafe → stub.")
        blocks_ok = False

    if blocks_ok:
        gdf_blocks = gpd.read_file(census_path)
        if gdf_blocks.crs is None:
            gdf_blocks.set_crs("EPSG:4326", inplace=True, allow_override=True)
        else:
            epsg = None
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
            print("⚠️ Block dosyasında GEOID benzeri sütun yok. Stub'a düşülecek.")
            blocks_ok = False
        else:
            gdf_blocks["GEOID"] = normalize_geoid(gdf_blocks[gcol], DEFAULT_GEOID_LEN)
    else:
        gdf_blocks = None

    if blocks_ok and not bus.empty:
        gdf_bus = gpd.GeoDataFrame(
            bus, geometry=gpd.points_from_xy(bus["stop_lon"], bus["stop_lat"]), crs="EPSG:4326"
        )
        try:
            gdf_bus = gpd.sjoin(gdf_bus, gdf_blocks[["GEOID", "geometry"]], how="left", predicate="within")
        except Exception as e:
            print(f"⚠️ sjoin(within) başarısız ({e}). sjoin_nearest(max_distance=5 m) deneniyor…")
            gdf_bus = gpd.sjoin_nearest(gdf_bus, gdf_blocks[["GEOID", "geometry"]], how="left", max_distance=5)

        gdf_bus = gdf_bus.drop(columns=["index_right"], errors="ignore")
        gdf_bus["GEOID"] = normalize_geoid(gdf_bus["GEOID"], DEFAULT_GEOID_LEN)
        bus_geo = pd.DataFrame(gdf_bus.drop(columns=["geometry"], errors="ignore")).copy()
        log_shape(bus_geo, "BUS⨯BLOCKS (GEOID atanmış)")
    else:
        bus_geo = bus.copy()
        bus_geo["GEOID"] = pd.NA

    # ham kaydet
    safe_save_csv(bus_geo, BUS_CANON_RAW)
    try:
        safe_save_csv(bus_geo, BUS_LEGACY_RAW_Y)
    except Exception as e:
        print(f"⚠️ Legacy bus_y.csv yazılamadı: {e}")

    # GEOID-level count & distance
    if blocks_ok and not bus_geo["GEOID"].isna().all():
        bus_count = (
            bus_geo.dropna(subset=["GEOID"])
                  .groupby("GEOID", as_index=False)
                  .agg(bus_stop_count=("stop_lat", "size"))
        )

        gdf_blocks_xy = gdf_blocks[["GEOID", "geometry"]].copy().to_crs(3857)
        gdf_blocks_xy["cx"] = gdf_blocks_xy.geometry.centroid.x
        gdf_blocks_xy["cy"] = gdf_blocks_xy.geometry.centroid.y

        bus_pts = bus_geo.dropna(subset=["stop_lat","stop_lon"])[["stop_lat","stop_lon"]].copy()
        gdf_bus_xy = gpd.GeoDataFrame(
            bus_pts, geometry=gpd.points_from_xy(bus_pts["stop_lon"], bus_pts["stop_lat"]), crs="EPSG:4326"
        ).to_crs(3857)

        if len(gdf_bus_xy) == 0:
            bus_dist = gdf_blocks_xy[["GEOID"]].copy()
            bus_dist["distance_to_bus"] = np.nan
        else:
            bus_coords = np.vstack([gdf_bus_xy.geometry.x.values, gdf_bus_xy.geometry.y.values]).T
            tree = cKDTree(bus_coords)
            centroids = np.vstack([gdf_blocks_xy["cx"].values, gdf_blocks_xy["cy"].values]).T
            distances, _ = tree.query(centroids, k=1)
            bus_dist = gdf_blocks_xy[["GEOID"]].copy()
            bus_dist["distance_to_bus"] = distances.astype(float)
    else:
        bus_count = pd.DataFrame(columns=["GEOID", "bus_stop_count"])
        bus_dist = pd.DataFrame({"GEOID": crime_geoids, "distance_to_bus": np.nan})

    bus_feat = pd.merge(bus_dist, bus_count, on="GEOID", how="left")
    bus_feat["bus_stop_count"] = bus_feat["bus_stop_count"].fillna(0).astype(int)
    bus_feat["GEOID"] = normalize_geoid(bus_feat["GEOID"], DEFAULT_GEOID_LEN)

    # sadece crime GEOID evreni
    bus_feat = bus_feat.merge(crime_geoids.to_frame(), on="GEOID", how="right")

    # stable bins: kaydet/uygula
    bins_json = load_bins(BUS_BINS_JSON)
    dist_range, dist_edges = make_or_apply_bins_distance(bus_feat["distance_to_bus"], bins_json)
    cnt_range,  cnt_edges  = make_or_apply_bins_count(bus_feat["bus_stop_count"], bins_json)

    bus_feat["distance_to_bus_range"] = dist_range
    bus_feat["bus_stop_count_range"]  = cnt_range

    # binleri ilk kez üretmişsek kaydet
    if not bins_json:
        save_bins({"distance_edges": dist_edges, "count_edges": cnt_edges}, BUS_BINS_JSON)
        print(f"✅ BUS bin kenarları kaydedildi → {BUS_BINS_JSON}")

    bus_feat = bus_feat.sort_values("GEOID").drop_duplicates(subset="GEOID", keep="first")
    assert bus_feat["GEOID"].is_unique, "BUS: GEOID hâlâ tekil değil!"

    safe_save_csv(bus_feat, BUS_SUMMARY_NAME)
    print(f"✅ BUS özet (GEOID-level) yazıldı → {BUS_SUMMARY_NAME}")

# =========================
# 4) MERGE: incremental ise sadece yeni satırlar; yoksa full
# =========================
def merge_bus(df: pd.DataFrame, bus_feat: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["GEOID"] = normalize_geoid(df["GEOID"], DEFAULT_GEOID_LEN)
    bus_feat = bus_feat.copy()
    bus_feat["GEOID"] = normalize_geoid(bus_feat["GEOID"], DEFAULT_GEOID_LEN)

    overlap = (set(df.columns) & set(bus_feat.columns)) - {"GEOID"}
    if overlap:
        bus_feat = bus_feat.drop(columns=list(overlap), errors="ignore")

    out = df.merge(bus_feat, on="GEOID", how="left", validate="many_to_one")
    if "bus_stop_count" in out.columns:
        out["bus_stop_count"] = pd.to_numeric(out["bus_stop_count"], errors="coerce").fillna(0).astype(int)
    return out

if RUN_MODE == "INCREMENTAL":
    before = crime_new.shape
    crime_new2 = merge_bus(crime_new, bus_feat)
    log_delta(before, crime_new2.shape, "CRIME_NEW ⨯ BUS (incremental)")
    log_shape(crime_new2, "CRIME_NEW (bus enrich)")

    # eski satırlar eski haliyle korunur
    # yeni satırlar güncel bus snapshot ile eklenir
    out = pd.concat([crime_old, crime_new2], ignore_index=True)

    # aynı key tekrar ederse yeni geleni tut
    out = out.drop_duplicates(subset=KEYS, keep="last")

    safe_save_csv(out, CRIME_OUTPUT)
    print(f"✅ INCREMENTAL güncelleme tamam → {CRIME_OUTPUT}")

else:
    before = crime_in.shape
    out = merge_bus(crime_in, bus_feat)
    log_delta(before, out.shape, "CRIME ⨯ BUS (FULL)")
    log_shape(out, "CRIME (bus enrich sonrası - FULL)")

    safe_save_csv(out, CRIME_OUTPUT)
    print(f"✅ FULL çıktı → {CRIME_OUTPUT}")

# örnek
try:
    print("sf_crime_04 — ilk 5 satır")
    print(out.head(5).to_string(index=False))
except Exception:
    pass
