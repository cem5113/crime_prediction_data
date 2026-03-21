# pipeline_make_sf_crime_06.py
# -----------------------------------------------------------------------------
# POI ENRICH — INCREMENTAL / APPEND-ONLY REVIZE
#
# SÖZLEŞME:
# 1) sf_crime_05.csv güncel kaynak kabul edilir.
# 2) sf_crime_06.csv varsa:
#    - eski satırlar korunur
#    - yalnızca yeni tarihli satırlar POI ile zenginleştirilir
#    - eski satırlar overwrite edilmez
# 3) POI risk skoru hesaplanırken (opsiyonel) son 5 yıllık suç verisi kullanılabilir,
#    fakat çıkan feature yalnızca yeni satırlara yazılır.
# 4) Merge mantığı GEOID bazlıdır; satır granülerliği GEOID × date × hour_range'dir.
# 5) POI geojson / clean CSV cache mantığı korunur.
# -----------------------------------------------------------------------------

import os
import ast
import json
import time
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import geopandas as gpd
import requests
from sklearn.neighbors import BallTree

try:
    from shapely.strtree import STRtree
except Exception:
    STRtree = None


# =============================================================================
# LOG HELPERS
# =============================================================================
def log_shape(df, label):
    r, c = df.shape
    print(f"📊 {label}: {r} satır × {c} sütun")


def log_delta(before_shape, after_shape, label):
    br, bc = before_shape
    ar, ac = after_shape
    print(f"🔗 {label}: {br}×{bc} → {ar}×{ac} (Δr={ar-br}, Δc={ac-bc})")


# =============================================================================
# 0) YOLLAR / ENV
# =============================================================================
BASE_DIR = os.getenv("CRIME_DATA_DIR", "crime_prediction_data")
Path(BASE_DIR).mkdir(parents=True, exist_ok=True)

POI_GEOJSON_1 = os.path.join(BASE_DIR, "sf_pois.geojson")
POI_GEOJSON_2 = os.path.join(".", "sf_pois.geojson")

BLOCK_PATH_1 = os.path.join(BASE_DIR, "sf_census_blocks_with_population.geojson")
BLOCK_PATH_2 = os.path.join(".", "sf_census_blocks_with_population.geojson")

POI_CLEAN_CSV = os.path.join(BASE_DIR, "sf_pois_cleaned_with_geoid.csv")
POI_RISK_JSON = os.path.join(BASE_DIR, "risky_pois_dynamic.json")
POI_GEOID_SUMMARY_CSV = os.path.join(BASE_DIR, "poi_geoid_summary.csv")

CRIME_IN = os.getenv("CRIME_IN", os.path.join(BASE_DIR, "sf_crime_05.csv"))
CRIME_OUT = os.getenv("CRIME_OUT", os.path.join(BASE_DIR, "sf_crime_06.csv"))

FORCE_POI_REFRESH = os.getenv("FORCE_POI_REFRESH", "0").strip().lower() in ("1", "true", "yes")
POI_RISK_LOOKBACK_YEARS = int(os.getenv("POI_RISK_LOOKBACK_YEARS", "5"))
INCLUDE_OFFICE_CRAFT = os.getenv("INCLUDE_OFFICE_CRAFT", "1").strip().lower() not in ("0", "false", "no")

# Eğer çıktı varsa, append-only mantık gereği eski dosyayı referans alacağız.
USE_EXISTING_OUTPUT_IF_ANY = True

# Anahtar kolonlar
PANEL_KEYS = ["GEOID", "date", "hour_range"]


# =============================================================================
# YARDIMCI FONKSİYONLAR
# =============================================================================
def _first_exists(*paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def _ensure_parent(path: str):
    Path(os.path.dirname(path) or ".").mkdir(parents=True, exist_ok=True)


def _safe_save_csv(df: pd.DataFrame, path: str):
    _ensure_parent(path)
    tmp = path + ".tmp"
    try:
        df.to_csv(tmp, index=False, encoding="utf-8-sig")
        os.replace(tmp, path)
        print(f"💾 Kaydedildi: {path}")
    except Exception as e:
        print(f"❌ Kaydetme hatası: {path}\n{e}")
        try:
            df.to_csv(path + ".bak", index=False, encoding="utf-8-sig")
            print(f"📁 Yedek oluşturuldu: {path}.bak")
        except Exception:
            pass


def _bbox_ok_sf(features, min_lon=-123.5, max_lon=-121.5, min_lat=37.0, max_lat=38.5):
    lons, lats = [], []
    for f in features[:min(200, len(features))]:
        try:
            lon, lat = f["geometry"]["coordinates"]
            lons.append(float(lon))
            lats.append(float(lat))
        except Exception:
            pass
    if not lons or not lats:
        return False
    return (min(lons) > min_lon and max(lons) < max_lon and min(lats) > min_lat and max(lats) < max_lat)


def _read_geojson_robust(path: str) -> gpd.GeoDataFrame:
    if path is None or not os.path.exists(path):
        raise FileNotFoundError(f"GeoJSON yok: {path}")

    try:
        gdf = gpd.read_file(path)
        return _ensure_crs(gdf, "EPSG:4326")
    except Exception as e:
        print(f"⚠️ gpd.read_file başarısız ({path}): {e}")
        txt = Path(path).read_text(encoding="utf-8", errors="ignore").strip()
        gj = None
        try:
            if "\n" in txt and txt.splitlines()[0].strip().startswith("{") and '"features"' not in txt:
                feats = [json.loads(line) for line in txt.splitlines() if line.strip()]
                gj = {"type": "FeatureCollection", "features": feats}
            else:
                gj = json.loads(txt)
        except Exception as e2:
            raise ValueError(f"GeoJSON parse edilemedi: {e2}")

        if "features" not in gj:
            raise ValueError("GeoJSON FeatureCollection bekleniyordu (features yok).")

        gdf = gpd.GeoDataFrame.from_features(gj["features"], crs="EPSG:4326")
        return _ensure_crs(gdf, "EPSG:4326")


def _ensure_crs(gdf, target="EPSG:4326"):
    if gdf.crs is None:
        return gdf.set_crs(target, allow_override=True)

    s = (gdf.crs.to_string() if hasattr(gdf.crs, "to_string") else str(gdf.crs)).upper()
    if s.endswith("CRS84"):
        return gdf.set_crs("EPSG:4326", allow_override=True)
    if s != target:
        return gdf.to_crs(target)
    return gdf


def _parse_tags(val):
    if isinstance(val, dict):
        return val
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
    for key in ("amenity", "shop", "leisure", "tourism", "office", "craft"):
        if key in tags and tags[key]:
            return key, tags[key], name
    return None, None, name


def _normalize_geoid(series: pd.Series, target_len: int = 11) -> pd.Series:
    s = pd.Series(series).astype(str).str.extract(r"(\d+)", expand=False).fillna("")
    s = s.str[:target_len].str.zfill(target_len)
    s = s.mask(s.eq("0" * target_len))
    return s


def _pick_existing(*paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def _ensure_datetime_date_col(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    if date_col not in out.columns:
        raise KeyError(f"❌ '{date_col}' kolonu yok.")
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce").dt.normalize()
    return out


def _drop_existing_poi_columns(df: pd.DataFrame) -> pd.DataFrame:
    poi_cols = [
        "poi_total_count",
        "poi_risk_score",
        "poi_dominant_type",
        "poi_total_count_range",
        "poi_risk_score_range",
    ]
    return df.drop(columns=[c for c in poi_cols if c in df.columns], errors="ignore")


def _value_range_label(x, edges, prefix="Q"):
    if pd.isna(x):
        if len(edges) >= 2:
            return f"{prefix}1 ({edges[0]:.1f}-{edges[1]:.1f})"
        return f"{prefix}1 (0-0)"
    for i in range(len(edges) - 1):
        if x <= edges[i + 1]:
            return f"{prefix}{i+1} ({edges[i]:.1f}-{edges[i+1]:.1f})"
    return f"{prefix}{len(edges)-1} ({edges[-2]:.1f}-{edges[-1]:.1f})"


def _make_dynamic_labels(series: pd.Series, bin_count=5):
    vals = pd.to_numeric(series, errors="coerce").dropna().values
    if vals.size == 0:
        def lab(_):
            return "Q1 (0-0)"
        return lab

    qs = np.quantile(vals, [i / bin_count for i in range(bin_count + 1)])

    def lab(x):
        if pd.isna(x):
            return f"Q1 ({qs[0]:.1f}-{qs[1]:.1f})"
        for i in range(bin_count):
            if x <= qs[i + 1]:
                return f"Q{i+1} ({qs[i]:.1f}-{qs[i+1]:.1f})"
        return f"Q{bin_count} ({qs[-2]:.1f}-{qs[-1]:.1f})"

    return lab


# =============================================================================
# 1) POI GEOJSON CACHE / DOWNLOAD
# =============================================================================
def ensure_sf_pois_geojson(
    out_path: str,
    include_office_craft: bool = True,
    force_refresh: bool = False,
    fallback_to_existing: bool = True,
):
    existing_ok = False
    if os.path.exists(out_path):
        try:
            gj0 = json.loads(Path(out_path).read_text(encoding="utf-8", errors="ignore"))
            feats0 = gj0.get("features", [])
            existing_ok = bool(feats0) and _bbox_ok_sf(feats0)
            if existing_ok and not force_refresh:
                print("✅ sf_pois.geojson mevcut ve SF bbox OK. (cache) → indirilmeyecek.")
                return out_path
            if existing_ok and force_refresh:
                print("♻️ force_refresh=True → SF POI yeniden indirilecek (hata olursa fallback var).")
            if not existing_ok:
                print("⚠️ sf_pois.geojson var ama SF dışı/bozuk → yeniden indirilecek.")
        except Exception as e:
            print(f"⚠️ sf_pois.geojson okunamadı ({e}) → yeniden indirilecek.")

    extra = ""
    if include_office_craft:
        extra = """
      nwr(area.sf)["office"];
      nwr(area.sf)["craft"];
    """

    query = f"""
    [out:json][timeout:600];
    area(3600111968)->.sf;
    (
      nwr(area.sf)["amenity"];
      nwr(area.sf)["shop"];
      nwr(area.sf)["leisure"];
      nwr(area.sf)["tourism"];
      {extra}
    );
    out center tags;
    """

    endpoints = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
        "https://overpass.openstreetmap.ru/api/interpreter",
    ]

    def overpass_post(q, eps, max_retries=4, backoff_base=1.8):
        headers = {"Accept": "application/json"}
        last_err = None
        for ep in eps:
            print(f"🌐 Overpass deneniyor: {ep}")
            for attempt in range(max_retries + 1):
                try:
                    r = requests.post(ep, data={"data": q}, headers=headers, timeout=900)

                    if r.status_code in (429,) or 500 <= r.status_code < 600:
                        if attempt >= max_retries:
                            r.raise_for_status()
                        time.sleep(backoff_base ** (attempt + 1))
                        continue

                    r.raise_for_status()
                    js = r.json()
                    if "elements" not in js:
                        raise ValueError("Overpass yanıtında 'elements' yok.")
                    return js

                except Exception as e:
                    last_err = e
                    if attempt >= max_retries:
                        print(f"❌ Endpoint başarısız: {ep} | hata: {e}")
                    else:
                        time.sleep(backoff_base ** (attempt + 1))
            print(f"↪️ Endpoint geçiliyor: {ep}")
        raise RuntimeError(f"❌ Tüm Overpass endpointleri başarısız oldu. Son hata: {last_err}")

    try:
        raw = overpass_post(query, endpoints)

        feats = []
        for el in raw.get("elements", []):
            tags = el.get("tags", {}) or {}

            if el.get("type") == "node":
                lat, lon = el.get("lat"), el.get("lon")
            else:
                c = el.get("center") or {}
                lat, lon = c.get("lat"), c.get("lon")

            if lat is None or lon is None:
                continue

            feats.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [float(lon), float(lat)]},
                "properties": {
                    "id": f'{el.get("type")}/{el.get("id")}',
                    "osm_type": el.get("type"),
                    "osm_id": el.get("id"),
                    "tags": tags,
                }
            })

        if (not feats) or (not _bbox_ok_sf(feats)):
            raise RuntimeError("❌ İndirilen POI SF bbox dışında (sanity fail).")

        gj = {"type": "FeatureCollection", "features": feats}
        _ensure_parent(out_path)
        Path(out_path).write_text(json.dumps(gj, ensure_ascii=False), encoding="utf-8")

        print(f"✅ SF POI indirildi: {out_path} | n={len(feats)}")
        return out_path

    except Exception as e:
        print(f"❌ Overpass indirme başarısız: {e}")

        if fallback_to_existing and os.path.exists(out_path):
            try:
                gj1 = json.loads(Path(out_path).read_text(encoding="utf-8", errors="ignore"))
                feats1 = gj1.get("features", [])
                if feats1 and _bbox_ok_sf(feats1):
                    print("🛡️ Fallback: mevcut sf_pois.geojson kullanılacak.")
                    return out_path
            except Exception:
                pass
        raise


# =============================================================================
# 2) POI CLEAN + GEOID BUILD
# =============================================================================
def build_poi_clean_with_geoid(blocks_path: str, poi_geojson_path: str) -> pd.DataFrame:
    print("📍 POI okunuyor → kategoriler çıkarılıyor → GEOID atanıyor...")

    if poi_geojson_path is None or not os.path.exists(poi_geojson_path):
        raise FileNotFoundError("❌ POI GeoJSON bulunamadı.")

    gdf = _read_geojson_robust(poi_geojson_path)

    if "tags" not in gdf.columns:
        gdf["tags"] = [{}] * len(gdf)
    gdf["tags"] = gdf["tags"].apply(_parse_tags)

    triples = gdf["tags"].apply(_extract_cat_sub_name)
    gdf[["poi_category", "poi_subcategory", "poi_name"]] = pd.DataFrame(triples.tolist(), index=gdf.index)

    if "geometry" not in gdf.columns:
        if {"lon", "lat"}.issubset(gdf.columns):
            gdf["geometry"] = gpd.points_from_xy(gdf["lon"], gdf["lat"])
        else:
            raise ValueError("GeoJSON 'geometry' veya 'lon/lat' içermiyor.")

    gdf = _ensure_crs(gdf, "EPSG:4326")
    gdf["lon"] = gdf.get("lon", pd.Series(index=gdf.index, dtype=float)).fillna(gdf.geometry.x)
    gdf["lat"] = gdf.get("lat", pd.Series(index=gdf.index, dtype=float)).fillna(gdf.geometry.y)

    if blocks_path is None or not os.path.exists(blocks_path):
        raise FileNotFoundError("❌ Nüfus blokları GeoJSON bulunamadı.")

    blocks = _read_geojson_robust(blocks_path)
    if "GEOID" not in blocks.columns:
        raise ValueError("Block dosyasında 'GEOID' yok.")

    blocks["GEOID"] = _normalize_geoid(blocks["GEOID"], 11)

    try:
        joined = gpd.sjoin(gdf, blocks[["GEOID", "geometry"]], how="left", predicate="within")
    except Exception as e:
        print("⚠️ gpd.sjoin başarısız, STRtree fallback →", e)
        if STRtree is None:
            raise RuntimeError("Shapely STRtree yok. 'shapely>=2.0' kurun veya rtree yükleyin.")

        geoms = list(blocks.geometry.values)
        tree = STRtree(geoms)
        geom_id_to_geoid = {id(g): geoid for g, geoid in zip(geoms, blocks["GEOID"])}

        geoid_list = []
        for pt in gdf.geometry.values:
            try:
                cands = tree.query(pt, predicate="contains")
            except TypeError:
                cands = [g for g in tree.query(pt) if g.contains(pt)]

            geoid_list.append(geom_id_to_geoid[id(cands[0])] if cands else None)

        joined = gdf.copy()
        joined["GEOID"] = geoid_list

    keep = [c for c in ["id", "lat", "lon", "poi_category", "poi_subcategory", "poi_name", "GEOID"] if c in joined.columns]
    df = joined[keep].copy()

    if "id" not in df.columns:
        df["id"] = np.arange(len(df))

    df = df.dropna(subset=["lat", "lon"]).copy()
    df["GEOID"] = _normalize_geoid(df["GEOID"], 11)

    before_n = len(df)
    df = df.dropna(subset=["GEOID"]).copy()
    if len(df) != before_n:
        print(f"⚠️ GEOID eşlenemeyen POI drop: {before_n - len(df)} / {before_n}")

    _safe_save_csv(df, POI_CLEAN_CSV)

    print(f"✅ Kaydedildi: {POI_CLEAN_CSV} | Satır: {len(df):,}")
    try:
        print(df.head(5).to_string(index=False))
    except Exception:
        pass

    return df


# =============================================================================
# 3) DİNAMİK POI RISK
# =============================================================================
def compute_dynamic_poi_risk(df_crime: pd.DataFrame, df_poi: pd.DataFrame, radius_m=300) -> dict:
    """
    POI alt-kategorileri için çevresindeki suç yoğunluğuna göre 0–3 arası skor.
    Bu sürümde risk hesabı dinamik olabilir.
    Ancak bu risk sonucu sadece yeni satırlara yazılacaktır.
    """

    dfp = df_poi.copy()
    dfp["lat"] = pd.to_numeric(dfp.get("lat"), errors="coerce")
    dfp["lon"] = pd.to_numeric(dfp.get("lon"), errors="coerce")
    dfp = dfp.dropna(subset=["lat", "lon"]).copy()

    if "poi_subcategory" in dfp.columns:
        dfp = dfp[~dfp["poi_subcategory"].isin(["police", "ranger_station"])]

    dfc = df_crime.copy()

    lat_candidates = ["latitude", "lat", "y"]
    lon_candidates = ["longitude", "lon", "long", "x"]

    lat_col = next((c for c in lat_candidates if c in dfc.columns), None)
    lon_col = next((c for c in lon_candidates if c in dfc.columns), None)

    if lat_col is None or lon_col is None:
        print("⚠️ Suç verisinde latitude/longitude bulunamadı. Risk sözlüğü boş üretilecek.")
        _ensure_parent(POI_RISK_JSON)
        with open(POI_RISK_JSON, "w", encoding="utf-8") as f:
            json.dump({}, f, indent=2, ensure_ascii=False)
        return {}

    dfc["latitude"] = pd.to_numeric(dfc[lat_col], errors="coerce")
    dfc["longitude"] = pd.to_numeric(dfc[lon_col], errors="coerce")
    dfc = dfc.dropna(subset=["latitude", "longitude"]).copy()

    print(f"POI noktaları: {len(dfp):,} | Suç noktaları: {len(dfc):,}")

    if dfc.empty or dfp.empty:
        print("⚠️ Risk için yeterli nokta yok. Boş sözlük yazılacak.")
        _ensure_parent(POI_RISK_JSON)
        with open(POI_RISK_JSON, "w", encoding="utf-8") as f:
            json.dump({}, f, indent=2, ensure_ascii=False)
        return {}

    crime_rad = np.radians(dfc[["latitude", "longitude"]].values)
    poi_rad = np.radians(dfp[["lat", "lon"]].values)

    tree = BallTree(crime_rad, metric="haversine")
    r = radius_m / 6371000.0

    poi_types = dfp["poi_subcategory"].fillna("")
    counts = []

    for pt, t in zip(poi_rad, poi_types):
        if not t:
            continue
        idx = tree.query_radius([pt], r=r)[0]
        counts.append((t, len(idx)))

    if not counts:
        _ensure_parent(POI_RISK_JSON)
        with open(POI_RISK_JSON, "w", encoding="utf-8") as f:
            json.dump({}, f, indent=2, ensure_ascii=False)
        return {}

    agg = defaultdict(list)
    for t, c in counts:
        agg[t].append(c)

    avg = {t: float(np.mean(v)) for t, v in agg.items()}

    vals = list(avg.values())
    vmin, vmax = min(vals), max(vals)

    if vmax - vmin < 1e-9:
        norm = {t: 1.5 for t in avg}
    else:
        norm = {t: round(3 * (x - vmin) / (vmax - vmin), 2) for t, x in avg.items()}

    _ensure_parent(POI_RISK_JSON)
    with open(POI_RISK_JSON, "w", encoding="utf-8") as f:
        json.dump(norm, f, indent=2, ensure_ascii=False)

    print("🔝 İlk 15 alt-kategori (skora göre):")
    for k, s in sorted(norm.items(), key=lambda x: -x[1])[:15]:
        print(f"  {k:<24} → {s:.2f}")

    return norm


# =============================================================================
# 4) GEOID DÜZEYİ POI ÖZET
# =============================================================================
def build_geoid_level_poi_features(df_poi: pd.DataFrame, poi_risk: dict) -> pd.DataFrame:
    """
    GEOID bazında:
      - poi_total_count
      - poi_risk_score
      - poi_dominant_type
      - range kolonları
    Not:
      Bu summary her çalışmada yeniden üretilebilir.
      Ancak append-only mantığı gereği sadece yeni satırlara yazılır.
    """

    dfp = df_poi.copy()
    dfp["GEOID"] = _normalize_geoid(dfp["GEOID"], 11) if "GEOID" in dfp.columns else pd.NA

    sub = dfp.get("poi_subcategory", "").astype(str)
    dfp["__risk__"] = sub.map(poi_risk).fillna(0.0)

    def _mode(arr):
        arr = [a for a in arr if pd.notna(a) and a != ""]
        if not arr:
            return "No_POI"
        c = Counter(arr)
        return c.most_common(1)[0][0]

    grp = dfp.groupby("GEOID", dropna=False)

    out = pd.DataFrame({
        "GEOID": grp.size().index,
        "poi_total_count": grp.size().values,
        "poi_risk_score": grp["__risk__"].sum().values,
        "poi_dominant_type": grp["poi_subcategory"].agg(_mode).values,
    })

    # Dinamik label burada kalabilir; çünkü eski satırlar overwrite edilmeyecek.
    lab_cnt = _make_dynamic_labels(out["poi_total_count"])
    lab_risk = _make_dynamic_labels(out["poi_risk_score"])

    out["poi_total_count_range"] = out["poi_total_count"].apply(lab_cnt)
    out["poi_risk_score_range"] = out["poi_risk_score"].apply(lab_risk)

    out["GEOID"] = _normalize_geoid(out["GEOID"], 11)
    out = out.dropna(subset=["GEOID"]).copy()
    out = out.sort_values("GEOID").drop_duplicates(subset=["GEOID"], keep="first")

    _safe_save_csv(out, POI_GEOID_SUMMARY_CSV)
    log_shape(out, "POI (GEOID-özet)")

    return out


# =============================================================================
# 5) SUÇA GEOID İLE POI MERGE
# =============================================================================
def enrich_crime_by_geoid(df_crime: pd.DataFrame, geoid_poi: pd.DataFrame) -> pd.DataFrame:
    out = df_crime.copy()
    out["GEOID"] = _normalize_geoid(out.get("GEOID"), 11)

    out = _drop_existing_poi_columns(out)

    before = out.shape

    _overlap = (set(out.columns) & set(geoid_poi.columns)) - {"GEOID"}
    if _overlap:
        print(f"🧹 POI merge overlap bulundu, geoid_poi'den düşürüldü: {sorted(_overlap)}")
        geoid_poi = geoid_poi.drop(columns=list(_overlap), errors="ignore")

    out = out.merge(geoid_poi, on="GEOID", how="left")

    out = out.fillna({
        "poi_total_count": 0,
        "poi_risk_score": 0.0,
        "poi_dominant_type": "No_POI",
        "poi_total_count_range": "Q1 (0-0)",
        "poi_risk_score_range": "Q1 (0-0)",
    })

    out["poi_total_count"] = pd.to_numeric(out["poi_total_count"], errors="coerce").fillna(0).astype(int)
    out["poi_risk_score"] = pd.to_numeric(out["poi_risk_score"], errors="coerce").fillna(0.0).astype(float)
    out["poi_dominant_type"] = out["poi_dominant_type"].astype(str)

    log_delta(before, out.shape, "CRIME ⨯ POI (GEOID-merge)")
    return out


# =============================================================================
# 6) APPEND-ONLY / NEW ROW TESPİTİ
# =============================================================================
def split_existing_and_new_rows(df_in: pd.DataFrame, existing_out_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Eğer sf_crime_06.csv mevcutsa:
      - eski satırları korur
      - df_in içindeki yeni satırları bulur
    Yeni satır tespiti PANEL_KEYS üzerinden yapılır.
    """

    if not os.path.exists(existing_out_path):
        print("ℹ️ Eski sf_crime_06.csv yok → tüm giriş satırları yeni kabul edilecek.")
        return pd.DataFrame(columns=df_in.columns), df_in.copy()

    old = pd.read_csv(existing_out_path, low_memory=False)
    if not set(PANEL_KEYS).issubset(old.columns):
        print("⚠️ Eski çıktı dosyasında panel anahtarları eksik → güvenli tarafta kalıp tüm girişi yeni kabul ediyorum.")
        return old, df_in.copy()

    old = old.copy()
    old["GEOID"] = _normalize_geoid(old["GEOID"], 11)
    old = _ensure_datetime_date_col(old, "date")

    cur = df_in.copy()
    cur["GEOID"] = _normalize_geoid(cur["GEOID"], 11)
    cur = _ensure_datetime_date_col(cur, "date")

    old_keys = old[PANEL_KEYS].drop_duplicates().copy()
    old_keys["__seen__"] = 1

    marked = cur.merge(old_keys, on=PANEL_KEYS, how="left")
    new_rows = marked[marked["__seen__"].isna()].drop(columns=["__seen__"]).copy()

    print(f"🧠 Eski çıktı bulundu: {existing_out_path}")
    log_shape(old, "MEVCUT sf_crime_06")
    log_shape(cur, "GÜNCEL sf_crime_05")
    log_shape(new_rows, "YENİ SATIRLAR (POI enrich edilecek)")

    return old, new_rows


def restrict_risk_history_window(df_crime_full: pd.DataFrame, reference_new_rows: pd.DataFrame, years: int = 5) -> pd.DataFrame:
    """
    POI risk skoru üretiminde kullanılacak suç tarih aralığını kısıtlar.
    Hesap evreni = son X yıl olabilir.
    Yazım evreni = yalnızca yeni satırlar.
    """

    full = df_crime_full.copy()
    full = _ensure_datetime_date_col(full, "date")

    if reference_new_rows.empty:
        return full

    ref = _ensure_datetime_date_col(reference_new_rows.copy(), "date")
    max_new_date = ref["date"].max()

    if pd.isna(max_new_date):
        return full

    min_allowed = max_new_date - pd.DateOffset(years=years)
    sliced = full[(full["date"] >= min_allowed) & (full["date"] <= max_new_date)].copy()

    print(f"🕒 POI risk lookback window: {min_allowed.date()} → {max_new_date.date()} | years={years}")
    log_shape(sliced, "RISK HESAP EVRENİ (son X yıl)")

    return sliced


def finalize_append_only_output(old_out: pd.DataFrame, new_enriched: pd.DataFrame) -> pd.DataFrame:
    """
    Eski çıktı + yeni enrich edilmiş satırlar birleştirilir.
    Eski satırlar değiştirilmez.
    """

    if old_out is None or len(old_out) == 0:
        final = new_enriched.copy()
    else:
        final = pd.concat([old_out, new_enriched], ignore_index=True)

    final["GEOID"] = _normalize_geoid(final.get("GEOID"), 11)

    if "date" in final.columns:
        final["date"] = pd.to_datetime(final["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    # Duplikasyon güvenliği
    if set(PANEL_KEYS).issubset(final.columns):
        before = len(final)
        final = final.drop_duplicates(subset=PANEL_KEYS, keep="first").copy()
        dropped = before - len(final)
        if dropped > 0:
            print(f"⚠️ PANEL_KEYS bazında {dropped} duplikasyon temizlendi (keep='first').")

    # Sıralama
    sort_cols = [c for c in ["date", "GEOID", "hour_range"] if c in final.columns]
    if sort_cols:
        final = final.sort_values(sort_cols).reset_index(drop=True)

    return final


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("🚀 Başlıyor (POI Enrich — Incremental / Append-Only Revize)...")

    # -------------------------------------------------------------------------
    # A) GÜNCEL GİRDİYİ OKU
    # -------------------------------------------------------------------------
    if not os.path.exists(CRIME_IN):
        candidates = [
            CRIME_IN,
            os.path.join(BASE_DIR, "sf_crime_05.csv"),
            os.path.join(BASE_DIR, "sf_crime_03.csv"),
            os.path.join(BASE_DIR, "sf_crime_02.csv"),
            os.path.join(BASE_DIR, "sf_crime.csv"),
            "sf_crime_05.csv",
            "sf_crime_03.csv",
            "sf_crime_02.csv",
            "sf_crime.csv",
        ]
        resolved = _first_exists(*candidates)
        if resolved:
            print(f"ℹ️ CRIME_IN bulunamadı, ilk mevcut aday seçildi → {resolved}")
            CRIME_IN = resolved
        else:
            raise FileNotFoundError(f"❌ Suç girdisi bulunamadı. Denenenler: {candidates}")

    df_crime_full = pd.read_csv(CRIME_IN, low_memory=False)
    if "GEOID" not in df_crime_full.columns:
        raise KeyError("❌ Suç verisinde GEOID yok.")
    if "date" not in df_crime_full.columns:
        raise KeyError("❌ Suç verisinde date yok.")

    df_crime_full["GEOID"] = _normalize_geoid(df_crime_full["GEOID"], 11)
    df_crime_full = _ensure_datetime_date_col(df_crime_full, "date")
    log_shape(df_crime_full, "CRIME FULL (sf_crime_05)")

    # -------------------------------------------------------------------------
    # B) ESKİ ÇIKTI VARSA YENİ SATIRLARI AYIR
    # -------------------------------------------------------------------------
    if USE_EXISTING_OUTPUT_IF_ANY:
        old_out, new_rows = split_existing_and_new_rows(df_crime_full, CRIME_OUT)
    else:
        old_out = pd.DataFrame(columns=df_crime_full.columns)
        new_rows = df_crime_full.copy()

    if new_rows.empty:
        print("✅ Yeni satır yok. Eski sf_crime_06 olduğu gibi korunuyor.")
        if os.path.exists(CRIME_OUT):
            print(f"📁 Mevcut çıktı korunuyor: {CRIME_OUT}")
        else:
            print("⚠️ Eski çıktı da yok; yazılacak veri bulunamadı.")
        raise SystemExit(0)

    # -------------------------------------------------------------------------
    # C) POI GEOJSON / BLOCK PATH
    # -------------------------------------------------------------------------
    blocks_path = _pick_existing(BLOCK_PATH_1, BLOCK_PATH_2)
    poi_geojson = _pick_existing(POI_GEOJSON_1, POI_GEOJSON_2)
    if poi_geojson is None:
        poi_geojson = os.path.join(BASE_DIR, "sf_pois.geojson")

    poi_geojson = ensure_sf_pois_geojson(
        poi_geojson,
        include_office_craft=INCLUDE_OFFICE_CRAFT,
        force_refresh=FORCE_POI_REFRESH,
        fallback_to_existing=True,
    )

    # -------------------------------------------------------------------------
    # D) POI CLEAN DOSYASI
    # -------------------------------------------------------------------------
    use_clean = os.path.exists(POI_CLEAN_CSV) and (not FORCE_POI_REFRESH)

    if use_clean:
        try:
            gj = json.loads(Path(poi_geojson).read_text(encoding="utf-8", errors="ignore"))
            feats = gj.get("features", [])
            if (not feats) or (not _bbox_ok_sf(feats)):
                print("⚠️ POI GeoJSON SF bbox dışında görünüyor → POI_CLEAN_CSV yeniden üretilecek.")
                use_clean = False
        except Exception as e:
            print(f"⚠️ POI GeoJSON okunamadı ({e}) → POI_CLEAN_CSV yeniden üretilecek.")
            use_clean = False

    if use_clean:
        print("ℹ️ Var olan temiz POI CSV kullanılacak:", POI_CLEAN_CSV)
        df_poi = pd.read_csv(POI_CLEAN_CSV, low_memory=False)

        if "lat" not in df_poi.columns and "latitude" in df_poi.columns:
            df_poi["lat"] = pd.to_numeric(df_poi["latitude"], errors="coerce")
        if "lon" not in df_poi.columns and "longitude" in df_poi.columns:
            df_poi["lon"] = pd.to_numeric(df_poi["longitude"], errors="coerce")
        if "poi_subcategory" not in df_poi.columns:
            guess = df_poi.get("poi_category", "Unknown")
            df_poi["poi_subcategory"] = guess.astype(str)

        df_poi["GEOID"] = _normalize_geoid(df_poi.get("GEOID"), 11)
    else:
        if blocks_path is None:
            raise FileNotFoundError("❌ sf_census_blocks_with_population.geojson yok. GEOID atamak için gerekli.")
        df_poi = build_poi_clean_with_geoid(blocks_path, poi_geojson)

    log_shape(df_poi, "POI clean")

    # -------------------------------------------------------------------------
    # E) RİSK HESAP EVRENİ = SON 5 YIL (veya env)
    # -------------------------------------------------------------------------
    risk_crime_universe = restrict_risk_history_window(
        df_crime_full=df_crime_full,
        reference_new_rows=new_rows,
        years=POI_RISK_LOOKBACK_YEARS,
    )

    # -------------------------------------------------------------------------
    # F) DİNAMİK RISK SÖZLÜĞÜ
    # -------------------------------------------------------------------------
    try:
        risk_dict = compute_dynamic_poi_risk(risk_crime_universe, df_poi, radius_m=300)
    except Exception as e:
        print(f"⚠️ Risk sözlüğü üretilemedi: {e}; boş sözlük kullanılacak.")
        risk_dict = {}

    print(f"🧪 Risk sözlüğü boyutu: {len(risk_dict)} alt-kategori")

    # -------------------------------------------------------------------------
    # G) GEOID DÜZEYİNDE POI FEATURE
    # -------------------------------------------------------------------------
    geoid_poi = build_geoid_level_poi_features(df_poi, risk_dict)

    # -------------------------------------------------------------------------
    # H) SADECE YENİ SATIRLARI ENRICH ET
    # -------------------------------------------------------------------------
    before_enrich = new_rows.shape
    new_rows_enriched = enrich_crime_by_geoid(new_rows, geoid_poi)
    log_delta(before_enrich, new_rows_enriched.shape, "YENİ SATIRLAR ⨯ POI")

    # -------------------------------------------------------------------------
    # I) ESKİ + YENİ BİRLEŞTİR (APPEND-ONLY)
    # -------------------------------------------------------------------------
    final_df = finalize_append_only_output(old_out, new_rows_enriched)
    log_shape(final_df, "FINAL sf_crime_06")

    # -------------------------------------------------------------------------
    # J) NaN RAPORU
    # -------------------------------------------------------------------------
    nan_counts = final_df.isna().sum()
    nan_counts = nan_counts[nan_counts > 0].sort_values(ascending=False)

    print("🔎 NaN sayıları (sf_crime_06 yazılmadan önce):")
    if nan_counts.empty:
        print("✅ NaN yok.")
    else:
        print(nan_counts.to_string())

    poi_cols = [
        "poi_total_count",
        "poi_risk_score",
        "poi_dominant_type",
        "poi_total_count_range",
        "poi_risk_score_range",
    ]
    poi_cols = [c for c in poi_cols if c in final_df.columns]
    if poi_cols:
        print("🔎 POI kolonları NaN sayıları:")
        print(final_df[poi_cols].isna().sum().to_string())

    # -------------------------------------------------------------------------
    # K) KAYDET
    # -------------------------------------------------------------------------
    final_to_save = final_df.copy()
    if "date" in final_to_save.columns:
        final_to_save["date"] = pd.to_datetime(final_to_save["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    _safe_save_csv(final_to_save, CRIME_OUT)
    print(f"✅ Yazıldı: {CRIME_OUT} | Satır: {len(final_to_save):,}")

    try:
        cols = [c for c in ["GEOID", "date", "hour_range", "poi_total_count", "poi_risk_score", "poi_dominant_type"] if c in final_to_save.columns]
        preview = final_to_save[cols].tail(10) if cols else final_to_save.tail(10)
        print("📌 Son 10 satır önizleme:")
        print(preview.to_string(index=False))
    except Exception as e:
        print(f"(info) Örnek yazdırılamadı: {e}")

    try:
        preview_file = pd.read_csv(CRIME_OUT, nrows=3, low_memory=False)
        print(f"{CRIME_OUT} — ilk 3 satır")
        print(preview_file.to_string(index=False))
    except Exception as e:
        print(f"(info) Kaydedilen dosya önizlemesi okunamadı: {e}")
