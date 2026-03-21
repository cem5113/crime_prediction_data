# scripts/update_bus.py
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import requests
from sklearn.neighbors import BallTree


# ============================================================
# AYARLAR
# ============================================================

SAVE_DIR = Path(os.getenv("CRIME_DATA_DIR", "crime_prediction_data"))
SAVE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_GEOID_LEN = int(os.getenv("GEOID_LEN", "11"))

# Girdi / çıktı
CRIME_INPUT_NAME = os.getenv("CRIME_INPUT_NAME", "sf_crime_03.csv")
CRIME_OUTPUT_NAME = os.getenv("CRIME_OUTPUT_NAME", "sf_crime_04.csv")

CRIME_INPUT_PATH = SAVE_DIR / CRIME_INPUT_NAME
CRIME_OUTPUT_PATH = SAVE_DIR / CRIME_OUTPUT_NAME
CRIME_OUTPUT_PARQUET = CRIME_OUTPUT_PATH.with_suffix(".parquet")

BUS_CACHE_NAME = os.getenv("BUS_CACHE_NAME", "sf_bus_stops_with_geoid.csv")
BUS_CACHE_PATH = SAVE_DIR / BUS_CACHE_NAME
BUS_CACHE_PARQUET = BUS_CACHE_PATH.with_suffix(".parquet")

# Socrata dataset id
BUS_DATASET_ID = os.getenv("BUS_DATASET_ID", "i28b-stks")  # varsayılan Muni stops
BUS_APP_TOKEN = os.getenv("SOCRATA_APP_TOKEN", "").strip()

# İndirme ayarları
SOCRATA_LIMIT = int(os.getenv("SOCRATA_LIMIT", "50000"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "60"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "4"))
RETRY_SLEEP = float(os.getenv("RETRY_SLEEP", "3"))

# SF bbox (yaklaşık)
SF_MIN_LAT = float(os.getenv("SF_MIN_LAT", "37.60"))
SF_MAX_LAT = float(os.getenv("SF_MAX_LAT", "37.84"))
SF_MIN_LON = float(os.getenv("SF_MIN_LON", "-122.55"))
SF_MAX_LON = float(os.getenv("SF_MAX_LON", "-122.34"))

EARTH_RADIUS_M = 6371000.0


# ============================================================
# YARDIMCI FONKSİYONLAR
# ============================================================

def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    return df


def normalize_geoid(series: pd.Series, geoid_len: int = DEFAULT_GEOID_LEN) -> pd.Series:
    s = series.astype(str).str.replace(".0", "", regex=False).str.strip()
    s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan, "NONE": np.nan})
    return s.where(s.isna(), s.str.zfill(geoid_len))


def ensure_geoid_column(df: pd.DataFrame, geoid_len: int = DEFAULT_GEOID_LEN) -> pd.DataFrame:
    df = _clean_columns(df)

    candidates = [
        "GEOID", "geoid", "Geoid",
        "TRACT11", "tract11",
        "tract", "TRACT",
        "GEOID_x", "GEOID_y"
    ]

    found = None
    for c in candidates:
        if c in df.columns:
            found = c
            break

    if found is None:
        upper_map = {str(c).strip().upper(): c for c in df.columns}
        if "GEOID" in upper_map:
            found = upper_map["GEOID"]

    if found is None:
        raise KeyError(f"GEOID benzeri kolon bulunamadı. Kolonlar: {list(df.columns)}")

    if found != "GEOID":
        print(f"🔁 GEOID kolon adı normalize edildi: {found} -> GEOID")
        df = df.rename(columns={found: "GEOID"})

    df["GEOID"] = normalize_geoid(df["GEOID"], geoid_len)

    if df["GEOID"].isna().all():
        raise ValueError("GEOID kolonunun tamamı NaN görünüyor.")

    return df


def find_first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def ensure_lat_lon_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = _clean_columns(df)

    lat_col = find_first_existing(df, ["latitude", "Latitude", "lat", "LAT", "y", "Y"])
    lon_col = find_first_existing(df, ["longitude", "Longitude", "lon", "LON", "long", "LONG", "x", "X"])

    if lat_col is None or lon_col is None:
        raise KeyError(f"Latitude/longitude kolonları bulunamadı. Kolonlar: {list(df.columns)}")

    if lat_col != "latitude":
        df = df.rename(columns={lat_col: "latitude"})
    if lon_col != "longitude":
        df = df.rename(columns={lon_col: "longitude"})

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    return df


def clip_sf_bbox(df: pd.DataFrame) -> pd.DataFrame:
    m = (
        df["latitude"].between(SF_MIN_LAT, SF_MAX_LAT, inclusive="both") &
        df["longitude"].between(SF_MIN_LON, SF_MAX_LON, inclusive="both")
    )
    return df.loc[m].copy()


def meters_to_range(series_m: pd.Series) -> pd.Series:
    bins = [-np.inf, 100, 250, 500, 1000, 2000, np.inf]
    labels = [
        "0-100m",
        "100-250m",
        "250-500m",
        "500m-1km",
        "1-2km",
        "2km+"
    ]
    return pd.cut(series_m, bins=bins, labels=labels)


def count_to_range(series_cnt: pd.Series) -> pd.Series:
    bins = [-np.inf, 0, 1, 3, 5, 10, np.inf]
    labels = [
        "0",
        "1",
        "2-3",
        "4-5",
        "6-10",
        "10+"
    ]
    return pd.cut(series_cnt, bins=bins, labels=labels)


def haversine_balltree_distance_m(
    src_lat: np.ndarray,
    src_lon: np.ndarray,
    ref_lat: np.ndarray,
    ref_lon: np.ndarray
) -> np.ndarray:
    src_rad = np.deg2rad(np.c_[src_lat, src_lon])
    ref_rad = np.deg2rad(np.c_[ref_lat, ref_lon])

    tree = BallTree(ref_rad, metric="haversine")
    dist_rad, _ = tree.query(src_rad, k=1)
    return dist_rad[:, 0] * EARTH_RADIUS_M


# ============================================================
# BUS API İNDİRME
# ============================================================

def fetch_bus_from_api() -> pd.DataFrame:
    print("🌐 BUS verisi Socrata API'den indiriliyor...")

    base_url = f"https://data.sfgov.org/resource/{BUS_DATASET_ID}.json"

    # En yaygın durak kolonlarını kapsayacak seçme
    select_cols = [
        "stop_id",
        "stop_name",
        "latitude",
        "longitude",
        "lat",
        "lon",
        "x",
        "y",
        "location",
        "location_1",
        "geoid"
    ]

    headers = {}
    if BUS_APP_TOKEN:
        headers["X-App-Token"] = BUS_APP_TOKEN

    all_rows = []
    offset = 0

    while True:
        params = {
            "$limit": SOCRATA_LIMIT,
            "$offset": offset,
            "$select": ",".join(select_cols)
        }

        ok = False
        last_err = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                r = requests.get(base_url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
                if r.status_code == 200:
                    rows = r.json()
                    ok = True
                    break

                last_err = RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
                print(f"⚠️ BUS API deneme {attempt}/{MAX_RETRIES} başarısız: HTTP {r.status_code}")
                time.sleep(RETRY_SLEEP)

            except Exception as e:
                last_err = e
                print(f"⚠️ BUS API deneme {attempt}/{MAX_RETRIES} exception: {e}")
                time.sleep(RETRY_SLEEP)

        if not ok:
            raise RuntimeError(f"BUS API başarısız. Son hata: {last_err}")

        if not rows:
            break

        chunk = pd.DataFrame(rows)
        all_rows.append(chunk)
        print(f"  + {len(chunk):,} kayıt indirildi...")

        if len(chunk) < SOCRATA_LIMIT:
            break

        offset += SOCRATA_LIMIT

    if not all_rows:
        raise ValueError("BUS API boş veri döndürdü.")

    df = pd.concat(all_rows, ignore_index=True)
    df = _clean_columns(df)
    print(f"📊 BUS raw(api): {df.shape[0]:,} satır × {df.shape[1]} sütun")

    # location / location_1 içinden lat-lon çıkarma fallback'i
    if "latitude" not in df.columns or "longitude" not in df.columns:
        if "location" in df.columns:
            loc = df["location"].astype(str)
            if loc.str.contains(",").any():
                parts = loc.str.replace("[()]", "", regex=True).str.split(",", expand=True)
                if parts.shape[1] >= 2:
                    if "latitude" not in df.columns:
                        df["latitude"] = parts[0]
                    if "longitude" not in df.columns:
                        df["longitude"] = parts[1]

        if ("latitude" not in df.columns or "longitude" not in df.columns) and "location_1" in df.columns:
            loc = df["location_1"].astype(str)
            if loc.str.contains(",").any():
                parts = loc.str.replace("[()]", "", regex=True).str.split(",", expand=True)
                if parts.shape[1] >= 2:
                    if "latitude" not in df.columns:
                        df["latitude"] = parts[0]
                    if "longitude" not in df.columns:
                        df["longitude"] = parts[1]

    df = ensure_lat_lon_columns(df)

    keep_cols = []
    for c in ["stop_id", "stop_name", "latitude", "longitude", "geoid"]:
        if c in df.columns:
            keep_cols.append(c)

    df = df[keep_cols].copy()
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"]).copy()
    df = clip_sf_bbox(df)

    # Eğer API geoid verirse tut, yoksa sonradan cache'de geoid zorunlu değil
    if "geoid" in df.columns:
        df["geoid"] = normalize_geoid(df["geoid"], DEFAULT_GEOID_LEN)

    df = df.drop_duplicates(subset=["latitude", "longitude"]).reset_index(drop=True)

    print(f"📊 BUS prepared(api): {df.shape[0]:,} satır × {df.shape[1]} sütun")
    return df


# ============================================================
# CACHE LOAD / SAVE
# ============================================================

def save_bus_cache(df_bus: pd.DataFrame) -> None:
    df_bus.to_csv(BUS_CACHE_PATH, index=False)
    try:
        df_bus.to_parquet(BUS_CACHE_PARQUET, index=False)
    except Exception as e:
        print(f"⚠️ BUS parquet yazılamadı: {e}")

    print(f"💾 BUS cache csv yazıldı: {BUS_CACHE_PATH}")
    print(f"💾 BUS cache parquet yazıldı: {BUS_CACHE_PARQUET}")


def load_bus_cache() -> pd.DataFrame:
    print("📂 BUS cache okunuyor...")

    if not BUS_CACHE_PATH.exists():
        raise FileNotFoundError(f"BUS cache bulunamadı: {BUS_CACHE_PATH}")

    df = pd.read_csv(BUS_CACHE_PATH, low_memory=False)
    df = _clean_columns(df)

    print(f"🔎 BUS cache kolonları: {list(df.columns)}")

    # geoid varsa standardize et, yoksa sorun çıkarma;
    # çünkü distance hesaplamak için geoid zorunlu değil.
    try:
        df = ensure_geoid_column(df, DEFAULT_GEOID_LEN)
    except Exception as e:
        print(f"⚠️ BUS cache içinde GEOID standardizasyonu atlandı: {e}")

    df = ensure_lat_lon_columns(df)
    df = df.dropna(subset=["latitude", "longitude"]).copy()
    df = clip_sf_bbox(df)
    df = df.drop_duplicates(subset=["latitude", "longitude"]).reset_index(drop=True)

    if "GEOID" in df.columns:
        print("✅ BUS cache GEOID hazır")
        print(df["GEOID"].head())

    print(f"📊 BUS cache loaded: {df.shape[0]:,} satır × {df.shape[1]} sütun")
    return df


def get_bus_data() -> pd.DataFrame:
    try:
        df_bus = fetch_bus_from_api()
        save_bus_cache(df_bus)
        return df_bus
    except Exception as e:
        print(f"⚠️ API başarısız; mevcut cache kullanılacak: {e}")
        return load_bus_cache()


# ============================================================
# CRIME INPUT LOAD
# ============================================================

def load_crime_input(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Crime input bulunamadı: {path}")

    print(f"📥 Crime input okunuyor: {path}")
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, low_memory=False)

    df = _clean_columns(df)
    df = ensure_geoid_column(df, DEFAULT_GEOID_LEN)
    df = ensure_lat_lon_columns(df)

    print(f"📊 CRIME input: {df.shape[0]:,} satır × {df.shape[1]} sütun")
    return df


# ============================================================
# ENRICHMENT
# ============================================================

def enrich_with_bus(crime_df: pd.DataFrame, bus_df: pd.DataFrame) -> pd.DataFrame:
    out = crime_df.copy()

    # 1) bus stop count per GEOID
    if "GEOID" in bus_df.columns:
        bus_count = (
            bus_df.dropna(subset=["GEOID"])
            .groupby("GEOID", as_index=False)
            .size()
            .rename(columns={"size": "bus_stop_count"})
        )
        out = out.merge(bus_count, on="GEOID", how="left")
    else:
        print("⚠️ BUS verisinde GEOID yok; bus_stop_count 0 atanacak.")
        out["bus_stop_count"] = np.nan

    out["bus_stop_count"] = pd.to_numeric(out["bus_stop_count"], errors="coerce").fillna(0).astype(int)

    # 2) en yakın otobüs durağı mesafesi
    if len(bus_df) == 0:
        print("⚠️ BUS verisi boş; distance_to_bus NaN atanacak.")
        out["distance_to_bus"] = np.nan
    else:
        print("📏 En yakın BUS mesafesi hesaplanıyor...")
        valid_mask = out["latitude"].notna() & out["longitude"].notna()
        out["distance_to_bus"] = np.nan

        if valid_mask.any():
            dist_m = haversine_balltree_distance_m(
                src_lat=out.loc[valid_mask, "latitude"].to_numpy(dtype=float),
                src_lon=out.loc[valid_mask, "longitude"].to_numpy(dtype=float),
                ref_lat=bus_df["latitude"].to_numpy(dtype=float),
                ref_lon=bus_df["longitude"].to_numpy(dtype=float),
            )
            out.loc[valid_mask, "distance_to_bus"] = dist_m

    # 3) range kolonları
    out["distance_to_bus_range"] = meters_to_range(out["distance_to_bus"])
    out["bus_stop_count_range"] = count_to_range(out["bus_stop_count"])

    return out


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("🚀 update_bus FINAL (ROBUST CACHE / GEOID SAFE)")

    bus_df = get_bus_data()
    crime_df = load_crime_input(CRIME_INPUT_PATH)

    enriched = enrich_with_bus(crime_df, bus_df)

    print(f"📊 ENRICHED: {enriched.shape[0]:,} satır × {enriched.shape[1]} sütun")

    enriched.to_csv(CRIME_OUTPUT_PATH, index=False)
    print(f"💾 csv yazıldı: {CRIME_OUTPUT_PATH}")

    try:
        enriched.to_parquet(CRIME_OUTPUT_PARQUET, index=False)
        print(f"💾 parquet yazıldı: {CRIME_OUTPUT_PARQUET}")
    except Exception as e:
        print(f"⚠️ parquet yazılamadı: {e}")

    must_have = [
        "GEOID",
        "distance_to_bus",
        "distance_to_bus_range",
        "bus_stop_count",
        "bus_stop_count_range",
    ]
    missing = [c for c in must_have if c not in enriched.columns]
    if missing:
        raise RuntimeError(f"Çıktıda eksik kolonlar var: {missing}")

    print("✅ DONE")


if __name__ == "__main__":
    main()
