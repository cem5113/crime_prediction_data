# update_sf_features_incremental.py
# =============================================================================
# SF FEATURE ENRICHMENT: sf_crime_09.csv + business/building -> sf_crime_10.csv
#
# Amaç:
# - sf_business_landuse.csv   (GEOID bazlı statik)
# - sf_building_permits_vacancy.csv (ham permit kayıtları)
# dosyalarını kullanarak sf_crime_09.csv'yi zenginleştirmek
#
# Çıktı:
# - sf_building_daily_features.csv   (ara çıktı)
# - sf_crime_10.csv
#
# Kritik not:
# - Building verisi aynı gün doğrudan merge edilmez.
# - Önce GEOID+date günlük aggregasyon yapılır.
# - Sonra strictly past-only feature'lar üretilir (shift(1) ile).
# - Böylece leakage azaltılır.
# =============================================================================

import os
import json
import re
import ast
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# AYARLAR
# =============================================================================
BASE_DIR = Path(os.getenv("CRIME_DATA_DIR", "crime_prediction_data"))
BASE_DIR.mkdir(parents=True, exist_ok=True)

# Ana panel
CRIME_IN = Path(os.getenv("CRIME_09_PATH", str(BASE_DIR / "sf_crime_09.csv")))
CRIME_OUT = Path(os.getenv("CRIME_10_PATH", str(BASE_DIR / "sf_crime_10.csv")))

# GitHub raw URL'ler
BUSINESS_URL = os.getenv(
    "BUSINESS_URL",
    "https://raw.githubusercontent.com/cem5113/crime_prediction_data/main/sf_business_landuse.csv"
)
BUILDING_URL = os.getenv(
    "BUILDING_URL",
    "https://raw.githubusercontent.com/cem5113/crime_prediction_data/main/sf_building_permits_vacancy.csv"
)

# Local fallback
BUSINESS_LOCAL = Path(os.getenv("BUSINESS_LOCAL", str(BASE_DIR / "sf_business_landuse.csv")))
BUILDING_LOCAL = Path(os.getenv("BUILDING_LOCAL", str(BASE_DIR / "sf_building_permits_vacancy.csv")))

# Ara çıktı
BUILDING_DAILY_OUT = Path(os.getenv(
    "BUILDING_DAILY_OUT",
    str(BASE_DIR / "sf_building_daily_features.csv")
))

GEOID_LEN = int(os.getenv("GEOID_LEN", "11"))

# Yalnız gerçekten faydalı, kompakt feature set
BUSINESS_KEEP = [
    "GEOID",
    "business_count",
    "landuse_mix_score",
]

BUILDING_FINAL_KEEP = [
    "GEOID",
    "date",
    "building_permit_count_prev_1d",
    "building_completed_count_prev_1d",
    "building_estimated_cost_sum_prev_1d",
    "building_permit_count_roll7",
    "building_completed_count_roll7",
    "building_estimated_cost_sum_roll7",
    "building_permit_count_roll28",
    "building_estimated_cost_sum_roll28",
    "building_permit_zscore_28d",
]


# =============================================================================
# HELPERS
# =============================================================================
def log(msg: str):
    print(msg, flush=True)


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def safe_save_csv(df: pd.DataFrame, path: Path):
    ensure_parent(path)
    tmp = str(path) + ".tmp"
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)
    log(f"💾 Yazıldı: {path}")


def normalize_geoid(series: pd.Series, target_len: int = GEOID_LEN) -> pd.Series:
    s = (
        series.astype(str)
        .str.extract(r"(\d+)", expand=False)
        .fillna("")
        .str[:target_len]
        .str.zfill(target_len)
    )
    return s.mask(s.eq("0" * target_len))


def normalize_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.strftime("%Y-%m-%d")


def clean_hour_range(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    m = s.str.extract(r"^\s*(\d{1,2})\s*[-:]\s*(\d{1,2})\s*$")
    ok = m[0].notna() & m[1].notna()

    out = pd.Series(np.nan, index=series.index, dtype=object)
    out.loc[ok] = (
        m.loc[ok, 0].astype(int).astype(str).str.zfill(2)
        + "-"
        + m.loc[ok, 1].astype(int).astype(str).str.zfill(2)
    )
    return out.fillna(s)


def parse_numeric(series: pd.Series) -> pd.Series:
    x = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(x, errors="coerce")


def read_csv_smart(local_path: Path, github_url: str, dtype=None) -> pd.DataFrame:
    if local_path.exists():
        log(f"📦 Local okunuyor: {local_path}")
        return pd.read_csv(local_path, low_memory=False, dtype=dtype)
    log(f"🌐 GitHub'dan okunuyor: {github_url}")
    return pd.read_csv(github_url, low_memory=False, dtype=dtype)


def log_shape(df: pd.DataFrame, name: str):
    log(f"📊 {name}: {df.shape[0]} satır × {df.shape[1]} sütun")


def drop_overlap_from_right(left: pd.DataFrame, right: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    overlap = (set(left.columns) & set(right.columns)) - set(keys)
    if overlap:
        log(f"🧹 overlap drop ({len(overlap)}): {sorted(overlap)}")
        right = right.drop(columns=list(overlap), errors="ignore")
    return right


# =============================================================================
# BUILDING LOCATION PARSE
# =============================================================================
def _parse_location_obj(val):
    if pd.isna(val):
        return None

    if isinstance(val, dict):
        return val

    if isinstance(val, str):
        txt = val.strip()
        if not txt:
            return None

        # JSON dict string
        try:
            return json.loads(txt)
        except Exception:
            pass

        # python dict string
        try:
            return ast.literal_eval(txt)
        except Exception:
            pass

    return None


def extract_lon_lat_from_location(val):
    # dict / json
    obj = _parse_location_obj(val)
    if isinstance(obj, dict):
        coords = obj.get("coordinates", None)
        if isinstance(coords, (list, tuple)) and len(coords) >= 2:
            return pd.to_numeric(coords[0], errors="coerce"), pd.to_numeric(coords[1], errors="coerce")

        lon = obj.get("longitude", None)
        lat = obj.get("latitude", None)
        if lon is not None and lat is not None:
            return pd.to_numeric(lon, errors="coerce"), pd.to_numeric(lat, errors="coerce")

    # POINT (lon lat)
    if isinstance(val, str):
        txt = val.strip()
        m = re.search(r"POINT\s*\(([-\d\.]+)\s+([-\d\.]+)\)", txt)
        if m:
            return pd.to_numeric(m.group(1), errors="coerce"), pd.to_numeric(m.group(2), errors="coerce")

    return np.nan, np.nan


# =============================================================================
# BUSINESS
# =============================================================================
def prepare_business(df_business: pd.DataFrame) -> pd.DataFrame:
    if df_business.empty:
        return pd.DataFrame(columns=BUSINESS_KEEP)

    df = df_business.copy()

    if "GEOID" not in df.columns:
        raise KeyError("❌ sf_business_landuse.csv içinde GEOID yok.")

    df["GEOID"] = normalize_geoid(df["GEOID"])

    for c in ["business_count", "landuse_mix_score"]:
        if c not in df.columns:
            df[c] = 0

    df["business_count"] = pd.to_numeric(df["business_count"], errors="coerce").fillna(0).astype("int32")
    df["landuse_mix_score"] = pd.to_numeric(df["landuse_mix_score"], errors="coerce").fillna(0).astype("int16")

    df = df[BUSINESS_KEEP].copy()
    df = df.drop_duplicates(subset=["GEOID"], keep="last").reset_index(drop=True)

    log_shape(df, "BUSINESS PREPARED")
    return df


# =============================================================================
# BUILDING -> DAILY AGG -> PAST-ONLY FEATURES
# =============================================================================
def prepare_building_daily_features(df_building_raw: pd.DataFrame) -> pd.DataFrame:
    if df_building_raw.empty:
        return pd.DataFrame(columns=BUILDING_FINAL_KEEP)

    df = df_building_raw.copy()
    log_shape(df, "BUILDING RAW")

    required_any = ["permit_number", "filed_date", "issued_date", "location"]
    if not any(c in df.columns for c in required_any):
        raise KeyError("❌ building csv beklenen şemada görünmüyor.")

    # location -> lon/lat
    if "location" in df.columns:
        coords = df["location"].apply(extract_lon_lat_from_location)
        df["longitude"] = coords.apply(lambda x: x[0] if isinstance(x, tuple) else np.nan)
        df["latitude"] = coords.apply(lambda x: x[1] if isinstance(x, tuple) else np.nan)
    else:
        df["longitude"] = np.nan
        df["latitude"] = np.nan

    # burada zaten building csv'de GEOID yok; sf_business gibi değil
    # senin verdiğin veri doğrudan point içeriyor.
    # crime panel GEOID bazlı olduğu için building kayıtlarını GEOID'e çevirmek gerekir.
    # Eğer building csv'de zaten GEOID yoksa ve ayrı geocoding adımı yoksa burada tract eşleşmesi yapılamaz.
    # Bu nedenle bir güvenli yol:
    # - eğer building csv'de sonradan GEOID eklenmişse onu kullan
    # - yoksa hata verip kullanıcıyı bilgilendir
    if "GEOID" in df.columns and df["GEOID"].notna().any():
        df["GEOID"] = normalize_geoid(df["GEOID"])
    else:
        raise KeyError(
            "❌ sf_building_permits_vacancy.csv içinde GEOID yok. "
            "Bu dosya point içeriyor ama GEOID içermiyor; önce GEOID eklenmiş versiyon kullanılmalı."
        )

    # tarih
    date_col = None
    if "issued_date" in df.columns:
        date_col = "issued_date"
    elif "filed_date" in df.columns:
        date_col = "filed_date"

    if date_col is None:
        raise KeyError("❌ building dosyasında issued_date / filed_date yok.")

    df["date"] = normalize_date(df[date_col])
    df = df.dropna(subset=["GEOID", "date"]).copy()

    if df.empty:
        return pd.DataFrame(columns=BUILDING_FINAL_KEEP)

    # completed
    if "completed_date" in df.columns:
        df["is_completed"] = pd.to_datetime(df["completed_date"], errors="coerce").notna().astype("int8")
    else:
        df["is_completed"] = 0

    # cost
    if "estimated_cost" in df.columns:
        df["estimated_cost_num"] = parse_numeric(df["estimated_cost"]).fillna(0.0)
    else:
        df["estimated_cost_num"] = 0.0

    # permit id
    if "permit_number" not in df.columns:
        df["permit_number"] = np.arange(len(df)).astype(str)

    # günlük agg
    daily = (
        df.groupby(["GEOID", "date"], as_index=False)
        .agg(
            building_permit_count=("permit_number", "nunique"),
            building_completed_count=("is_completed", "sum"),
            building_estimated_cost_sum=("estimated_cost_num", "sum"),
        )
    )

    daily["GEOID"] = normalize_geoid(daily["GEOID"])
    daily["date"] = normalize_date(daily["date"])
    daily = daily.sort_values(["GEOID", "date"]).reset_index(drop=True)

    # geçmişe dayalı feature
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    grp = daily.groupby("GEOID", group_keys=False)

    daily["building_permit_count_prev_1d"] = grp["building_permit_count"].shift(1)
    daily["building_completed_count_prev_1d"] = grp["building_completed_count"].shift(1)
    daily["building_estimated_cost_sum_prev_1d"] = grp["building_estimated_cost_sum"].shift(1)

    daily["building_permit_count_roll7"] = grp["building_permit_count"].shift(1).rolling(7).mean()
    daily["building_completed_count_roll7"] = grp["building_completed_count"].shift(1).rolling(7).mean()
    daily["building_estimated_cost_sum_roll7"] = grp["building_estimated_cost_sum"].shift(1).rolling(7).mean()

    daily["building_permit_count_roll28"] = grp["building_permit_count"].shift(1).rolling(28).mean()
    daily["building_estimated_cost_sum_roll28"] = grp["building_estimated_cost_sum"].shift(1).rolling(28).mean()

    roll28_mean = grp["building_permit_count"].shift(1).rolling(28).mean()
    roll28_std = grp["building_permit_count"].shift(1).rolling(28).std()

    daily["building_permit_zscore_28d"] = (
        (daily["building_permit_count"] - roll28_mean) / (roll28_std + 1e-6)
    )

    # fill
    feat_cols = [c for c in BUILDING_FINAL_KEEP if c not in ["GEOID", "date"]]
    for c in feat_cols:
        daily[c] = pd.to_numeric(daily[c], errors="coerce").fillna(0)

    # raw same-day kolonları merge etmeyeceğiz
    daily["date"] = daily["date"].dt.strftime("%Y-%m-%d")
    out = daily[BUILDING_FINAL_KEEP].copy()

    # dtype
    int_like = [
        "building_permit_count_prev_1d",
        "building_completed_count_prev_1d",
    ]
    for c in int_like:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype("int32")

    float_like = [c for c in feat_cols if c not in int_like]
    for c in float_like:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype("float32")

    log_shape(out, "BUILDING DAILY FEATURES")
    safe_save_csv(out, BUILDING_DAILY_OUT)
    return out


# =============================================================================
# MAIN
# =============================================================================
def main():
    log("🚀 update_sf_features_incremental.py başladı...")

    if not CRIME_IN.exists():
        raise FileNotFoundError(f"❌ sf_crime_09.csv bulunamadı: {CRIME_IN}")

    # -------------------------------------------------------------------------
    # 1) crime panel oku
    # -------------------------------------------------------------------------
    crime = pd.read_csv(CRIME_IN, low_memory=False, dtype={"GEOID": str})
    log_shape(crime, "CRIME INPUT")

    if "GEOID" not in crime.columns:
        raise KeyError("❌ sf_crime_09.csv içinde GEOID yok.")
    if "date" not in crime.columns:
        raise KeyError("❌ sf_crime_09.csv içinde date yok.")

    crime["GEOID"] = normalize_geoid(crime["GEOID"])
    crime["date"] = normalize_date(crime["date"])

    if "hour_range" in crime.columns:
        crime["hour_range"] = clean_hour_range(crime["hour_range"])

    # -------------------------------------------------------------------------
    # 2) business oku ve hazırla
    # -------------------------------------------------------------------------
    business_raw = read_csv_smart(BUSINESS_LOCAL, BUSINESS_URL, dtype={"GEOID": str})
    business = prepare_business(business_raw)

    # -------------------------------------------------------------------------
    # 3) building oku ve hazırla
    # -------------------------------------------------------------------------
    building_raw = read_csv_smart(BUILDING_LOCAL, BUILDING_URL, dtype={"GEOID": str})
    building = prepare_building_daily_features(building_raw)

    # -------------------------------------------------------------------------
    # 4) business merge (GEOID)
    # -------------------------------------------------------------------------
    business = drop_overlap_from_right(crime, business, keys=["GEOID"])
    before = crime.shape
    crime = crime.merge(business, on="GEOID", how="left", validate="many_to_one")
    log(f"🔗 CRIME × BUSINESS: {before} -> {crime.shape}")

    # -------------------------------------------------------------------------
    # 5) building merge (GEOID + date)
    # -------------------------------------------------------------------------
    building = drop_overlap_from_right(crime, building, keys=["GEOID", "date"])
    before = crime.shape
    crime = crime.merge(building, on=["GEOID", "date"], how="left", validate="many_to_one")
    log(f"🔗 CRIME × BUILDING: {before} -> {crime.shape}")

    # -------------------------------------------------------------------------
    # 6) fill
    # -------------------------------------------------------------------------
    for c in ["business_count", "landuse_mix_score"]:
        if c in crime.columns:
            crime[c] = pd.to_numeric(crime[c], errors="coerce").fillna(0)

    for c in [c for c in BUILDING_FINAL_KEEP if c not in ["GEOID", "date"]]:
        if c in crime.columns:
            crime[c] = pd.to_numeric(crime[c], errors="coerce").fillna(0)

    if "business_count" in crime.columns:
        crime["business_count"] = crime["business_count"].astype("int32")
    if "landuse_mix_score" in crime.columns:
        crime["landuse_mix_score"] = crime["landuse_mix_score"].astype("int16")

    for c in [
        "building_permit_count_prev_1d",
        "building_completed_count_prev_1d",
    ]:
        if c in crime.columns:
            crime[c] = crime[c].astype("int32")

    for c in [
        "building_estimated_cost_sum_prev_1d",
        "building_permit_count_roll7",
        "building_completed_count_roll7",
        "building_estimated_cost_sum_roll7",
        "building_permit_count_roll28",
        "building_estimated_cost_sum_roll28",
        "building_permit_zscore_28d",
    ]:
        if c in crime.columns:
            crime[c] = crime[c].astype("float32")

    # -------------------------------------------------------------------------
    # 7) çıktı
    # -------------------------------------------------------------------------
    log_shape(crime, "FINAL sf_crime_10")
    safe_save_csv(crime, CRIME_OUT)

    # kısa özet
    preview_cols = [
        c for c in [
            "GEOID", "date", "hour_range",
            "business_count", "landuse_mix_score",
            "building_permit_count_prev_1d",
            "building_completed_count_prev_1d",
            "building_estimated_cost_sum_prev_1d",
            "building_permit_count_roll7",
            "building_estimated_cost_sum_roll7",
            "building_permit_zscore_28d",
        ] if c in crime.columns
    ]

    if preview_cols:
        log("📌 Önizleme:")
        try:
            print(crime[preview_cols].head(10).to_string(index=False))
        except Exception:
            pass

    log("✅ sf_crime_10 üretildi.")


if __name__ == "__main__":
    main()
