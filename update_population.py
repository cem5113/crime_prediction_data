# update_population.py
# =============================================================================
# DEMOGRAPHIC / POPULATION ENRICH (APPEND-ONLY, TRACT/GEOID-BASED)
#
# AMAÇ
# ----
# 1) crime_prediction_data/sf_population.csv dosyasından tract-level
#    demografik öznitelikler üretmek
# 2) sf_crime_02.csv -> sf_crime_03.csv zenginleştirmesini yapmak
# 3) İlk koşuda tüm veriyi enrich etmek
# 4) Sonraki koşularda eski sf_crime_03 satırlarını değiştirmeden,
#    yalnızca yeni crime satırlarını yeni demografi snapshot'ı ile eklemek
#
# SÖZLEŞME
# --------
# - GEOID seviyesi: census tract (11 hane)
# - İlk kez çalışırsa: tüm sf_crime_02 -> sf_crime_03
# - sf_crime_03 zaten varsa:
#     * eski satırlar korunur
#     * yalnızca yeni satırlar enrich edilir
# - Demografi dosyası güncellenirse:
#     * eski crime satırları değişmez
#     * yeni crime satırları yeni demografiyle eşleşir
#
# NOT
# ---
# Bu script, verdiğin demografik veri şemasına göre tasarlanmıştır:
# - geography == tract
# - geography_id = GEOID
# - estimate = değer
# - acs_table / acs_label / demographic_category_label = feature çıkarımı
# =============================================================================

import os
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

pd.options.mode.copy_on_write = True


# =============================================================================
# HELPERS
# =============================================================================
def log_shape(df: pd.DataFrame, label: str):
    r, c = df.shape
    print(f"📊 {label}: {r} satır × {c} sütun")


def log_delta(before_shape, after_shape, label: str):
    br, bc = before_shape
    ar, ac = after_shape
    print(f"🔗 {label}: {br}×{bc} → {ar}×{ac} (Δr={ar-br}, Δc={ac-bc})")


def ensure_parent(path: str):
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def safe_save_csv(df: pd.DataFrame, path: str):
    ensure_parent(path)
    tmp = str(path) + ".tmp"
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)
    print(f"💾 Kaydedildi: {path}")

def safe_save_parquet(df: pd.DataFrame, path: str):
    ensure_parent(path)
    tmp = str(path) + ".tmp.parquet"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    print(f"💾 Parquet kaydedildi: {path}")

def read_existing_output(parquet_path: str) -> pd.DataFrame | None:
    if os.path.exists(parquet_path):
        print(f"📥 Mevcut çıktı Parquet bulundu: {parquet_path}")
        return pd.read_parquet(parquet_path)

    csv_path = parquet_path.replace(".parquet", ".csv")
    if os.path.exists(csv_path):
        print(f"📥 Mevcut çıktı CSV bulundu: {csv_path}")
        return pd.read_csv(csv_path, low_memory=False)

    return None
    
def digits_only(s: pd.Series) -> pd.Series:
    return s.astype(str).str.extract(r"(\d+)", expand=False).fillna("")


def normalize_geoid(series: pd.Series, target_len: int = 11) -> pd.Series:
    s = digits_only(series)
    s = s.str[:target_len].str.zfill(target_len)
    s = s.mask(s.eq("0" * target_len))
    return s


def parse_numeric_series(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.strip().str.replace(" ", "", regex=False)

    # yalnız virgül varsa: decimal virgül kabul et
    only_comma = x.str.contains(",", regex=False) & ~x.str.contains(r"\.", regex=True)
    x.loc[only_comma] = x.loc[only_comma].str.replace(",", ".", regex=False)

    # hem nokta hem virgül varsa: noktayı binlik, virgülü decimal kabul et
    both = x.str.contains(",", regex=False) & x.str.contains(r"\.", regex=True)
    x.loc[both] = (
        x.loc[both]
         .str.replace(".", "", regex=False)
         .str.replace(",", ".", regex=False)
    )

    return pd.to_numeric(x, errors="coerce")

def find_crime_input(base_dir: Path) -> str:
    cands = [
        base_dir / "sf_crime_02.parquet",
        Path.cwd() / "sf_crime_02.parquet",
        Path("sf_crime_02.parquet"),
        base_dir / "sf_crime_02.csv",
        Path.cwd() / "sf_crime_02.csv",
        Path("sf_crime_02.csv"),
    ]
    for p in cands:
        if p.exists():
            print(f"📥 Crime input bulundu: {p}")
            return str(p)
    raise FileNotFoundError("❌ sf_crime_02.parquet / sf_crime_02.csv bulunamadı.")

def ensure_date_col(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    out = df.copy()
    if col not in out.columns:
        raise KeyError(f"❌ '{col}' kolonu yok.")
    out[col] = pd.to_datetime(out[col], errors="coerce").dt.normalize()
    return out


def detect_panel_keys(df: pd.DataFrame) -> list[str]:
    keys = []
    for c in ["GEOID", "date", "hour_range"]:
        if c in df.columns:
            keys.append(c)
    return keys


def dedupe_by_panel_keys(df: pd.DataFrame, keys: list[str], keep: str = "first") -> pd.DataFrame:
    if not keys:
        return df.copy()
    before = len(df)
    out = df.drop_duplicates(subset=keys, keep=keep).copy()
    dropped = before - len(out)
    if dropped > 0:
        print(f"⚠️ Panel anahtarlarına göre {dropped} duplikasyon temizlendi.")
    return out


# =============================================================================
# PATHS / ENV
# =============================================================================
BASE_DIR = Path(os.getenv("CRIME_DATA_DIR", "crime_prediction_data"))
BASE_DIR.mkdir(parents=True, exist_ok=True)

CRIME_INPUT = os.getenv("CRIME_INPUT", "").strip() or find_crime_input(BASE_DIR)
CRIME_OUTPUT_PARQUET = str(BASE_DIR / "sf_crime_03.parquet")

DEMOGRAPHIC_PATH = (os.getenv("POPULATION_PATH", "") or "").strip()
if not DEMOGRAPHIC_PATH:
    cand = BASE_DIR / "sf_population.parquet"
    if cand.exists():
        DEMOGRAPHIC_PATH = str(cand)
    elif Path("sf_population.parquet").exists():
        DEMOGRAPHIC_PATH = "sf_population.parquet"
    else:
        raise FileNotFoundError("❌ sf_population.parquet bulunamadı. Lütfen crime_prediction_data altına ekleyin.")

DEMOGRAPHIC_FEATURES_PARQUET = str(BASE_DIR / "sf_demographic_features.parquet")

GEOID_LEN = int(os.getenv("GEOID_LEN", "11"))
APPEND_ONLY = os.getenv("APPEND_ONLY", "1").strip().lower() not in ("0", "false", "no")


# =============================================================================
# DEMOGRAPHIC FEATURE EXTRACTION
# =============================================================================
def select_latest_rows_per_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aynı GEOID + feature birden çok ACS döneminde gelebilir.
    En güncel satırı seçmek için:
      - end_year büyük olan
      - data_loaded_at büyük olan
    öncelikli alınır.
    """
    x = df.copy()

    if "end_year" in x.columns:
        x["end_year_num"] = pd.to_numeric(x["end_year"], errors="coerce")
    else:
        x["end_year_num"] = np.nan

    if "data_loaded_at" in x.columns:
        x["data_loaded_at_ts"] = pd.to_datetime(x["data_loaded_at"], errors="coerce")
    else:
        x["data_loaded_at_ts"] = pd.NaT

    if "data_as_of" in x.columns:
        x["data_as_of_ts"] = pd.to_datetime(x["data_as_of"], errors="coerce")
    else:
        x["data_as_of_ts"] = pd.NaT

    sort_cols = ["GEOID", "feature_name", "end_year_num", "data_loaded_at_ts", "data_as_of_ts"]
    sort_cols = [c for c in sort_cols if c in x.columns]

    x = x.sort_values(sort_cols)
    x = x.drop_duplicates(subset=["GEOID", "feature_name"], keep="last")
    return x


def map_feature_name(row: pd.Series) -> Optional[str]:
    """
    Verilen demografi şemasından kullanılabilir tract-level feature adı üretir.
    """
    acs_table = str(row.get("acs_table", "")).strip()
    acs_label = str(row.get("acs_label", "")).strip().lower()
    demo_label = str(row.get("demographic_category_label", "")).strip().lower()
    acs_concept = str(row.get("acs_concept", "")).strip().lower()

    # -------------------------
    # TOPLAM NÜFUS
    # -------------------------
    # B01001 total veya B03002 total gibi
    if acs_table == "B01001" and "estimate!!total:" == acs_label:
        return "population_total"

    if acs_table == "B03002" and acs_label == "estimate!!total:":
        return "population_total"

    # -------------------------
    # YAŞ DAĞILIMI (B06001)
    # -------------------------
    # Bu veri setinde yaş grupları B06001 içinde tract-level geliyor.
    # Şimdilik doğrudan label bazlı alıyoruz.
    if acs_table == "B06001":
        if demo_label == "under 5 years":
            return "pop_age_under_5"
        if demo_label == "5 to 17 years":
            return "pop_age_5_17"
        if demo_label == "18 to 24 years":
            return "pop_age_18_24"
        if demo_label == "25 to 34 years":
            return "pop_age_25_34"
        if demo_label == "35 to 44 years":
            return "pop_age_35_44"
        if demo_label == "45 to 54 years":
            return "pop_age_45_54"
        if demo_label == "55 to 64 years":
            return "pop_age_55_64"
        if demo_label == "65 to 74 years":
            return "pop_age_65_74"
        if demo_label == "75 to 84 years":
            return "pop_age_75_84"
        if demo_label == "85 years and over":
            return "pop_age_85_plus"

    # -------------------------
    # IRK / ETNİSİTE (B03002)
    # -------------------------
    # Bu tabloda label pattern matching ile feature çıkarıyoruz.
    if acs_table == "B03002":
        s = acs_label

        # Hispanic total
        if "estimate!!total:!!hispanic or latino:" == s:
            return "pop_hispanic"

        # Not Hispanic racial groups
        if "not hispanic or latino:!!white alone" in s:
            return "pop_nh_white"
        if "not hispanic or latino:!!black or african american alone" in s:
            return "pop_nh_black"
        if "not hispanic or latino:!!asian alone" in s:
            return "pop_nh_asian"
        if "not hispanic or latino:!!american indian and alaska native alone" in s:
            return "pop_nh_native"
        if "not hispanic or latino:!!native hawaiian and other pacific islander alone" in s:
            return "pop_nh_pacific"
        if "not hispanic or latino:!!some other race alone" in s:
            return "pop_nh_other"
        if "not hispanic or latino:!!two or more races" in s:
            return "pop_nh_multiracial"

    return None


def build_demographic_features(demo_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Tract-level demografik feature tablosu üretir.
    Çıktı: GEOID bazında tekil feature tablosu
    """

    df = demo_raw.copy()
    log_shape(df, "DEMOGRAPHIC RAW")

    required_cols = ["geography", "geography_id", "estimate"]
    for c in required_cols:
        if c not in df.columns:
            raise KeyError(f"❌ Demografik dosyada zorunlu kolon eksik: {c}")

    # Sadece tract
    df = df[df["geography"].astype(str).str.lower().eq("tract")].copy()
    log_shape(df, "DEMOGRAPHIC (yalnız tract)")

    # GEOID normalize
    df["GEOID"] = normalize_geoid(df["geography_id"], GEOID_LEN)

    # estimate numeric
    df["estimate_num"] = parse_numeric_series(df["estimate"])

    # feature name eşle
    df["feature_name"] = df.apply(map_feature_name, axis=1)
    df = df.dropna(subset=["GEOID", "feature_name"]).copy()
    log_shape(df, "DEMOGRAPHIC (feature-mapped)")

    # Aynı feature için en güncel snapshot'ı al
    df = select_latest_rows_per_feature(df)

    # Pivot
    feat = (
        df.pivot_table(
            index="GEOID",
            columns="feature_name",
            values="estimate_num",
            aggfunc="last"
        )
        .reset_index()
    )
    feat.columns.name = None

    # -----------------------------
    # Temel eksikler ve oranlar
    # -----------------------------
    if "population_total" not in feat.columns:
        feat["population_total"] = np.nan

    age_cols = [
        "pop_age_under_5",
        "pop_age_5_17",
        "pop_age_18_24",
        "pop_age_25_34",
        "pop_age_35_44",
        "pop_age_45_54",
        "pop_age_55_64",
        "pop_age_65_74",
        "pop_age_75_84",
        "pop_age_85_plus",
    ]
    for c in age_cols:
        if c not in feat.columns:
            feat[c] = 0.0

    race_cols = [
        "pop_hispanic",
        "pop_nh_white",
        "pop_nh_black",
        "pop_nh_asian",
        "pop_nh_native",
        "pop_nh_pacific",
        "pop_nh_other",
        "pop_nh_multiracial",
    ]
    for c in race_cols:
        if c not in feat.columns:
            feat[c] = 0.0

    # Yaş toplulaştırmaları
    feat["pop_age_18_34"] = feat["pop_age_18_24"].fillna(0) + feat["pop_age_25_34"].fillna(0)
    feat["pop_age_35_64"] = (
        feat["pop_age_35_44"].fillna(0)
        + feat["pop_age_45_54"].fillna(0)
        + feat["pop_age_55_64"].fillna(0)
    )
    feat["pop_age_65_plus"] = (
        feat["pop_age_65_74"].fillna(0)
        + feat["pop_age_75_84"].fillna(0)
        + feat["pop_age_85_plus"].fillna(0)
    )

    # Güvenli payda
    denom = pd.to_numeric(feat["population_total"], errors="coerce").replace(0, np.nan)

    # Oranlar
    feat["pct_age_under_5"] = feat["pop_age_under_5"] / denom
    feat["pct_age_5_17"] = feat["pop_age_5_17"] / denom
    feat["pct_age_18_24"] = feat["pop_age_18_24"] / denom
    feat["pct_age_25_34"] = feat["pop_age_25_34"] / denom
    feat["pct_age_18_34"] = feat["pop_age_18_34"] / denom
    feat["pct_age_35_64"] = feat["pop_age_35_64"] / denom
    feat["pct_age_65_plus"] = feat["pop_age_65_plus"] / denom

    feat["pct_hispanic"] = feat["pop_hispanic"] / denom
    feat["pct_nh_white"] = feat["pop_nh_white"] / denom
    feat["pct_nh_black"] = feat["pop_nh_black"] / denom
    feat["pct_nh_asian"] = feat["pop_nh_asian"] / denom
    feat["pct_nh_multiracial"] = feat["pop_nh_multiracial"] / denom

    # NaN temizliği
    num_cols = [c for c in feat.columns if c != "GEOID"]
    for c in num_cols:
        feat[c] = pd.to_numeric(feat[c], errors="coerce")

    # Absolute count kolonları NaN ise 0, ratio kolonları NaN ise 0
    abs_cols = [c for c in num_cols if not c.startswith("pct_")]
    pct_cols = [c for c in num_cols if c.startswith("pct_")]

    feat[abs_cols] = feat[abs_cols].fillna(0)
    feat[pct_cols] = feat[pct_cols].fillna(0.0)

    # population int
    if "population_total" in feat.columns:
        feat["population_total"] = feat["population_total"].round().astype(int)

    # Uyum için eski kolon adını da koy
    feat["population"] = feat["population_total"]

    feat = feat.sort_values("GEOID").drop_duplicates(subset="GEOID", keep="last")
    log_shape(feat, "DEMOGRAPHIC FEATURES (GEOID-level)")

    # Kaydet
    safe_save_parquet(feat, DEMOGRAPHIC_FEATURES_PARQUET)

    return feat

# =============================================================================
# APPEND-ONLY LOGIC
# =============================================================================
def split_old_and_new_rows(
    crime_in: pd.DataFrame,
    crime_out_parquet: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    sf_crime_03 mevcutsa:
      - eski satırları korur
      - sf_crime_02'de olup sf_crime_03'te olmayan satırları yeni kabul eder
    parquet öncelikli, csv fallback destekli.
    """
    if not APPEND_ONLY:
        print("ℹ️ APPEND_ONLY kapalı → tüm satırlar yeni kabul edilecek.")
        return pd.DataFrame(columns=crime_in.columns), crime_in.copy()

    old = read_existing_output(crime_out_parquet)
    if old is None:
        print("ℹ️ Önceki çıktı yok → tüm satırlar yeni kabul edilecek.")
        print("⚠️ İlk koşu: full enrich yapılacak.")
        return pd.DataFrame(columns=crime_in.columns), crime_in.copy()

    if "GEOID" not in old.columns:
        print("⚠️ Eski sf_crime_03 içinde GEOID yok → tüm giriş yeni kabul edilecek.")
        return old, crime_in.copy()

    old = old.copy()
    crime_in = crime_in.copy()

    old["GEOID"] = normalize_geoid(old["GEOID"], GEOID_LEN)
    crime_in["GEOID"] = normalize_geoid(crime_in["GEOID"], GEOID_LEN)

    keys = detect_panel_keys(crime_in)
    required = {"GEOID", "date", "hour_range"}

    if required.issubset(keys) and required.issubset(old.columns):
        crime_in = ensure_date_col(crime_in, "date")
        old = ensure_date_col(old, "date")

        old_keys = old[["GEOID", "date", "hour_range"]].drop_duplicates().copy()
        old_keys["__seen__"] = 1

        marked = crime_in.merge(old_keys, on=["GEOID", "date", "hour_range"], how="left")
        new_rows = marked[marked["__seen__"].isna()].drop(columns=["__seen__"]).copy()

        print("🧠 Yeni satır tespiti: GEOID + date + hour_range")
        log_shape(old, "MEVCUT sf_crime_03")
        log_shape(crime_in, "GÜNCEL sf_crime_02")
        log_shape(new_rows, "YENİ CRIME SATIRLARI")
        return old, new_rows

    print("⚠️ Panel anahtarları eksik → güvenli tarafta kalıp tüm sf_crime_02'yi yeni kabul ediyorum.")
    return old, crime_in.copy()

def merge_demographics(df_crime: pd.DataFrame, demo_feat: pd.DataFrame) -> pd.DataFrame:
    out = df_crime.copy()
    out["GEOID"] = normalize_geoid(out["GEOID"], GEOID_LEN)

    # overlap temizliği
    overlap = (set(out.columns) & set(demo_feat.columns)) - {"GEOID"}
    if overlap:
        print(f"🧹 DEMOGRAPHIC merge overlap bulundu, demo_feat'ten düşürüldü: {sorted(overlap)}")
        demo_feat = demo_feat.drop(columns=list(overlap), errors="ignore")

    before = out.shape
    out = out.merge(demo_feat, on="GEOID", how="left", validate="many_to_one")
    log_delta(before, out.shape, "CRIME ⨯ DEMOGRAPHIC")

    # Güvenli fill
    for c in out.columns:
        if c.startswith("pct_"):
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    for c in out.columns:
        if c.startswith("pop_") or c in ("population", "population_total"):
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)

    if "population" in out.columns:
        out["population"] = pd.to_numeric(out["population"], errors="coerce").fillna(0).round().astype("int32")
    if "population_total" in out.columns:
        out["population_total"] = pd.to_numeric(out["population_total"], errors="coerce").fillna(0).round().astype("int32")
    
    for c in out.columns:
        if c.startswith("pct_"):
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype("float32")
        elif c.startswith("pop_"):
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype("float32")

    return out


def finalize_output(old_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    if old_df is None or len(old_df) == 0:
        out = new_df
    else:
        out = pd.concat([old_df, new_df], ignore_index=True, copy=False)
    if "GEOID" in out.columns:
        out["GEOID"] = normalize_geoid(out["GEOID"], GEOID_LEN)

    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    keys = detect_panel_keys(out)
    out = dedupe_by_panel_keys(out, keys, keep="first")

    sort_cols = [c for c in ["date", "GEOID", "hour_range"] if c in out.columns]
    if sort_cols and len(out) <= 1_000_000:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    else:
        out = out.reset_index(drop=True)

    return out

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("🚀 Demographic update başlıyor...")
    print("🔎 CWD:", os.getcwd())
    print("🔎 BASE_DIR:", BASE_DIR.resolve())
    print("🔎 CRIME_INPUT:", CRIME_INPUT)
    
    if not Path(CRIME_INPUT).exists():
        raise FileNotFoundError(f"❌ CRIME_INPUT bulunamadı: {CRIME_INPUT}")
    if not Path(DEMOGRAPHIC_PATH).exists():
        raise FileNotFoundError(f"❌ sf_population.csv bulunamadı: {DEMOGRAPHIC_PATH}")

    # -------------------------------------------------------------------------
    # 1) Crime input oku
    # -------------------------------------------------------------------------
    if CRIME_INPUT.lower().endswith(".parquet"):
        crime_in = pd.read_parquet(CRIME_INPUT)
    else:
        crime_in = pd.read_csv(CRIME_INPUT, low_memory=False)
    if "GEOID" not in crime_in.columns:
        raise KeyError("❌ Suç verisinde GEOID kolonu yok.")
    crime_in["GEOID"] = normalize_geoid(crime_in["GEOID"], GEOID_LEN)
    log_shape(crime_in, "CRIME INPUT")

    # -------------------------------------------------------------------------
    # 2) Önce append-only split yap
    #    Böylece yeni satır yoksa demographic dosyasını boşuna okumayız
    # -------------------------------------------------------------------------
    old_out, new_rows = split_old_and_new_rows(
        crime_in,
        CRIME_OUTPUT_PARQUET
    )

    if new_rows.empty:
        print("✅ Yeni crime satırı yok → demographic enrich atlandı.")
        print("✅ Eski sf_crime_03 aynen korunuyor.")
        if os.path.exists(CRIME_OUTPUT_PARQUET):
            print(f"📁 Mevcut Parquet çıktı: {CRIME_OUTPUT_PARQUET}")
        return

    # -------------------------------------------------------------------------
    # 3) Yalnızca gerçekten yeni satır varsa demographic oku
    # -------------------------------------------------------------------------
    demo_raw = pd.read_parquet(DEMOGRAPHIC_PATH)
    demo_raw = demo_raw.astype(str)
    log_shape(demo_raw, "DEMOGRAPHIC PARQUET")

    # -------------------------------------------------------------------------
    # 4) Feature üret
    #    Bu feature set, SADECE yeni gelen crime satırlarına uygulanacak
    # -------------------------------------------------------------------------
    demo_feat = build_demographic_features(demo_raw)

    # -------------------------------------------------------------------------
    # 5) Sadece yeni satırları enrich et
    # -------------------------------------------------------------------------
    before = new_rows.shape
    new_rows_enriched = merge_demographics(new_rows, demo_feat)
    log_delta(before, new_rows_enriched.shape, "YENİ SATIRLAR ⨯ DEMOGRAPHIC")
    log_shape(new_rows_enriched, "NEW ENRICHED ROWS")

    # -------------------------------------------------------------------------
    # 6) Eski + yeni birleştir
    #    Eski satırlar eski haliyle kalır
    #    Yeni satırlar güncel demographic snapshot ile eklenir
    # -------------------------------------------------------------------------
    final_df = finalize_output(old_out, new_rows_enriched)
    log_shape(final_df, "FINAL sf_crime_03")

    # -------------------------------------------------------------------------
    # 7) NaN / coverage raporu
    # -------------------------------------------------------------------------
    if "population" in final_df.columns:
        zero_pop = int((pd.to_numeric(final_df["population"], errors="coerce").fillna(0) == 0).sum())
        print(f"🔎 population=0 olan satır sayısı: {zero_pop}")

    dem_cols = [
        "population",
        "population_total",
        "pct_age_under_5",
        "pct_age_5_17",
        "pct_age_18_24",
        "pct_age_25_34",
        "pct_age_18_34",
        "pct_age_35_64",
        "pct_age_65_plus",
        "pct_hispanic",
        "pct_nh_white",
        "pct_nh_black",
        "pct_nh_asian",
        "pct_nh_multiracial",
    ]
    dem_cols = [c for c in dem_cols if c in final_df.columns]
    if dem_cols:
        print("🔎 Demographic kolon NaN sayıları:")
        print(final_df[dem_cols].isna().sum().to_string())

    # -------------------------------------------------------------------------
    # 8) Kaydet
    # -------------------------------------------------------------------------
    safe_save_parquet(final_df, CRIME_OUTPUT_PARQUET)
    print("ℹ️ CSV tamamen kapatıldı; yalnız parquet kaydedildi.")

    try:
        preview_cols = [c for c in [
            "GEOID", "date", "hour_range",
            "population",
            "pct_age_18_24",
            "pct_age_25_34",
            "pct_age_65_plus",
            "pct_hispanic",
            "pct_nh_white",
            "pct_nh_black",
            "pct_nh_asian",
        ] if c in final_df.columns]
        if preview_cols:
            print("📌 Önizleme:")
            print(final_df[preview_cols].tail(10).to_string(index=False))
    except Exception as e:
        print(f"ℹ️ Önizleme atlandı: {e}")

if __name__ == "__main__":
    main()
