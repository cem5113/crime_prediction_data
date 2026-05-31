# update_population.py
# =============================================================================
# DEMOGRAPHIC / POPULATION ENRICH (LEAN + APPEND-ONLY + RAM-FRIENDLY)
# =============================================================================

import os
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


def digits_only(s: pd.Series) -> pd.Series:
    return s.astype(str).str.extract(r"(\d+)", expand=False).fillna("")


def normalize_geoid(series: pd.Series, target_len: int = 11) -> pd.Series:
    s = digits_only(series)
    s = s.str[:target_len].str.zfill(target_len)
    s = s.mask(s.eq("0" * target_len))
    return s


def parse_numeric_series(s: pd.Series) -> pd.Series:
    x = (
        s.astype(str)
        .str.replace(".", "", regex=False)   # binlik ayracı
        .str.replace(",", ".", regex=False)  # 0,123 -> 0.123
        .str.replace(" ", "", regex=False)
    )
    return pd.to_numeric(x, errors="coerce")


def find_crime_input(base_dir: Path) -> str:
    cands = [
        base_dir / "sf_crime_02.csv",
        Path("sf_crime_02.csv"),
        base_dir / "sf_crime.csv",
        Path("sf_crime.csv"),
    ]
    for p in cands:
        if p.exists():
            return str(p)
    raise FileNotFoundError("❌ sf_crime_02.csv veya sf_crime.csv bulunamadı.")


def ensure_date_col(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    out = df.copy()
    if col not in out.columns:
        raise KeyError(f"❌ '{col}' kolonu yok.")
    out[col] = pd.to_datetime(out[col], errors="coerce").dt.strftime("%Y-%m-%d")
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


def make_panel_key_series(df: pd.DataFrame) -> pd.Series:
    missing = [c for c in ["GEOID", "date", "hour_range"] if c not in df.columns]
    if missing:
        raise KeyError(f"❌ Panel key için eksik kolon(lar): {missing}")

    tmp = df[["GEOID", "date", "hour_range"]].copy()
    tmp["GEOID"] = normalize_geoid(tmp["GEOID"], GEOID_LEN)
    tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    tmp["hour_range"] = tmp["hour_range"].astype(str)
    return tmp["GEOID"] + "|" + tmp["date"] + "|" + tmp["hour_range"]


# =============================================================================
# PATHS / ENV
# =============================================================================
BASE_DIR = Path(os.getenv("CRIME_DATA_DIR", "crime_prediction_data"))
BASE_DIR.mkdir(parents=True, exist_ok=True)

CRIME_INPUT = os.getenv("CRIME_INPUT", "").strip() or find_crime_input(BASE_DIR)
CRIME_OUTPUT = str(BASE_DIR / "sf_crime_03.csv")

DEMOGRAPHIC_PATH = (os.getenv("POPULATION_PATH", "") or "").strip()
if not DEMOGRAPHIC_PATH:
    cand = BASE_DIR / "sf_population.csv"
    if cand.exists():
        DEMOGRAPHIC_PATH = str(cand)
    elif Path("sf_population.csv").exists():
        DEMOGRAPHIC_PATH = "sf_population.csv"
    else:
        raise FileNotFoundError("❌ sf_population.csv bulunamadı. Lütfen crime_prediction_data altına ekleyin.")

DEMOGRAPHIC_FEATURES_CSV = str(BASE_DIR / "sf_demographic_features.csv")

GEOID_LEN = int(os.getenv("GEOID_LEN", "11"))
APPEND_ONLY = os.getenv("APPEND_ONLY", "1").strip().lower() not in ("0", "false", "no")

# Stacking için budanmış son demografi seti
FINAL_DEMOGRAPHIC_COLS = [
    "GEOID",
    "population",
    "pct_age_18_34",
    "pct_age_65_plus",
]

# =============================================================================
# DEMOGRAPHIC FEATURE EXTRACTION
# =============================================================================
def select_latest_rows_per_feature(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()

    x["end_year_num"] = pd.to_numeric(x["end_year"], errors="coerce") if "end_year" in x.columns else np.nan
    x["data_loaded_at_ts"] = pd.to_datetime(x["data_loaded_at"], errors="coerce") if "data_loaded_at" in x.columns else pd.NaT
    x["data_as_of_ts"] = pd.to_datetime(x["data_as_of"], errors="coerce") if "data_as_of" in x.columns else pd.NaT

    sort_cols = ["GEOID", "feature_name", "end_year_num", "data_loaded_at_ts", "data_as_of_ts"]
    sort_cols = [c for c in sort_cols if c in x.columns]

    x = x.sort_values(sort_cols)
    x = x.drop_duplicates(subset=["GEOID", "feature_name"], keep="last")
    return x


def map_feature_name(row: pd.Series) -> Optional[str]:
    acs_table = str(row.get("acs_table", "")).strip()
    acs_label = str(row.get("acs_label", "")).strip().lower()
    demo_label = str(row.get("demographic_category_label", "")).strip().lower()

    # total population
    if acs_table == "B01001" and acs_label == "estimate!!total:":
        return "population_total"
    if acs_table == "B03002" and acs_label == "estimate!!total:":
        return "population_total"

    # age
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

    # race / ethnicity
    if acs_table == "B03002":
        s = acs_label
        if "estimate!!total:!!hispanic or latino:" == s:
            return "pop_hispanic"
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
    df = demo_raw.copy()
    log_shape(df, "DEMOGRAPHIC RAW")

    required_cols = ["geography", "geography_id", "estimate"]
    for c in required_cols:
        if c not in df.columns:
            raise KeyError(f"❌ Demografik dosyada zorunlu kolon eksik: {c}")

    df = df[df["geography"].astype(str).str.lower().eq("tract")].copy()
    log_shape(df, "DEMOGRAPHIC (yalnız tract)")

    df["GEOID"] = normalize_geoid(df["geography_id"], GEOID_LEN)
    df["estimate_num"] = parse_numeric_series(df["estimate"])
    df["feature_name"] = df.apply(map_feature_name, axis=1)

    df = df.dropna(subset=["GEOID", "feature_name"]).copy()
    log_shape(df, "DEMOGRAPHIC (feature-mapped)")

    df = select_latest_rows_per_feature(df)

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

    feat["pop_age_18_34"] = feat["pop_age_18_24"].fillna(0) + feat["pop_age_25_34"].fillna(0)
    feat["pop_age_65_plus"] = (
        feat["pop_age_65_74"].fillna(0)
        + feat["pop_age_75_84"].fillna(0)
        + feat["pop_age_85_plus"].fillna(0)
    )

    denom = pd.to_numeric(feat["population_total"], errors="coerce").replace(0, np.nan)

    feat["pct_age_18_34"] = feat["pop_age_18_34"] / denom
    feat["pct_age_65_plus"] = feat["pop_age_65_plus"] / denom
    feat["pct_hispanic"] = feat["pop_hispanic"] / denom
    feat["pct_nh_white"] = feat["pop_nh_white"] / denom
    feat["pct_nh_black"] = feat["pop_nh_black"] / denom
    feat["pct_nh_asian"] = feat["pop_nh_asian"] / denom
    feat["pct_nh_multiracial"] = feat["pop_nh_multiracial"] / denom

    feat["population"] = pd.to_numeric(feat["population_total"], errors="coerce")

    # Sadece lean final set
    for c in FINAL_DEMOGRAPHIC_COLS:
        if c not in feat.columns:
            feat[c] = 0.0 if c != "GEOID" else None

    feat = feat[FINAL_DEMOGRAPHIC_COLS].copy()

    # temiz tipler
    for c in feat.columns:
        if c == "GEOID":
            continue
        feat[c] = pd.to_numeric(feat[c], errors="coerce")

    if "population" in feat.columns:
        feat["population"] = feat["population"].fillna(0).round().astype("int32")

    pct_cols = [c for c in feat.columns if c.startswith("pct_")]
    for c in pct_cols:
        feat[c] = feat[c].fillna(0.0).astype("float32")

    feat = feat.sort_values("GEOID").drop_duplicates(subset="GEOID", keep="last")
    log_shape(feat, "DEMOGRAPHIC FEATURES (LEAN GEOID-level)")

    safe_save_csv(feat, DEMOGRAPHIC_FEATURES_CSV)
    return feat


# =============================================================================
# APPEND-ONLY LOGIC
# =============================================================================
def split_old_and_new_rows(crime_in: pd.DataFrame, crime_out_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    sf_crime_03 varsa sadece yeni anahtarları bulur.
    RAM dostu: full merge yerine key set membership kullanır.
    """
    if not APPEND_ONLY or not os.path.exists(crime_out_path):
        print("ℹ️ Append-only çıktı yok → tüm satırlar yeni kabul edilecek.")
        return pd.DataFrame(columns=crime_in.columns), crime_in

    old = pd.read_csv(crime_out_path, low_memory=False)

    if "GEOID" not in old.columns:
        print("⚠️ Eski sf_crime_03 içinde GEOID yok → tüm giriş yeni kabul edilecek.")
        return old, crime_in

    old["GEOID"] = normalize_geoid(old["GEOID"], GEOID_LEN)
    crime_in = crime_in.copy()
    crime_in["GEOID"] = normalize_geoid(crime_in["GEOID"], GEOID_LEN)

    if set(["GEOID", "date", "hour_range"]).issubset(crime_in.columns) and set(["GEOID", "date", "hour_range"]).issubset(old.columns):
        crime_in = ensure_date_col(crime_in, "date")
        old = ensure_date_col(old, "date")

        old_keys = set(make_panel_key_series(old).dropna().unique())
        crime_keys = make_panel_key_series(crime_in)

        is_new = ~crime_keys.isin(old_keys)
        new_rows = crime_in.loc[is_new].copy()

        print("🧠 Yeni satır tespiti: GEOID + date + hour_range")
        log_shape(old, "MEVCUT sf_crime_03")
        log_shape(crime_in, "GÜNCEL sf_crime_02")
        log_shape(new_rows, "YENİ CRIME SATIRLARI")
        return old, new_rows

    print("⚠️ Panel anahtarları eksik → tüm sf_crime_02 yeni kabul ediliyor.")
    return old, crime_in

def merge_demographics(df_crime: pd.DataFrame, demo_feat: pd.DataFrame) -> pd.DataFrame:
    out = df_crime.copy()
    out["GEOID"] = normalize_geoid(out["GEOID"], GEOID_LEN)

    overlap = (set(out.columns) & set(demo_feat.columns)) - {"GEOID"}
    if overlap:
        print(f"🧹 DEMOGRAPHIC overlap bulundu, demo_feat'ten düşürüldü: {sorted(overlap)}")
        demo_feat = demo_feat.drop(columns=list(overlap), errors="ignore")

    before = out.shape
    out = out.merge(demo_feat, on="GEOID", how="left", validate="many_to_one")
    log_delta(before, out.shape, "CRIME ⨯ DEMOGRAPHIC")

    if "population" in out.columns:
        out["population"] = (
            pd.to_numeric(out["population"], errors="coerce")
            .fillna(0)
            .round()
            .astype("int32")
        )

    pct_cols = [c for c in out.columns if c.startswith("pct_")]
    for c in pct_cols:
        out[c] = (
            pd.to_numeric(out[c], errors="coerce")
            .fillna(0.0)
            .astype("float32")
        )

    # NEW FEATURE
    if ("pct_age_18_34" in out.columns) and ("population" in out.columns):
        out["young_pop_pressure"] = (
            pd.to_numeric(out["pct_age_18_34"], errors="coerce").fillna(0.0) *
            pd.to_numeric(out["population"], errors="coerce").fillna(0.0)
        ).astype("float32")
    else:
        out["young_pop_pressure"] = np.float32(0.0)

    return out

def finalize_output(old_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    if old_df is None or len(old_df) == 0:
        return new_df.copy()

    out = pd.concat([old_df, new_df], ignore_index=True)

    if "GEOID" in out.columns:
        out["GEOID"] = normalize_geoid(out["GEOID"], GEOID_LEN)

    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    keys = detect_panel_keys(out)
    out = dedupe_by_panel_keys(out, keys, keep="first")

    # burada tam sort zorunlu değil; istersen açarsın
    # sort_cols = [c for c in ["date", "GEOID", "hour_range"] if c in out.columns]
    # if sort_cols:
    #     out = out.sort_values(sort_cols).reset_index(drop=True)

    return out


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("🚀 Demographic update başlıyor...")

    if not Path(CRIME_INPUT).exists():
        raise FileNotFoundError(f"❌ CRIME_INPUT bulunamadı: {CRIME_INPUT}")
    if not Path(DEMOGRAPHIC_PATH).exists():
        raise FileNotFoundError(f"❌ sf_population.csv bulunamadı: {DEMOGRAPHIC_PATH}")

    crime_in = pd.read_csv(CRIME_INPUT, low_memory=False)
    if "GEOID" not in crime_in.columns:
        raise KeyError("❌ Suç verisinde GEOID kolonu yok.")
    crime_in["GEOID"] = normalize_geoid(crime_in["GEOID"], GEOID_LEN)
    log_shape(crime_in, "CRIME INPUT")

    monthly_refresh = pd.Timestamp.utcnow().day == 1
    force_population_refresh = (
        os.getenv("FORCE_POPULATION_REFRESH", "0")
        .strip()
        .lower()
        in ("1", "true", "yes", "on")
    )
    
    if (
        os.path.exists(DEMOGRAPHIC_FEATURES_CSV)
        and not monthly_refresh
        and not force_population_refresh
    ):
        print(f"📦 Demographic cache kullanılıyor: {DEMOGRAPHIC_FEATURES_CSV}")
        demo_feat = pd.read_csv(DEMOGRAPHIC_FEATURES_CSV, dtype={"GEOID": str}, low_memory=False)
        demo_feat["GEOID"] = normalize_geoid(demo_feat["GEOID"], GEOID_LEN)
    else:
        print("♻️ Demographic features yeniden üretilecek.")
        print(f"monthly_refresh={monthly_refresh}, force_population_refresh={force_population_refresh}")
    
        demo_raw = pd.read_csv(DEMOGRAPHIC_PATH, low_memory=False, dtype=str)
        log_shape(demo_raw, "DEMOGRAPHIC CSV")
    
        demo_feat = build_demographic_features(demo_raw)

    old_out, new_rows = split_old_and_new_rows(crime_in, CRIME_OUTPUT)

    if new_rows.empty:
        print("✅ Yeni crime satırı yok. Eski sf_crime_03 korunuyor.")
        if os.path.exists(CRIME_OUTPUT):
            print(f"📁 Mevcut çıktı: {CRIME_OUTPUT}")
        return

    before = new_rows.shape
    new_rows_enriched = merge_demographics(new_rows, demo_feat)
    log_delta(before, new_rows_enriched.shape, "YENİ SATIRLAR ⨯ DEMOGRAPHIC")
    log_shape(new_rows_enriched, "NEW ENRICHED ROWS")

    # İlk full run ise gereksiz concat/sort/dedupe yapma
    if old_out is None or len(old_out) == 0:
        final_df = new_rows_enriched
        print("⚡ İlk/full run: finalize_output bypass edildi.")
    else:
        final_df = finalize_output(old_out, new_rows_enriched)

    log_shape(final_df, "FINAL sf_crime_03")

    if "population" in final_df.columns:
        zero_pop = int((pd.to_numeric(final_df["population"], errors="coerce").fillna(0) == 0).sum())
        print(f"🔎 population=0 olan satır sayısı: {zero_pop}")

    dem_cols = [c for c in FINAL_DEMOGRAPHIC_COLS if c != "GEOID" and c in final_df.columns]
    if dem_cols:
        print("🔎 Demographic kolon NaN sayıları:")
        print(final_df[dem_cols].isna().sum().to_string())

    safe_save_csv(final_df, CRIME_OUTPUT)

    try:
        preview_cols = [c for c in ["GEOID", "date", "hour_range"] + [c for c in FINAL_DEMOGRAPHIC_COLS if c != "GEOID"] if c in final_df.columns]
        print("📌 Önizleme:")
        print(final_df[preview_cols].tail(10).to_string(index=False))
    except Exception as e:
        print(f"ℹ️ Önizleme atlandı: {e}")


if __name__ == "__main__":
    main()
