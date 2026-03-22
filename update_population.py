# update_population.py
# =============================================================================
# DEMOGRAPHIC / POPULATION ENRICH (APPEND-ONLY, TRACT/GEOID-BASED)
#
# AMAÇ
# ----
# 1) sf_population.csv dosyasından tract-level demografik öznitelikler üretmek
# 2) sf_crime_02.csv -> sf_crime_03.csv zenginleştirmesini yapmak
# 3) İlk koşuda tüm veriyi enrich etmek
# 4) Sonraki koşularda eski sf_crime_03 satırlarını değiştirmeden,
#    yalnızca yeni crime satırlarını yeni demografi snapshot'ı ile eklemek
#
# NOTLAR
# ------
# - Vektörel feature mapping kullanır (apply(axis=1) yok)
# - Remote population kaynağı varsa onu dener; başarısızsa yerelden devam eder
# - Eski crime satırları append-only mantıkla korunur
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
    print(f"📊 {label}: {r} satır × {c} sütun", flush=True)


def log_delta(before_shape, after_shape, label: str):
    br, bc = before_shape
    ar, ac = after_shape
    print(f"🔗 {label}: {br}×{bc} → {ar}×{ac} (Δr={ar-br}, Δc={ac-bc})", flush=True)


def ensure_parent(path: str):
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def safe_save_csv(df: pd.DataFrame, path: str):
    ensure_parent(path)
    tmp = str(path) + ".tmp"
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)
    print(f"💾 Kaydedildi: {path}", flush=True)


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
         .str.replace(",", "", regex=False)
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
        print(f"⚠️ Panel anahtarlarına göre {dropped} duplikasyon temizlendi.", flush=True)
    return out


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
        DEMOGRAPHIC_PATH = str(BASE_DIR / "sf_population.csv")

DEMOGRAPHIC_FEATURES_CSV = str(BASE_DIR / "sf_demographic_features.csv")

GEOID_LEN = int(os.getenv("GEOID_LEN", "11"))
APPEND_ONLY = os.getenv("APPEND_ONLY", "1").strip().lower() not in ("0", "false", "no")

POP_REMOTE_URL = os.getenv(
    "POP_REMOTE_URL",
    "https://data.sfgov.org/api/views/4qbq-hvtt/rows.csv?accessType=DOWNLOAD"
).strip()
USE_REMOTE_POP = os.getenv("USE_REMOTE_POP", "1").strip().lower() not in ("0", "false", "no")


# =============================================================================
# POPULATION SOURCE
# =============================================================================
def load_population_source() -> pd.DataFrame:
    """
    Önce remote source dener.
    Başarılıysa yerel sf_population.csv dosyasını günceller.
    Başarısızsa mevcut yerel dosya ile devam eder.
    """
    local_path = Path(DEMOGRAPHIC_PATH)

    if USE_REMOTE_POP:
        try:
            print(f"🌐 Remote population deneniyor: {POP_REMOTE_URL}", flush=True)
            remote_df = pd.read_csv(POP_REMOTE_URL, low_memory=False, dtype=str)
            if remote_df is not None and not remote_df.empty:
                print("✅ Remote population okundu.", flush=True)
                log_shape(remote_df, "REMOTE POPULATION")
                safe_save_csv(remote_df, str(local_path))
                return remote_df
            else:
                print("⚠️ Remote population boş döndü, yerelden devam edilecek.", flush=True)
        except Exception as e:
            print(f"⚠️ Remote population okunamadı: {e}", flush=True)

    if local_path.exists():
        print(f"📂 Yerel population okunuyor: {local_path}", flush=True)
        local_df = pd.read_csv(local_path, low_memory=False, dtype=str)
        print("✅ Yerel population okundu.", flush=True)
        log_shape(local_df, "LOCAL POPULATION")
        return local_df

    raise FileNotFoundError(
        f"❌ Ne remote population okunabildi ne de yerel dosya bulundu: {local_path}"
    )


# =============================================================================
# DEMOGRAPHIC FEATURE EXTRACTION
# =============================================================================
def select_latest_rows_per_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aynı GEOID + feature birden çok ACS döneminde gelebilir.
    En güncel satırı seç:
      - end_year büyük
      - data_loaded_at büyük
      - data_as_of büyük
    """
    x = df.copy()

    x["end_year_num"] = pd.to_numeric(x["end_year"], errors="coerce") if "end_year" in x.columns else np.nan
    x["data_loaded_at_ts"] = pd.to_datetime(x["data_loaded_at"], errors="coerce") if "data_loaded_at" in x.columns else pd.NaT
    x["data_as_of_ts"] = pd.to_datetime(x["data_as_of"], errors="coerce") if "data_as_of" in x.columns else pd.NaT

    sort_cols = ["GEOID", "feature_name", "end_year_num", "data_loaded_at_ts", "data_as_of_ts"]
    sort_cols = [c for c in sort_cols if c in x.columns]

    x = x.sort_values(sort_cols)
    x = x.drop_duplicates(subset=["GEOID", "feature_name"], keep="last")
    return x


def map_feature_names_vectorized(df: pd.DataFrame) -> pd.Series:
    """
    apply(axis=1) yerine vektörel feature name üretimi.
    """
    acs_table = df.get("acs_table", pd.Series("", index=df.index)).astype(str).str.strip()
    acs_label = df.get("acs_label", pd.Series("", index=df.index)).astype(str).str.strip().str.lower()
    demo_label = df.get("demographic_category_label", pd.Series("", index=df.index)).astype(str).str.strip().str.lower()

    feat = pd.Series(pd.NA, index=df.index, dtype="object")

    # population_total
    m = (acs_table.eq("B01001") & acs_label.eq("estimate!!total:")) | \
        (acs_table.eq("B03002") & acs_label.eq("estimate!!total:"))
    feat = feat.mask(m, "population_total")

    # B06001 age groups
    age_map = {
        "under 5 years": "pop_age_under_5",
        "5 to 17 years": "pop_age_5_17",
        "18 to 24 years": "pop_age_18_24",
        "25 to 34 years": "pop_age_25_34",
        "35 to 44 years": "pop_age_35_44",
        "45 to 54 years": "pop_age_45_54",
        "55 to 64 years": "pop_age_55_64",
        "65 to 74 years": "pop_age_65_74",
        "75 to 84 years": "pop_age_75_84",
        "85 years and over": "pop_age_85_plus",
    }
    m_b06001 = acs_table.eq("B06001")
    for k, v in age_map.items():
        feat = feat.mask(m_b06001 & demo_label.eq(k), v)

    # B03002 race/ethnicity
    m_b03002 = acs_table.eq("B03002")
    feat = feat.mask(m_b03002 & acs_label.eq("estimate!!total:!!hispanic or latino:"), "pop_hispanic")
    feat = feat.mask(m_b03002 & acs_label.str.contains("not hispanic or latino:!!white alone", na=False), "pop_nh_white")
    feat = feat.mask(m_b03002 & acs_label.str.contains("not hispanic or latino:!!black or african american alone", na=False), "pop_nh_black")
    feat = feat.mask(m_b03002 & acs_label.str.contains("not hispanic or latino:!!asian alone", na=False), "pop_nh_asian")
    feat = feat.mask(m_b03002 & acs_label.str.contains("not hispanic or latino:!!american indian and alaska native alone", na=False), "pop_nh_native")
    feat = feat.mask(m_b03002 & acs_label.str.contains("not hispanic or latino:!!native hawaiian and other pacific islander alone", na=False), "pop_nh_pacific")
    feat = feat.mask(m_b03002 & acs_label.str.contains("not hispanic or latino:!!some other race alone", na=False), "pop_nh_other")
    feat = feat.mask(m_b03002 & acs_label.str.contains("not hispanic or latino:!!two or more races", na=False), "pop_nh_multiracial")

    return feat


def build_demographic_features(demo_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Tract-level demografik feature tablosu üretir.
    Çıktı: GEOID bazında tekil feature tablosu.
    """
    print("   -> build_demographic_features başladı", flush=True)

    df = demo_raw.copy()
    log_shape(df, "DEMOGRAPHIC RAW")

    required_cols = ["geography", "geography_id", "estimate"]
    for c in required_cols:
        if c not in df.columns:
            raise KeyError(f"❌ Demografik dosyada zorunlu kolon eksik: {c}")

    print("   -> tract filter", flush=True)
    df = df[df["geography"].astype(str).str.lower().eq("tract")].copy()
    log_shape(df, "DEMOGRAPHIC (yalnız tract)")

    print("   -> GEOID normalize", flush=True)
    df["GEOID"] = normalize_geoid(df["geography_id"], GEOID_LEN)

    print("   -> estimate numeric", flush=True)
    df["estimate_num"] = parse_numeric_series(df["estimate"])

    print("   -> vectorized feature mapping", flush=True)
    df["feature_name"] = map_feature_names_vectorized(df)
    df = df.dropna(subset=["GEOID", "feature_name"]).copy()
    log_shape(df, "DEMOGRAPHIC (feature-mapped)")

    print("   -> latest snapshot select", flush=True)
    df = select_latest_rows_per_feature(df)

    print("   -> pivot", flush=True)
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

    denom = pd.to_numeric(feat["population_total"], errors="coerce").replace(0, np.nan)

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

    num_cols = [c for c in feat.columns if c != "GEOID"]
    for c in num_cols:
        feat[c] = pd.to_numeric(feat[c], errors="coerce")

    abs_cols = [c for c in num_cols if not c.startswith("pct_")]
    pct_cols = [c for c in num_cols if c.startswith("pct_")]

    feat[abs_cols] = feat[abs_cols].fillna(0)
    feat[pct_cols] = feat[pct_cols].fillna(0.0)

    if "population_total" in feat.columns:
        feat["population_total"] = feat["population_total"].round().astype(int)

    feat["population"] = feat["population_total"]

    feat = feat.sort_values("GEOID").drop_duplicates(subset="GEOID", keep="last")
    log_shape(feat, "DEMOGRAPHIC FEATURES (GEOID-level)")

    safe_save_csv(feat, DEMOGRAPHIC_FEATURES_CSV)
    print("   -> build_demographic_features bitti", flush=True)
    return feat


# =============================================================================
# APPEND-ONLY LOGIC
# =============================================================================
def split_old_and_new_rows(crime_in: pd.DataFrame, crime_out_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not APPEND_ONLY or not os.path.exists(crime_out_path):
        print("ℹ️ Append-only çıktı yok → tüm satırlar yeni kabul edilecek.", flush=True)
        return pd.DataFrame(columns=crime_in.columns), crime_in.copy()

    old = pd.read_csv(crime_out_path, low_memory=False)
    if "GEOID" not in old.columns:
        print("⚠️ Eski sf_crime_03 içinde GEOID yok → tüm giriş yeni kabul edilecek.", flush=True)
        return old, crime_in.copy()

    old["GEOID"] = normalize_geoid(old["GEOID"], GEOID_LEN)
    crime_in = crime_in.copy()
    crime_in["GEOID"] = normalize_geoid(crime_in["GEOID"], GEOID_LEN)

    keys = detect_panel_keys(crime_in)
    if set(["GEOID", "date", "hour_range"]).issubset(keys) and set(["GEOID", "date", "hour_range"]).issubset(old.columns):
        crime_in = ensure_date_col(crime_in, "date")
        old = ensure_date_col(old, "date")

        old_keys = old[["GEOID", "date", "hour_range"]].drop_duplicates().copy()
        old_keys["__seen__"] = 1

        marked = crime_in.merge(old_keys, on=["GEOID", "date", "hour_range"], how="left")
        new_rows = marked[marked["__seen__"].isna()].drop(columns=["__seen__"]).copy()

        print("🧠 Yeni satır tespiti: GEOID + date + hour_range", flush=True)
        log_shape(old, "MEVCUT sf_crime_03")
        log_shape(crime_in, "GÜNCEL sf_crime_02")
        log_shape(new_rows, "YENİ CRIME SATIRLARI")
        return old, new_rows

    print("⚠️ Panel anahtarları eksik → tüm sf_crime_02 yeni kabul ediliyor.", flush=True)
    return old, crime_in.copy()


def merge_demographics(df_crime: pd.DataFrame, demo_feat: pd.DataFrame) -> pd.DataFrame:
    out = df_crime.copy()
    out["GEOID"] = normalize_geoid(out["GEOID"], GEOID_LEN)

    overlap = (set(out.columns) & set(demo_feat.columns)) - {"GEOID"}
    if overlap:
        print(f"🧹 DEMOGRAPHIC merge overlap bulundu, düşürülüyor: {sorted(overlap)}", flush=True)
        demo_feat = demo_feat.drop(columns=list(overlap), errors="ignore")

    before = out.shape
    out = out.merge(demo_feat, on="GEOID", how="left", validate="many_to_one")
    log_delta(before, out.shape, "CRIME ⨯ DEMOGRAPHIC")

    for c in out.columns:
        if c.startswith("pct_"):
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    for c in out.columns:
        if c.startswith("pop_") or c in ("population", "population_total"):
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)

    if "population" in out.columns:
        out["population"] = out["population"].round().astype(int)
    if "population_total" in out.columns:
        out["population_total"] = out["population_total"].round().astype(int)

    return out


def finalize_output(old_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    if old_df is None or len(old_df) == 0:
        out = new_df.copy()
    else:
        out = pd.concat([old_df, new_df], ignore_index=True)

    if "GEOID" in out.columns:
        out["GEOID"] = normalize_geoid(out["GEOID"], GEOID_LEN)

    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    keys = detect_panel_keys(out)
    out = dedupe_by_panel_keys(out, keys, keep="first")

    sort_cols = [c for c in ["date", "GEOID", "hour_range"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)

    return out


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("🚀 Demographic update başlıyor...", flush=True)
    print(f"CRIME_INPUT={CRIME_INPUT}", flush=True)
    print(f"DEMOGRAPHIC_PATH={DEMOGRAPHIC_PATH}", flush=True)
    print(f"CRIME_OUTPUT={CRIME_OUTPUT}", flush=True)
    print(f"USE_REMOTE_POP={USE_REMOTE_POP}", flush=True)

    if not Path(CRIME_INPUT).exists():
        raise FileNotFoundError(f"❌ CRIME_INPUT bulunamadı: {CRIME_INPUT}")

    # 1) Crime input oku
    print("1) crime okunacak", flush=True)
    crime_in = pd.read_csv(CRIME_INPUT, low_memory=False)
    print("1) crime okundu", flush=True)

    if "GEOID" not in crime_in.columns:
        raise KeyError("❌ Suç verisinde GEOID kolonu yok.")

    crime_in["GEOID"] = normalize_geoid(crime_in["GEOID"], GEOID_LEN)
    log_shape(crime_in, "CRIME INPUT")

    # 2) Demographic source oku
    print("2) demographic source hazırlanıyor", flush=True)
    demo_raw = load_population_source()
    print("2) demographic source hazır", flush=True)
    log_shape(demo_raw, "DEMOGRAPHIC CSV")

    # 3) Feature üret
    print("3) feature build başlayacak", flush=True)
    demo_feat = build_demographic_features(demo_raw)
    print("3) feature build bitti", flush=True)

    # 4) Append-only split
    old_out, new_rows = split_old_and_new_rows(crime_in, CRIME_OUTPUT)

    if new_rows.empty:
        print("✅ Yeni crime satırı yok. Eski sf_crime_03 korunuyor.", flush=True)
        if os.path.exists(CRIME_OUTPUT):
            print(f"📁 Mevcut çıktı: {CRIME_OUTPUT}", flush=True)
        return

    # 5) Sadece yeni satırları enrich et
    before = new_rows.shape
    new_rows_enriched = merge_demographics(new_rows, demo_feat)
    log_delta(before, new_rows_enriched.shape, "YENİ SATIRLAR ⨯ DEMOGRAPHIC")
    log_shape(new_rows_enriched, "NEW ENRICHED ROWS")

    # 6) Eski + yeni birleştir
    final_df = finalize_output(old_out, new_rows_enriched)
    log_shape(final_df, "FINAL sf_crime_03")

    # 7) Coverage
    if "population" in final_df.columns:
        zero_pop = int((pd.to_numeric(final_df["population"], errors="coerce").fillna(0) == 0).sum())
        print(f"🔎 population=0 olan satır sayısı: {zero_pop}", flush=True)

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
        print("🔎 Demographic kolon NaN sayıları:", flush=True)
        print(final_df[dem_cols].isna().sum().to_string(), flush=True)

    # 8) Kaydet
    safe_save_csv(final_df, CRIME_OUTPUT)

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
            print("📌 Önizleme:", flush=True)
            print(final_df[preview_cols].tail(10).to_string(index=False), flush=True)
    except Exception as e:
        print(f"ℹ️ Önizleme atlandı: {e}", flush=True)


if __name__ == "__main__":
    main()
