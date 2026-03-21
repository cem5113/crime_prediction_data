# ============================================================
# ✅ enrich_with_neighbors_fr.py (FULL REVIZE v2.0 — PANEL-SAFE / LEAK-FREE / PUBLISH-LAG AWARE)
#
# Amaç:
# sf_crime_08.csv (panel) -> neighbor_crime_1h/3h/6h/24h/3d/7d ekleyip sf_crime_09.csv üretmek
#
# Kritik düzeltmeler:
#   ✅ PANEL-SAFE base_cnt: "tüm satırlar olay" YANLIŞTI -> artık y_count / crime_count / Y_label üzerinden
#   ✅ Leak-free + publish lag: 1h/3h/6h/24h/3d/7d geçmiş pencereleri publish lag ile hesaplanır
#   ✅ Kolon garanti: merge sonrası neighbor_crime_1h/3h/6h/24h/3d/7d yoksa HARD FAIL
#   ✅ Sadece komşu GEOID’lerin günlük olaylarını toplar (neighbor_cnt_day)
#   ✅ Günlük seri asfreq('D') ile tamamlanır (eksik gün=0)
#   ✅ Merge geri: GEOID×date anahtarıyla panelin tüm hour_range satırlarına yayılır
#   ✅ Opsiyonel legacy alias: nei_7d_sum istersen neighbor_crime_7d ile doldurulur

#
# Not:
#   Senin istediğin "son suç tarihine göre 1d/3d/7d" mantığı publish_lag_days ile karşılanır:
#     - suçlar 24–48 saat sonra yayımlanıyorsa publish_lag_days=2 (default)
#     - D gününün feature'ı, en fazla D-1-2 = D-3 gününe kadar "bilinen" suçları kullanır
# ============================================================

from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd
import numpy as np

pd.options.mode.copy_on_write = True

# ---------- utils ----------
def log(m: str):
    print(m, flush=True)

def _read_table(p: Path) -> pd.DataFrame:
    if not p.exists():
        log(f"ℹ️ Yok: {p}")
        return pd.DataFrame()

    if p.suffix.lower() == ".parquet":
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p, low_memory=False)

    log(f"📖 Okundu: {p} ({len(df):,}×{df.shape[1]})")
    return df

def _safe_write_table(df: pd.DataFrame, p: Path, write_csv: bool = False, csv_path: Path | None = None):
    p.parent.mkdir(parents=True, exist_ok=True)

    df2 = df.copy()
    for c in df2.select_dtypes(include=["float64"]).columns:
        df2[c] = pd.to_numeric(df2[c], downcast="float")
    for c in df2.select_dtypes(include=["int64", "Int64"]).columns:
        df2[c] = pd.to_numeric(df2[c], downcast="integer")

    tmp = Path(str(p) + ".tmp.parquet")
    df2.to_parquet(tmp, index=False, engine="pyarrow", compression="snappy")
    tmp.replace(p)
    log(f"💾 Parquet yazıldı: {p} ({len(df2):,}×{df2.shape[1]})")

    if write_csv and csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_csv = Path(str(csv_path) + ".tmp")
        df2.to_csv(tmp_csv, index=False, encoding="utf-8-sig")
        tmp_csv.replace(csv_path)
        log(f"💾 CSV yazıldı: {csv_path} ({len(df2):,}×{df2.shape[1]})")

def _norm_geoid(s: pd.Series, L: int = 11) -> pd.Series:
    return (
        s.astype(str)
         .str.extract(r"(\d+)", expand=False)
         .fillna("")
         .str[:L].str.zfill(L)
    )

# --- tarih yardımcıları: HER ZAMAN datetime64[ns] (SF local day) ---
def _as_date64(s: pd.Series) -> pd.Series:
    # 1) mixed tz/naive olasılığına karşı her şeyi UTC parse et
    dt = pd.to_datetime(s, errors="coerce", utc=True)

    # 2) SF gün anahtarı: UTC -> America/Los_Angeles
    dt = dt.dt.tz_convert("America/Los_Angeles")

    # 3) Gün başına indir (00:00) ve tz bilgisini at
    return dt.dt.normalize().dt.tz_localize(None)

def _ensure_date_col(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    cand = None
    for c in ["date", "incident_date", "incident_datetime", "datetime", "time", "timestamp"]:
        if c in d.columns:
            cand = c
            break

    if cand is None and {"year", "month", "day"}.issubset(d.columns):
        d["date"] = _as_date64(pd.to_datetime(d[["year", "month", "day"]], errors="coerce"))
    elif cand is not None:
        d["date"] = _as_date64(d[cand])
    else:
        d["date"] = pd.NaT

    return d

def _pick(cols, *cands):
    low = {c.lower(): c for c in cols}
    for k in cands:
        if k.lower() in low:
            return low[k.lower()]
    return None

def _normalize_keys(df: pd.DataFrame, geoid_len: int = 11) -> pd.DataFrame:
    d = df.copy()

    # GEOID kolonunu bul / normalize et
    geoid_col = None
    for cand in ["GEOID", "geoid", "grid_id", "gridid"]:
        if cand in d.columns:
            geoid_col = cand
            break
    if geoid_col is None:
        raise RuntimeError("❌ Girdi dosyasında GEOID/geoid kolonu yok!")

    if geoid_col != "GEOID":
        d["GEOID"] = d[geoid_col]

    d["GEOID"] = _norm_geoid(d["GEOID"], geoid_len)
    d = _ensure_date_col(d)
    return d

# ---------- config ----------
BASE_DIR = Path(os.environ.get("CRIME_DATA_DIR", "crime_prediction_data"))

FR_IN_ENV      = os.environ.get("sf_CRIME_IN", "sf_crime_08.parquet")
FR_OUT_ENV     = os.environ.get("sf_CRIME_OUT", "sf_crime_09.parquet")
FR_OUT_CSV_ENV = os.environ.get("sf_CRIME_OUT_CSV", "sf_crime_09.csv")

NEIGH_FILE_ENV = os.environ.get("NEIGH_FILE", "neighbors.csv")
GEOID_LEN = int(os.environ.get("GEOID_LEN", "11"))

PUBLISH_LAG_DAYS = int(os.environ.get("PUBLISH_LAG_DAYS", "2"))
MAKE_LEGACY_NEI7 = os.environ.get("MAKE_LEGACY_NEI7", "0").lower() in ("1", "true", "yes", "on")
WRITE_CSV = os.environ.get("WRITE_CSV", "0").lower() in ("1", "true", "yes", "on")


# ---------- core ----------
def _build_base_daily_counts(panel: pd.DataFrame) -> pd.DataFrame:
    """
    PANEL (GEOID×date×hour_range) içinden GEOID×date günlük gerçek olay sayısını üretir.
    Öncelik:
      1) y_count
      2) crime_count
      3) Y_label (0/1)
    """
    d = panel.copy()

    # en iyi kolon hangisi?
    if "y_count" in d.columns:
        src = "y_count"
        s = pd.to_numeric(d["y_count"], errors="coerce").fillna(0.0)
    elif "crime_count" in d.columns:
        src = "crime_count"
        s = pd.to_numeric(d["crime_count"], errors="coerce").fillna(0.0)
    elif "Y_label" in d.columns:
        src = "Y_label"
        s = pd.to_numeric(d["Y_label"], errors="coerce").fillna(0.0)
    else:
        raise RuntimeError("❌ base_cnt üretmek için y_count / crime_count / Y_label bulunamadı.")

    d["_cnt_src_"] = s

    base = (
        d.groupby(["GEOID", "date"], dropna=False)["_cnt_src_"]
         .sum()
         .reset_index(name="base_cnt")
    )

    # int'e indir (negatif varsa 0'a kırp)
    base["base_cnt"] = pd.to_numeric(base["base_cnt"], errors="coerce").fillna(0).clip(lower=0).round().astype("int64")

    log(f"🧮 base_cnt üretildi (kaynak={src}) | rows={len(base):,} | base_cnt.sum={int(base['base_cnt'].sum()):,}")
    return base


def neighbor_daily_features(
    base: pd.DataFrame,
    neigh: pd.DataFrame,
    publish_lag_days: int = 2,
) -> pd.DataFrame:
    """
    base: GEOID×date×base_cnt (günlük gerçek olay sayısı)  ✅ base_cnt doğru olmalı
    neigh: geoid, neighbor

    publish_lag_days:
      - suçların sisteme düşme/yayımlanma gecikmesi (tipik 1–3 gün)
      - D günü feature'ı: en fazla D-1-publish_lag_days gününe kadar bilinen olaylar
        => shift_k = 1 + publish_lag_days

    Çıktı:
      GEOID×date:
        neighbor_crime_1h
        neighbor_crime_3h
        neighbor_crime_6h
        neighbor_crime_24h
        neighbor_crime_3d
        neighbor_crime_7d
    """
    # ---- normalize base ----
    b = base.copy()
    if "GEOID" not in b.columns or "date" not in b.columns or "base_cnt" not in b.columns:
        raise RuntimeError("❌ base beklenen kolonlar: GEOID, date, base_cnt")

    b["GEOID"]    = _norm_geoid(b["GEOID"], GEOID_LEN)
    b["date"]     = _as_date64(b["date"])
    b["base_cnt"] = pd.to_numeric(b["base_cnt"], errors="coerce").fillna(0).clip(lower=0).round().astype("int64")
    b = b.dropna(subset=["GEOID", "date"])

    # ---- komşuluk normalize ----
    nb = neigh.rename(columns={
        _pick(neigh.columns, "geoid", "src", "source"): "geoid",
        _pick(neigh.columns, "neighbor", "dst", "target"): "neighbor"
    }).copy()

    if "geoid" not in nb.columns or "neighbor" not in nb.columns:
        raise RuntimeError("❌ neighbors.csv kolonları geoid/neighbor değil. Kolonları kontrol et.")

    nb["geoid"]    = _norm_geoid(nb["geoid"], GEOID_LEN)
    nb["neighbor"] = _norm_geoid(nb["neighbor"], GEOID_LEN)
    nb = nb.dropna(subset=["geoid", "neighbor"]).drop_duplicates()

    # ---- neighbor'ın base_cnt'sini ana geoid'e taşı ----
    b_neighbor = b.rename(columns={"GEOID": "neighbor"})
    nb_merge = nb.merge(b_neighbor, on="neighbor", how="left")

    # neighbor tarafında base_cnt yoksa 0
    nb_merge["base_cnt"] = pd.to_numeric(nb_merge["base_cnt"], errors="coerce").fillna(0).astype("int64")

    day_sum = (
        nb_merge.groupby(["geoid", "date"], dropna=False)["base_cnt"]
                .sum().reset_index(name="neighbor_cnt_day")
    )

    # ---- GEOID bazında tam günlük seri + publish lag'li rolling ----
    shift_k = int(publish_lag_days) + 1  # leak-free (dünü) + yayın gecikmesi

    def _per_geoid(gdf: pd.DataFrame) -> pd.DataFrame:
        gdf = gdf.sort_values("date").set_index("date")
        gdf = gdf.asfreq("D", fill_value=0)  # eksik günler 0
    
        s = gdf["neighbor_cnt_day"].astype("int64")
    
        # Not:
        # Bu seri günlük olduğu için 1h/3h/6h/24h kolonları gerçek saatlik değil,
        # publish-lag-aware "en son bilinen gün" mantığıyla yaklaşık kısa dönem temsilidir.
        s_shift = s.shift(shift_k)
    
        gdf["neighbor_crime_1h"]  = s_shift.fillna(0).astype("int64")
        gdf["neighbor_crime_3h"]  = s_shift.fillna(0).astype("int64")
        gdf["neighbor_crime_6h"]  = s_shift.fillna(0).astype("int64")
        gdf["neighbor_crime_24h"] = s_shift.fillna(0).astype("int64")
        gdf["neighbor_crime_3d"]  = s_shift.rolling(3, min_periods=1).sum().fillna(0).astype("int64")
        gdf["neighbor_crime_7d"]  = s_shift.rolling(7, min_periods=1).sum().fillna(0).astype("int64")
    
        return gdf.reset_index()

    out = (
        day_sum.groupby("geoid", group_keys=False)
               .apply(_per_geoid)
               .reset_index(drop=True)
    )

    out = out.rename(columns={"geoid": "GEOID"})[
        [
            "GEOID",
            "date",
            "neighbor_crime_1h",
            "neighbor_crime_3h",
            "neighbor_crime_6h",
            "neighbor_crime_24h",
            "neighbor_crime_3d",
            "neighbor_crime_7d",
        ]
    ]

    out["GEOID"] = _norm_geoid(out["GEOID"], GEOID_LEN)
    out["date"]  = _as_date64(out["date"])

    for c in [
        "neighbor_crime_1h",
        "neighbor_crime_3h",
        "neighbor_crime_6h",
        "neighbor_crime_24h",
        "neighbor_crime_3d",
        "neighbor_crime_7d",
    ]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).clip(lower=0).round().astype("int64")

    return out


def main() -> int:
    log("🚀 enrich_with_neighbors_fr.py (sf_crime_08 → sf_crime_09) — FULL REVIZE v2.0")

    BASE_DIR.mkdir(parents=True, exist_ok=True)

    fr_in = BASE_DIR / FR_IN_ENV if not Path(FR_IN_ENV).is_absolute() else Path(FR_IN_ENV)
    fr_out = BASE_DIR / FR_OUT_ENV if not Path(FR_OUT_ENV).is_absolute() else Path(FR_OUT_ENV)
    fr_out_csv = BASE_DIR / FR_OUT_CSV_ENV if not Path(FR_OUT_CSV_ENV).is_absolute() else Path(FR_OUT_CSV_ENV)

    neigh_path = BASE_DIR / NEIGH_FILE_ENV if not Path(NEIGH_FILE_ENV).is_absolute() else Path(NEIGH_FILE_ENV)

    # parquet yoksa csv fallback
    if not fr_in.exists():
        alt_csv = fr_in.with_suffix(".csv")
        if alt_csv.exists():
            fr_in = alt_csv

    if not neigh_path.exists():
        alt_neigh = Path("neighbors.csv")
        if alt_neigh.exists():
            neigh_path = alt_neigh

    log(f"📥 FR input : {fr_in}")
    log(f"📤 FR output: {fr_out}")
    log(f"📂 NEIGH PATH: {neigh_path.resolve()}")
    log(f"⏳ PUBLISH_LAG_DAYS={PUBLISH_LAG_DAYS}  (shift_k={PUBLISH_LAG_DAYS+1})")

    df_raw = _read_table(fr_in)
    if df_raw.empty:
        raise RuntimeError("❌ Girdi dosyası boş veya okunamadı.")

    if not neigh_path.exists():
        raise FileNotFoundError(f"❌ neighbors.csv bulunamadı: {neigh_path.resolve()}")

    neigh = pd.read_csv(neigh_path, low_memory=False).dropna()
    if neigh.empty:
        raise RuntimeError("❌ neighbors.csv boş.")

    # ---- normalize input ----
    df = _normalize_keys(df_raw, GEOID_LEN)
    df = df.dropna(subset=["GEOID", "date"])
    log(f"🧹 Normalize sonrası satır: {len(df):,}")

    # ---- base_cnt: PANELDEN DOĞRU ÜRET ----
    base = _build_base_daily_counts(df)

    # ---- neighbor features ----
    feats = neighbor_daily_features(base, neigh, publish_lag_days=PUBLISH_LAG_DAYS)
    log(
        f"✨ neighbor feats: {len(feats):,} satır (GEOID×date) — kolonlar: "
        f"[neighbor_crime_1h, neighbor_crime_3h, neighbor_crime_6h, "
        f"neighbor_crime_24h, neighbor_crime_3d, neighbor_crime_7d]"
    )

    # ---- merge back to panel (her hour_range satırına aynı günlük değerler yayılır) ----
    feats["GEOID"] = _norm_geoid(feats["GEOID"], GEOID_LEN)
    feats["date"]  = _as_date64(feats["date"])

    nb_cols = [
        "neighbor_crime_1h",
        "neighbor_crime_3h",
        "neighbor_crime_6h",
        "neighbor_crime_24h",
        "neighbor_crime_3d",
        "neighbor_crime_7d",
    ]
    df = df.drop(columns=[c for c in nb_cols if c in df.columns], errors="ignore")

    df_out = df.merge(feats, on=["GEOID", "date"], how="left")

    # fill
    for c in nb_cols:
        if c in df_out.columns:
            df_out[c] = pd.to_numeric(df_out[c], errors="coerce").fillna(0).clip(lower=0).round().astype("int64")

    # ---- legacy alias (opsiyonel) ----
    if MAKE_LEGACY_NEI7:
        if "nei_7d_sum" not in df_out.columns:
            df_out["nei_7d_sum"] = df_out["neighbor_crime_7d"].astype("int64")
            log("🧷 legacy: nei_7d_sum = neighbor_crime_7d eklendi.")
        else:
            # varsa güncellemek istersen:
            # df_out["nei_7d_sum"] = df_out["neighbor_crime_7d"].astype("int64")
            log("🧷 legacy: nei_7d_sum zaten var (dokunmadım).")

    # ---- HARD GUARANTEE: kolonlar var mı? ----
    missing = [c for c in nb_cols if c not in df_out.columns]
    if missing:
        raise RuntimeError(f"❌ Merge sonrası komşu kolonları yok: {missing}")

    # ---- kalite kontrol: beklenen aralıklar ----
    log("🔎 QC:")
    log(f"  neighbor_crime_1h sum={int(df_out['neighbor_crime_1h'].sum()):,}")
    log(f"  neighbor_crime_3h sum={int(df_out['neighbor_crime_3h'].sum()):,}")
    log(f"  neighbor_crime_6h sum={int(df_out['neighbor_crime_6h'].sum()):,}")
    log(f"  neighbor_crime_24h sum={int(df_out['neighbor_crime_24h'].sum()):,}")
    log(f"  neighbor_crime_3d sum={int(df_out['neighbor_crime_3d'].sum()):,}")
    log(f"  neighbor_crime_7d sum={int(df_out['neighbor_crime_7d'].sum()):,}")

    _safe_write_table(df_out, fr_out, write_csv=WRITE_CSV, csv_path=fr_out_csv)

    # Preview
    try:
        cols = ["GEOID", "date"] + nb_cols
        log("— OUTPUT preview —")
        log(df_out[cols].head(10).to_string(index=False))
    except Exception:
        pass

    log("✅ Tamam.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
