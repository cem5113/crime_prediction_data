#!/usr/bin/env python3
# =============================================================================
# ✅ scripts/update_neighbors.py (FULL REVIZE v2.1 — PANEL-SAFE / LEAK-FREE / PUBLISH-LAG AWARE)
#
# Üretir:
#   neighbor_crime_1h, neighbor_crime_3h, neighbor_crime_6h,
#   neighbor_crime_24h, neighbor_crime_3d, neighbor_crime_7d  (GEOID×date)
#   + panelin tüm hour_range satırlarına GEOID×date ile yayar
# Opsiyonel:
#   nei_7d_sum = neighbor_crime_7d (legacy)
#
# ENV:
#   CRIME_DATA_DIR
#   NEIGHBOR_INPUT_CSV   (default: sf_crime_08.csv)
#   NEIGHBOR_OUTPUT_CSV  (default: sf_crime_09.csv)
#   NEIGHBOR_FILE        (default: neighbors.csv)
#   GEOID_LEN            (default: 11)
#   PUBLISH_LAG_DAYS     (default: 2)   # 24–48h için 2
#   MAKE_LEGACY_NEI7     (default: 1)   # legacy nei_7d_sum üret
# =============================================================================

from __future__ import annotations

import os
import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("CRIME_DATA_DIR", str(ROOT / "crime_prediction_data"))).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)

IN_CSV_ENV  = os.environ.get("NEIGHBOR_INPUT_CSV",  "sf_crime_08.csv")
OUT_CSV_ENV = os.environ.get("NEIGHBOR_OUTPUT_CSV", "sf_crime_09.csv")
NEIGHBOR_FILE_ENV = os.environ.get("NEIGHBOR_FILE", "neighbors.csv")

GEOID_LEN = int(os.environ.get("GEOID_LEN", "11"))

PUBLISH_LAG_DAYS = int(os.environ.get("PUBLISH_LAG_DAYS", "2"))
SHIFT_K = 1 + PUBLISH_LAG_DAYS   # leak-free (d-1) + publish lag

MAKE_LEGACY_NEI7 = os.environ.get("MAKE_LEGACY_NEI7", "1").lower() in ("1","true","yes","on")

def _resolve(p: str) -> Path:
    q = Path(p)
    return q.resolve() if q.is_absolute() else (DATA_DIR / q).resolve()

IN_CSV = _resolve(IN_CSV_ENV)
OUT_CSV = _resolve(OUT_CSV_ENV)
NEIGHBOR_FILE = _resolve(NEIGHBOR_FILE_ENV)

def _norm_geoid(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.extract(r"(\d+)", expand=False)
         .fillna("")
         .str[:GEOID_LEN]
         .str.zfill(GEOID_LEN)
    )

def _pick_col(cols, *cands):
    low = {c.lower(): c for c in cols}
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    return None

def _as_date64_any(s: pd.Series) -> pd.Series:
    """
    SF local day -> tz-naive midnight (datetime64[ns])
    date string ise gün kayması yapmaz; datetime ise SF gününe çevirir.
    """
    # date gibi geliyorsa (YYYY-MM-DD), utc convert yapmadan parse
    dt = pd.to_datetime(s, errors="coerce")
    # tz-aware ise SF'ye çevir
    try:
        if getattr(dt.dt, "tz", None) is not None:
            dt = dt.dt.tz_convert("America/Los_Angeles")
        else:
            # naive ama saat içeriyorsa: SF kabul et
            # (tam date ise zaten 00:00)
            pass
    except Exception:
        pass
    return dt.dt.normalize()

def _build_base_daily_counts(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Günlük gerçek olay sayısı (GEOID×date -> base_cnt)
    Öncelik: y_count > crime_count > Y_label
    """
    if "y_count" in panel.columns:
        src = "y_count"
        s = pd.to_numeric(panel["y_count"], errors="coerce").fillna(0)
    elif "crime_count" in panel.columns:
        src = "crime_count"
        s = pd.to_numeric(panel["crime_count"], errors="coerce").fillna(0)
    elif "Y_label" in panel.columns:
        src = "Y_label"
        s = pd.to_numeric(panel["Y_label"], errors="coerce").fillna(0)
    else:
        raise RuntimeError("❌ y_count / crime_count / Y_label yok; base_cnt üretilemedi.")

    tmp = panel[["GEOID","date"]].copy()
    tmp["_cnt_"] = s.clip(lower=0)

    base = (
        tmp.groupby(["GEOID","date"], as_index=False)["_cnt_"].sum()
           .rename(columns={"_cnt_":"base_cnt"})
    )
    base["base_cnt"] = pd.to_numeric(base["base_cnt"], errors="coerce").fillna(0).clip(lower=0).round().astype("int64")
    print(f"🧮 base_cnt source={src} | rows={len(base):,} | sum={int(base['base_cnt'].sum()):,}", flush=True)
    return base

def _neighbor_daily_features(base: pd.DataFrame, nbr: pd.DataFrame) -> pd.DataFrame:
    """
    base: GEOID,date,base_cnt  (date: datetime64[ns] midnight)
    nbr: geoid, neighbor
    """
    b = base.copy()
    b["GEOID"] = _norm_geoid(b["GEOID"])
    b["date"] = pd.to_datetime(b["date"], errors="coerce")
    b["base_cnt"] = pd.to_numeric(b["base_cnt"], errors="coerce").fillna(0).clip(lower=0).round().astype("int64")
    b = b.dropna(subset=["GEOID","date"])

    n = nbr.copy()
    n["geoid"] = _norm_geoid(n["geoid"])
    n["neighbor"] = _norm_geoid(n["neighbor"])
    n = n.dropna().drop_duplicates()
    n = n[n["geoid"] != n["neighbor"]]

    # neighbor->base bağla
    b_nei = b.rename(columns={"GEOID":"neighbor"})
    m = n.merge(b_nei, on="neighbor", how="left")
    m["base_cnt"] = pd.to_numeric(m["base_cnt"], errors="coerce").fillna(0).astype("int64")

    day_sum = (
        m.groupby(["geoid","date"], as_index=False)["base_cnt"]
         .sum()
         .rename(columns={"geoid":"GEOID", "base_cnt":"neighbor_cnt_day"})
    )

    day_sum = day_sum.sort_values(["GEOID","date"])

    def per_geoid(g: pd.DataFrame) -> pd.DataFrame:
        g = g.set_index("date").asfreq("D", fill_value=0)
        s = g["neighbor_cnt_day"].astype("int64")
    
        # günlük seri olduğu için 1h/3h/6h/24h kolonları
        # gerçek saatlik değil, publish-lag-aware kısa dönem approx'tır
        s_shift = s.shift(SHIFT_K)
    
        g["neighbor_crime_1h"]  = s_shift.fillna(0).astype("int64")
        g["neighbor_crime_3h"]  = s_shift.fillna(0).astype("int64")
        g["neighbor_crime_6h"]  = s_shift.fillna(0).astype("int64")
        g["neighbor_crime_24h"] = s_shift.fillna(0).astype("int64")
        g["neighbor_crime_3d"]  = s_shift.rolling(3, min_periods=1).sum().fillna(0).astype("int64")
        g["neighbor_crime_7d"]  = s_shift.rolling(7, min_periods=1).sum().fillna(0).astype("int64")
    
        return g.reset_index()
    
    out = day_sum.groupby("GEOID", group_keys=False).apply(per_geoid).reset_index(drop=True)
    return out[
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

def main():
    print("=============================================================", flush=True)
    print("🧭 update_neighbors.py — FULL REVIZE v2.1", flush=True)
    print("DATA_DIR     :", DATA_DIR, flush=True)
    print("IN_CSV       :", IN_CSV, flush=True)
    print("OUT_CSV      :", OUT_CSV, flush=True)
    print("NEIGHBOR_FILE:", NEIGHBOR_FILE, flush=True)
    print(f"PUBLISH_LAG_DAYS={PUBLISH_LAG_DAYS} (SHIFT_K={SHIFT_K})", flush=True)
    print("=============================================================", flush=True)

    if not IN_CSV.exists():
        raise FileNotFoundError(f"❌ IN_CSV yok: {IN_CSV}")
    if not NEIGHBOR_FILE.exists():
        raise FileNotFoundError(f"❌ neighbors.csv yok: {NEIGHBOR_FILE}")

    df = pd.read_csv(IN_CSV, low_memory=False)
    df.columns = df.columns.str.replace("\ufeff", "", regex=False)

    # GEOID
    gcol = _pick_col(df.columns, "GEOID", "geoid", "geography_id", "geoid10")
    if not gcol:
        raise RuntimeError("❌ GEOID kolonu yok")
    df["GEOID"] = _norm_geoid(df[gcol])

    # date
    dcol = _pick_col(df.columns, "date", "datetime", "time", "timestamp")
    if not dcol:
        raise RuntimeError("❌ date/datetime/time yok")
    df["date"] = _as_date64_any(df[dcol])

    df = df.dropna(subset=["GEOID","date"])
    print(f"📖 panel rows={len(df):,} cols={df.shape[1]}", flush=True)

    # base_cnt
    base = _build_base_daily_counts(df)

    # neighbors
    nbr = pd.read_csv(NEIGHBOR_FILE, low_memory=False, dtype=str)
    s = _pick_col(nbr.columns, "geoid", "GEOID", "src", "source")
    t = _pick_col(nbr.columns, "neighbor", "neighbor_geoid", "dst", "target")
    if not s or not t:
        raise RuntimeError(f"❌ neighbors başlıkları tanınmadı: {nbr.columns.tolist()}")
    nbr = nbr.rename(columns={s:"geoid", t:"neighbor"})[["geoid","neighbor"]]

    feats = _neighbor_daily_features(base, nbr)
    print(f"✨ feats rows={len(feats):,} cols={feats.shape[1]}", flush=True)

    # merge back
    df_out = df.copy()
    # aynı kolonları varsa temizle
    for c in [
        "neighbor_crime_1h",
        "neighbor_crime_3h",
        "neighbor_crime_6h",
        "neighbor_crime_24h",
        "neighbor_crime_3d",
        "neighbor_crime_7d",
        "nei_7d_sum",
    ]:
        if c in df_out.columns:
            df_out = df_out.drop(columns=[c])

    df_out = df_out.merge(feats, on=["GEOID","date"], how="left")
    for c in [
        "neighbor_crime_1h",
        "neighbor_crime_3h",
        "neighbor_crime_6h",
        "neighbor_crime_24h",
        "neighbor_crime_3d",
        "neighbor_crime_7d",
    ]:
        if c not in df_out.columns:
            raise RuntimeError(f"❌ HARD FAIL: {c} merge sonrası yok!")
        df_out[c] = pd.to_numeric(df_out[c], errors="coerce").fillna(0).clip(lower=0).round().astype("int64")

    if MAKE_LEGACY_NEI7:
        df_out["nei_7d_sum"] = df_out["neighbor_crime_7d"].astype("int64")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT_CSV, index=False)
    print(f"✅ wrote: {OUT_CSV} (rows={len(df_out):,}, cols={df_out.shape[1]})", flush=True)

if __name__ == "__main__":
    main()
