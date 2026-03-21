#!/usr/bin/env python3
# =============================================================================
# ✅ scripts/update_neighbors.py (FINAL — NO 1H)
# PARQUET-FIRST / PANEL-SAFE / LEAK-FREE / PUBLISH-LAG AWARE
#
# ÜRETİLEN KOLONLAR
# -----------------
# - neighbor_crime_3h
# - neighbor_crime_6h
# - neighbor_crime_24h
# - neighbor_crime_3d
# - neighbor_crime_7d
#
# NOT:
# - 1h kaldırıldı (günlük seri olduğu için anlamsızdı)
# =============================================================================

from __future__ import annotations
import os
from pathlib import Path
import pandas as pd

pd.options.mode.copy_on_write = True

# =============================================================================
# CONFIG
# =============================================================================
BASE_DIR = Path(os.environ.get("CRIME_DATA_DIR", "crime_prediction_data")).resolve()
BASE_DIR.mkdir(parents=True, exist_ok=True)

FR_IN_ENV = os.environ.get("sf_CRIME_IN", "sf_crime_08.parquet")
FR_OUT_ENV = os.environ.get("sf_CRIME_OUT", "sf_crime_09.parquet")
FR_OUT_CSV_ENV = os.environ.get("sf_CRIME_OUT_CSV", "sf_crime_09.csv")

NEIGH_FILE_ENV = os.environ.get("NEIGH_FILE", "neighbors.csv")

GEOID_LEN = int(os.environ.get("GEOID_LEN", "11"))
PUBLISH_LAG_DAYS = int(os.environ.get("PUBLISH_LAG_DAYS", "2"))
SHIFT_K = PUBLISH_LAG_DAYS + 1

MAKE_LEGACY_NEI7 = os.environ.get("MAKE_LEGACY_NEI7", "0").lower() in ("1","true","yes","on")
WRITE_CSV = os.environ.get("WRITE_CSV", "0").lower() in ("1","true","yes","on")

# =============================================================================
# HELPERS
# =============================================================================
def log(x): print(x, flush=True)

def _resolve(base, p):
    p = Path(p)
    return p if p.is_absolute() else base / p

def _read(p):
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p, low_memory=False)

def _norm_geoid(s):
    return s.astype(str).str.extract(r"(\d+)", expand=False).fillna("").str[:GEOID_LEN].str.zfill(GEOID_LEN)

def _as_date(s):
    dt = pd.to_datetime(s, errors="coerce", utc=True)
    dt = dt.dt.tz_convert("America/Los_Angeles")
    return dt.dt.normalize().dt.tz_localize(None)

def _pick(cols, *cands):
    m = {c.lower(): c for c in cols}
    for c in cands:
        if c.lower() in m:
            return m[c.lower()]
    return None

# =============================================================================
# CORE
# =============================================================================
def build_base(df):
    if "y_count" in df:
        s = df["y_count"]
    elif "crime_count" in df:
        s = df["crime_count"]
    elif "Y_label" in df:
        s = df["Y_label"]
    else:
        raise RuntimeError("❌ base yok")

    tmp = df[["GEOID","date"]].copy()
    tmp["cnt"] = pd.to_numeric(s, errors="coerce").fillna(0).clip(lower=0)

    base = tmp.groupby(["GEOID","date"], as_index=False)["cnt"].sum()
    base["cnt"] = base["cnt"].round().astype("int64")

    return base.rename(columns={"cnt":"base_cnt"})

def neighbor_features(base, neigh):
    base["GEOID"] = _norm_geoid(base["GEOID"])
    base["date"] = pd.to_datetime(base["date"])

    neigh["geoid"] = _norm_geoid(neigh["geoid"])
    neigh["neighbor"] = _norm_geoid(neigh["neighbor"])

    b2 = base.rename(columns={"GEOID":"neighbor"})
    m = neigh.merge(b2, on="neighbor", how="left")
    m["base_cnt"] = m["base_cnt"].fillna(0)

    day = m.groupby(["geoid","date"])["base_cnt"].sum().reset_index()
    day = day.rename(columns={"geoid":"GEOID","base_cnt":"neighbor_cnt_day"})

    def per_g(g):
        g = g.sort_values("date").set_index("date").asfreq("D", fill_value=0)
        s = g["neighbor_cnt_day"]

        s_shift = s.shift(SHIFT_K)

        g["neighbor_crime_3h"]  = s_shift.fillna(0)
        g["neighbor_crime_6h"]  = s_shift.fillna(0)
        g["neighbor_crime_24h"] = s_shift.fillna(0)
        g["neighbor_crime_3d"]  = s_shift.rolling(3,1).sum().fillna(0)
        g["neighbor_crime_7d"]  = s_shift.rolling(7,1).sum().fillna(0)

        return g.reset_index()

    out = day.groupby("GEOID", group_keys=False).apply(per_g).reset_index(drop=True)

    return out

# =============================================================================
# MAIN
# =============================================================================
def main():
    log("🚀 update_neighbors FINAL (NO 1H)")

    IN = _resolve(BASE_DIR, FR_IN_ENV)
    OUT = _resolve(BASE_DIR, FR_OUT_ENV)
    OUT_CSV = _resolve(BASE_DIR, FR_OUT_CSV_ENV)
    NEI = _resolve(BASE_DIR, NEIGH_FILE_ENV)

    if not IN.exists():
        alt = IN.with_suffix(".csv")
        if alt.exists():
            IN = alt
        else:
            raise FileNotFoundError(f"❌ input yok: {IN}")

    df = _read(IN)

    gcol = _pick(df.columns,"GEOID","geoid")
    dcol = _pick(df.columns,"date","datetime")

    df["GEOID"] = _norm_geoid(df[gcol])
    df["date"] = _as_date(df[dcol])

    df = df.dropna(subset=["GEOID","date"])

    base = build_base(df)

    neigh = pd.read_csv(NEI)
    s = _pick(neigh.columns,"geoid","src")
    t = _pick(neigh.columns,"neighbor","dst")

    neigh = neigh.rename(columns={s:"geoid",t:"neighbor"})[["geoid","neighbor"]]

    feats = neighbor_features(base, neigh)

    cols = [
        "neighbor_crime_3h",
        "neighbor_crime_6h",
        "neighbor_crime_24h",
        "neighbor_crime_3d",
        "neighbor_crime_7d",
    ]

    df = df.drop(columns=[c for c in cols if c in df.columns], errors="ignore")

    df = df.merge(feats, on=["GEOID","date"], how="left")

    for c in cols:
        df[c] = df[c].fillna(0).astype("int64")

    if MAKE_LEGACY_NEI7:
        df["nei_7d_sum"] = df["neighbor_crime_7d"]

    df.to_parquet(OUT, index=False)

    if WRITE_CSV:
        df.to_csv(OUT_CSV, index=False)

    log("✅ DONE")

if __name__ == "__main__":
    main()
