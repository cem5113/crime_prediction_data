#!/usr/bin/env python3
# scripts/update_neighbors.py
from __future__ import annotations

import os, re
from pathlib import Path
import pandas as pd

# =============================================================================
# PATHS (FIXED)
#   - Workflow env: CRIME_DATA_DIR zaten set ediliyor (github.workspace)
#   - Nested "crime_prediction_data/crime_prediction_data/..." hatasını önler
# =============================================================================
ROOT = Path(__file__).resolve().parent.parent

# ✅ DATA_DIR: önce CRIME_DATA_DIR (workflow), yoksa repo içi default
DATA_DIR = Path(os.environ.get("CRIME_DATA_DIR", str(ROOT / "crime_prediction_data"))).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# ENV (input/output)
# =============================================================================
IN_CSV_ENV  = os.environ.get("NEIGHBOR_INPUT_CSV", "sf_crime_08.csv")
OUT_CSV_ENV = os.environ.get("NEIGHBOR_OUTPUT_CSV", "sf_crime_09.csv")  # <-- 09 olarak yaz

NEIGHBOR_FILE_ENV = os.environ.get("NEIGHBOR_FILE", "neighbors.csv")

GEOID_LEN = int(os.environ.get("GEOID_LEN", "11"))
WINDOW_DAYS = int(os.environ.get("NEIGHBOR_WINDOW_DAYS", "7"))
LAG_DAYS    = int(os.environ.get("NEIGHBOR_LAG_DAYS", "1"))

# =============================================================================
# HELPERS (PATH RESOLVE)
# =============================================================================
def _pick_latest_sf(base: Path) -> Path | None:
    """
    base içinde sf_crime_*.csv dosyalarından en yüksek versiyon numarasını döndürür.
    Örn: sf_crime_07.csv, sf_crime_08.csv, sf_crime_09.csv ...
    """
    files = list(base.glob("sf_crime_*.csv"))
    if not files:
        return None

    def ver(p: Path) -> int:
        m = re.search(r"sf_crime_(\d+)\.csv$", p.name)
        return int(m.group(1)) if m else -1

    files.sort(key=ver, reverse=True)
    return files[0]

def _resolve_path(p: Path) -> Path:
    """
    Env ile gelen path:
      - absolute ise direkt kullan
      - relative ise DATA_DIR altında arar
    """
    if p.is_absolute():
        return p.resolve()
    return (DATA_DIR / p).resolve()

def _resolve_in_csv(p: Path) -> Path:
    """
    IN_CSV için:
      1) env absolute + exists
      2) env relative -> DATA_DIR altında exists
      3) 08 <-> 8 toleransı
      4) latest sf_crime_*.csv fallback
    """
    # 1) absolute
    if p.is_absolute() and p.exists():
        return p.resolve()

    # 2) relative -> DATA_DIR
    cand = _resolve_path(p)
    if cand.exists():
        return cand

    # 3) 08 <-> 8 toleransı
    name = cand.name
    if name == "sf_crime_08.csv":
        alt = cand.with_name("sf_crime_8.csv")
        if alt.exists():
            return alt.resolve()
    if name == "sf_crime_8.csv":
        alt = cand.with_name("sf_crime_08.csv")
        if alt.exists():
            return alt.resolve()

    # 4) latest fallback
    latest = _pick_latest_sf(DATA_DIR)
    if latest and latest.exists():
        print(f"ℹ️ IN_CSV bulunamadı: {cand} | latest fallback: {latest}", flush=True)
        return latest.resolve()

    raise FileNotFoundError(f"Girdi bulunamadı: {cand}")

# Resolve final paths
IN_CSV  = _resolve_in_csv(Path(IN_CSV_ENV))
OUT_CSV = _resolve_path(Path(OUT_CSV_ENV))

NEIGHBOR_FILE = _resolve_path(Path(NEIGHBOR_FILE_ENV))

# =============================================================================
# CORE HELPERS (UNCHANGED LOGIC)
# =============================================================================
def _norm_geoid(s: pd.Series, L: int = GEOID_LEN) -> pd.Series:
    return (s.astype(str).str.extract(r"(\d+)", expand=False).str[:L].str.zfill(L))

def _pick_col(dcols, *cands):
    low = {c.lower(): c for c in dcols}
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    return None

# =============================================================================
# MAIN
# =============================================================================
def main():
    # Debug prints (runner loglarında path sorunları anında görünür)
    print("=============================================================", flush=True)
    print("🧭 update_neighbors.py", flush=True)
    print(f"ROOT        : {ROOT}", flush=True)
    print(f"DATA_DIR    : {DATA_DIR}", flush=True)
    print(f"IN_CSV      : {IN_CSV}", flush=True)
    print(f"OUT_CSV     : {OUT_CSV}", flush=True)
    print(f"NEIGHBOR_FILE: {NEIGHBOR_FILE}", flush=True)
    print("=============================================================", flush=True)

    if not IN_CSV.exists():
        raise FileNotFoundError(f"Girdi bulunamadı: {IN_CSV}")
    if not NEIGHBOR_FILE.exists():
        raise FileNotFoundError(f"Komşuluk dosyası bulunamadı: {NEIGHBOR_FILE}")

    df = pd.read_csv(IN_CSV, low_memory=False, dtype=str)

    # tarih alanı
    dcol = _pick_col(df.columns, "date", "datetime", "time")
    if not dcol:
        raise RuntimeError("Tarih kolonu bulunamadı (date/datetime/time)")

    dt = pd.to_datetime(df[dcol], errors="coerce", utc=True).dt.tz_convert("America/Los_Angeles")
    df["date"] = dt.dt.date

    # GEOID alanı
    gcol = _pick_col(df.columns, "GEOID", "geoid", "geography_id", "geoid10")
    if not gcol:
        raise RuntimeError("GEOID kolonu bulunamadı")
    df["GEOID"] = _norm_geoid(df[gcol])

    # crime_count yoksa 0
    ccol = _pick_col(df.columns, "crime_count")
    if not ccol:
        df["crime_count"] = 0
        ccol = "crime_count"
    df[ccol] = pd.to_numeric(df[ccol], errors="coerce").fillna(0).astype(int)

    # Günlük toplam
    daily = (df[["GEOID", "date", ccol]]
             .groupby(["GEOID", "date"], as_index=False)[ccol].sum()
             .rename(columns={ccol: "crime_count"}))
    daily["date"] = pd.to_datetime(daily["date"])

    # neighbors.csv oku (çeşitli başlık varyantları)
    nbr = pd.read_csv(NEIGHBOR_FILE, dtype=str)
    s = _pick_col(nbr.columns, "geoid", "GEOID", "src", "source")
    t = _pick_col(nbr.columns, "neighbor", "NEIGHBOR_GEOID", "neighbor_geoid", "dst", "target")
    if not s or not t:
        raise RuntimeError(f"neighbors.csv başlıkları anlaşılamadı: {nbr.columns.tolist()}")

    nbr = nbr.rename(columns={s: "geoid", t: "neighbor"})[["geoid", "neighbor"]].dropna()
    for c in ("geoid", "neighbor"):
        nbr[c] = _norm_geoid(nbr[c])

    # Komşu serilerini bağla
    d2 = nbr.merge(
        daily.rename(columns={"GEOID": "neighbor"}),
        left_on="neighbor",
        right_on="neighbor",
        how="left"
    )
    d2 = d2.rename(columns={"geoid": "GEOID"})  # ana GEOID

    # Rolling + lag
    d2 = d2.sort_values(["GEOID", "date"])

    def _agg(x: pd.DataFrame) -> pd.DataFrame:
        x = x.set_index("date").asfreq("D", fill_value=0)
        roll = x["crime_count"].rolling(WINDOW_DAYS).sum().shift(LAG_DAYS)
        x["nei_7d_sum"] = roll
        return x.reset_index()

    d3 = (d2.groupby("GEOID", group_keys=False).apply(_agg).reset_index(drop=True))
    d3["date"] = d3["date"].dt.date
    d4 = (d3.groupby(["GEOID", "date"], as_index=False)["nei_7d_sum"].sum())
    d4["nei_7d_sum"] = pd.to_numeric(d4["nei_7d_sum"], errors="coerce").fillna(0.0)

    # Orijinal tabloya merge → 09
    df_out = df.copy()
    df_out["date"] = pd.to_datetime(df_out["date"]).dt.date

    df_out = df_out.merge(d4, on=["GEOID", "date"], how="left")

    # 🧪 NEIGHBOR coverage (fillna ÖNCESİ gerçek coverage)
    cov_nei = df_out["nei_7d_sum"].notna().mean() if "nei_7d_sum" in df_out.columns else 0.0
    print(f"🧪 NEI 7d coverage (nei_7d_sum notna): {cov_nei:.3%}", flush=True)

    df_out["nei_7d_sum"] = pd.to_numeric(df_out["nei_7d_sum"], errors="coerce").fillna(0.0)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT_CSV, index=False)

    print(f"✅ 08 → 09 tamam: {IN_CSV.name} → {OUT_CSV.name} (rows={len(df_out)})", flush=True)

if __name__ == "__main__":
    main()
