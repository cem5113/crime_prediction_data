#!/usr/bin/env python3
# scripts/update_neighbors.py
from __future__ import annotations

import os, re
from pathlib import Path
import pandas as pd

# =============================================================================
# PATHS (FIXED)
# =============================================================================
ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.environ.get("CRIME_DATA_DIR", str(ROOT / "crime_prediction_data"))).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# ENV (input/output)
# =============================================================================
IN_CSV_ENV  = os.environ.get("NEIGHBOR_INPUT_CSV", "sf_crime_08.csv")
OUT_CSV_ENV = os.environ.get("NEIGHBOR_OUTPUT_CSV", "sf_crime_09.csv")
NEIGHBOR_FILE_ENV = os.environ.get("NEIGHBOR_FILE", "neighbors.csv")

GEOID_LEN = int(os.environ.get("GEOID_LEN", "11"))
WINDOW_DAYS = int(os.environ.get("NEIGHBOR_WINDOW_DAYS", "7"))
LAG_DAYS    = int(os.environ.get("NEIGHBOR_LAG_DAYS", "1"))

# =============================================================================
# HELPERS (PATH RESOLVE)
# =============================================================================
def _pick_latest_sf(base: Path) -> Path | None:
    files = list(base.glob("sf_crime_*.csv"))
    if not files:
        return None

    def ver(p: Path) -> int:
        m = re.search(r"sf_crime_(\d+)\.csv$", p.name)
        return int(m.group(1)) if m else -1

    files.sort(key=ver, reverse=True)
    return files[0]

def _resolve_path(p: Path) -> Path:
    if p.is_absolute():
        return p.resolve()
    return (DATA_DIR / p).resolve()

def _resolve_in_csv(p: Path) -> Path:
    if p.is_absolute() and p.exists():
        return p.resolve()

    cand = _resolve_path(p)
    if cand.exists():
        return cand

    name = cand.name
    if name == "sf_crime_08.csv":
        alt = cand.with_name("sf_crime_8.csv")
        if alt.exists():
            return alt.resolve()
    if name == "sf_crime_8.csv":
        alt = cand.with_name("sf_crime_08.csv")
        if alt.exists():
            return alt.resolve()

    latest = _pick_latest_sf(DATA_DIR)
    if latest and latest.exists():
        print(f"ℹ️ IN_CSV bulunamadı: {cand} | latest fallback: {latest}", flush=True)
        return latest.resolve()

    raise FileNotFoundError(f"Girdi bulunamadı: {cand}")

IN_CSV  = _resolve_in_csv(Path(IN_CSV_ENV))
OUT_CSV = _resolve_path(Path(OUT_CSV_ENV))
NEIGHBOR_FILE = _resolve_path(Path(NEIGHBOR_FILE_ENV))

# =============================================================================
# CORE HELPERS
# =============================================================================
def _norm_geoid(s: pd.Series, L: int = GEOID_LEN) -> pd.Series:
    return (s.astype(str).str.extract(r"(\d+)", expand=False).str[:L].str.zfill(L))

def _pick_col(dcols, *cands):
    low = {c.lower(): c for c in dcols}
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    return None

def _safe_build_date(df: pd.DataFrame, dcol: str) -> pd.Series:
    """
    ✅ FIX-1: date kolonu 'YYYY-MM-DD' ise UTC convert yapma -> gün kayması olmasın.
    datetime/time ise tz-aware ise convert; naive ise LA lokalize.
    """
    x = pd.to_datetime(df[dcol], errors="coerce")

    if dcol.lower() == "date":
        return x.dt.date

    # datetime/time için:
    # tz-aware ise LA'ya çevir, naive ise LA kabul et
    try:
        tz = x.dt.tz
    except Exception:
        tz = None

    if tz is None:
        x = x.dt.tz_localize("America/Los_Angeles", nonexistent="shift_forward", ambiguous="NaT")
    else:
        x = x.dt.tz_convert("America/Los_Angeles")

    return x.dt.date

def _safe_pick_count_col(df: pd.DataFrame) -> str:
    """
    ✅ FIX-2: crime_count yoksa 0'a düşme.
    Sende tipik adaylar: y_count/hr_cnt/Y_label/y_event
    """
    return (
        _pick_col(df.columns, "crime_count", "y_count", "count", "hr_cnt", "Y_label", "y_event")
        or ""
    )

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=============================================================", flush=True)
    print("🧭 update_neighbors.py", flush=True)
    print(f"ROOT         : {ROOT}", flush=True)
    print(f"DATA_DIR     : {DATA_DIR}", flush=True)
    print(f"IN_CSV       : {IN_CSV}", flush=True)
    print(f"OUT_CSV      : {OUT_CSV}", flush=True)
    print(f"NEIGHBOR_FILE: {NEIGHBOR_FILE}", flush=True)
    print("=============================================================", flush=True)

    if not IN_CSV.exists():
        raise FileNotFoundError(f"Girdi bulunamadı: {IN_CSV}")
    if not NEIGHBOR_FILE.exists():
        raise FileNotFoundError(f"Komşuluk dosyası bulunamadı: {NEIGHBOR_FILE}")

    df = pd.read_csv(IN_CSV, low_memory=False, dtype=str, encoding="utf-8-sig")
    df.columns = df.columns.str.replace("\ufeff", "", regex=False)

    # --- tarih alanı ---
    dcol = _pick_col(df.columns, "date", "datetime", "time")
    if not dcol:
        raise RuntimeError("Tarih kolonu bulunamadı (date/datetime/time)")

    df["date"] = _safe_build_date(df, dcol)

    # --- GEOID alanı ---
    gcol = _pick_col(df.columns, "GEOID", "geoid", "geography_id", "geoid10")
    if not gcol:
        raise RuntimeError("GEOID kolonu bulunamadı")
    df["GEOID"] = _norm_geoid(df[gcol])

    # --- crime_count türet ---
    ccol = _safe_pick_count_col(df)
    if not ccol:
        raise RuntimeError("crime_count/y_count/hr_cnt/Y_label/y_event kolonlarından hiçbiri bulunamadı.")

    # Standartlaştır: crime_count
    df["crime_count"] = pd.to_numeric(df[ccol], errors="coerce").fillna(0).astype(int)

    # Günlük toplam
    daily = (df[["GEOID", "date", "crime_count"]]
             .groupby(["GEOID", "date"], as_index=False)["crime_count"].sum())
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")

    # neighbors.csv oku (çeşitli başlık varyantları)
    nbr = pd.read_csv(NEIGHBOR_FILE, dtype=str)
    s = _pick_col(nbr.columns, "geoid", "GEOID", "src", "source")
    t = _pick_col(nbr.columns, "neighbor", "NEIGHBOR_GEOID", "neighbor_geoid", "dst", "target")
    if not s or not t:
        raise RuntimeError(f"neighbors.csv başlıkları anlaşılamadı: {nbr.columns.tolist()}")

    nbr = nbr.rename(columns={s: "geoid", t: "neighbor"})[["geoid", "neighbor"]].dropna().drop_duplicates()
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

    # ✅ FIX-3: date NaT olanları at (asfreq patlamasın)
    d2 = d2.dropna(subset=["date"])

    d2 = d2.sort_values(["GEOID", "date"])

    def _agg(x: pd.DataFrame) -> pd.DataFrame:
        x = x.dropna(subset=["date"])
        if x.empty:
            # hiç gün yoksa boş dön
            return x.assign(nei_7d_sum=0).reset_index(drop=True)

        x = x.set_index("date").asfreq("D", fill_value=0)
        roll = x["crime_count"].rolling(WINDOW_DAYS).sum().shift(LAG_DAYS)
        x["nei_7d_sum"] = roll
        return x.reset_index()

    d3 = (d2.groupby("GEOID", group_keys=False).apply(_agg).reset_index(drop=True))
    d3["date"] = pd.to_datetime(d3["date"], errors="coerce").dt.date

    d4 = (d3.groupby(["GEOID", "date"], as_index=False)["nei_7d_sum"].sum())
    d4["nei_7d_sum"] = pd.to_numeric(d4["nei_7d_sum"], errors="coerce").fillna(0.0)

    # Orijinal tabloya merge → 09
    df_out = df.copy()
    df_out["date"] = pd.to_datetime(df_out["date"], errors="coerce").dt.date
    df_out = df_out.merge(d4, on=["GEOID", "date"], how="left")

    cov_nei = df_out["nei_7d_sum"].notna().mean() if "nei_7d_sum" in df_out.columns else 0.0
    print(f"🧪 NEI 7d coverage (nei_7d_sum notna): {cov_nei:.3%}", flush=True)

    df_out["nei_7d_sum"] = pd.to_numeric(df_out["nei_7d_sum"], errors="coerce").fillna(0.0)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT_CSV, index=False)

    print(f"✅ 08 → 09 tamam: {IN_CSV.name} → {OUT_CSV.name} (rows={len(df_out)})", flush=True)

if __name__ == "__main__":
    main()
