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

def _read_csv(p: Path) -> pd.DataFrame:
    if not p.exists():
        log(f"ℹ️ Yok: {p}")
        return pd.DataFrame()
    df = pd.read_csv(p, low_memory=False)
    log(f"📖 Okundu: {p} ({len(df):,}×{df.shape[1]})")
    return df

def _safe_write_csv(df: pd.DataFrame, p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

    # küçük downcast
    for c in df.select_dtypes(include=["float64"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="float")
    for c in df.select_dtypes(include=["int64","Int64"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="integer")

    tmp = p.with_suffix(p.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(p)
    log(f"💾 Yazıldı: {p} ({len(df):,}×{df.shape[1]})")

def _norm_geoid(s: pd.Series, L: int = 11) -> pd.Series:
    return (
        s.astype(str)
         .str.extract(r"(\d+)", expand=False)
         .fillna("")
         .str[:L].str.zfill(L)
    )

# --- tarih yardımcıları: HER ZAMAN datetime64[ns] ---
def _as_date64(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()

def _ensure_date_col(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    cand = None
    for c in ["date", "incident_date", "incident_datetime", "datetime", "time", "timestamp"]:
        if c in d.columns:
            cand = c
            break

    if cand is None and {"year","month","day"}.issubset(d.columns):
        d["date"] = _as_date64(pd.to_datetime(d[["year","month","day"]], errors="coerce"))
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

def find_latest_fr_crime(base_dir: Path) -> Path:
    files = sorted(base_dir.glob("fr_crime_*.csv"))
    if not files:
        raise FileNotFoundError(f"❌ fr_crime_*.csv bulunamadı: {base_dir}")

    def _ver(p: Path):
        m = re.search(r"fr_crime_(\d+)\.csv", p.name)
        return int(m.group(1)) if m else -1

    files.sort(key=_ver, reverse=True)
    return files[0]

def bump_version_name(p: Path) -> Path:
    m = re.search(r"(fr_crime_)(\d+)(\.csv)", p.name)
    if not m:
        return p.with_name("fr_crime_01.csv")
    prefix, num, suffix = m.groups()
    new_num = int(num) + 1
    return p.with_name(f"{prefix}{new_num:02d}{suffix}")

# ---------- config ----------
BASE_DIR = Path(os.environ.get("CRIME_DATA_DIR", "crime_prediction_data"))

# fr_crime_08 → fr_crime_09 sabit, ama istersen env ile override edebilirsin
FR_IN_ENV  = os.environ.get("FR_CRIME_IN",  "fr_crime_08.csv")
FR_OUT_ENV = os.environ.get("FR_CRIME_OUT", "fr_crime_09.csv")

# neighbors.csv için: env verilmemişse BASE_DIR / "neighbors.csv" kullan
NEIGH_PATH = Path(
    os.environ.get("NEIGH_FILE", "neighbors.csv")
)

GEOID_LEN = int(os.environ.get("GEOID_LEN", "11"))

# ---------- core ----------
def neighbor_daily_features(base: pd.DataFrame, neigh: pd.DataFrame) -> pd.DataFrame:
    """
    base: GEOID×date×base_cnt
    neigh: geoid, neighbor
    Adımlar:
      - neighbors ⨯ base (neighbor→date→count)
      - geoid×date toplamla → neighbor_cnt_day
      - geoid bazında tarihe göre sırala, shift(1) ve rolling(3,7)
    """
    b = base.copy()
    b["GEOID"]    = _norm_geoid(b["GEOID"], GEOID_LEN)
    b["date"]     = _as_date64(b["date"])
    b["base_cnt"] = pd.to_numeric(b["base_cnt"], errors="coerce").fillna(0).astype("int64")

    # Tam tarih kapsaması: her GEOID için min→max arası tüm günler (eksikler 0)
    full = []
    for g, gdf in b.groupby("GEOID", dropna=False):
        gdf = gdf.sort_values("date")
        rng = pd.date_range(gdf["date"].min(), gdf["date"].max(), freq="D")
        aux = pd.DataFrame({"GEOID": g, "date": rng})
        aux = aux.merge(gdf[["date","base_cnt"]], on="date", how="left")
        aux["base_cnt"] = aux["base_cnt"].fillna(0).astype("int64")
        full.append(aux)

    b2 = pd.concat(full, ignore_index=True)

    # Komşuluk: geoid (src) → neighbor (dst)
    nb = neigh.rename(columns={
        _pick(neigh.columns, "geoid","src","source"): "geoid",
        _pick(neigh.columns, "neighbor","dst","target"): "neighbor"
    }).copy()

    if "geoid" not in nb.columns or "neighbor" not in nb.columns:
        raise RuntimeError("❌ neighbors.csv kolonları geoid/neighbor değil. Kolonları kontrol et.")

    nb["geoid"]    = _norm_geoid(nb["geoid"], GEOID_LEN)
    nb["neighbor"] = _norm_geoid(nb["neighbor"], GEOID_LEN)
    nb = nb.dropna()

    # Komşu günlük sayıları: nb (geoid, neighbor) ⨯ base (neighbor, date, base_cnt)
    nb_merge = nb.merge(b2.rename(columns={"GEOID":"neighbor"}), on="neighbor", how="left")

    day_sum = (
        nb_merge.groupby(["geoid","date"], dropna=False)["base_cnt"]
                .sum().reset_index(name="neighbor_cnt_day")
    )

    # Rolling pencereler: geoid bazında tarih sırasıyla, dünü dahil (shift1)
    day_sum = day_sum.sort_values(["geoid","date"])
    day_sum["nb_last1d"] = day_sum.groupby("geoid")["neighbor_cnt_day"].shift(1).fillna(0)

    for W in (3,7):
        day_sum[f"nb_last{W}d"] = (
            day_sum.groupby("geoid")["neighbor_cnt_day"]
                   .shift(1)  # sızıntı yok
                   .rolling(W, min_periods=1).sum()
        ).fillna(0)

    out = day_sum.rename(columns={
        "geoid": "GEOID",
        "nb_last1d": "neighbor_crime_1d",
        "nb_last3d": "neighbor_crime_3d",
        "nb_last7d": "neighbor_crime_7d",
    })[["GEOID","date","neighbor_crime_1d","neighbor_crime_3d","neighbor_crime_7d"]]

    out["date"] = _as_date64(out["date"])
    for c in ["neighbor_crime_1d","neighbor_crime_3d","neighbor_crime_7d"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype("int64")

    return out

def build_hour_range_artifact(df: pd.DataFrame, out_path: Path) -> None:
    """
    df_out içinden (GEOID, date, neighbor_crime_1d/3d/7d) al,
    her gün için 8 tane hour range üret, her range için
    bitiş saatine göre slot_end_ts üret ve yeni CSV'ye yaz.
    """

    # 8 adet 3-saatlik range (örnek konfig; istersen değiştirirsin)
    HOUR_RANGES = [
        ("00:00", "03:00"),
        ("03:00", "06:00"),
        ("06:00", "09:00"),
        ("09:00", "12:00"),
        ("12:00", "15:00"),
        ("15:00", "18:00"),
        ("18:00", "21:00"),
        ("21:00", "24:00"),
    ]

    base_cols = ["GEOID", "date", "neighbor_crime_1d", "neighbor_crime_3d", "neighbor_crime_7d"]
    missing = [c for c in base_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Hour-range artifact için eksik kolon(lar): {missing}")

    d = df[base_cols].dropna(subset=["GEOID", "date"]).copy()
    d["GEOID"] = _norm_geoid(d["GEOID"], GEOID_LEN)
    d["date"] = _as_date64(d["date"])

    blocks = []
    for i, (start_h, end_h) in enumerate(HOUR_RANGES, start=1):
        tmp = d.copy()
        tmp["hour_slot_id"] = i
        tmp["hour_start"] = start_h
        tmp["hour_end"] = end_h

        # hour range'in bitiş anı: aynı günün end_h saati
        if end_h == "24:00":
            # 24:00'ı bir sonraki gün 00:00 olarak yorumla
            tmp["slot_end_ts"] = tmp["date"] + pd.Timedelta(days=1)
        else:
            tmp["slot_end_ts"] = tmp["date"] + pd.to_timedelta(end_h + ":00")

        blocks.append(tmp)

    artifact = pd.concat(blocks, ignore_index=True)

    # CSV'ye yaz
    _safe_write_csv(artifact, out_path)
    log(
        f"✨ hour-range artifact: {out_path} — "
        f"{len(artifact):,} satır, {artifact['hour_slot_id'].nunique()} hour range"
    )

def main() -> int:
    log("🚀 enrich_with_neighbors_fr.py (fr_crime_08 → fr_crime_09)")
    
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    
    # ---- input/output fr_crime ----
    fr_in = BASE_DIR / FR_IN_ENV if not Path(FR_IN_ENV).is_absolute() else Path(FR_IN_ENV)
    fr_out = BASE_DIR / FR_OUT_ENV if not Path(FR_OUT_ENV).is_absolute() else Path(FR_OUT_ENV)
    
    log(f"📥 FR input : {fr_in}")
    log(f"📤 FR output: {fr_out}")
    log(f"📂 NEIGH PATH: {NEIGH_PATH}")
    
    df_raw = _read_csv(fr_in)
    if df_raw.empty:
        raise RuntimeError("❌ Girdi CSV boş veya okunamadı.")

    # Komşuluk dosyası
    if not NEIGH_PATH.exists():
        raise FileNotFoundError(f"❌ neighbors.csv bulunamadı: {NEIGH_PATH.resolve()}")

    neigh = pd.read_csv(NEIGH_PATH, low_memory=False).dropna()
    if neigh.empty:
        raise RuntimeError("❌ neighbors.csv boş.")

    # ---- normalize input ----
    df = _normalize_keys(df_raw, GEOID_LEN)
    df = df.dropna(subset=["GEOID","date"])
    log(f"🧹 Normalize sonrası satır: {len(df):,}")

    # ---- base_cnt üret ----
    base_cols = [c for c in ["crime_count","Y_day","Y_label"] if c in df.columns]

    if base_cols:
        for c in base_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int64")
        df["__base__"] = df[base_cols].sum(axis=1)
        base = (
            df.groupby(["GEOID","date"], dropna=False)["__base__"]
              .sum().reset_index(name="base_cnt")
        )
    else:
        base = (
            df.groupby(["GEOID","date"], dropna=False)
              .size().reset_index(name="base_cnt")
        )

    log(f"🧮 base_cnt hazır: {len(base):,} satır (GEOID×date)")

    # ---- neighbor features ----
    feats = neighbor_daily_features(base, neigh)
    log(
        f"✨ neighbor feats: {len(feats):,} satır (GEOID×date) — eklenecek kolonlar: "
        f"[neighbor_crime_1d, neighbor_crime_3d, neighbor_crime_7d]"
    )

    # ---- merge back to original ----
    feats["GEOID"] = _norm_geoid(feats["GEOID"], GEOID_LEN)
    feats["date"]  = _as_date64(feats["date"])

    df_out = df.merge(feats, on=["GEOID","date"], how="left")

    for c in ["neighbor_crime_1d","neighbor_crime_3d","neighbor_crime_7d"]:
        if c in df_out.columns:
            df_out[c] = pd.to_numeric(df_out[c], errors="coerce").fillna(0).astype("int64")

    _safe_write_csv(df_out, fr_out)

    hour_artifact_path = fr_out.with_name(fr_out.stem + "_hour_ranges.csv")
    build_hour_range_artifact(df_out, hour_artifact_path)

    # Preview
    try:
        cols = ["GEOID","date","neighbor_crime_1d","neighbor_crime_3d","neighbor_crime_7d"]
        log("— OUTPUT preview —")
        log(df_out[cols].head(8).to_string(index=False))
    except Exception:
        pass

    log("✅ Tamam.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
