# update_911.py
# -*- coding: utf-8 -*-

import os
import re
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd


# =========================================================
# CONFIG
# =========================================================
DEFAULT_GEOID_LEN = 11

BASE_DIR = Path(os.getenv("CRIME_DATA_DIR", ".")).resolve()
SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR

CRIME_IN_CANDIDATES = [
    BASE_DIR / "sf_crime_y.csv",
    BASE_DIR / "sf_crime_y.parquet",
    BASE_DIR / "sf_crime.csv",
    BASE_DIR / "sf_crime.parquet",
]

RAW_911_CANDIDATES = [
    BASE_DIR / "sf_911_last_5_year_y.parquet",
    BASE_DIR / "sf_911_last_5_year_y.csv",
    BASE_DIR / "sf_911_full_raw.parquet",
    BASE_DIR / "sf_911_full_raw.csv",
    BASE_DIR / "sf_911_last_5_year.parquet",
    BASE_DIR / "sf_911_last_5_year.csv",
]

SUMMARY_OUT_PARQUET = OUT_DIR / "sf_911_last_5_year.parquet"
SUMMARY_OUT_CSV = OUT_DIR / "sf_911_last_5_year.csv"
SUMMARY_Y_OUT_PARQUET = OUT_DIR / "sf_911_last_5_year_y.parquet"
SUMMARY_Y_OUT_CSV = OUT_DIR / "sf_911_last_5_year_y.csv"

CRIME_OUT_CSV = OUT_DIR / "sf_crime_01.csv"
CRIME_OUT_PARQUET = OUT_DIR / "sf_crime_01.parquet"

NEIGHBOR_CANDIDATES = [
    BASE_DIR / "neighbors.csv",
    SCRIPT_DIR / "neighbors.csv",
]

HOUR_ORDER = [
    "00-03", "03-06", "06-09", "09-12",
    "12-15", "15-18", "18-21", "21-24"
]
HOUR_TO_SLOT = {h: i for i, h in enumerate(HOUR_ORDER)}


# =========================================================
# UTILS
# =========================================================
def log(msg: str) -> None:
    print(msg, flush=True)


def read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def write_both(df: pd.DataFrame, parquet_path: Path, csv_path: Path) -> None:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")


def pick_existing(candidates: List[Path]) -> Optional[Path]:
    for p in candidates:
        if p.exists():
            return p
    return None


def to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.date


def normalize_geoid(s: pd.Series, width: int = DEFAULT_GEOID_LEN) -> pd.Series:
    s = s.astype("string").str.strip()
    s = s.replace({"<NA>": pd.NA, "nan": pd.NA, "None": pd.NA, "": pd.NA})
    s = s.str.replace(r"\.0$", "", regex=True)
    s = s.str.replace(r"\D", "", regex=True)
    s = s.where(s.str.len() > 0, pd.NA)
    s = s.str.zfill(width)
    return s


def safe_div(numer, denom):
    numer = pd.to_numeric(numer, errors="coerce").fillna(0)
    denom = pd.to_numeric(denom, errors="coerce").fillna(0)
    out = np.where(denom > 0, numer / denom, 0.0)
    return pd.Series(out, index=numer.index, dtype="float32")


def canonicalize_hour_range(x) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s in HOUR_ORDER:
        return s

    s = s.replace("–", "-").replace("—", "-").replace("_", "-").replace(" ", "")
    m = re.match(r"^(\d{1,2})[:-]?(\d{2})?-(\d{1,2})[:-]?(\d{2})?$", s)
    if m:
        h1 = int(m.group(1))
        h2 = int(m.group(3))
        key = f"{h1:02d}-{h2:02d}"
        if key in HOUR_ORDER:
            return key

    m2 = re.match(r"^(\d{1,2})$", s)
    if m2:
        h = int(m2.group(1))
        slot = (h // 3) * 3
        end = slot + 3
        return f"{slot:02d}-{24 if end == 24 else end:02d}"

    return None


def hour_to_range_from_datetime(dt_series: pd.Series) -> pd.Series:
    hrs = pd.to_datetime(dt_series, errors="coerce").dt.hour
    slot_start = (hrs // 3) * 3
    slot_end = slot_start + 3
    out = np.where(
        hrs.notna(),
        [f"{int(a):02d}-{24 if int(b) == 24 else int(b):02d}" for a, b in zip(slot_start.fillna(0), slot_end.fillna(0))],
        None
    )
    return pd.Series(out, index=dt_series.index, dtype="object")


def load_neighbor_file() -> Optional[Path]:
    p = pick_existing(NEIGHBOR_CANDIDATES)
    if p is not None:
        log(f"🧭 Neighbor file bulundu: {p}")
    else:
        log("⚠️ Neighbor file bulunamadı; fallback kullanılacak.")
    return p


# =========================================================
# EVENT-LEVEL PREP
# =========================================================
def choose_datetime_col(df: pd.DataFrame) -> Optional[str]:
    for c in [
        "received_datetime",
        "entry_datetime",
        "dispatch_datetime",
        "onscene_datetime",
        "close_datetime",
        "datetime",
        "date",
    ]:
        if c in df.columns:
            return c
    return None


def normalize_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for c in [
        "agency",
        "priority_final",
        "priority_original",
        "disposition",
        "call_type_final_desc",
        "call_type_original_desc",
        "call_type_final",
        "call_type_original",
    ]:
        if c in df.columns:
            df[c] = df[c].astype("string").str.strip()

    agency_txt = df["agency"].astype("string").str.upper().fillna("") if "agency" in df.columns else pd.Series("", index=df.index)
    df["police_calls"] = agency_txt.str.contains(r"\bPD\b|POLICE", regex=True).astype("int8")
    df["mta_calls"] = agency_txt.str.contains(r"\bMTA\b|TRANSIT", regex=True).astype("int8")
    df["sheriff_calls"] = agency_txt.str.contains(r"SHERIFF", regex=True).astype("int8")

    pr = None
    for c in ["priority_final", "priority_original"]:
        if c in df.columns:
            pr = df[c].astype("string").str.upper().str.strip()
            break
    if pr is None:
        pr = pd.Series("", index=df.index, dtype="string")

    df["priority_A_calls"] = pr.str.startswith("A").astype("int8")
    df["priority_B_calls"] = pr.str.startswith("B").astype("int8")
    df["priority_C_calls"] = pr.str.startswith("C").astype("int8")

    disp = df["disposition"].astype("string").str.upper().fillna("") if "disposition" in df.columns else pd.Series("", index=df.index, dtype="string")
    df["disp_han_count"] = disp.str.contains("HAN", regex=False).astype("int8")
    df["disp_utl_count"] = disp.str.contains("UTL", regex=False).astype("int8")
    df["disp_adv_count"] = disp.str.contains("ADV", regex=False).astype("int8")
    df["disp_arr_count"] = disp.str.contains("ARR", regex=False).astype("int8")

    return df


def classify_call_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    src = None
    for c in ["call_type_final_desc", "call_type_original_desc", "call_type_final", "call_type_original"]:
        if c in df.columns:
            src = c
            break

    txt = df[src].astype("string").str.lower().fillna("") if src else pd.Series("", index=df.index)

    patterns = {
        "type_traffic_count": r"traffic|vehicle|collision|accident|road",
        "type_assault_count": r"assault|battery|fight|stabbing|shooting",
        "type_theft_count": r"theft|burglary|robbery|larceny|shoplift",
        "type_disturbance_count": r"disturb|noise|trespass|suspicious|dispute",
        "type_domestic_count": r"domestic|family violence|dv",
        "type_weapon_count": r"weapon|gun|knife|armed",
    }

    for col, pat in patterns.items():
        df[col] = txt.str.contains(pat, case=False, regex=True, na=False).astype("int8")

    return df


def build_event_level(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    dt_col = choose_datetime_col(df)
    if dt_col is None:
        raise ValueError("911 ham verisinde datetime kolonu bulunamadı.")

    df["_event_dt"] = pd.to_datetime(df[dt_col], errors="coerce")
    df["date"] = df["_event_dt"].dt.date
    df["hour_range"] = hour_to_range_from_datetime(df["_event_dt"])

    if "GEOID" not in df.columns:
        raise ValueError("911 ham verisinde GEOID kolonu yok.")

    df["GEOID"] = normalize_geoid(df["GEOID"], DEFAULT_GEOID_LEN)

    df = df[
        df["date"].notna()
        & df["GEOID"].notna()
        & df["hour_range"].isin(HOUR_ORDER)
    ].copy()

    df = normalize_categoricals(df)
    df = classify_call_types(df)

    return df


# =========================================================
# SUMMARY
# =========================================================
def make_standard_summary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Zaten summary ise normalize et
    if {"date", "hour_range"}.issubset(df.columns) and (
        "call_count" in df.columns or "911_request_count_hour_range" in df.columns
    ):
        log("ℹ️ Girdi zaten summary gibi görünüyor, kolonlar normalize edilecek.")

        if "call_count" not in df.columns and "911_request_count_hour_range" in df.columns:
            df["call_count"] = pd.to_numeric(df["911_request_count_hour_range"], errors="coerce").fillna(0)

        if "call_count" not in df.columns:
            df["call_count"] = 0.0

        df["date"] = to_date(df["date"])
        df["GEOID"] = normalize_geoid(df["GEOID"], DEFAULT_GEOID_LEN)
        df["hour_range"] = df["hour_range"].map(canonicalize_hour_range)

        df = df[
            df["GEOID"].notna()
            & df["date"].notna()
            & df["hour_range"].isin(HOUR_ORDER)
        ].copy()

        if "slot_index" not in df.columns:
            df["slot_index"] = df["hour_range"].map(HOUR_TO_SLOT)

        if "daily_cnt" not in df.columns:
            day = (
                df.groupby(["GEOID", "date"], dropna=False, observed=True)["call_count"]
                .sum()
                .reset_index(name="daily_cnt")
            )
            df = df.merge(day, on=["GEOID", "date"], how="left")

        if "911_request_count_daily(before_24_hours)" not in df.columns:
            day_tbl = (
                df[["GEOID", "date", "daily_cnt"]]
                .drop_duplicates()
                .sort_values(["GEOID", "date"])
                .copy()
            )
            day_tbl["911_request_count_daily(before_24_hours)"] = (
                day_tbl.groupby("GEOID", sort=False)["daily_cnt"].shift(1).fillna(0)
            )
            df = df.merge(
                day_tbl[["GEOID", "date", "911_request_count_daily(before_24_hours)"]],
                on=["GEOID", "date"],
                how="left",
            )

        return df

    # Raw event-level -> compact summary
    df = build_event_level(df)

    call_type_col = None
    for c in ["call_type_final_desc", "call_type_original_desc", "call_type_final", "call_type_original"]:
        if c in df.columns:
            call_type_col = c
            break

    priority_col = None
    for c in ["priority_final", "priority_original"]:
        if c in df.columns:
            priority_col = c
            break

    agg_map = {
        "call_count": ("GEOID", "size"),
        "police_calls": ("police_calls", "sum"),
        "mta_calls": ("mta_calls", "sum"),
        "sheriff_calls": ("sheriff_calls", "sum"),
        "priority_A_calls": ("priority_A_calls", "sum"),
        "priority_B_calls": ("priority_B_calls", "sum"),
        "priority_C_calls": ("priority_C_calls", "sum"),
        "disp_han_count": ("disp_han_count", "sum"),
        "disp_utl_count": ("disp_utl_count", "sum"),
        "disp_adv_count": ("disp_adv_count", "sum"),
        "disp_arr_count": ("disp_arr_count", "sum"),
        "type_traffic_count": ("type_traffic_count", "sum"),
        "type_assault_count": ("type_assault_count", "sum"),
        "type_theft_count": ("type_theft_count", "sum"),
        "type_disturbance_count": ("type_disturbance_count", "sum"),
        "type_domestic_count": ("type_domestic_count", "sum"),
        "type_weapon_count": ("type_weapon_count", "sum"),
    }

    panel = df.groupby(["GEOID", "date", "hour_range"], dropna=False, observed=True).agg(
        **agg_map
    ).reset_index()

    if call_type_col is not None:
        tmp = (
            df.groupby(["GEOID", "date", "hour_range"], dropna=False, observed=True)[call_type_col]
            .nunique(dropna=True)
            .reset_index(name="unique_call_type")
        )
        panel = panel.merge(tmp, on=["GEOID", "date", "hour_range"], how="left")
    else:
        panel["unique_call_type"] = 0

    if priority_col is not None:
        tmp = (
            df.groupby(["GEOID", "date", "hour_range"], dropna=False, observed=True)[priority_col]
            .nunique(dropna=True)
            .reset_index(name="unique_priority")
        )
        panel = panel.merge(tmp, on=["GEOID", "date", "hour_range"], how="left")
    else:
        panel["unique_priority"] = 0

    if "disposition" in df.columns:
        tmp = (
            df.groupby(["GEOID", "date", "hour_range"], dropna=False, observed=True)["disposition"]
            .nunique(dropna=True)
            .reset_index(name="unique_disposition")
        )
        panel = panel.merge(tmp, on=["GEOID", "date", "hour_range"], how="left")
    else:
        panel["unique_disposition"] = 0

    panel["date"] = to_date(panel["date"])
    panel["GEOID"] = normalize_geoid(panel["GEOID"], DEFAULT_GEOID_LEN)
    panel["hour_range"] = panel["hour_range"].map(canonicalize_hour_range)

    panel = panel[
        panel["GEOID"].notna()
        & panel["date"].notna()
        & panel["hour_range"].isin(HOUR_ORDER)
    ].copy()

    panel["slot_index"] = panel["hour_range"].map(HOUR_TO_SLOT)
    panel = panel.sort_values(["GEOID", "date", "slot_index"]).reset_index(drop=True)

    # Günlük toplam
    daily = (
        panel.groupby(["GEOID", "date"], dropna=False, observed=True)["call_count"]
        .sum()
        .reset_index(name="daily_cnt")
    )
    panel = panel.merge(daily, on=["GEOID", "date"], how="left")

    # Önceki gün toplamı
    day_tbl = daily.sort_values(["GEOID", "date"]).copy()
    day_tbl["911_request_count_daily(before_24_hours)"] = (
        day_tbl.groupby("GEOID", sort=False)["daily_cnt"].shift(1).fillna(0)
    )
    panel = panel.merge(
        day_tbl[["GEOID", "date", "911_request_count_daily(before_24_hours)"]],
        on=["GEOID", "date"],
        how="left",
    )

    # Temporal compact features
    g = panel.groupby("GEOID", sort=False)
    gs = panel.groupby(["GEOID", "hour_range"], sort=False)

    panel["911_prev_slot"] = g["call_count"].shift(1).fillna(0)
    panel["911_prev_2slot"] = (
        g["call_count"].rolling(2, min_periods=1).sum().shift(1).reset_index(level=0, drop=True).fillna(0)
    )
    panel["911_prev_8slot"] = (
        g["call_count"].rolling(8, min_periods=1).sum().shift(1).reset_index(level=0, drop=True).fillna(0)
    )

    panel["911_roll_1d"] = (
        g["call_count"].rolling(8, min_periods=1).sum().shift(1).reset_index(level=0, drop=True).fillna(0)
    )
    panel["911_roll_3d"] = (
        g["call_count"].rolling(24, min_periods=1).sum().shift(1).reset_index(level=0, drop=True).fillna(0)
    )
    panel["911_roll_7d"] = (
        g["call_count"].rolling(56, min_periods=1).sum().shift(1).reset_index(level=0, drop=True).fillna(0)
    )

    panel["911_same_slot_prev_1d"] = gs["call_count"].shift(1).fillna(0)
    panel["911_same_slot_prev_3d"] = gs["call_count"].shift(3).fillna(0)
    panel["911_same_slot_prev_7d"] = gs["call_count"].shift(7).fillna(0)
    panel["911_same_slot_roll_4"] = (
        gs["call_count"].rolling(4, min_periods=1).mean().shift(1).reset_index(level=[0, 1], drop=True).fillna(0)
    )

    prev2 = g["call_count"].shift(2).fillna(0)
    panel["911_growth_prev_slot"] = safe_div(panel["911_prev_slot"] - prev2, prev2)

    roll_mean = g["call_count"].rolling(8, min_periods=2).mean().shift(1).reset_index(level=0, drop=True)
    roll_std = g["call_count"].rolling(8, min_periods=2).std().shift(1).reset_index(level=0, drop=True)

    panel["911_zscore_1d"] = np.where(
        roll_std.fillna(0) > 0,
        (panel["911_prev_slot"] - roll_mean.fillna(0)) / roll_std.fillna(0),
        0.0
    )
    panel["911_zscore_1d"] = pd.Series(panel["911_zscore_1d"], index=panel.index).fillna(0).astype("float32")
    panel["911_spike_flag_1d"] = (panel["911_zscore_1d"] >= 2.0).astype("int8")

    panel["has_past_crime"] = (panel["911_prev_slot"] > 0).astype("int8")

    panel["row_no_geoid"] = g.cumcount()
    last_pos = np.where(panel["call_count"] > 0, panel["row_no_geoid"], np.nan)
    last_pos = pd.Series(last_pos, index=panel.index)
    last_pos = last_pos.groupby(panel["GEOID"]).ffill().shift(1)
    panel["slots_since_last_crime"] = panel["row_no_geoid"] - last_pos
    panel["slots_since_last_crime"] = panel["slots_since_last_crime"].fillna(9999).astype("float32")
    panel = panel.drop(columns=["row_no_geoid"], errors="ignore")

    for c in panel.columns:
        if c in ["GEOID", "date", "hour_range"]:
            continue
        if not pd.api.types.is_numeric_dtype(panel[c]):
            panel[c] = pd.to_numeric(panel[c], errors="coerce").fillna(0)

    return panel


# =========================================================
# NEIGHBOR FEATURES
# =========================================================
def add_neighbor_features_fallback(summary: pd.DataFrame) -> pd.DataFrame:
    summary = summary.copy()
    summary["date"] = pd.to_datetime(summary["date"], errors="coerce").dt.date
    log("ℹ️ Neighbor fallback: aynı date-hour_range'te diğer GEOID ortalaması kullanılacak.")

    base_cols = ["911_roll_3d", "911_roll_7d"]
    rename_map = {
        "911_roll_3d": "911_neighbors_last3d",
        "911_roll_7d": "911_neighbors_last7d",
    }

    grp = summary.groupby(["date", "hour_range"], observed=True)

    for c in base_cols:
        if c not in summary.columns:
            summary[c] = 0.0

        total = grp[c].transform("sum")
        cnt = grp[c].transform("count")
        neigh = np.where(cnt > 1, (total - summary[c]) / (cnt - 1), 0)

        summary[rename_map[c]] = (
            pd.Series(neigh, index=summary.index).fillna(0).astype("float32")
        )

    return summary


def add_neighbor_features(summary: pd.DataFrame) -> pd.DataFrame:
    summary = summary.copy()
    summary["date"] = pd.to_datetime(summary["date"], errors="coerce").dt.date

    neighbor_path = load_neighbor_file()
    needed = ["GEOID", "date", "hour_range", "911_roll_3d", "911_roll_7d"]
    for c in needed:
        if c not in summary.columns:
            summary[c] = 0.0

    if neighbor_path is None:
        return add_neighbor_features_fallback(summary)

    try:
        adj = pd.read_csv(neighbor_path, low_memory=False)
        cols_lower = {c.lower(): c for c in adj.columns}

        if "geoid" in cols_lower and "neighbor_geoid" in cols_lower:
            adj = adj.rename(columns={
                cols_lower["geoid"]: "GEOID",
                cols_lower["neighbor_geoid"]: "neighbor_GEOID",
            })
        elif "geoid" in cols_lower and "neighbor" in cols_lower:
            adj = adj.rename(columns={
                cols_lower["geoid"]: "GEOID",
                cols_lower["neighbor"]: "neighbor_GEOID",
            })
        elif "GEOID" in adj.columns and "neighbor_GEOID" in adj.columns:
            pass
        else:
            raise ValueError(f"neighbors.csv beklenen kolonları taşımıyor: {adj.columns.tolist()}")

        adj["GEOID"] = normalize_geoid(adj["GEOID"], DEFAULT_GEOID_LEN)
        adj["neighbor_GEOID"] = normalize_geoid(adj["neighbor_GEOID"], DEFAULT_GEOID_LEN)
        adj = adj.dropna(subset=["GEOID", "neighbor_GEOID"]).drop_duplicates().copy()

        log(f"🧭 Neighbor pair sayısı         : {len(adj):,}")
        log(f"🧭 Unique GEOID sayısı         : {adj['GEOID'].nunique():,}")
        log(f"🧭 Unique neighbor_GEOID sayısı: {adj['neighbor_GEOID'].nunique():,}")

        if adj.empty:
            return add_neighbor_features_fallback(summary)

        base = summary[["GEOID", "date", "hour_range", "911_roll_3d", "911_roll_7d"]].copy()
        base = base.groupby(["GEOID", "date", "hour_range"], as_index=False, observed=True).mean(numeric_only=True)

        merged = adj.merge(
            base.rename(columns={"GEOID": "neighbor_GEOID"}),
            on="neighbor_GEOID",
            how="left"
        )

        agg = merged.groupby(["GEOID", "date", "hour_range"], observed=True).agg(
            **{
                "911_neighbors_last3d": ("911_roll_3d", "mean"),
                "911_neighbors_last7d": ("911_roll_7d", "mean"),
            }
        ).reset_index()

        summary = summary.merge(agg, on=["GEOID", "date", "hour_range"], how="left")

        for c in ["911_neighbors_last3d", "911_neighbors_last7d"]:
            if c in summary.columns:
                summary[c] = summary[c].fillna(0).astype("float32")

        return summary

    except Exception as e:
        log(f"⚠️ neighbors.csv okunamadı / işlenemedi, fallback kullanılacak: {e}")
        return add_neighbor_features_fallback(summary)


# =========================================================
# MERGE WITH CRIME
# =========================================================
def merge_with_crime(df_crime: pd.DataFrame, df_911: pd.DataFrame) -> pd.DataFrame:
    df_crime = df_crime.copy()
    df_911 = df_911.copy()

    if "GEOID" not in df_crime.columns:
        raise ValueError("Crime dosyasında GEOID kolonu yok.")

    if "date" not in df_crime.columns:
        dt_candidates = [c for c in ["date", "datetime", "occurred_at", "incident_datetime"] if c in df_crime.columns]
        if dt_candidates:
            df_crime["date"] = to_date(df_crime[dt_candidates[0]])
        else:
            raise ValueError("Crime dosyasında date/datetime kolonu bulunamadı.")

    if "hour_range" not in df_crime.columns:
        if "event_hour" in df_crime.columns:
            event_hour = pd.to_numeric(df_crime["event_hour"], errors="coerce").fillna(-1).astype(int)
            slot_start = (event_hour // 3) * 3
            slot_end = slot_start + 3
            df_crime["hour_range"] = np.where(
                event_hour >= 0,
                [f"{int(a):02d}-{24 if int(b) == 24 else int(b):02d}" for a, b in zip(slot_start, slot_end)],
                None
            )
        elif "datetime" in df_crime.columns:
            df_crime["hour_range"] = hour_to_range_from_datetime(pd.to_datetime(df_crime["datetime"], errors="coerce"))
        else:
            raise ValueError("Crime dosyasında hour_range/event_hour/datetime bulunamadı.")

    df_crime["GEOID"] = normalize_geoid(df_crime["GEOID"], DEFAULT_GEOID_LEN)
    df_crime["date"] = to_date(df_crime["date"])
    df_crime["hour_range"] = df_crime["hour_range"].map(canonicalize_hour_range)

    df_911["GEOID"] = normalize_geoid(df_911["GEOID"], DEFAULT_GEOID_LEN)
    df_911["date"] = to_date(df_911["date"])
    df_911["hour_range"] = df_911["hour_range"].map(canonicalize_hour_range)

    df_crime = df_crime[
        df_crime["GEOID"].notna()
        & df_crime["date"].notna()
        & df_crime["hour_range"].isin(HOUR_ORDER)
    ].copy()

    df_911 = df_911[
        df_911["GEOID"].notna()
        & df_911["date"].notna()
        & df_911["hour_range"].isin(HOUR_ORDER)
    ].copy()

    log("🔗 Join modu: DATE-BASED (GEOID, date, hour_range)")

    crime_shape_before = df_crime.shape
    merged = df_crime.merge(
        df_911,
        on=["GEOID", "date", "hour_range"],
        how="left"
    )

    log(f"📐 CRIME⨯911 merged: {merged.shape}")

    if "call_count" in merged.columns:
        matched = merged["call_count"].notna().sum()
        unmatched = merged["call_count"].isna().sum()
        log(f"🔍 911 match olan satır    : {matched:,}")
        log(f"🔍 911 match olmayan satır : {unmatched:,}")
        log(f"🔍 911 match oranı         : %{(matched / len(merged) * 100):.2f}")

    fill_cols = [
        "call_count",
        "daily_cnt",
        "911_request_count_daily(before_24_hours)",

        "911_prev_slot",
        "911_prev_2slot",
        "911_prev_8slot",
        "911_roll_1d",
        "911_roll_3d",
        "911_roll_7d",

        "911_same_slot_prev_1d",
        "911_same_slot_prev_3d",
        "911_same_slot_prev_7d",
        "911_same_slot_roll_4",

        "911_growth_prev_slot",
        "911_zscore_1d",
        "911_spike_flag_1d",

        "911_neighbors_last3d",
        "911_neighbors_last7d",

        "police_calls",
        "mta_calls",
        "sheriff_calls",

        "priority_A_calls",
        "priority_B_calls",
        "priority_C_calls",

        "type_traffic_count",
        "type_assault_count",
        "type_theft_count",
        "type_disturbance_count",
        "type_domestic_count",
        "type_weapon_count",

        "disp_han_count",
        "disp_utl_count",
        "disp_adv_count",
        "disp_arr_count",

        "unique_call_type",
        "unique_priority",
        "unique_disposition",

        "has_past_crime",
        "slots_since_last_crime",
    ]

    for c in fill_cols:
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors="coerce").fillna(0)

    log(f"📏 crime input satır/sütun   : {crime_shape_before}")
    log(f"📏 sf_crime_01 satır/sütun   : {merged.shape}")
    if merged.shape[0] == crime_shape_before[0]:
        log("✅ Merge sonrası satır kaybı yok.")
    else:
        log("⚠️ Merge sonrası satır sayısı değişti.")

    nan_total = int(merged.isna().sum().sum())
    nan_cols = merged.isna().sum()
    nan_cols = nan_cols[nan_cols > 0].sort_values(ascending=False)

    log(f"🧪 sf_crime_01 toplam NaN sayısı: {nan_total:,}")
    log(f"🧪 sf_crime_01 NaN içeren kolon sayısı: {len(nan_cols):,}")
    if len(nan_cols) > 0:
        topn = min(10, len(nan_cols))
        log(f"🧪 sf_crime_01 en çok NaN içeren ilk {topn} kolon:")
        for col, cnt in nan_cols.head(topn).items():
            pct = (cnt / len(merged) * 100) if len(merged) else 0
            log(f"   - {col}: {cnt:,} (%{pct:.2f})")

    return merged


# =========================================================
# MAIN
# =========================================================
def main():
    log(f"📂 SCRIPT_DIR = {SCRIPT_DIR}")
    log(f"📂 BASE_DIR   = {BASE_DIR}")
    log(f"📂 OUT_DIR    = {OUT_DIR}")
    log(f"🧾 DAILY_OUT seen as: {os.getenv('DAILY_OUT', '(unset)')}")

    p_911 = pick_existing(RAW_911_CANDIDATES)
    if p_911 is None:
        raise FileNotFoundError(f"911 input bulunamadı. Adaylar: {[str(x) for x in RAW_911_CANDIDATES]}")

    log(f"📦 911 base bulundu: {p_911}")
    log(f"📥 Yerel 911 tabanı okunuyor: {p_911}")
    df_911_raw = read_any(p_911)
    log(f"📐 911 input raw/summary: {df_911_raw.shape}")

    df_911 = make_standard_summary(df_911_raw)
    df_911 = add_neighbor_features(df_911)

    write_both(df_911, SUMMARY_OUT_PARQUET, SUMMARY_OUT_CSV)
    log(f"✅ Yerel 911 özet kaydedildi → {SUMMARY_OUT_PARQUET}")

    write_both(df_911, SUMMARY_Y_OUT_PARQUET, SUMMARY_Y_OUT_CSV)
    log(f"✅ Y-özet kaydedildi        → {SUMMARY_Y_OUT_PARQUET}")
    log(f"📐 final_911: {df_911.shape}")

    p_crime = pick_existing(CRIME_IN_CANDIDATES)
    if p_crime is None:
        raise FileNotFoundError(f"Crime input bulunamadı. Adaylar: {[str(x) for x in CRIME_IN_CANDIDATES]}")

    log(f"✅ Ana crime input bulundu: {p_crime}")
    df_crime = read_any(p_crime)
    log(f"📐 crime input: {df_crime.shape}")

    merged = merge_with_crime(df_crime, df_911)

    merged.to_csv(CRIME_OUT_CSV, index=False, encoding="utf-8-sig")
    merged.to_parquet(CRIME_OUT_PARQUET, index=False)

    log(f"✅ sf_crime_01.csv yazıldı     → {CRIME_OUT_CSV}")
    log(f"✅ sf_crime_01.parquet yazıldı → {CRIME_OUT_PARQUET}")


if __name__ == "__main__":
    main()
