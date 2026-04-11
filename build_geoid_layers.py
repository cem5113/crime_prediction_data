"""
build_geoid_layers.py

Amaç:
1) sf_crime_09.parquet içinden GEOID bazlı sabit/yavaş değişen profil tablosu üretmek
   -> geoid_profile.parquet

2) GEOID bazlı yakın dönem özet tablosu üretmek
   -> geoid_dashboard_snapshot.parquet

Beklenen ana kaynak:
- /content/drive/MyDrive/crime_inputs/sf_crime_09.parquet

Çıktılar:
- /content/drive/MyDrive/crime_inputs/geoid_profile.parquet
- /content/drive/MyDrive/crime_inputs/geoid_dashboard_snapshot.parquet
"""

import os
import re
import json
import warnings
from typing import List, Optional, Dict, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# PATH AYARLARI
# ============================================================
BASE_DIR = "/content/drive/MyDrive/crime_inputs"

INPUT_PATHS = [
    os.path.join(BASE_DIR, "sf_crime_09.parquet"),
    os.path.join(BASE_DIR, "sf_crime_09.csv"),
]

OUT_PROFILE = os.path.join(BASE_DIR, "geoid_profile.parquet")
OUT_SNAPSHOT = os.path.join(BASE_DIR, "geoid_dashboard_snapshot.parquet")
OUT_DEBUG_JSON = os.path.join(BASE_DIR, "geoid_layer_build_debug.json")


# ============================================================
# YARDIMCI FONKSİYONLAR
# ============================================================
def first_existing(cols: List[str], candidates: List[str]) -> Optional[str]:
    colset = set(cols)
    for c in candidates:
        if c in colset:
            return c
    return None


def existing_many(cols: List[str], candidates: List[str]) -> List[str]:
    colset = set(cols)
    return [c for c in candidates if c in colset]


def safe_mode(series: pd.Series):
    s = series.dropna()
    if s.empty:
        return np.nan
    m = s.mode(dropna=True)
    if len(m) == 0:
        return np.nan
    return m.iloc[0]


def to_numeric_if_possible(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def normalize_geoid(s: pd.Series) -> pd.Series:
    """
    GEOID'yi 11 haneli string'e normalize eder.
    Regex extract yerine sayısal temizlik + zfill kullanılır.
    """
    s = s.astype(str).str.strip()

    # sadece rakamları koru
    s = s.str.replace(r"\.0$", "", regex=True)
    s = s.str.replace(r"[^\d]", "", regex=True)

    # boşları NaN yap
    s = s.replace("", np.nan)

    def _fix(x):
        if pd.isna(x):
            return np.nan
        try:
            return str(int(float(x))).zfill(11)
        except Exception:
            x = str(x)
            return x.zfill(11) if x.isdigit() else np.nan

    return s.map(_fix)


def normalize_hour_range(x):
    """
    00-03 formatına normalize eder.
    """
    if pd.isna(x):
        return np.nan

    x = str(x).strip()

    # zaten 00-03 gibi ise
    m = re.match(r"^\s*(\d{1,2})\s*[-_]\s*(\d{1,2})\s*$", x)
    if m:
        a = int(m.group(1))
        b = int(m.group(2))
        return f"{a:02d}-{b:02d}"

    # tek sayı ise slot başlangıcı say
    if x.isdigit():
        a = int(x)
        b = (a + 3) % 24
        return f"{a:02d}-{b:02d}"

    return x


def load_main_panel() -> Tuple[pd.DataFrame, str]:
    for p in INPUT_PATHS:
        if os.path.exists(p):
            if p.endswith(".parquet"):
                df = pd.read_parquet(p)
            else:
                df = pd.read_csv(p)
            return df, p
    raise FileNotFoundError(
        f"Ana panel bulunamadı. Beklenen yollardan hiçbiri yok: {INPUT_PATHS}"
    )


def aggregate_static_feature(g: pd.core.groupby.SeriesGroupBy, how: str):
    if how == "median":
        return g.median()
    elif how == "mean":
        return g.mean()
    elif how == "max":
        return g.max()
    elif how == "min":
        return g.min()
    elif how == "mode":
        return g.apply(safe_mode)
    elif how == "last":
        return g.last()
    else:
        return g.apply(safe_mode)


def summarize_window(
    df: pd.DataFrame,
    geoid_col: str,
    date_col: str,
    event_col: Optional[str],
    col_911: Optional[str],
    col_311: Optional[str],
    neighbor_col: Optional[str],
    end_date: pd.Timestamp,
    days: int,
    suffix: str,
) -> pd.DataFrame:
    """
    Son N gün için GEOID bazlı özet üretir.
    """
    start_date = end_date - pd.Timedelta(days=days - 1)
    mask = (df[date_col] >= start_date) & (df[date_col] <= end_date)
    part = df.loc[mask].copy()

    out = pd.DataFrame({geoid_col: sorted(df[geoid_col].dropna().unique())})

    if len(part) == 0:
        return out

    grp = part.groupby(geoid_col, dropna=False)

    # Suç sayısı / olay oranı
    if event_col is not None:
        part[event_col] = pd.to_numeric(part[event_col], errors="coerce").fillna(0)
        crime_sum = grp[event_col].sum().rename(f"crime_count_{suffix}")
        crime_mean = grp[event_col].mean().rename(f"crime_rate_{suffix}")
        out = out.merge(crime_sum.reset_index(), on=geoid_col, how="left")
        out = out.merge(crime_mean.reset_index(), on=geoid_col, how="left")

    # 911
    if col_911 is not None:
        part[col_911] = pd.to_numeric(part[col_911], errors="coerce").fillna(0)
        call_911_sum = grp[col_911].sum().rename(f"calls_911_{suffix}")
        call_911_mean = grp[col_911].mean().rename(f"calls_911_rate_{suffix}")
        out = out.merge(call_911_sum.reset_index(), on=geoid_col, how="left")
        out = out.merge(call_911_mean.reset_index(), on=geoid_col, how="left")

    # 311
    if col_311 is not None:
        part[col_311] = pd.to_numeric(part[col_311], errors="coerce").fillna(0)
        call_311_sum = grp[col_311].sum().rename(f"calls_311_{suffix}")
        call_311_mean = grp[col_311].mean().rename(f"calls_311_rate_{suffix}")
        out = out.merge(call_311_sum.reset_index(), on=geoid_col, how="left")
        out = out.merge(call_311_mean.reset_index(), on=geoid_col, how="left")

    # Komşu suç
    if neighbor_col is not None:
        part[neighbor_col] = pd.to_numeric(part[neighbor_col], errors="coerce").fillna(0)
        neigh_mean = grp[neighbor_col].mean().rename(f"neighbor_crime_mean_{suffix}")
        neigh_sum = grp[neighbor_col].sum().rename(f"neighbor_crime_sum_{suffix}")
        out = out.merge(neigh_mean.reset_index(), on=geoid_col, how="left")
        out = out.merge(neigh_sum.reset_index(), on=geoid_col, how="left")

    return out


# ============================================================
# ANA İŞ AKIŞI
# ============================================================
def main():
    # --------------------------------------------------------
    # 1) Veri oku
    # --------------------------------------------------------
    df, used_input_path = load_main_panel()
    print(f"✅ Ana panel okundu: {used_input_path}")
    print(f"shape: {df.shape}")

    original_cols = df.columns.tolist()

    # --------------------------------------------------------
    # 2) Ana kolonları bul
    # --------------------------------------------------------
    geoid_col = first_existing(original_cols, ["GEOID", "geoid", "tract_geoid"])
    date_col = first_existing(original_cols, ["date", "datetime", "Date", "DATE"])
    hour_range_col = first_existing(original_cols, ["hour_range", "hour_bin", "slot"])

    if geoid_col is None:
        raise ValueError("❌ GEOID kolonu bulunamadı.")
    if date_col is None:
        raise ValueError("❌ date/datetime kolonu bulunamadı.")

    # olay kolonu
    event_col = first_existing(original_cols, [
        "y_event", "Y_label", "label", "crime_event", "target"
    ])

    # 911 / 311
    col_911 = first_existing(original_cols, [
        "911_request_count_hour_range",
        "911_request_count_daily(before_24_hours)",
        "911_request_count_daily",
        "call_911_count",
        "calls_911",
        "count_911",
    ])
    col_311 = first_existing(original_cols, [
        "311_request_count",
        "call_311_count",
        "calls_311",
        "count_311",
    ])

    # komşu suç
    neighbor_col = first_existing(original_cols, [
        "neighbor_crime_7d",
        "neighbor_crime_3d",
        "neighbor_crime_1d",
        "neighbor_crime",
    ])

    # --------------------------------------------------------
    # 3) Normalize et
    # --------------------------------------------------------
    df[geoid_col] = normalize_geoid(df[geoid_col])
    df = df[df[geoid_col].notna()].copy()

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df[df[date_col].notna()].copy()
    df[date_col] = df[date_col].dt.normalize()

    if hour_range_col is not None:
        df[hour_range_col] = df[hour_range_col].map(normalize_hour_range)

    # event kolonu yoksa geçmiş suç proxy olarak 0/1 üret
    if event_col is None:
        print("⚠️ y_event / Y_label bulunamadı. 0 değerli geçici kolon üretilecek.")
        event_col = "_tmp_event_col"
        df[event_col] = 0

    df[event_col] = pd.to_numeric(df[event_col], errors="coerce").fillna(0).clip(lower=0)

    print(f"✅ GEOID kolonu    : {geoid_col}")
    print(f"✅ tarih kolonu    : {date_col}")
    print(f"✅ hour_range      : {hour_range_col}")
    print(f"✅ olay kolonu     : {event_col}")
    print(f"✅ 911 kolonu      : {col_911}")
    print(f"✅ 311 kolonu      : {col_311}")
    print(f"✅ komşu suç kolonu: {neighbor_col}")

    # --------------------------------------------------------
    # 4) GEOID PROFILE için aday kolonlar
    # --------------------------------------------------------
    static_candidates: Dict[str, Dict[str, List[str] or str]] = {
        "population": {
            "cands": ["population", "pop_total", "total_population", "nufus"],
            "how": "median",
        },
        "population_density": {
            "cands": ["population_density", "pop_density", "density"],
            "how": "median",
        },

        "median_income": {
            "cands": ["median_income", "income_median"],
            "how": "median",
        },

        "poi_total_count": {
            "cands": ["poi_total_count", "poi_count_total", "total_poi_count"],
            "how": "median",
        },
        "poi_risk_score": {
            "cands": ["poi_risk_score", "poi_risk_score_mean", "poi_risk"],
            "how": "median",
        },
        "poi_dominant_type": {
            "cands": ["poi_dominant_type", "dominant_poi_type", "poi_top_type"],
            "how": "mode",
        },

        "distance_to_bus": {
            "cands": ["distance_to_bus", "bus_distance"],
            "how": "median",
        },
        "bus_stop_count": {
            "cands": ["bus_stop_count", "count_bus_stops"],
            "how": "median",
        },
        "distance_to_train": {
            "cands": ["distance_to_train", "train_distance"],
            "how": "median",
        },
        "train_stop_count": {
            "cands": ["train_stop_count", "count_train_stops"],
            "how": "median",
        },

        "distance_to_police": {
            "cands": ["distance_to_police", "police_distance"],
            "how": "median",
        },
        "distance_to_government_building": {
            "cands": [
                "distance_to_government_building",
                "distance_to_government",
                "government_distance",
            ],
            "how": "median",
        },

        "is_near_police": {
            "cands": ["is_near_police"],
            "how": "max",
        },
        "is_near_government": {
            "cands": ["is_near_government", "is_near_government_building"],
            "how": "max",
        },

        "latitude": {
            "cands": ["latitude", "lat", "centroid_lat"],
            "how": "median",
        },
        "longitude": {
            "cands": ["longitude", "lon", "lng", "centroid_lon"],
            "how": "median",
        },

        "season_mode": {
            "cands": ["season"],
            "how": "mode",
        },
    }

    # Demografi/demografik yapı için olası kategorik kolonlar
    demographic_candidates = [
        "race_majority",
        "ethnicity_majority",
        "age_group_dominant",
        "household_type",
        "income_group",
        "education_level",
        "demographic_profile",
    ]

    # --------------------------------------------------------
    # 5) GEOID PROFILE üret
    # --------------------------------------------------------
    geoid_values = sorted(df[geoid_col].dropna().unique())
    geoid_profile = pd.DataFrame({geoid_col: geoid_values})

    grp = df.groupby(geoid_col, dropna=False)

    used_profile_cols = {}

    for out_name, meta in static_candidates.items():
        src = first_existing(original_cols, meta["cands"])
        if src is None:
            continue

        how = meta["how"]
        if how in ["median", "mean", "max", "min"]:
            tmp = to_numeric_if_possible(df[src])
            agg = tmp.groupby(df[geoid_col]).agg(how).rename(out_name)
            geoid_profile = geoid_profile.merge(
                agg.reset_index(), on=geoid_col, how="left"
            )
        else:
            agg = grp[src].apply(safe_mode).rename(out_name)
            geoid_profile = geoid_profile.merge(
                agg.reset_index(), on=geoid_col, how="left"
            )

        used_profile_cols[out_name] = src

    # Demografik yapı kolonları varsa ekle
    for c in demographic_candidates:
        if c in original_cols:
            agg = grp[c].apply(safe_mode).rename(c)
            geoid_profile = geoid_profile.merge(
                agg.reset_index(), on=geoid_col, how="left"
            )
            used_profile_cols[c] = c

    # Uzun dönem suç / çağrı ortalamaları
    long_term_parts = []

    crime_rate_lt = grp[event_col].mean().rename("long_term_crime_rate")
    crime_count_lt = grp[event_col].sum().rename("long_term_crime_count")
    long_term_parts.extend([crime_rate_lt, crime_count_lt])

    if col_911 is not None:
        x911 = pd.to_numeric(df[col_911], errors="coerce").fillna(0)
        long_term_parts.append(x911.groupby(df[geoid_col]).mean().rename("long_term_911_rate"))
        long_term_parts.append(x911.groupby(df[geoid_col]).sum().rename("long_term_911_sum"))

    if col_311 is not None:
        x311 = pd.to_numeric(df[col_311], errors="coerce").fillna(0)
        long_term_parts.append(x311.groupby(df[geoid_col]).mean().rename("long_term_311_rate"))
        long_term_parts.append(x311.groupby(df[geoid_col]).sum().rename("long_term_311_sum"))

    if neighbor_col is not None:
        xng = pd.to_numeric(df[neighbor_col], errors="coerce").fillna(0)
        long_term_parts.append(xng.groupby(df[geoid_col]).mean().rename("long_term_neighbor_crime_mean"))
        long_term_parts.append(xng.groupby(df[geoid_col]).sum().rename("long_term_neighbor_crime_sum"))

    for part in long_term_parts:
        geoid_profile = geoid_profile.merge(
            part.reset_index(), on=geoid_col, how="left"
        )

    # veri kapsamı
    coverage = grp[date_col].agg(["min", "max", "nunique"]).reset_index()
    coverage.columns = [geoid_col, "panel_date_min", "panel_date_max", "n_days_present"]
    geoid_profile = geoid_profile.merge(coverage, on=geoid_col, how="left")

    # slot coverage
    if hour_range_col is not None:
        slot_cov = grp[hour_range_col].nunique().rename("n_hour_ranges_present")
        geoid_profile = geoid_profile.merge(slot_cov.reset_index(), on=geoid_col, how="left")

    # --------------------------------------------------------
    # 6) SNAPSHOT üret
    # --------------------------------------------------------
    max_date = df[date_col].max()
    print(f"✅ Panel tarih aralığı: {df[date_col].min().date()} -> {max_date.date()}")

    snap_7d = summarize_window(
        df=df,
        geoid_col=geoid_col,
        date_col=date_col,
        event_col=event_col,
        col_911=col_911,
        col_311=col_311,
        neighbor_col=neighbor_col,
        end_date=max_date,
        days=7,
        suffix="7d",
    )

    snap_30d = summarize_window(
        df=df,
        geoid_col=geoid_col,
        date_col=date_col,
        event_col=event_col,
        col_911=col_911,
        col_311=col_311,
        neighbor_col=neighbor_col,
        end_date=max_date,
        days=30,
        suffix="30d",
    )

    snap_90d = summarize_window(
        df=df,
        geoid_col=geoid_col,
        date_col=date_col,
        event_col=event_col,
        col_911=col_911,
        col_311=col_311,
        neighbor_col=neighbor_col,
        end_date=max_date,
        days=90,
        suffix="90d",
    )

    geoid_snapshot = pd.DataFrame({geoid_col: geoid_values})
    for piece in [snap_7d, snap_30d, snap_90d]:
        geoid_snapshot = geoid_snapshot.merge(piece, on=geoid_col, how="left")

    # son görülen tarih bilgisi
    last_seen = grp[date_col].max().rename("last_seen_date")
    geoid_snapshot = geoid_snapshot.merge(last_seen.reset_index(), on=geoid_col, how="left")

    geoid_snapshot["snapshot_end_date"] = max_date
    geoid_snapshot["updated_at"] = pd.Timestamp.utcnow()

    # --------------------------------------------------------
    # 7) NULL / tip temizliği
    # --------------------------------------------------------
    for frame in [geoid_profile, geoid_snapshot]:
        for c in frame.columns:
            if c == geoid_col:
                continue
            if str(frame[c].dtype) == "object":
                # object kalacaksa kalsın; sadece çok boş stringleri NaN yap
                frame[c] = frame[c].replace("", np.nan)

    # --------------------------------------------------------
    # 8) Kaydet
    # --------------------------------------------------------
    geoid_profile.to_parquet(OUT_PROFILE, index=False)
    geoid_snapshot.to_parquet(OUT_SNAPSHOT, index=False)

    debug_info = {
        "used_input_path": used_input_path,
        "input_shape": list(df.shape),
        "geoid_col": geoid_col,
        "date_col": date_col,
        "hour_range_col": hour_range_col,
        "event_col": event_col,
        "col_911": col_911,
        "col_311": col_311,
        "neighbor_col": neighbor_col,
        "used_profile_cols": used_profile_cols,
        "profile_shape": list(geoid_profile.shape),
        "snapshot_shape": list(geoid_snapshot.shape),
        "panel_date_min": str(df[date_col].min()),
        "panel_date_max": str(df[date_col].max()),
        "profile_path": OUT_PROFILE,
        "snapshot_path": OUT_SNAPSHOT,
    }

    with open(OUT_DEBUG_JSON, "w", encoding="utf-8") as f:
        json.dump(debug_info, f, ensure_ascii=False, indent=2, default=str)

    # --------------------------------------------------------
    # 9) Ekran çıktısı
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("✅ GEOID LAYER ÜRETİMİ TAMAMLANDI")
    print("=" * 70)
    print(f"profile  path  : {OUT_PROFILE}")
    print(f"profile  shape : {geoid_profile.shape}")
    print(f"snapshot path  : {OUT_SNAPSHOT}")
    print(f"snapshot shape : {geoid_snapshot.shape}")
    print(f"debug json     : {OUT_DEBUG_JSON}")

    show_cols_profile = [c for c in [
        geoid_col,
        "population",
        "poi_total_count",
        "poi_dominant_type",
        "distance_to_bus",
        "bus_stop_count",
        "distance_to_train",
        "train_stop_count",
        "long_term_crime_rate",
        "long_term_911_rate",
        "long_term_311_rate",
    ] if c in geoid_profile.columns]

    show_cols_snapshot = [c for c in [
        geoid_col,
        "crime_count_7d",
        "crime_count_30d",
        "calls_911_7d",
        "calls_911_30d",
        "calls_311_7d",
        "calls_311_30d",
        "neighbor_crime_mean_7d",
        "neighbor_crime_mean_30d",
        "snapshot_end_date",
    ] if c in geoid_snapshot.columns]

    print("\n[PROFILE PREVIEW]")
    print(geoid_profile[show_cols_profile].head(5).to_string(index=False) if show_cols_profile else geoid_profile.head(5).to_string(index=False))

    print("\n[SNAPSHOT PREVIEW]")
    print(geoid_snapshot[show_cols_snapshot].head(5).to_string(index=False) if show_cols_snapshot else geoid_snapshot.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
