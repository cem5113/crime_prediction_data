# app.py
# Streamlit UI for "Full SF Crime Pipeline" outputs
# Terminology: "Verseti" is used instead of "dataset".

from __future__ import annotations

import io
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

# Optional speed-ups (polars / pyarrow). If not installed, pandas fallback.
try:
    import polars as pl  # type: ignore
except Exception:
    pl = None

try:
    import pydeck as pdk  # type: ignore
except Exception:
    pdk = None


# =========================
# Config
# =========================
st.set_page_config(
    page_title="SUTAM – SF Crime Pipeline (Verseti Görüntüleyici)",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_TITLE = "SUTAM – Full SF Crime Pipeline | Verseti Görüntüleyici"

DEFAULT_SEARCH_DIRS = [
    Path(os.environ.get("CRIME_DATA_DIR", ".")),
    Path("."),
    Path("./crime_prediction_data"),
    Path("./data"),
    Path("./outputs"),
]

# Verseti candidates (prefer later stages)
VERSETE_FILES_ORDER = [
    "sf_crime_09.csv",
    "sf_crime_08.csv",
    "sf_crime_07.csv",
    "sf_crime_06.csv",
    "sf_crime_05.csv",
    "sf_crime_04.csv",
    "sf_crime_03.csv",
    "sf_crime_02.csv",
    "sf_crime_01.csv",
    "sf_crime.csv",
    # event cache / base
    "sf_crime_y.csv",
]

AUX_FILES = [
    "neighbors.csv",
    "sf_crime_grid_full_labeled.csv",
    "sf_911_last_5_year.csv",
    "sf_311_last_5_years.csv",
    "sf_population.csv",
    "sf_bus_stops_with_geoid.csv",
    "sf_train_stops_with_geoid.csv",
    "sf_pois_cleaned_with_geoid.csv",
    "sf_weather_5years.csv",
    "week.csv",
    # optional operational outputs (if you add later)
    "ops_brief_topk.csv",
    "ops_brief_geoid_summary.csv",
    "ops_brief_geoid_hour_range.csv",
]


# =========================
# Utilities
# =========================
def human_bytes(n: int) -> str:
    step = 1024.0
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < step:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= step
    return f"{n:.1f} PB"


def find_first_existing(filename: str, search_dirs: list[Path]) -> Path | None:
    for d in search_dirs:
        p = (d / filename).resolve()
        if p.exists() and p.is_file():
            return p
    return None


def list_available_versetis(search_dirs: list[Path]) -> dict[str, Path]:
    out = {}
    for f in VERSETE_FILES_ORDER:
        p = find_first_existing(f, search_dirs)
        if p:
            out[f] = p
    return out


def file_meta(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path),
        "size": human_bytes(stat.st_size),
        "mtime": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
    }


def _read_csv_pandas(path: Path) -> pd.DataFrame:
    # robust reading: handles BOM, large files, mixed types
    return pd.read_csv(path, low_memory=False, encoding="utf-8-sig")


def _read_csv_polars(path: Path) -> pd.DataFrame:
    # polars -> pandas for Streamlit compatibility
    df = pl.read_csv(str(path), ignore_errors=True)
    return df.to_pandas()


@st.cache_data(show_spinner=False)
def load_verseti(path_str: str) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame()

    # Prefer polars if available (large CSV speed)
    try:
        if pl is not None and path.stat().st_size > 50 * 1024 * 1024:
            return _read_csv_polars(path)
    except Exception:
        pass

    return _read_csv_pandas(path)


def safe_to_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def ensure_hour_range(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates hour_range (00-03, 03-06, ..., 21-00) if missing.
    Uses event_hour if available; otherwise tries datetime/time.
    """
    if "hour_range" in df.columns:
        return df

    hour = None
    if "event_hour" in df.columns:
        hour = pd.to_numeric(df["event_hour"], errors="coerce")
    elif "datetime" in df.columns:
        dt = safe_to_datetime(df["datetime"])
        hour = dt.dt.hour
    elif "time" in df.columns:
        # try parse HH:MM:SS
        t = df["time"].astype(str).str.slice(0, 2)
        hour = pd.to_numeric(t, errors="coerce")

    if hour is None:
        return df

    # bins: 0-3, 3-6, ..., 21-24(0)
    bins = [0, 3, 6, 9, 12, 15, 18, 21, 24]
    labels = ["00-03", "03-06", "06-09", "09-12", "12-15", "15-18", "18-21", "21-00"]
    hr = hour.fillna(-1).astype(int)
    hr_clip = hr.clip(lower=0, upper=23)
    df["hour_range"] = pd.cut(hr_clip, bins=bins, labels=labels, right=False, include_lowest=True).astype(str)
    return df


def ensure_date(df: pd.DataFrame) -> pd.DataFrame:
    if "date" in df.columns:
        df["date"] = safe_to_datetime(df["date"]).dt.date
        return df
    if "date_only" in df.columns:
        df["date"] = safe_to_datetime(df["date_only"]).dt.date
        return df
    if "datetime" in df.columns:
        df["date"] = safe_to_datetime(df["datetime"]).dt.date
        return df
    return df


def pick_lat_lon_cols(df: pd.DataFrame) -> tuple[str | None, str | None]:
    # common names
    lat_candidates = ["latitude", "lat", "y", "centroid_lat"]
    lon_candidates = ["longitude", "lon", "lng", "x", "centroid_lon"]
    lat = next((c for c in lat_candidates if c in df.columns), None)
    lon = next((c for c in lon_candidates if c in df.columns), None)
    return lat, lon


def pick_geoid_col(df: pd.DataFrame) -> str | None:
    for c in ["GEOID", "geoid", "geoid10", "tract_geoid"]:
        if c in df.columns:
            return c
    return None


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def metric_card(col, label: str, value: str, help_text: str | None = None):
    col.metric(label=label, value=value, help=help_text)


# =========================
# UI
# =========================
st.title(APP_TITLE)

search_dirs = DEFAULT_SEARCH_DIRS
available = list_available_versetis(search_dirs)

with st.sidebar:
    st.header("Ayarlar")

    if not available:
        st.error(
            "Hiç Verseti bulunamadı. Repo kökünde veya crime_prediction_data/ altında "
            "sf_crime_09.csv (veya önceki aşamalar) olmalı."
        )
        st.stop()

    verseti_name = st.selectbox(
        "Verseti seç",
        options=list(available.keys()),
        index=0,
        help="Pipeline çıktılarından görüntülemek istediğin Verseti’yi seç.",
    )
    verseti_path = available[verseti_name]
    st.caption(f"📄 {verseti_path}")

    st.divider()

    st.subheader("Filtreler (opsiyonel)")
    use_sampling = st.checkbox("Harita için örnekleme kullan", value=True)
    map_sample_n = st.slider("Harita örneklem satırı", min_value=5_000, max_value=200_000, value=50_000, step=5_000)

    st.divider()

    st.subheader("Yardımcı dosyalar")
    aux_found = []
    for f in AUX_FILES:
        p = find_first_existing(f, search_dirs)
        if p:
            aux_found.append((f, p))
    if aux_found:
        for f, p in aux_found[:12]:
            m = file_meta(p)
            st.caption(f"✅ {f} • {m['size']} • {m['mtime']}")
    else:
        st.caption("Henüz yardımcı dosya bulunmadı (opsiyonel).")


# Load
df = load_verseti(str(verseti_path))
if df.empty:
    st.error(f"Verseti okunamadı veya boş: {verseti_path}")
    st.stop()

# Normalize key columns
df = ensure_date(df)
df = ensure_hour_range(df)

geoid_col = pick_geoid_col(df)
lat_col, lon_col = pick_lat_lon_cols(df)

# Top summary
meta = file_meta(verseti_path)
top_left, top_mid, top_right, top_4 = st.columns([1.1, 1.0, 1.0, 1.2])

metric_card(top_left, "Verseti", verseti_name, help_text=meta["path"])
metric_card(top_mid, "Satır", f"{len(df):,}")
metric_card(top_right, "Sütun", f"{df.shape[1]:,}")
metric_card(top_4, "Dosya", f"{meta['size']} • {meta['mtime']}")

# Filters UI (main area)
f1, f2, f3, f4, f5 = st.columns([1.2, 1.2, 1.2, 1.2, 1.2])

date_min, date_max = None, None
if "date" in df.columns and df["date"].notna().any():
    date_min = df["date"].min()
    date_max = df["date"].max()
    sel_date = f1.date_input("Tarih", value=(date_min, date_max))
else:
    sel_date = None
    f1.caption("Tarih kolonu yok.")

if "hour_range" in df.columns:
    hour_opts = sorted([h for h in df["hour_range"].dropna().unique().tolist() if h != "nan"])
    sel_hour = f2.multiselect("Saat dilimi (hour_range)", options=hour_opts, default=hour_opts)
else:
    sel_hour = []
    f2.caption("hour_range yok.")

if geoid_col:
    geoid_opts = sorted(df[geoid_col].astype(str).dropna().unique().tolist())
    # keep it light: show empty as "All"
    sel_geoid = f3.multiselect("GEOID", options=geoid_opts, default=[])
else:
    sel_geoid = []
    f3.caption("GEOID yok.")

cat_col = "category" if "category" in df.columns else None
subcat_col = "subcategory" if "subcategory" in df.columns else None

if cat_col:
    cat_opts = sorted(df[cat_col].astype(str).dropna().unique().tolist())
    sel_cat = f4.multiselect("Kategori", options=cat_opts, default=[])
else:
    sel_cat = []
    f4.caption("category yok.")

if subcat_col:
    sub_opts = sorted(df[subcat_col].astype(str).dropna().unique().tolist())
    sel_sub = f5.multiselect("Alt kategori", options=sub_opts, default=[])
else:
    sel_sub = []
    f5.caption("subcategory yok.")

# Apply filters
fdf = df.copy()

if sel_date and isinstance(sel_date, (tuple, list)) and len(sel_date) == 2:
    d0, d1 = sel_date
    if "date" in fdf.columns:
        fdf = fdf[(fdf["date"] >= d0) & (fdf["date"] <= d1)]

if sel_hour:
    fdf = fdf[fdf["hour_range"].astype(str).isin(sel_hour)]

if geoid_col and sel_geoid:
    fdf = fdf[fdf[geoid_col].astype(str).isin(sel_geoid)]

if cat_col and sel_cat:
    fdf = fdf[fdf[cat_col].astype(str).isin(sel_cat)]

if subcat_col and sel_sub:
    fdf = fdf[fdf[subcat_col].astype(str).isin(sel_sub)]

st.divider()

# Filtered summary
s1, s2, s3, s4 = st.columns([1, 1, 1, 1])

metric_card(s1, "Filtreli satır", f"{len(fdf):,}")
nan_total = int(fdf.isna().sum().sum())
metric_card(s2, "Toplam NaN", f"{nan_total:,}")
metric_card(s3, "GEOID sayısı", f"{fdf[geoid_col].nunique():,}" if geoid_col else "—")
metric_card(s4, "Kategori sayısı", f"{fdf[cat_col].nunique():,}" if cat_col else "—")

# Downloads
dl1, dl2, dl3 = st.columns([1, 1, 2])
with dl1:
    st.download_button(
        "⬇️ Filtreli Verseti (CSV)",
        data=to_csv_bytes(fdf),
        file_name=f"filtered_{verseti_name}",
        mime="text/csv",
        use_container_width=True,
    )
with dl2:
    st.download_button(
        "⬇️ Tüm Verseti (CSV)",
        data=to_csv_bytes(df),
        file_name=f"{verseti_name}",
        mime="text/csv",
        use_container_width=True,
    )
with dl3:
    st.caption("Not: Büyük Versetilerde CSV indirme tarayıcıda yavaş olabilir.")

# Tabs
tab1, tab2, tab3 = st.tabs(["Önizleme", "Harita", "Kolon/İstatistik"])

with tab1:
    st.subheader("Filtreli Verseti önizleme")
    st.dataframe(fdf.head(2000), use_container_width=True, height=520)

with tab2:
    st.subheader("Harita görünümü")

    if pdk is None:
        st.warning("pydeck yüklü değil. Harita için requirements.txt içine pydeck ekleyebilirsin (Streamlit zaten çoğu zaman getirir).")
    elif lat_col is None or lon_col is None:
        st.warning("Bu Verseti’de latitude/longitude bulunamadı. Harita çizilemedi.")
    else:
        mdf = fdf.copy()

        # Clean coords
        mdf[lat_col] = pd.to_numeric(mdf[lat_col], errors="coerce")
        mdf[lon_col] = pd.to_numeric(mdf[lon_col], errors="coerce")
        mdf = mdf.dropna(subset=[lat_col, lon_col])

        if mdf.empty:
            st.warning("Seçili filtrelerde koordinatlı satır yok.")
        else:
            if use_sampling and len(mdf) > map_sample_n:
                mdf = mdf.sample(map_sample_n, random_state=42)

            # center
            center_lat = float(mdf[lat_col].mean())
            center_lon = float(mdf[lon_col].mean())

            # tooltip fields
            tooltip_parts = []
            if geoid_col:
                tooltip_parts.append(f"<b>GEOID:</b> {{{geoid_col}}}")
            if "date" in mdf.columns:
                tooltip_parts.append("<b>Date:</b> {date}")
            if "hour_range" in mdf.columns:
                tooltip_parts.append("<b>Hour:</b> {hour_range}")
            if cat_col:
                tooltip_parts.append(f"<b>Category:</b> {{{cat_col}}}")
            if subcat_col:
                tooltip_parts.append(f"<b>Subcategory:</b> {{{subcat_col}}}")
            tooltip_html = "<br/>".join(tooltip_parts) if tooltip_parts else "Kayıt"

            layer = pdk.Layer(
                "ScatterplotLayer",
                data=mdf,
                get_position=f"[{lon_col}, {lat_col}]",
                get_radius=35,
                pickable=True,
                opacity=0.6,
            )

            view_state = pdk.ViewState(
                latitude=center_lat,
                longitude=center_lon,
                zoom=10.6,
                pitch=0,
            )

            st.pydeck_chart(
                pdk.Deck(
                    layers=[layer],
                    initial_view_state=view_state,
                    tooltip={"html": tooltip_html},
                ),
                use_container_width=True,
            )

            st.caption(
                f"Haritada gösterilen satır: {len(mdf):,}  |  "
                f"Koordinat kolonları: {lat_col}, {lon_col}"
            )

with tab3:
    st.subheader("Kolonlar ve hızlı kontroller")

    c1, c2 = st.columns([1.2, 1.8])

    with c1:
        st.write("**Kolon listesi**")
        st.code(", ".join(df.columns.tolist()))

        st.write("**NaN (ilk 25 kolon)**")
        nan_s = df.isna().sum().sort_values(ascending=False).head(25)
        st.dataframe(nan_s.rename("nan_count").to_frame(), use_container_width=True, height=420)

    with c2:
        st.write("**Sayısal kolon özetleri (ilk 20)**")
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            st.caption("Sayısal kolon yok.")
        else:
            desc = df[num_cols].describe().T
            st.dataframe(desc.head(20), use_container_width=True, height=520)

        st.write("**Kategori dağılımı (varsa)**")
        if cat_col:
            vc = fdf[cat_col].astype(str).value_counts().head(20)
            st.dataframe(vc.rename("count").to_frame(), use_container_width=True, height=420)
        else:
            st.caption("category kolonu yok.")


# Footer
st.divider()
st.caption(
    "Bu arayüz SUTAM uygulamasının verisetini (sf_crime_01…sf_crime_09) incelemek içindir. "
)
