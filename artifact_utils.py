# 3 Saatlik Bloklar (≤7 gün; 3-saatlik aralık) ve Günlük (≤365 gün) risk görünümleri

import os
from streamlit_folium import st_folium
import folium
import io
import json
import posixpath
import zipfile
from io import BytesIO
from datetime import datetime, timedelta

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

import requests
import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk

# ------------------------------------------------------------
# ⚙️ GitHub repo ve artifact bilgisi
# ------------------------------------------------------------
REPOSITORY_OWNER = "cem5113"
REPOSITORY_NAME  = "crime_prediction_data"
# Sadece SF risk çıktıları artifact'i
ARTIFACT_NAME_SHOULD_CONTAIN = "sf-crime-outputs-parquet"

ARTIFACT_MEMBER_HOURLY = "risk_3h_next7d_top3"
ARTIFACT_MEMBER_DAILY  = "risk_daily_next365d_top5"

# 3-saatlik CSV için yerel yol (SF tahmin çıktısı)
CSV_HOURLY_FRSTYLE = "crime_forecast_7days_all_geoids_FRstyle.csv"

# Yerel GeoJSON (2_🗺️_Risk_Haritası.py ile aynı)
GEOJSON_LOCAL = "data/sf_cells.geojson"

# ------------------------------------------------------------
# 🔑 Token / Header
# ------------------------------------------------------------
def resolve_github_token() -> str | None:
    if os.getenv("GITHUB_TOKEN"):
        return os.getenv("GITHUB_TOKEN")
    for key in ("github_token", "GH_TOKEN", "GITHUB_TOKEN"):
        try:
            if key in st.secrets and st.secrets[key]:
                os.environ["GITHUB_TOKEN"] = str(st.secrets[key])
                return os.environ["GITHUB_TOKEN"]
        except Exception:
            pass
    return None

def github_api_headers() -> dict:
    headers = {"Accept": "application/vnd.github+json"}
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers

# ------------------------------------------------------------
# 📦 Artifact ZIP alma (en güncel ve süresi dolmamış)
# ------------------------------------------------------------
def resolve_latest_artifact_zip_url(owner: str, repo: str, name_contains: str):
    token = resolve_github_token()
    if not token:
        return None, {}
    base = f"https://api.github.com/repos/{owner}/{repo}"
    response = requests.get(
        f"{base}/actions/artifacts?per_page=100",
        headers=github_api_headers(),
        timeout=60,
    )
    response.raise_for_status()
    artifacts = (response.json() or {}).get("artifacts", []) or []
    artifacts = [
        a for a in artifacts
        if (name_contains in a.get("name", "")) and not a.get("expired")
    ]
    if not artifacts:
        return None, {}
    artifacts.sort(key=lambda a: a.get("updated_at", ""), reverse=True)
    url = f"{base}/actions/artifacts/{artifacts[0]['id']}/zip"
    return url, github_api_headers()

# ------------------------------------------------------------
# 🧰 ZIP içinden üye okuma (nested zip + parquet/csv fallback)
# ------------------------------------------------------------
def read_member_from_zip_bytes(zip_bytes: bytes, member_path: str) -> pd.DataFrame:
    """
    Artifact ZIP'inde:
      - önce doğrudan dosyayı arar
      - yoksa içerdeki .zip (örn. sf_parquet_outputs.zip) dosyalarını açıp orada arar.

    member_path: "risk_hourly_next24h_top3" gibi gövde adı.
    """

    def read_any_table(raw_bytes: bytes, name_hint: str) -> pd.DataFrame:
        buf = BytesIO(raw_bytes)
        name_l = name_hint.lower()
        if name_l.endswith(".csv"):
            return pd.read_csv(buf)
        # Önce parquet dene, hata olursa csv'e düş
        try:
            buf.seek(0)
            return pd.read_parquet(buf)
        except Exception:
            buf.seek(0)
            return pd.read_csv(buf)

    def scan_zip(zf: zipfile.ZipFile, member_path: str) -> pd.DataFrame | None:
        """Verilen ZipFile içinde stem'i geçen ilk dosyayı bulup DataFrame döndürür."""
        names = zf.namelist()
        base  = posixpath.basename(member_path)
        stem  = base.split(".")[0]
        stemL = stem.lower()

        for n in names:
            bn = posixpath.basename(n)
            if stemL in bn.lower():
                with zf.open(n) as f:
                    return read_any_table(f.read(), bn)
        return None

    # 1) Dış ZIP'i aç
    with zipfile.ZipFile(BytesIO(zip_bytes)) as outer:
        # Önce dış zip içinde ara
        df = scan_zip(outer, member_path)
        if df is not None:
            return df

        # 2) Bulunamazsa: içerdeki .zip dosyalarını sırayla dene
        for name in outer.namelist():
            if name.lower().endswith(".zip"):
                with outer.open(name) as f_z:
                    inner_bytes = f_z.read()
                try:
                    with zipfile.ZipFile(BytesIO(inner_bytes)) as inner:
                        df_inner = scan_zip(inner, member_path)
                        if df_inner is not None:
                            return df_inner
                except zipfile.BadZipFile:
                    continue

    # Hiçbir eşleşme bulunamadıysa:
    raise FileNotFoundError(
        f"ZIP içinde '{member_path}' gövdesini içeren bir CSV/PARQUET dosyası bulunamadı."
    )

@st.cache_data(show_spinner=False)
def load_artifact_member(member: str) -> pd.DataFrame:
    url, headers = resolve_latest_artifact_zip_url(
        REPOSITORY_OWNER, REPOSITORY_NAME, ARTIFACT_NAME_SHOULD_CONTAIN
    )
    if not url:
        raise RuntimeError("Artifact bulunamadı veya GITHUB_TOKEN yok.")
    r = requests.get(url, headers=headers, timeout=120, allow_redirects=True)
    r.raise_for_status()
    return read_member_from_zip_bytes(r.content, member)

# ------------------------------------------------------------
# 🧭 Şema doğrulayıcılar (hourly/daily)
# ------------------------------------------------------------
def normalize_hourly_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    risk_3h_next7d_top3 veya crime_forecast_7days_all_geoids_FRstyle.csv için
    saatlik (3-saatlik blok) şema normalizasyonu.

    Desteklenen kolonlar:
      - date
      - geoid
      - risk_score / p_stack / prob / probability / score / risk
      - hour  veya  hour_range_3h / hour_range / hour_block

    Eğer hour yoksa, hour_range_3h içinden başlangıç saati (0,3,6,...) çıkarılır
    ve 'hour' kolonuna yazılır. 'timestamp' = date + hour (saat) olarak üretilir.
    """
    df = df.copy()
    cols = {c.lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n in df.columns:
                return n
            if n.lower() in cols:
                return cols[n.lower()]
        return None

    c_date   = pick("date")
    c_hour   = pick("hour", "hour_idx", "hour_of_day", "hour_index")
    c_hrange = pick("hour_range_3h", "hour_range", "hour_block")
    c_geoid  = pick("geoid", "GEOID", "cell_id", "id")
    c_risk   = pick("risk_score", "p_stack", "prob", "probability", "score", "risk")

    if not (c_date and c_geoid and c_risk and (c_hour or c_hrange)):
        raise ValueError(
            "Saatlik veri için 'date, geoid, risk_score' ve 'hour' veya "
            "'hour_range_3h' benzeri bir kolon zorunlu."
        )

    # Tarih
    df["date"] = pd.to_datetime(df[c_date], errors="coerce")

    # GEOID ve risk skoru
    df["geoid"] = df[c_geoid].astype(str)
    df["risk_score"] = pd.to_numeric(df[c_risk], errors="coerce")

    # Saat: varsa doğrudan 'hour', yoksa hour_range_3h içinden başlangıç saati
    if c_hour:
        df["hour"] = (
            pd.to_numeric(df[c_hour], errors="coerce")
            .astype("Int64")
            .clip(0, 23)
        )
    else:
        def parse_start_hour(val) -> float:
            if pd.isna(val):
                return np.nan
            s = str(val).strip()
            # farklı tire karakterlerini normalize et
            s = s.replace("–", "-").replace("—", "-")
            if "-" not in s:
                return np.nan
            a, _ = s.split("-", 1)
            try:
                h0 = int(a.strip())
                # 0–23 aralığına zorla
                h0 = max(0, min(23, h0))
                return h0
            except Exception:
                return np.nan

        df["hour"] = df[c_hrange].map(parse_start_hour).astype("Int64")

    # İsteğe bağlı: hour_range stringini de sakla (ileride lazım olursa)
    if c_hrange:
        df["hour_range_3h"] = df[c_hrange].astype(str)

    # Geçersiz satırları at
    df = df.dropna(subset=["date", "hour", "geoid"]).copy()

    # Zaman damgası: tarih + saat
    df["timestamp"] = df["date"].dt.floor("D") + pd.to_timedelta(
        df["hour"].fillna(0).astype(int),
        unit="h",
    )

    return df

def normalize_daily_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols = {c.lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n in df.columns:
                return n
            if n.lower() in cols:
                return cols[n.lower()]
        return None

    c_date  = pick("date")
    c_geoid = pick("geoid", "GEOID", "cell_id", "id")
    c_risk  = pick("risk_score", "p_stack", "prob", "probability", "score", "risk")

    if not (c_date and c_geoid and c_risk):
        raise ValueError("Günlük veri için 'date, geoid, risk_score' zorunlu.")

    df["date"] = pd.to_datetime(df[c_date], errors="coerce").dt.floor("D")
    df["geoid"] = df[c_geoid].astype(str)
    df["risk_score"] = pd.to_numeric(df[c_risk], errors="coerce")

    df = df.dropna(subset=["date", "geoid"]).copy()
    return df

def rgba_to_hex(rgba):
    """[r,g,b,a] → '#rrggbb'"""
    try:
        r, g, b, _ = rgba
        return "#{:02x}{:02x}{:02x}".format(int(r), int(g), int(b))
    except Exception:
        return "#dddddd"
