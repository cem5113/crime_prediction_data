from __future__ import annotations
from typing import Optional, Union, Dict, List, Tuple, Any

import streamlit as st
import pandas as pd
import requests
import re
import os, json, subprocess, sys
from pathlib import Path
import io, zipfile
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np  # ▶️ eklendi: aşağıda np.nanmean vb. kullanılıyor

# --- Forensic rapor yardımcı (varsa import et, yoksa stub kullan) ---
try:
    from scripts.forensic_report import build_forensic_report
except Exception:
    def build_forensic_report(**kwargs):
        return None

# -----------------------------------------------------------------------------
# Global Kurulum
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Suç Tahmin Modeli - Veri Güncelleme", layout="wide")

try:
    ROOT = Path(__file__).resolve().parent
except NameError:
    ROOT = Path.cwd()

DATA_DIR = ROOT / "crime_prediction_data"
SCRIPTS_DIR = ROOT / "scripts"
DATA_DIR.mkdir(parents=True, exist_ok=True)
SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
SEARCH_DIRS = [SCRIPTS_DIR, ROOT]

PIPELINE = [
    {"title": "Suç Olay Verisi (Temel) + Grid", "name": "update_crime",      "alts": ["build_crime_grid", "crime_grid_build"]},
    {"title": "911 Çağrıları (Polis İlişkili)", "name": "update_911",        "alts": ["enrich_911"]},
    {"title": "311 Şikayetleri (Kolluk İlişkili)", "name": "update_311",     "alts": ["enrich_311"]},
    {"title": "Nüfus & Demografi",             "name": "update_population", "alts": ["enrich_population"]},
    {"title": "Otobüs Durakları",              "name": "update_bus",        "alts": ["enrich_bus"]},
    {"title": "Tren Durakları (BART)",         "name": "update_train",      "alts": ["enrich_train"]},
    {"title": "POI (İlgi Noktaları)",          "name": "update_poi",        "alts": ["pipeline_make_sf_crime_06", "app_poi_to_06", "enrich_poi"]},
    {"title": "Polis & Kamu Binaları",         "name": "update_police_gov", "alts": ["enrich_police_gov_06_to_07", "enrich_police_gov", "enrich_police"]},
    {"title": "Hava Durumu",                   "name": "update_weather",    "alts": ["enrich_weather"]},
]

# -----------------------------------------------------------------------------
# Yardımcılar
# -----------------------------------------------------------------------------
def pick_url(key: str, default: str) -> str:
    # Öncelik: 1) st.secrets  2) ENV  3) default
    try:
        if key in st.secrets and st.secrets[key]:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.getenv(key, default)

def _mask_token(u: str) -> str:
    try:
        return re.sub(r'(\$\$app_token=)[^&]+', r'\1•••', str(u))
    except Exception:
        return str(u)

def _human_bytes(n: int) -> str:
    if n is None:
        return "-"
    step = 1024.0
    for u in ["B", "KB", "MB", "GB", "TB"]:
        if n < step:
            return f"{n:.0f} {u}" if u == "B" else f"{n:.1f} {u}"
        n /= step
    return f"{n:.1f} PB"

def _fmt_dt(ts: Optional[float]) -> str:
    if ts is None:
        return "-"
    dt = datetime.fromtimestamp(ts, tz=timezone.utc).astimezone()
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def _age_str(ts: Optional[float]) -> str:
    if ts is None:
        return "-"
    delta = datetime.now().timestamp() - ts
    if delta < 60:
        return f"{int(delta)} sn"
    if delta < 3600:
        return f"{int(delta // 60)} dk"
    if delta < 86400:
        return f"{int(delta // 3600)} sa"
    return f"{int(delta // 86400)} g"

def ensure_script(filename: str) -> Optional[Path]:
    """
    Verilen dosya adını SEARCH_DIRS içinde arar; bulursa Path döner, yoksa None.
    """
    cand = filename if filename.endswith(".py") else f"{filename}.py"
    for base in SEARCH_DIRS:
        p = Path(base) / cand
        if p.exists() and p.is_file():
            return p
    return None

def _candidate_names(base: str, locale: str) -> List[str]:
    if locale and locale != "default":
        return [
            f"{base}.fr.py",
            f"{base}_fr.py",
            f"{base}.{locale}.py",
            f"{base}-{locale}.py",
            f"{base}.py",
        ]
    return [f"{base}.py"]

def resolve_script(entry: dict, locale: str = "default") -> Optional[Path]:
    # 1) asıl ad için
    for cand in _candidate_names(entry["name"], locale):
        p = ensure_script(cand)
        if p:
            return p
    # 2) alternatif adlar için
    for alt in entry.get("alts", []):
        for cand in _candidate_names(alt, locale):
            pp = ensure_script(cand)
            if pp:
                return pp
    return None

def run_script(path: Path) -> bool:
    st.write(f"▶️ {path.name} çalıştırılıyor…")
    placeholder = st.empty()
    lines: List[str] = []
    try:
        proc = subprocess.Popen(
            [sys.executable, "-u", str(path)],
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        import time
        
        last_ui = time.time()
        
        while True:
            line = proc.stdout.readline()
            if not line and proc.poll() is not None:
                break
            if line:
                lines.append(line.rstrip())
                lines = lines[-400:]
        
                # UI'yı her satırda değil, 0.5 sn'de bir güncelle
                if time.time() - last_ui >= 0.5:
                    placeholder.code("\n".join(lines))
                    last_ui = time.time()
        
        # final update (bittiyse son hali bas)
        placeholder.code("\n".join(lines))

        rc = proc.wait()
        if rc == 0:
            st.success(f"✅ {path.name} tamamlandı")
            return True
        else:
            st.error(f"❌ {path.name} hata verdi (exit={rc})")
            return False
    except Exception as e:
        st.error(f"🚨 {path.name} çağrılamadı: {e}")
        return False
        
def _human_mb(n_bytes: int) -> float:
    if n_bytes is None:
        return 0.0
    return round(n_bytes / (1024**2), 3)

@st.cache_data(show_spinner=False)
def _file_cols_only(p_str: str) -> int | None:
    try:
        p = Path(p_str)
        return len(pd.read_csv(p, nrows=0).columns)
    except Exception:
        return None

def build_stage_summary(data_dir: Path) -> pd.DataFrame:
    STAGES = [
        (0, "Suç Olay Verisi (Ham)", "sf_crime.csv"),
        (1, "Zaman Özellikleri Eklenmiş", "sf_crime_01.csv"),
        (2, "911 Çağrıları (Polis İlişkili) Eklenmiş", "sf_crime_02.csv"),
        (3, "311 Şikayetleri (Kolluk İlişkili) Eklenmiş", "sf_crime_03.csv"),
        (4, "Nüfus & Demografi Eklenmiş", "sf_crime_04.csv"),
        (5, "Ulaşım (Otobüs + Tren) Özellikleri Eklenmiş", "sf_crime_05.csv"),
        (6, "POI (İlgi Noktaları) + Risk Skoru Eklenmiş", "sf_crime_06.csv"),
        (7, "Polis & Kamu Binaları Mesafe/Yakınlık Eklenmiş", "sf_crime_07.csv"),
        (8, "Hava Durumu Eklenmiş", "sf_crime_08.csv"),
        (9, "Komşuluk & Near-Repeat Özellikleri Eklenmiş Nihai Model Girdisi", "sf_crime_09.csv"),
    ]

    rows = []
    prev_cols = None
    prev_mb = None

    for stage_no, title, fname in STAGES:
        p = data_dir / fname
        exists = p.exists()

        if exists:
            n_cols = _file_cols_only(str(p))
            n_rows = "-"   # satır sayımı kaldırıldı (UI donmasın)
            size_mb = _human_mb(p.stat().st_size)
        else:
            n_rows, n_cols, size_mb = None, None, None

        d_cols = (n_cols - prev_cols) if (exists and prev_cols is not None and n_cols is not None) else None
        if exists and prev_mb is not None and size_mb is not None:
            d_mb = round(size_mb - prev_mb, 3)
            pct = round((d_mb / prev_mb * 100.0), 2) if prev_mb > 0 else None
        else:
            d_mb, pct = None, None

        rows.append({
            "Aşama": stage_no,
            "Tanım": title,
            "Dosya": fname,
            "Durum": "✅ Var" if exists else "⚠️ Yok",
            "Satır Sayısı": n_rows if n_rows is not None else "-",
            "Sütun Sayısı": n_cols if n_cols is not None else "-",
            "Δ Sütun": d_cols if d_cols is not None else "-",
            "Boyut (MB)": size_mb if size_mb is not None else "-",
            "Δ Boyut (MB)": d_mb if d_mb is not None else "-",
            "% Artış": f"%{pct}" if pct is not None else "-",
        })

        if exists and (n_cols is not None) and (size_mb is not None):
            prev_cols = n_cols
            prev_mb = size_mb

    return pd.DataFrame(rows)

# -----------------------------------------------------------------------------
# ENV ve URL’ler
# -----------------------------------------------------------------------------
RAW_911_URL = pick_url(
    "RAW_911_URL",
    "https://github.com/cem5113/crime_prediction_data/releases/download/v1.0.1/sf_911_last_5_year.csv",
)

CRIME_CSV_URL = "https://github.com/cem5113/crime_prediction_data/raw/main/sf_crime.csv"
CRIME_CSV_LATEST = CRIME_CSV_URL 
SF311_URL     = "https://github.com/cem5113/crime_prediction_data/raw/main/sf_311_last_5_years.csv"


# CSV-ONLY: Nüfus verisi yerel dosyadan okunacak
DEFAULT_POP_CSV = str((Path(os.environ.get("CRIME_DATA_DIR", "crime_prediction_data")) / "sf_population.csv").resolve())
POPULATION_PATH = pick_url("POPULATION_PATH", DEFAULT_POP_CSV)

# Güvenlik: URL verilirse reddet (CSV-only mod)
if re.match(r"^https?://", str(POPULATION_PATH), flags=re.I):
    POPULATION_PATH = DEFAULT_POP_CSV

os.environ["POPULATION_PATH"] = str(POPULATION_PATH)

# ⬇️ 911 artımlı çekim için API ayarları
SF911_API_URL       = pick_url("SF911_API_URL", "https://data.sfgov.org/resource/2zdj-bwza.json")
SF911_AGENCY_FILTER = pick_url("SF911_AGENCY_FILTER", "agency like '%Police%'")
SF911_API_TOKEN     = pick_url("SF911_API_TOKEN", "")

# Çocuk süreçlerin de aynı değerleri görmesi için ENV
os.environ["CRIME_CSV_URL"] = CRIME_CSV_LATEST
os.environ["RAW_911_URL"]   = RAW_911_URL
os.environ["SF311_URL"]     = SF311_URL
os.environ["GEOID_LEN"]     = os.environ.get("GEOID_LEN", "11")

GEOID_LEN = int(os.environ.get("GEOID_LEN", "11"))

def _norm_geoid(s: pd.Series, L: int = GEOID_LEN) -> pd.Series:
    return (
        s.astype(str)
         .str.extract(r"(\d+)", expand=False)
         .str[:L]
         .str.zfill(L)
    )

os.environ["SF911_API_URL"]       = SF911_API_URL
os.environ["SF911_AGENCY_FILTER"] = SF911_AGENCY_FILTER
if SF911_API_TOKEN:
    os.environ["SF911_API_TOKEN"] = SF911_API_TOKEN

SOCS_APP_TOKEN = st.secrets.get("SOCS_APP_TOKEN", os.environ.get("SOCS_APP_TOKEN", ""))
if SOCS_APP_TOKEN:
    os.environ["SOCS_APP_TOKEN"] = SOCS_APP_TOKEN

# LATEST veya 2022/2023 gibi belirli yıl
os.environ["ACS_YEAR"] = st.secrets.get("ACS_YEAR", os.environ.get("ACS_YEAR", "LATEST"))

# Virgülle filtre (boş bırak = tüm kategoriler)
os.environ["DEMOG_WHITELIST"] = st.secrets.get(
    "DEMOG_WHITELIST",
    os.environ.get("DEMOG_WHITELIST", "")
)

GITHUB_REPO = os.environ.get("GITHUB_REPO", "cem5113/crime_prediction_data")

# ENV'de yoksa varsayılanı ENV'e de yaz (dispatch tarafı için güvenli)
os.environ["GITHUB_WORKFLOW"] = os.environ.get("GITHUB_WORKFLOW", "full_pipeline.yml")
GITHUB_WORKFLOW = os.environ["GITHUB_WORKFLOW"]

ARTIFACT_NAMES = [
    "crime-pipeline-output",       # ✅ yeni
    "sf-crime-pipeline-output",    # (geri uyum)
]

def make_sf_crime_L(data_dir: Path, unique_per_geoid: bool = True) -> Path:
    """
    sf_crime_y.csv veya sf_crime.csv içinden sadece GEOID ve Y_label kolonlarını
    alıp crime_prediction_data/sf_crime_L.csv olarak yazar.
    unique_per_geoid=True ise aynı GEOID için Y_label'ların max'ını alır.
    """
    import pandas as pd, os

    cdir = Path(data_dir)
    candidates = [cdir / "sf_crime_y.csv", cdir / "sf_crime.csv"]
    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        raise FileNotFoundError("sf_crime_y.csv veya sf_crime.csv bulunamadı.")

    df = pd.read_csv(src, low_memory=False)

    low = {c.lower(): c for c in df.columns}
    geoid_col = low.get("geoid") or low.get("geography_id") or low.get("geoid11") or low.get("geoid_11")
    y_col     = low.get("y_label") or low.get("y") or low.get("label") or low.get("target")
    if not geoid_col or not y_col:
        raise ValueError(f"Gerekli kolonlar yok. Mevcut kolonlar: {list(df.columns)}")

    out = df[[geoid_col, y_col]].copy()
    out.columns = ["GEOID", "Y_label"]

    # GEOID normalize (11 haneli; senin ortam değişkenine uy)
    L = int(os.environ.get("GEOID_LEN", "11"))
    out["GEOID"] = (
        out["GEOID"].astype(str).str.extract(r"(\d+)", expand=False).str[:L].str.zfill(L)
    )

    if unique_per_geoid:
        out = out.groupby("GEOID", as_index=False)["Y_label"].max()

    dest = cdir / "sf_crime_L.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    return dest

def _resolve_workflow_id(target: str):
    import requests, os
    url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/workflows?per_page=100"
    r = requests.get(url, headers=_gh_headers(), timeout=30); r.raise_for_status()
    ws = r.json().get("workflows", [])
    # 1) Dosya adıyla eşleşme
    for w in ws:
        if os.path.basename(str(w.get("path",""))) == target:
            return w.get("id")
    # 2) Görünen adla eşleşme (yedek)
    for w in ws:
        if str(w.get("name","")).strip().lower() == target.strip().lower():
            return w.get("id")
    return None

def _get_last_run_by_workflow():
    url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/workflows/{GITHUB_WORKFLOW}/runs?per_page=1"
    r = requests.get(url, headers=_gh_headers(), timeout=30)
    if r.status_code != 200:
        return None, r.status_code, r.text
    arr = r.json().get("workflow_runs", [])
    return (arr[0] if arr else None), 200, ""

def _render_last_run_status(container):
    if not (st.secrets.get("GH_TOKEN") or os.environ.get("GH_TOKEN")):
        container.info("GH_TOKEN yok; GitHub durumunu okuyamıyorum.")
        return
    try:
        run, code, msg = _get_last_run_by_workflow()
        if not run:
            container.info("Bu workflow için run bulunamadı.")
            return
        status = run.get("status")
        concl  = run.get("conclusion") or "-"
        started = run.get("run_started_at")
        html_url = run.get("html_url")
        container.markdown(
            f"**Son koşum:** `{status}` / `{concl}` · başlama: `{started}`  ·  [GitHub’da aç]({html_url})"
        )
    except Exception as e:
        container.warning(f"Durum okunamadı: {e}")
# -----------------------------------------------------------------------------
# 08 → 09 Dönüşüm Yardımcıları
# -----------------------------------------------------------------------------
def _group_rare_labels(
    df: pd.DataFrame,
    col: str,
    min_prop: Optional[float] = None,
    min_count: Optional[int] = None,
    other_label: str = "Other",
    out_stats_path: Optional[Path] = None,
) -> pd.Series:
    if col not in df.columns:
        return pd.Series([None] * len(df), index=df.index)

    s = df[col].astype(str).str.strip()
    total = len(s)
    vc = s.value_counts(dropna=False)

    env_prop = os.environ.get("RARE_MIN_PROP")
    env_count = os.environ.get("RARE_MIN_COUNT")
    if min_prop is None and env_prop:
        try:
            min_prop = float(env_prop)
        except Exception:
            pass
    if min_count is None and env_count:
        try:
            min_count = int(env_count)
        except Exception:
            pass

    if min_prop is None and min_count is None:
        min_prop, min_count = 0.01, 200

    rare_mask = pd.Series(False, index=vc.index)
    if min_prop is not None:
        rare_mask |= (vc / max(total, 1)) < float(min_prop)
    if min_count is not None:
        rare_mask |= vc < int(min_count)

    rare_values = set(vc[rare_mask].index)
    grouped = s.where(~s.isin(rare_values), other_label)

    if out_stats_path is not None:
        out_stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_df = pd.DataFrame({
            col: vc.index,
            "count": vc.values,
            "prop": vc.values / max(total, 1),
            "is_rare": vc.index.map(lambda v: v in rare_values)
        })
        try:
            stats_df.to_csv(out_stats_path, index=False)
        except Exception:
            pass

    return grouped

def clean_and_save_crime_09(input_obj: Union[str, pd.DataFrame] = "sf_crime_08.csv", output_path: str = "sf_crime_09.csv"):
    # input_obj hem DataFrame hem de dosya yolu olabilir
    if isinstance(input_obj, pd.DataFrame):
        df = input_obj.copy()
    else:
        df = pd.read_csv(input_obj, dtype={"GEOID": str})

    # 🛠 FIX-1: GEOID normalizasyonu (zfill kaldırıldı, sadece sayısal çekirdek)
    if "GEOID" in df.columns:
        df["GEOID"] = (
            df["GEOID"]
            .astype(str)
            .str.extract(r"(\d+)", expand=False)
        )

    if "category" in df.columns:
        df["category"] = df["category"].astype(str).str.strip().str.title()

    # Rare class grouping
    try:
        out_dir = Path(output_path).parent if isinstance(output_path, str) else Path(".")
        if "category" in df.columns:
            df["category_grouped"] = _group_rare_labels(
                df, "category", min_prop=None, min_count=None,
                other_label="Other", out_stats_path=out_dir / "rare_stats_category.csv"
            )

        if "subcategory" in df.columns:
            df["subcategory"] = df["subcategory"].astype(str).str.strip().str.title()
            df["subcategory_grouped"] = _group_rare_labels(
                df, "subcategory", min_prop=None, min_count=None,
                other_label="Other", out_stats_path=out_dir / "rare_stats_subcategory.csv"
            )

        try:
            st.caption("🔎 Rare grouping uygulandı (category/subcategory). İstatistikler CSV olarak kaydedildi.")
        except Exception:
            pass
    except Exception as _e:
        try:
            st.warning(f"Rare grouping atlandı: {str(_e)}")
        except Exception:
            print(f"Rare grouping atlandı: {_e}")

    # Tip dönüştürücüler
    def to_int(df_, col, default=0):
        if col in df_.columns:
            df_[col] = (
                pd.to_numeric(df_[col], errors="coerce")
                .fillna(default)
                .round()
                .astype("Int64")
            )

    def to_float(df_, col, default=0.0):
        if col in df_.columns:
            df_[col] = (
                pd.to_numeric(df_[col], errors="coerce")
                .fillna(default)
                .astype(float)
            )

    # Sayaç kolonları
    int_count_cols = [
        "crime_count",
        "911_request_count_hour_range",
        "911_request_count_daily(before_24_hours)",
        "311_request_count",
        "bus_stop_count",
        "train_stop_count",
        "poi_total_count",
    ]
    for c in int_count_cols:
        to_int(df, c, default=0)

    # Risk skoru
    to_float(df, "poi_risk_score", default=0.0)

    # Binary kolonlar
    def to_binary(df_, col):
        if col in df_.columns:
            m = {
                "true": 1, "t": 1, "yes": 1, "y": 1, "1": 1, "evet": 1,
                "false": 0, "f": 0, "no": 0, "n": 0, "0": 0, "hayır": 0, "hayir": 0
            }
            s = df_[col].replace({True: 1, False: 0})
            s = s.astype(str).str.strip().str.lower().map(m)
            df_[col] = pd.to_numeric(s, errors="coerce").fillna(0).astype("Int64")

    for c in ["is_near_police", "is_near_government"]:
        to_binary(df, c)

    # Mesafe kolonları
    for c in ["distance_to_bus", "distance_to_train", "distance_to_police", "distance_to_government_building"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(9999.0).astype(float)

    # Range kolonları (int kategoriler)
    for c in ["bus_stop_count_range", "train_stop_count_range", "poi_total_count_range", "poi_risk_score_range"]:
        to_int(df, c, default=0)

    for c in ["distance_to_bus_range", "distance_to_train_range", "distance_to_police_range", "distance_to_government_building_range"]:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            max_cat = int(s.max(skipna=True)) if pd.notna(s.max(skipna=True)) else 3
            df[c] = s.fillna(max_cat).round().astype("Int64")

    # Nüfus (median ile doldur)
    if "population" in df.columns:
        df["population"] = pd.to_numeric(df["population"], errors="coerce")
        median_pop = df["population"].median(skipna=True)
        median_pop = 0 if pd.isna(median_pop) else median_pop
        df["population"] = df["population"].fillna(median_pop)

    # POI dominant type
    if "poi_dominant_type" in df.columns:
        # 🛠 FIX-2: Boş stringleri de None say, sonra "None" ile doldur
        df["poi_dominant_type"] = (
            df["poi_dominant_type"]
            .replace({"": np.nan})
            .fillna("None")
            .astype(str)
        )

    # Tarih normalize
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    elif "datetime" in df.columns and "date" not in df.columns:
        df["date"] = pd.to_datetime(df["datetime"], errors="coerce").dt.date

    # 🛠 FIX-3: Near-repeat güvenli hale getirildi (crime_count yoksa Y_label'dan türet)
    try:
        need = {"date", "GEOID"}
        has_crime_count = "crime_count" in df.columns

        if not has_crime_count and "Y_label" in df.columns:
            df["crime_count"] = (
                pd.to_numeric(df["Y_label"], errors="coerce")
                .fillna(0)
                .astype(int)
            )
            has_crime_count = True

        if need.issubset(df.columns) and has_crime_count:
            # kategori kolonu öncelik sırası
            cat_col = None
            for cc in ["category_grouped", "subcategory_grouped", "category"]:
                if cc in df.columns:
                    cat_col = cc
                    break

            if cat_col:
                tmp = df[["date", "GEOID", cat_col, "crime_count"]].copy()
                tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce").dt.date
                g = (tmp.groupby(["GEOID", cat_col, "date"], as_index=False)["crime_count"].sum())
                g["date"] = pd.to_datetime(g["date"])
                g = g.sort_values(["GEOID", cat_col, "date"])

                def _roll_counts(x):
                    x = x.set_index("date").asfreq("D", fill_value=0)
                    x["nr_7d"]  = x["crime_count"].rolling("7D").sum().shift(1)
                    x["nr_14d"] = x["crime_count"].rolling("14D").sum().shift(1)
                    return x.reset_index()

                g2 = (g.groupby(["GEOID", cat_col]).apply(_roll_counts).reset_index(level=[0, 1]).reset_index(drop=True))
                g2["date"] = g2["date"].dt.date

                df = df.merge(
                    g2[["GEOID", cat_col, "date", "nr_7d", "nr_14d"]],
                    on=["GEOID", cat_col, "date"], how="left"
                )
                for c in ["nr_7d", "nr_14d"]:
                    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(float)
    except Exception as _e:
        print(f"near-repeat uyarı: {_e}")

    # Dışsal değişken örnek skoru
    try:
        ext_map_path = Path(DATA_DIR) / "crime_type_externals_map.json"
        if ext_map_path.exists():
            with open(ext_map_path, "r", encoding="utf-8") as f:
                type_map = json.load(f)
            key_col = "category_grouped" if "category_grouped" in df.columns else (
                      "category" if "category" in df.columns else None)
            if key_col:
                def _ext_score(row):
                    cols = type_map.get(str(row[key_col]), [])
                    vals = []
                    for c in cols:
                        if c in df.columns:
                            try:
                                vals.append(float(row.get(c, 0)))
                            except Exception:
                                pass
                    return float(np.nanmean(vals)) if len(vals) > 0 else np.nan
                df["externals_type_score"] = df.apply(_ext_score, axis=1)
                df["externals_type_score"] = df["externals_type_score"].fillna(0.0).astype(float)
    except Exception as _e:
        print(f"dışsal değişken uyarı: {_e}")

    preview_cols = [c for c in ["nr_7d", "nr_14d", "nei_7d_sum", "externals_type_score"] if c in df.columns]
    if preview_cols:
        try:
            st.caption("🧩 Yeni mekânsal-zamansal özellikler (ilk 20 satır):")
            st.dataframe(df[preview_cols].head(20))
        except Exception:
            pass

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ {output_path} kaydedildi. Satır sayısı: {len(df)}")
    return df

def load_sf_crime_08(local_path: Path) -> Optional[pd.DataFrame]:
    """Önce yerel dosyayı dene; yoksa artifact’tan çek. Ardından crime_mix varsa grid ile merge et."""
    def _normalize_date_cols(df_: pd.DataFrame) -> pd.DataFrame:
        if "date" in df_.columns:
            df_["date"] = pd.to_datetime(df_["date"], errors="coerce").dt.date
        elif "datetime" in df_.columns and "date" not in df_.columns:
            df_["date"] = pd.to_datetime(df_["datetime"], errors="coerce").dt.date
        return df_

    df: Optional[pd.DataFrame] = None
    try:
        if local_path.exists():
            df = pd.read_csv(local_path, low_memory=False)
            df = _normalize_date_cols(df)
    except Exception as e:
        st.warning(f"Yerel sf_crime_08.csv okunamadı: {e}")

    if df is None:
        df = fetch_latest_artifact_df()
        if df is None:
            return None
        df = _normalize_date_cols(df)

    # crime_mix merge (grid) — opsiyonel
    try:
        _out_dir  = local_path.parent
        _grid_path = _out_dir / "sf_crime_grid_full_labeled.csv"
        if _grid_path.exists():
            grid = pd.read_csv(_grid_path, dtype={"GEOID": str}, low_memory=False)

            keys = ["GEOID", "season", "day_of_week", "event_hour"]
            if set(keys).issubset(grid.columns) and "crime_mix" in grid.columns and set(keys).issubset(df.columns):

                df["GEOID"]   = _norm_geoid(df["GEOID"])
                grid["GEOID"] = _norm_geoid(grid["GEOID"])

                for c in ["day_of_week", "event_hour"]:
                    df[c]   = pd.to_numeric(df[c], errors="coerce").astype("Int64")
                    grid[c] = pd.to_numeric(grid[c], errors="coerce").astype("Int64")

                merged = df.merge(
                    grid[keys + ["crime_mix"]],
                    on=keys, how="left", suffixes=("", "_grid"), validate="many_to_one"
                )

                if "crime_mix_grid" in merged.columns:
                    if "crime_mix" not in merged.columns:
                        merged["crime_mix"] = ""
                    merged["crime_mix"] = merged["crime_mix"].astype(str)
                    merged["crime_mix"] = merged["crime_mix"].where(
                        merged["crime_mix"].str.len() > 0,
                        merged["crime_mix_grid"].fillna("")
                    )
                    merged = merged.drop(columns=["crime_mix_grid"], errors="ignore")
                df = merged
        else:
            print(f"crime_mix merge atlandı: grid bulunamadı → {_grid_path}")
    except Exception as _e:
        print(f"crime_mix merge uyarısı: {_e}")

    return df

def load_city_crime_08(prefix: str, data_dir: Path) -> Optional[pd.DataFrame]:
    """{prefix}_crime_08.csv'yi yükler ve date kolonunu normalize eder."""
    path = data_dir / f"{prefix}_crime_08.csv"
    if prefix.lower() == "sf":
        return load_sf_crime_08(path)
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, low_memory=False)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
        elif "datetime" in df.columns:
            df["date"] = pd.to_datetime(df["datetime"], errors="coerce").dt.date
        return df
    except Exception as e:
        st.warning(f"{path.name} okunamadı: {e}")
        return None

def process_city_to_09(prefix: str, data_dir: Path) -> Optional[pd.DataFrame]:
    """{prefix}_crime_08 → temizle (_08_clean) → neighbors → {prefix}_crime_09 üretir."""
    df08 = load_city_crime_08(prefix, data_dir)
    if df08 is None:
        st.info(f"{prefix}_crime_08.csv bulunamadı.")
        return None

    out_clean = data_dir / f"{prefix}_crime_08_clean.csv"
    clean_and_save_crime_09(df08, str(out_clean))
    st.success(f"✅ {out_clean.name} kaydedildi.")

    graph_script = resolve_script({"name": "update_neighbors_graph", "alts": ["update_neighbors_graph.py", "neighbors_graph", "neighbors_graph.py"]})
    feat_script  = resolve_script({"name": "update_neighbors",       "alts": ["update_neighbors.py"]})

    # prefix'e özel neighbors varsa onu kullan; yoksa genel
    neighbor_file_pref = data_dir / f"{prefix}_neighbors.csv"
    neighbor_file_gen  = data_dir / "neighbors.csv"
    neighbor_file_use  = neighbor_file_pref if neighbor_file_pref.exists() else neighbor_file_gen

    if not neighbor_file_use.exists() and graph_script:
        ok_graph = run_script(graph_script)
        st.success("🗺️ neighbors.csv üretildi.") if ok_graph else st.warning("neighbors graph başarısız.")
        if ok_graph:
            neighbor_file_use = neighbor_file_gen  # grafikten sonra genel üretildi varsayalım

    if feat_script:
        os.environ["NEIGHBOR_FILE"]        = os.environ.get("NEIGHBOR_FILE", str(neighbor_file_use))
        os.environ["NEIGHBOR_INPUT_CSV"]   = str(out_clean)
        os.environ["NEIGHBOR_OUTPUT_CSV"]  = str(data_dir / f"{prefix}_crime_09.csv")
        os.environ["NEIGHBOR_WINDOW_DAYS"] = os.environ.get("NEIGHBOR_WINDOW_DAYS", "7")
        os.environ["NEIGHBOR_LAG_DAYS"]    = os.environ.get("NEIGHBOR_LAG_DAYS", "1")

        ok_feat = run_script(feat_script)
        if ok_feat:
            st.success(f"🧩 {prefix}_crime_09.csv üretildi (nei_7d_sum eklendi).")
            try:
                return pd.read_csv(data_dir / f"{prefix}_crime_09.csv", low_memory=False)
            except Exception:
                return None
        else:
            st.warning("update_neighbors.py çalıştırılamadı; logu kontrol edin.")
    else:
        st.info("update_neighbors.py bulunamadı (scripts klasörüne ekleyin).")

    return None

# -----------------------------------------------------------------------------
# Dosya Listeleme / Dönüştürme Yardımcıları
# -----------------------------------------------------------------------------
def list_files_sorted(
    include: Optional[List[Union[str, Path]]] = None,
    base_dir: Optional[Path] = None,
    pattern: str = "*.csv",
    ascending: bool = True,
    include_missing: bool = True,
) -> pd.DataFrame:
    bdir = base_dir or DATA_DIR
    rows: List[Dict[str, Any]] = []

    if include is None:
        include = []
        for prefix in ["sf"]:  # fr kaldırıldı
            include += [str(bdir / f"{prefix}_crime_{i:02d}.csv") for i in range(1, 10)]
            include += [str(bdir / f"{prefix}_crime_y.csv")]
        include += [str(bdir / "sf_crime_grid_full_labeled.csv")]
        for p in bdir.glob(pattern):
            include.append(str(p))

    seen = set()
    for x in include:
        p = Path(x)
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen:
            continue
        seen.add(key)

        exists = p.exists()
        if p.name.startswith("fr_"):
            continue
        try:
            st_ = p.stat() if exists else None
            mtime = st_.st_mtime if st_ else None
            size  = st_.st_size  if st_ else None
        except Exception:
            mtime, size = None, None

        if exists or include_missing:
            rows.append({
                "file": p.name,
                "path": str(p),
                "exists": bool(exists),
                "size": _human_bytes(size),
                "modified": _fmt_dt(mtime),
                "age": _age_str(mtime),
                "_mtime": mtime,
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("_mtime", ascending=ascending, na_position="last").drop(columns=["_mtime"])
        df = df.reset_index(drop=True)
        df.insert(0, "Sıra", range(1, len(df) + 1))
    return df

def convert_csv_dir_to_parquet(
    input_dir: Path,
    output_dir: Path,
    pattern: str = "*.csv",
    compression: str = "zstd",
    stats: bool = True
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    try:
        import polars as pl
        use_polars = True
    except Exception:
        use_polars = False

    if not use_polars:
        try:
            import pyarrow  # noqa: F401
        except Exception:
            raise RuntimeError("Ne polars ne de pyarrow mevcut. Lütfen 'pip install polars pyarrow' kurun.")

    for p in sorted(Path(input_dir).glob(pattern)):
        if not p.is_file():
            continue
        out = Path(output_dir) / (p.stem + ".parquet")
        try:
            if use_polars:
                df_pl = pl.read_csv(str(p))
                df_pl.write_parquet(str(out), compression=compression)
                n_rows = df_pl.height
            else:
                df_pd = pd.read_csv(p, low_memory=False)
                df_pd.to_parquet(out, compression=compression, index=False, engine="pyarrow")
                n_rows = len(df_pd)

            src_sz = p.stat().st_size if p.exists() else None
            dst_sz = out.stat().st_size if out.exists() else None
            rows.append({
                "src": str(p.name),
                "dst": str(out.name),
                "rows": n_rows,
                "src_size": src_sz,
                "dst_size": dst_sz,
            })
        except Exception as e:
            rows.append({
                "src": str(p.name),
                "dst": str(out.name),
                "rows": None,
                "src_size": None,
                "dst_size": None,
                "error": str(e),
            })

    res = pd.DataFrame(rows)
    if stats and not res.empty:
        try:
            res["src_size_mb"] = (res["src_size"].astype("float") / (1024**2)).round(3)
            res["dst_size_mb"] = (res["dst_size"].astype("float") / (1024**2)).round(3)
            res["ratio"] = (res["dst_size"].astype("float") / res["src_size"].astype("float")).round(3)
        except Exception:
            pass
    return res

# -----------------------------------------------------------------------------
# İndirilebilir kaynaklar
# -----------------------------------------------------------------------------
DOWNLOADS = {
    "Suç Taban CSV (Release latest)": {
        "url": "https://raw.githubusercontent.com/cem5113/crime_prediction_data/main/sf_crime.csv",
        "path": str(DATA_DIR / "sf_crime.csv"),
    },
    "Tahmin Grid Verisi (GEOID × Zaman + Y_label)": {
        "url": "https://raw.githubusercontent.com/cem5113/crime_prediction_data/main/sf_crime_grid_full_labeled.csv",
        "path": str(DATA_DIR / "sf_crime_grid_full_labeled.csv"),
        "allow_artifact": True,
        "artifact_picks": ["sf_crime_grid_full_labeled.csv"],
    },
    "911 Çağrıları (özet)": {
        "url": RAW_911_URL,
        "path": str(DATA_DIR / "sf_911_last_5_year.csv"),
    },
    "311 Çağrıları (özet)": {
        "url": "https://raw.githubusercontent.com/cem5113/crime_prediction_data/main/sf_311_last_5_years.csv",
        "path": str(DATA_DIR / "sf_311_last_5_years.csv"),
    },
    "Otobüs Durakları": {
        "url": "https://raw.githubusercontent.com/cem5113/crime_prediction_data/main/sf_bus_stops_with_geoid.csv",
        "path": str(DATA_DIR / "sf_bus_stops_with_geoid.csv"),
    },
    "Tren Durakları": {
        "url": "https://raw.githubusercontent.com/cem5113/crime_prediction_data/main/sf_train_stops_with_geoid.csv",
        "path": str(DATA_DIR / "sf_train_stops_with_geoid.csv"),
    },
    "POI GeoJSON": {
        "url": "https://raw.githubusercontent.com/cem5113/crime_prediction_data/main/sf_pois.geojson",
        "path": str(DATA_DIR / "sf_pois.geojson"),
        "is_json": True,
    },
    "Nüfus Verisi": {
        "url": "",
        "path": str(DATA_DIR / "sf_population.csv"),
        "local_src": str(POPULATION_PATH),
        "is_local_csv": True,
    },
    "POI Risk Skorları": {
        "url": "https://raw.githubusercontent.com/cem5113/crime_prediction_data/main/risky_pois_dynamic.json",
        "path": str(DATA_DIR / "risky_pois_dynamic.json"),
        "is_json": True,
    },
    "Polis İstasyonları": {
        "url": "https://raw.githubusercontent.com/cem5113/crime_prediction_data/main/sf_police_stations.csv",
        "path": str(DATA_DIR / "sf_police_stations.csv"),
    },
    "Devlet Binaları": {
        "url": "https://raw.githubusercontent.com/cem5113/crime_prediction_data/main/sf_government_buildings.csv",
        "path": str(DATA_DIR / "sf_government_buildings.csv"),
    },
    "Hava Durumu": {
        "url": "https://raw.githubusercontent.com/cem5113/crime_prediction_data/main/sf_weather_5years.csv",
        "path": str(DATA_DIR / "sf_weather_5years.csv"),
    },
}

def download_and_preview(name, url, file_path, is_json=False, allow_artifact_fallback=False, artifact_picks=None):
    st.markdown(f"### 🔹 {name}")
    st.caption(f"URL: {_mask_token(url)}")
    ok = False

    # Yerel kopya (Nüfus) — CSV-only mod
    meta = DOWNLOADS.get(name, {})
    if meta.get("is_local_csv"):
        try:
            src = Path(meta["local_src"])
            dst = Path(file_path)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.exists():
                dst.write_bytes(src.read_bytes())
                ok = True
                st.info("Yerel CSV kopyalandı.")
            else:
                st.warning(f"Yerel dosya bulunamadı: {src}")
        except Exception as e:
            st.warning(f"Yerel kopya hatası: {e}")

    if not ok and url:
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            if is_json:
                Path(file_path).write_text(r.text, encoding="utf-8")
            else:
                with open(file_path, "wb") as f:
                    f.write(r.content)
            ok = True
        except Exception as e:
            st.warning(f"Raw indirme başarısız: {e}")

    if not ok and allow_artifact_fallback:
        try:
            blob = fetch_file_from_latest_artifact(artifact_picks or [os.path.basename(file_path)])
            if blob:
                Path(file_path).parent.mkdir(parents=True, exist_ok=True)
                with open(file_path, "wb") as f:
                    f.write(blob)
                ok = True
                st.info("Dosya artifact'tan alındı.")
        except Exception as e:
            st.warning(f"Artifact fallback başarısız: {e}")

    if not ok:
        st.error(f"❌ {name} indirilemedi.")
        return

    # Önizleme
    try:
        if is_json:
            try:
                data = json.loads(Path(file_path).read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    st.json(data)
                elif isinstance(data, list):
                    st.json(data[:3])
                else:
                    st.code(str(data)[:1000])
            except Exception:
                st.code(Path(file_path).read_text(encoding="utf-8")[:2000])
        else:
            head = pd.read_csv(file_path, nrows=3)
            cols = head.columns.tolist()
            st.dataframe(head)
            st.caption(f"📌 Sütunlar: {cols}")
        st.success("✅ İndirildi.")
    except Exception as e:
        st.info("Önizleme başarısız; dosya indirildi.")
        st.code(f"Önizleme hatası: {e}")

# -----------------------------------------------------------------------------
# UI — Başlık ve Sidebar
# -----------------------------------------------------------------------------
st.title("📦 Veri Güncelleme Paneli")
st.caption("Bu ekran yalnızca veri katmanlarını günceller ve birleştirir.")

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### Ayarlar")
    with st.sidebar.expander("Demografi (ACS) Ayarları"):
        acs_year_default = os.environ.get("ACS_YEAR", "LATEST")
        whitelist_default = os.environ.get("DEMOG_WHITELIST", "")
        level_default = os.environ.get("CENSUS_GEO_LEVEL", "auto")
    
        acs_year_in = st.text_input(
            label="ACS_YEAR (LATEST veya YYYY)",
            value=str(acs_year_default or "LATEST"),
            key="acs_year_in_sb",
            help="5-year ACS için en son yılı kullanmak genelde uygundur."
        )
    
        whitelist_in = st.text_input(
            label="DEMOG_WHITELIST (virgüllü; boş = hepsi)",
            value=str(whitelist_default or ""),
            key="demog_whitelist_in_sb",
            help='Örn: "population,median_income,education". Metin eşleşmesiyle filtreler.'
        )
    
        levels = ["auto", "tract", "blockgroup", "block"]
        try:
            idx = levels.index(level_default) if level_default in levels else 0
        except Exception:
            idx = 0
        level_in_main = st.selectbox(
            "CENSUS_GEO_LEVEL",
            levels,
            index=idx,
            key="census_geo_level_in_main",
            help="Nüfus GEOID eşleşme seviyesi. `auto` çoğu durumda yeterlidir."
        )
        os.environ["CENSUS_GEO_LEVEL"] = level_in_main

        pop_default = os.environ.get("POPULATION_PATH", str(POPULATION_PATH))
        pop_url_in_main = st.text_input(
            label="POPULATION_PATH (YEREL CSV YOLU)",
            value=str(pop_default or ""),
            key="population_path_in_main",
            help="Örn: crime_prediction_data/sf_population.csv (URL kabul edilmez)."
        )

        _v = str(acs_year_in).strip()
        if _v.upper() == "LATEST":
            os.environ["ACS_YEAR"] = "LATEST"
        else:
            _digits = re.sub(r"\D", "", _v)
            os.environ["ACS_YEAR"] = _digits if len(_digits) == 4 else "LATEST"

        os.environ["DEMOG_WHITELIST"] = str(whitelist_in or "")

        if re.match(r"^https?://", str(pop_url_in_main), flags=re.I):
            st.error("CSV-only mod: URL kabul edilmez. Yerel bir CSV yolu girin.")
        else:
            os.environ["POPULATION_PATH"] = pop_url_in_main or str(POPULATION_PATH)

# -----------------------------------------------------------------------------
# 1) (Opsiyonel) Verileri indir ve önizle
# -----------------------------------------------------------------------------
st.markdown("### Veri Katmanları (Durum Özeti)")

LAYER_FILES = [
    ("Suç Olay Verisi (Temel)", "sf_crime.csv", "Dinamik (Günlük)"),
    ("911 Çağrıları (Polis İlişkili)", "sf_911_last_5_year.csv", "Dinamik (Günlük)"),
    ("311 Şikayetleri (Kolluk İlişkili)", "sf_311_last_5_years.csv", "Dinamik (Günlük)"),
    ("Nüfus & Demografi", "sf_population.csv", "Statik / Yıllık"),
    ("Otobüs Durakları", "sf_bus_stops_with_geoid.csv", "Yarı-dinamik (Aylık)"),
    ("Tren Durakları (BART)", "sf_train_stops_with_geoid.csv", "Yarı-dinamik (Aylık)"),
    ("POI (İlgi Noktaları)", "sf_pois_cleaned_with_geoid.csv", "Yarı-dinamik"),
    ("Polis & Kamu Binaları", "sf_police_stations.csv", "Yarı-dinamik"),
    ("Hava Durumu", "sf_weather_5years.csv", "Dinamik (Günlük)"),
    ("Komşuluk (Neighbors)", "neighbors.csv", "Yapısal"),
    ("Son Birleşik Çıktı", "sf_crime_09.csv", "Çıktı (En güncel)"),
]

rows = []
for title, fname, typ in LAYER_FILES:
    p = DATA_DIR / fname
    rows.append({
        "Konu": title,
        "Dosya": fname,
        "Tür": typ,
        "Durum": "✅ Var" if p.exists() else "⚠️ Yok",
        "Güncellenme": _fmt_dt(p.stat().st_mtime) if p.exists() else "-",
        "Yaş": _age_str(p.stat().st_mtime) if p.exists() else "-",
    })

st.dataframe(pd.DataFrame(rows), use_container_width=True)
st.info("📌 Not: En son güncellenen/birleşen çıktı genellikle **sf_crime_09.csv** dosyasıdır. Analizde kullanılabilecek güncel veri kaynağıdır.")

with st.expander("📊 Tez/Savunma: Veri İşleme Aşamaları Özeti", expanded=False):
    if st.checkbox("Aşama özetini hesapla (yavaş olabilir)", value=False):
        df_stage = build_stage_summary(DATA_DIR)
        st.dataframe(df_stage, use_container_width=True)
    else:
        st.info("Kapalı: UI donmaması için hesap yapılmadı.")

    # küçük “etkili” özet
    last_ok = df_stage[(df_stage["Aşama"] == 9) & (df_stage["Durum"] == "✅ Var")]
    if not last_ok.empty:
        st.success("✅ En güncel nihai model girdisi hazır: **Aşama 9**")
    else:
        st.warning("⚠️ Nihai model girdisi (Aşama 9) henüz yok. Komşuluk/Near-Repeat adımı çalıştırılmalı.")

    try:
        tmp = df_stage[df_stage["Δ Sütun"] != "-"].copy()
        if not tmp.empty:
            tmp["Δ Sütun"] = pd.to_numeric(tmp["Δ Sütun"], errors="coerce")
            top = tmp.sort_values("Δ Sütun", ascending=False).head(1)
            if not top.empty:
                st.info(
                    f"📌 En yüksek özellik artışı: **Aşama {int(top['Aşama'].iloc[0])}** "
                    f"({top['Tanım'].iloc[0]}) → Δ Sütun: **{int(top['Δ Sütun'].iloc[0])}**"
                )
    except Exception:
        pass
        
# -----------------------------------------------------------------------------
# 1.5) Dosyaları tarihe göre sırala
# -----------------------------------------------------------------------------
st.markdown("### Dosyaları tarihe göre sırala")
colA, colB, colC = st.columns([1, 1, 2])
with colA:
    order = st.radio("Sıralama", ["Eski → Yeni", "Yeni → Eski"], horizontal=True, index=0)
with colB:
    show_missing = st.checkbox("Eksikleri de göster", value=True)
with colC:
    patt = st.text_input(
        "Desen (glob)", "*.csv",
        help="Örn: sf_crime_*.csv",
        key="glob_list_files"
    )

asc = (order == "Eski → Yeni")
if st.button("📂 Listeyi Oluştur", key="make_file_list"):
    df_files = list_files_sorted(pattern=patt, ascending=asc, include_missing=show_missing)
    st.dataframe(df_files, use_container_width=True, hide_index=True)
else:
    st.info("Liste hazır değil. 'Listeyi Oluştur'a bas.")

# -----------------------------------------------------------------------------
# 1.6) CSV → Parquet dönüştür
# -----------------------------------------------------------------------------
st.markdown("### CSV → Parquet dönüştür")
with st.expander("🔄 CSV’leri Parquet’e çevir (zstd)"):
    in_dir = st.text_input(
        "Girdi klasörü", value=str(DATA_DIR),
        help="Örn: crime_prediction_data/",
        key="csv2parquet_in_dir"
    )
    out_dir = st.text_input(
        "Çıktı klasörü", value=str(ROOT / "parquet_out"),
        help="Örn: parquet_out/",
        key="csv2parquet_out_dir"
    )
    patt_in = st.text_input(
        "Desen (glob)", "*.csv",
        help="Örn: sf_crime_*.csv",
        key="csv2parquet_glob"
    )
    comp = st.selectbox(
        "Sıkıştırma",
        ["zstd", "snappy", "gzip", "brotli", "uncompressed"],
        index=0,
        key="csv2parquet_codec"
    )
    want_stats = st.checkbox(
        "Özet/stats üret", value=True,
        key="csv2parquet_stats"
    )

    if st.button("🧰 Dönüştür (CSV → Parquet)", key="csv2parquet_run"):
        try:
            res = convert_csv_dir_to_parquet(
                input_dir=Path(in_dir),
                output_dir=Path(out_dir),
                pattern=patt_in,
                compression=comp,
                stats=want_stats
            )
            if res.empty:
                st.info("Eşleşen CSV bulunamadı.")
            else:
                st.success("Dönüşüm tamamlandı.")
                st.dataframe(res)
        except Exception as e:
            st.error(f"Dönüşüm hatası: {e}")


# -----------------------------------------------------------------------------
# 2) Veri Güncelleme (Tek Akış)
# -----------------------------------------------------------------------------
st.markdown("### Veri Güncelle ve Birleştir (Tek Akış)")

if st.button("⚙️ Güncelleme İşlemini Başlat"):
    with st.spinner("⏳ Veri katmanları güncelleniyor ve birleştiriliyor..."):
        all_ok = True

        # 1) Katman scriptleri
        for entry in PIPELINE:
            st.markdown(f"#### 🔹 {entry['title']}")
            sp = resolve_script(entry, locale="default")
            if not sp:
                st.warning("⏭️ Script bulunamadı, adım atlandı.")
                all_ok = False
                continue
            ok = run_script(sp)
            all_ok = all_ok and ok

        # 2) Komşuluk + nihai çıktı (opsiyonel, varsa)
        st.markdown("#### 🔹 Komşuluk Özellikleri + Nihai Çıktı")
        try:
            _ = process_city_to_09("sf", DATA_DIR)
        except Exception as e:
            st.warning(f"Komşuluk/Nihai çıktı adımı atlandı: {e}")
            all_ok = False

    if all_ok:
        st.success("🎉 Güncelleme tamamlandı. Güncel birleşik veri hazır.")
    else:
        st.warning("ℹ️ Güncelleme tamamlandı; bazı adımlar atlandı veya hata verdi.")

# -----------------------------------------------------------------------------
# 3) Güncel Birleşik Veri (Model Girdisi) — Önizleme
# -----------------------------------------------------------------------------
st.markdown("### Güncel Birleşik Veri (Model Girdisi)")

candidates = [
    DATA_DIR / "sf_crime_09.csv",
    DATA_DIR / "sf_crime_08.csv",
    DATA_DIR / "sf_crime.csv",
]

final_path = next((p for p in candidates if p.exists()), None)

if final_path is None:
    st.info("Henüz birleşik çıktı yok. Önce güncellemeyi çalıştırın.")
else:
    st.caption(f"📌 Seçilen dosya: **{final_path.name}**")
    try:
        st.dataframe(pd.read_csv(final_path, nrows=20, low_memory=False), use_container_width=True)
    except Exception as e:
        st.info(f"Önizleme okunamadı: {e}")
