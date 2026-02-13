# app.py
# SUTAM – Veri Hazırlama Süreci (Pipeline Summary)
# - Amaç: Sadece “Aşama–Dosya–Satır–Sütun–NaN hücre(%)–Tam boş satır–Not” tablosunu üretip göstermek
# - Kaynak: (1) Local (repo içinde dosyalar)  (2) GitHub raw (commit)  (3) GitHub release asset  (4) GitHub artifact (token ile)
#
# ÇALIŞTIRMA NOTU (Streamlit Cloud):
# - Eğer workflow "persist=artifact" ise app dosyaları repo’da olmayacağı için app.py bunları göremez.
#   O zaman: Secrets’a GITHUB_TOKEN ekleyip DATA_SOURCE="artifact" kullanmalısın.
# - Eğer "persist=commit" ise DATA_SOURCE="raw" yeterli (token gerekmez).
#
# ENV / SECRETS:
#   DATA_SOURCE: local|raw|release|artifact   (default: raw)
#   GH_OWNER, GH_REPO, GH_BRANCH             (default: repo bilgisi)
#   GITHUB_TOKEN                             (artifact için şart; release/private ise gerekebilir)
#   WORKFLOW_NAME                            (default: Full SF Crime Pipeline)
#   ARTIFACT_NAME                            (default: sf-crime-pipeline-output)
#   RELEASE_ASSET_NAME                        (default: sf-crime-pipeline-output.zip)  # release senaryosu
#   RELEASE_TAG                               (default: latest)
#
# Dosya listesi senin akışına göre:
# 00 sf_crime.csv
# 01 sf_crime_01.csv
# 02 sf_crime_02.csv
# 03 sf_crime_03.csv
# 04 sf_crime_04.csv
# 05 sf_crime_05.csv
# 06 sf_crime_06.csv
# 07 sf_crime_07.csv
# 08 sf_crime_08.csv
# 09 sf_crime_09.csv

import os
import io
import re
import json
import time
import zipfile
import tempfile
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

import pandas as pd
import requests
import streamlit as st


# =========================
# 0) UI / Stil
# =========================
st.set_page_config(page_title="SUTAM – Veri Hazırlama Süreci", layout="wide")

# “Times 12 formatında klasik başlık”
st.markdown(
    """
<style>
h1, h2, h3, h4, h5, h6, p, div, span, label {
  font-family: "Times New Roman", Times, serif !important;
}
.block-container { padding-top: 1.25rem; }
table, th, td { font-family: "Times New Roman", Times, serif !important; font-size: 12pt !important; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("SUTAM – Veri Hazırlama Süreci")


# =========================
# 1) Konfigürasyon
# =========================
DATA_SOURCE = os.getenv("DATA_SOURCE", "raw").strip().lower()  # local|raw|release|artifact

GH_OWNER = os.getenv("GH_OWNER", "")  # boşsa otomatik deneyeceğiz
GH_REPO = os.getenv("GH_REPO", "")
GH_BRANCH = os.getenv("GH_BRANCH", "main")

WORKFLOW_NAME = os.getenv("WORKFLOW_NAME", "Full SF Crime Pipeline")
ARTIFACT_NAME = os.getenv("ARTIFACT_NAME", "sf-crime-pipeline-output")

RELEASE_TAG = os.getenv("RELEASE_TAG", "latest")
RELEASE_ASSET_NAME = os.getenv("RELEASE_ASSET_NAME", "sf-crime-pipeline-output.zip")

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")  # artifact için şart

# Local / repo içi varsayılan data dizini (istersen değiştir)
LOCAL_DATA_DIR = os.getenv("LOCAL_DATA_DIR", "crime_prediction_data")  # repo içinde

# İstersen tabloya dahil etmek için ek dosyalar (örn. week.csv)
EXTRA_FILES = os.getenv("EXTRA_FILES", "").strip()  # "week.csv;neighbors.csv" gibi


@dataclass
class StageSpec:
    stage: str
    filename: str
    note: str


PIPELINE_SPECS: List[StageSpec] = [
    StageSpec("00", "sf_crime.csv", "Ham + temiz + GEOID + zaman feature"),
    StageSpec("01", "sf_crime_01.csv", "+ 911"),
    StageSpec("02", "sf_crime_02.csv", "+ 311"),
    StageSpec("03", "sf_crime_03.csv", "+ nüfus/demografi"),
    StageSpec("04", "sf_crime_04.csv", "+ otobüs mesafe/yoğunluk"),
    StageSpec("05", "sf_crime_05.csv", "+ tren mesafe/yoğunluk"),
    StageSpec("06", "sf_crime_06.csv", "+ POI risk/yoğunluk"),
    StageSpec("07", "sf_crime_07.csv", "+ police/gov mesafe/yakınlık"),
    StageSpec("08", "sf_crime_08.csv", "(senin akışında burası netleştirilecek)"),
    StageSpec("09", "sf_crime_09.csv", "+ neighbors/otokorelasyon"),
]


# =========================
# 2) Yardımcılar: GitHub erişimi
# =========================
def _guess_owner_repo() -> Tuple[str, str]:
    """
    Streamlit Cloud’da repo bilgisi yoksa env üzerinden set et.
    Local’de kullanıcı set etmediyse boş döner.
    """
    return GH_OWNER, GH_REPO


def _gh_headers() -> Dict[str, str]:
    h = {"Accept": "application/vnd.github+json"}
    if GITHUB_TOKEN:
        h["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return h


def _http_get(url: str, headers: Optional[Dict[str, str]] = None, stream: bool = False) -> requests.Response:
    resp = requests.get(url, headers=headers or {}, stream=stream, timeout=60)
    resp.raise_for_status()
    return resp


def _download_bytes(url: str, headers: Optional[Dict[str, str]] = None) -> bytes:
    r = _http_get(url, headers=headers, stream=True)
    buf = io.BytesIO()
    for chunk in r.iter_content(chunk_size=1024 * 1024):
        if chunk:
            buf.write(chunk)
    return buf.getvalue()


def _raw_file_url(owner: str, repo: str, branch: str, path: str) -> str:
    # raw github content
    return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}"


def _api_url(owner: str, repo: str, endpoint: str) -> str:
    return f"https://api.github.com/repos/{owner}/{repo}/{endpoint.lstrip('/')}"


@st.cache_data(show_spinner=False, ttl=600)
def gh_find_latest_success_run_id(owner: str, repo: str, workflow_name: str) -> Optional[int]:
    """
    Workflow adına göre son başarılı run_id bulur.
    Token yoksa private repo’da çalışmaz.
    """
    if not owner or not repo:
        return None

    # Workflow'ları listele → name eşleşmesini bul → workflow_id
    wf_url = _api_url(owner, repo, "actions/workflows")
    r = _http_get(wf_url, headers=_gh_headers())
    data = r.json()
    workflow_id = None
    for wf in data.get("workflows", []):
        if (wf.get("name") or "").strip().lower() == workflow_name.strip().lower():
            workflow_id = wf.get("id")
            break
    if not workflow_id:
        return None

    runs_url = _api_url(owner, repo, f"actions/workflows/{workflow_id}/runs?status=success&per_page=1")
    r2 = _http_get(runs_url, headers=_gh_headers())
    j = r2.json()
    runs = j.get("workflow_runs", [])
    if not runs:
        return None
    return runs[0].get("id")


@st.cache_data(show_spinner=False, ttl=600)
def gh_get_artifact_download_url(owner: str, repo: str, run_id: int, artifact_name: str) -> Optional[str]:
    """
    Verilen run_id içindeki artifact_name’in download URL’sini bulur.
    """
    if not owner or not repo or not run_id:
        return None
    arts_url = _api_url(owner, repo, f"actions/runs/{run_id}/artifacts?per_page=100")
    r = _http_get(arts_url, headers=_gh_headers())
    data = r.json()
    for a in data.get("artifacts", []):
        if (a.get("name") or "").strip() == artifact_name.strip():
            return a.get("archive_download_url")
    return None


@st.cache_data(show_spinner=False, ttl=600)
def gh_download_and_extract_artifact(owner: str, repo: str, run_id: int, artifact_name: str) -> str:
    """
    Artifact zip indirir, temp dir’e açar, extracted dir path döner.
    """
    url = gh_get_artifact_download_url(owner, repo, run_id, artifact_name)
    if not url:
        raise RuntimeError(f"Artifact bulunamadı: {artifact_name} (run_id={run_id})")

    # GitHub artifact download URL’si API auth ister
    zbytes = _download_bytes(url, headers=_gh_headers())

    tmpdir = tempfile.mkdtemp(prefix="sutam_artifact_")
    with zipfile.ZipFile(io.BytesIO(zbytes), "r") as zf:
        zf.extractall(tmpdir)
    return tmpdir


@st.cache_data(show_spinner=False, ttl=600)
def gh_download_and_extract_release_asset(owner: str, repo: str, tag: str, asset_name: str) -> str:
    """
    Release asset zip indirip açar.
    Public ise token gerekmez; private ise token gerekir.
    """
    rel_url = _api_url(owner, repo, f"releases/{tag}")
    r = _http_get(rel_url, headers=_gh_headers())
    rel = r.json()
    assets = rel.get("assets", [])
    dl = None
    for a in assets:
        if (a.get("name") or "").strip() == asset_name.strip():
            dl = a.get("browser_download_url")
            break
    if not dl:
        raise RuntimeError(f"Release asset bulunamadı: {asset_name} (tag={tag})")

    zbytes = _download_bytes(dl, headers=_gh_headers() if GITHUB_TOKEN else None)
    tmpdir = tempfile.mkdtemp(prefix="sutam_release_")
    with zipfile.ZipFile(io.BytesIO(zbytes), "r") as zf:
        zf.extractall(tmpdir)
    return tmpdir


# =========================
# 3) CSV profil hesaplama
# =========================
def profile_csv(path: str) -> Dict[str, object]:
    """
    Satır/sütun/NaN oranı/tam boş satır.
    Büyük dosyada da hızlı olması için:
    - dtype=str ile okunur (NaN tespiti için yeterli)
    - low_memory False
    """
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception:
        # Bazı CSV’lerde delimiter/encoding sorun olabilir
        df = pd.read_csv(path, low_memory=False, encoding="utf-8", on_bad_lines="skip")

    n_rows, n_cols = df.shape

    # NaN hücre (%): toplam NaN / (rows*cols) *100
    if n_rows == 0 or n_cols == 0:
        nan_pct = 0.0
    else:
        nan_cells = int(df.isna().sum().sum())
        total_cells = int(n_rows * n_cols)
        nan_pct = (nan_cells / total_cells) * 100.0

    # Tam boş satır: tüm kolonları NaN olan satır sayısı
    empty_rows = int(df.isna().all(axis=1).sum()) if n_rows else 0

    return {
        "rows": int(n_rows),
        "cols": int(n_cols),
        "nan_pct": float(nan_pct),
        "empty_rows": int(empty_rows),
    }


def fmt_int(x) -> str:
    if x is None:
        return "-"
    try:
        return f"{int(x):,}".replace(",", ".")  # TR binlik
    except Exception:
        return "-"


def fmt_nan_pct(x) -> str:
    if x is None:
        return "-"
    try:
        # küçükse bilimsel gibi görünmesin diye 6 basamak
        return f"{float(x):.6f}".rstrip("0").rstrip(".")
    except Exception:
        return "-"


# =========================
# 4) Dosya çözümleme: hangi kaynaktan okuyacağız?
# =========================
def resolve_base_dir() -> Tuple[str, str]:
    """
    (base_dir, info_text)
    base_dir: dosyaların bulunduğu yer (local klasör veya extracted temp dir)
    """
    owner, repo = _guess_owner_repo()

    if DATA_SOURCE == "local":
        return LOCAL_DATA_DIR, f"Kaynak: LOCAL → `{LOCAL_DATA_DIR}/`"

    if DATA_SOURCE == "raw":
        # raw için base_dir yok; URL üzerinden indireceğiz (temp’e atacağız)
        if not owner or not repo:
            return "", "Kaynak: RAW ama GH_OWNER/GH_REPO boş. (Env ile set et)"
        return "", f"Kaynak: GITHUB RAW → {owner}/{repo}@{GH_BRANCH}"

    if DATA_SOURCE == "release":
        if not owner or not repo:
            return "", "Kaynak: RELEASE ama GH_OWNER/GH_REPO boş. (Env ile set et)"
        d = gh_download_and_extract_release_asset(owner, repo, RELEASE_TAG, RELEASE_ASSET_NAME)
        return d, f"Kaynak: RELEASE asset → `{RELEASE_TAG}` / `{RELEASE_ASSET_NAME}`"

    if DATA_SOURCE == "artifact":
        if not owner or not repo:
            return "", "Kaynak: ARTIFACT ama GH_OWNER/GH_REPO boş. (Env ile set et)"
        if not GITHUB_TOKEN:
            return "", "Kaynak: ARTIFACT seçili ama GITHUB_TOKEN yok. (Streamlit secrets’a ekle)"
        run_id = gh_find_latest_success_run_id(owner, repo, WORKFLOW_NAME)
        if not run_id:
            return "", f"Artifact için başarılı run bulunamadı: `{WORKFLOW_NAME}`"
        d = gh_download_and_extract_artifact(owner, repo, run_id, ARTIFACT_NAME)
        return d, f"Kaynak: ARTIFACT → run_id={run_id} / `{ARTIFACT_NAME}`"

    return "", f"Bilinmeyen DATA_SOURCE: {DATA_SOURCE}"


@st.cache_data(show_spinner=False, ttl=600)
def download_raw_to_temp(owner: str, repo: str, branch: str, rel_path: str) -> Optional[str]:
    """
    raw GitHub üzerinden dosyayı indirir, temp klasöre yazar, path döndürür.
    """
    url = _raw_file_url(owner, repo, branch, rel_path)
    try:
        b = _download_bytes(url)
    except Exception:
        return None
    tmpdir = tempfile.mkdtemp(prefix="sutam_raw_")
    out = os.path.join(tmpdir, os.path.basename(rel_path))
    with open(out, "wb") as f:
        f.write(b)
    return out


def find_file_in_dir(base_dir: str, filename: str) -> Optional[str]:
    """
    base_dir içinde filename’i bul.
    Artifact/release zip’lerinde bazen alt klasöre düşebilir: recursive ara.
    """
    if not base_dir:
        return None
    direct = os.path.join(base_dir, filename)
    if os.path.isfile(direct):
        return direct
    # recursive search
    for root, _, files in os.walk(base_dir):
        if filename in files:
            return os.path.join(root, filename)
    return None


# =========================
# 5) Tabloyu üret
# =========================
def build_pipeline_table() -> Tuple[pd.DataFrame, str]:
    base_dir, info = resolve_base_dir()
    owner, repo = _guess_owner_repo()

    rows = []
    missing = []

    for spec in PIPELINE_SPECS:
        rec = {
            "Aşama": spec.stage,
            "Dosya": spec.filename,
            "Satır": "-",
            "Sütun": "-",
            "NaN hücre (%)": "-",
            "Tam boş satır": "-",
            "Not": spec.note,
        }

        fpath = None

        if DATA_SOURCE == "raw":
            if owner and repo:
                # raw path’leri: önce crime_prediction_data/, sonra repo root fallback
                cand1 = f"{LOCAL_DATA_DIR}/{spec.filename}".replace("\\", "/")
                cand2 = spec.filename
                p = download_raw_to_temp(owner, repo, GH_BRANCH, cand1) or download_raw_to_temp(owner, repo, GH_BRANCH, cand2)
                fpath = p
        else:
            # local/release/artifact → directory araması
            fpath = find_file_in_dir(base_dir, spec.filename)

        if not fpath:
            missing.append(spec.filename)
            rows.append(rec)
            continue

        try:
            prof = profile_csv(fpath)
            rec["Satır"] = fmt_int(prof["rows"])
            rec["Sütun"] = fmt_int(prof["cols"])
            rec["NaN hücre (%)"] = fmt_nan_pct(prof["nan_pct"])
            rec["Tam boş satır"] = fmt_int(prof["empty_rows"])
        except Exception:
            missing.append(spec.filename)
        rows.append(rec)

    df = pd.DataFrame(rows, columns=["Aşama", "Dosya", "Satır", "Sütun", "NaN hücre (%)", "Tam boş satır", "Not"])

    msg = info
    if missing:
        msg += f"\n\nEksik/erişilemeyen dosyalar ({len(missing)}): " + ", ".join(missing)
        msg += "\n\nNot: Workflow 'persist=artifact' ise dosyalar repo’da olmaz. Bu durumda DATA_SOURCE=artifact + GITHUB_TOKEN gerekir."
    return df, msg


# =========================
# 6) UI
# =========================
with st.expander("Kaynak ayarları", expanded=False):
    st.write("**DATA_SOURCE**:", DATA_SOURCE)
    st.write("**LOCAL_DATA_DIR**:", LOCAL_DATA_DIR)
    st.write("**WORKFLOW_NAME**:", WORKFLOW_NAME)
    st.write("**ARTIFACT_NAME**:", ARTIFACT_NAME)
    st.write("**GH_BRANCH**:", GH_BRANCH)
    st.write("**Release**:", f"{RELEASE_TAG} / {RELEASE_ASSET_NAME}")
    st.caption(
        "İpucu: Workflow artifact üretiyorsa app’nin görmesi için DATA_SOURCE=artifact ve GITHUB_TOKEN gerekir. "
        "Commit’e basıyorsan DATA_SOURCE=raw yeter."
    )

with st.spinner("Pipeline tablo üretiliyor..."):
    table_df, info_text = build_pipeline_table()

st.info(info_text)

# tablo (sadece bu!)
st.dataframe(table_df, use_container_width=True, hide_index=True)

# csv indir
csv_bytes = table_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Tabloyu indir (CSV)",
    data=csv_bytes,
    file_name="sutam_pipeline_summary.csv",
    mime="text/csv",
)

# Opsiyonel: hızlı toplamlar
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Aşama sayısı", len(table_df))
with col2:
    filled = (table_df["Satır"] != "-").sum()
    st.metric("Dolu profil satırı", int(filled))
with col3:
    st.metric("Eksik dosya", int(len(table_df) - filled))
