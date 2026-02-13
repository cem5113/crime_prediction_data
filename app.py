# app.py — SUTAM | Veri Hazırlama Süreci (artifact/raw/local fallback)
# Python 3.11 + Streamlit

from __future__ import annotations

import os
import io
import re
import zipfile
import shutil
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

try:
    import requests
except Exception:
    requests = None

# PyGithub (requirements.txt içinde pygithub var)
try:
    from github import Github
except Exception:
    Github = None


# =========================
# UI / STYLE (Helvetica)
# =========================
st.set_page_config(page_title="SUTAM – Veri Hazırlama Süreci", layout="wide")

st.markdown(
    """
    <style>
      html, body, [class*="css"]  {
        font-family: Helvetica, Arial, sans-serif !important;
      }
      .sutam-title {
        font-size: 22px;        /* daha küçük */
        font-weight: 700;
        margin: 0.2rem 0 0.6rem 0;
        letter-spacing: 0.2px;
      }
      .sutam-sub {
        font-size: 13px;
        opacity: 0.8;
        margin-bottom: 0.7rem;
      }
      .sutam-foot {
        font-size: 13px;
        opacity: 0.85;
        margin-top: 0.6rem;
      }
      .stDataFrame {
        font-size: 13px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="sutam-title">SUTAM – Veri Hazırlama Süreci</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sutam-sub">Pipeline çıktılarının dosya bazlı sağlık özeti (satır/sütun, NaN %, tam boş satır) ve güncellik bilgisi.</div>',
    unsafe_allow_html=True,
)


# =========================
# CONFIG (ENV)
# =========================
def env(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return v.strip() if isinstance(v, str) and v.strip() else default

DATA_SOURCE     = env("DATA_SOURCE", "raw").lower()  # artifact | raw | local
LOCAL_DATA_DIR  = env("LOCAL_DATA_DIR", "crime_prediction_data")
GH_OWNER        = env("GH_OWNER", "")
GITHUB_REPO     = env("GITHUB_REPO", "")  # "owner/repo" (opsiyonel)
GH_TOKEN        = env("GH_TOKEN", "") or env("GITHUB_TOKEN", "")
GH_BRANCH       = env("GH_BRANCH", "main")
WORKFLOW_NAME   = env("WORKFLOW_NAME", "Full SF Crime Pipeline")
ARTIFACT_NAME   = env("ARTIFACT_NAME", "sf-crime-pipeline-output")

# opsiyonel fallback (release zip vs)
RELEASE_ZIP_URL = env("RELEASE_ZIP_URL", "")  # örn: https://.../sf-crime-pipeline-output.zip

ALLOW_PIPELINE  = env("ALLOW_PIPELINE", "0")  # sadece bilgi amaçlı

# Eğer kullanıcı sadece "owner/repo" verdi ise ayrıştır
if (not GH_OWNER) and GITHUB_REPO and "/" in GITHUB_REPO:
    GH_OWNER = GITHUB_REPO.split("/")[0].strip()

if (not GITHUB_REPO) and GH_OWNER:
    # kullanıcı bazen sadece GH_OWNER set ediyor; repo adı gerekli
    # burada tahmin etmiyoruz, boş bırakıyoruz
    pass


# =========================
# FILES / STAGES (TABLE)
# =========================
STAGES = [
    ("00", "sf_crime.csv",    "Ham + temiz + GEOID + zaman feature"),
    ("01", "sf_crime_01.csv", "+ 911"),
    ("02", "sf_crime_02.csv", "+ 311"),
    ("03", "sf_crime_03.csv", "+ nüfus/demografi"),
    ("04", "sf_crime_04.csv", "+ otobüs mesafe/yoğunluk"),
    ("05", "sf_crime_05.csv", "+ tren mesafe/yoğunluk"),
    ("06", "sf_crime_06.csv", "+ POI risk/yoğunluk"),
    ("07", "sf_crime_07.csv", "+ police/gov mesafe/yakınlık"),
    ("08", "sf_crime_08.csv", "(akışa göre netleşecek/ara çıktı)"),
    ("09", "sf_crime_09.csv", "+ neighbors/otokorelasyon"),
]


# =========================
# HELPERS
# =========================
def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def repo_owner_and_name() -> tuple[str, str]:
    """
    Öncelik:
      1) GITHUB_REPO = owner/repo
      2) GH_OWNER + repo env (GITHUB_REPO yoksa hata)
    """
    if GITHUB_REPO and "/" in GITHUB_REPO:
        o, r = GITHUB_REPO.split("/", 1)
        return o.strip(), r.strip()

    # Bu noktada repo adı bilinmiyor → crash etmeyelim
    return "", ""

def safe_read_csv(path: Path) -> pd.DataFrame:
    # büyük dosyalarda daha stabil
    return pd.read_csv(path, low_memory=False)

def calc_file_stats(csv_path: Path) -> dict:
    """
    Satır/Sütun/NaN%/Tam boş satır hesaplar.
    """
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return {"rows": None, "cols": None, "nan_pct": None, "empty_rows": None, "ok": False}

    try:
        df = safe_read_csv(csv_path)
        rows, cols = df.shape
        # NaN yüzdesi: tüm hücreler içinde boş olanların oranı
        total_cells = rows * cols if rows and cols else 0
        nan_cells = int(df.isna().sum().sum()) if total_cells else 0
        nan_pct = (nan_cells / total_cells * 100.0) if total_cells else 0.0
        empty_rows = int(df.isna().all(axis=1).sum()) if rows else 0
        return {
            "rows": rows,
            "cols": cols,
            "nan_pct": round(float(nan_pct), 3),
            "empty_rows": empty_rows,
            "ok": True,
        }
    except Exception as e:
        return {"rows": None, "cols": None, "nan_pct": None, "empty_rows": None, "ok": False, "err": str(e)}

def fmt_int(x):
    return "—" if x is None else f"{x:,}".replace(",", ".")

def fmt_float(x):
    return "—" if x is None else f"{x:.3f}"

def get_latest_file_mtime(base_dir: Path) -> tuple[datetime | None, str | None]:
    candidates = [s[1] for s in STAGES[::-1]]  # 09 -> 00
    for f in candidates:
        p = base_dir / f
        if p.exists():
            ts = p.stat().st_mtime
            return datetime.fromtimestamp(ts, tz=timezone.utc), f
    return None, None


# =========================
# ARTIFACT DOWNLOAD
# =========================
def download_and_extract_latest_artifact(
    out_dir: Path,
    owner: str,
    repo: str,
    workflow_name: str,
    artifact_name: str,
    branch: str,
    token: str,
) -> dict:
    """
    Son başarılı workflow run'ından artifact zip indirip out_dir'e açar.
    """
    if Github is None:
        raise RuntimeError("PyGithub (github) import edilemedi. requirements.txt içinde pygithub olmalı.")
    if not token:
        raise RuntimeError("DATA_SOURCE=artifact için GH_TOKEN veya GITHUB_TOKEN gerekli.")
    if not owner or not repo:
        raise RuntimeError("Repo bilgisi eksik. GITHUB_REPO='owner/repo' veya GH_OWNER + repo gerekli.")

    gh = Github(token)
    r = gh.get_repo(f"{owner}/{repo}")

    # Workflow seç (adı ile)
    workflows = list(r.get_workflows())
    wf = None
    for w in workflows:
        if (w.name or "").strip() == workflow_name.strip():
            wf = w
            break
    if wf is None:
        # isim eşleşmediyse id bulunamadı
        # kullanıcı bazen dosya adı verir: full_pipeline.yml
        # o zaman workflow_path ile deneyelim
        for w in workflows:
            if workflow_name.strip().lower() in (w.path or "").lower():
                wf = w
                break
    if wf is None:
        raise RuntimeError(f"Workflow bulunamadı: '{workflow_name}'. Repo içindeki Actions workflow adını kontrol et.")

    # Son başarılı run
    runs = wf.get_runs(branch=branch, status="success")
    run = None
    for x in runs:
        run = x
        break
    if run is None:
        raise RuntimeError(f"'{workflow_name}' için branch='{branch}' üzerinde SUCCESS run bulunamadı.")

    # Run artifacts
    arts = run.get_artifacts()
    target = None
    for a in arts:
        if (a.name or "").strip() == artifact_name.strip():
            target = a
            break
    if target is None:
        # isim tutmadıysa, ilk artifact
        for a in arts:
            target = a
            break
    if target is None:
        raise RuntimeError("Bu run içinde artifact bulunamadı.")

    # Download (archive_download_url) — requests ile
    if requests is None:
        raise RuntimeError("requests import edilemedi.")

    url = target.archive_download_url
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}
    resp = requests.get(url, headers=headers, timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"Artifact indirilemedi. HTTP {resp.status_code}: {resp.text[:300]}")

    ensure_dir(out_dir)
    zbytes = io.BytesIO(resp.content)
    with zipfile.ZipFile(zbytes) as z:
        z.extractall(out_dir)

    return {
        "run_id": getattr(run, "id", None),
        "run_number": getattr(run, "run_number", None),
        "artifact_found": target.name,
        "artifact_size_in_bytes": getattr(target, "size_in_bytes", None),
    }


def maybe_fetch_data(base_dir: Path) -> dict:
    """
    DATA_SOURCE:
      - raw/local: repo içi klasörden okur (base_dir)
      - artifact: GitHub Actions artifact indirip base_dir'e extract eder
    """
    base_dir = ensure_dir(base_dir)
    info = {"mode": DATA_SOURCE, "base_dir": str(base_dir)}

    owner, repo = repo_owner_and_name()

    if DATA_SOURCE in ("raw", "local"):
        # hiçbir şey indirme, sadece dizin var mı kontrol et
        info["note"] = "RAW/LOCAL: repo içindeki dosyalar okunacak."
        return info

    if DATA_SOURCE == "artifact":
        # daha önce extract edilmişse tekrar indirme (basit kontrol)
        sentinel = base_dir / ".artifact_ok"
        if sentinel.exists():
            info["note"] = "Artifact zaten hazır (sentinel bulundu)."
            return info

        # artifact indir
        meta = download_and_extract_latest_artifact(
            out_dir=base_dir,
            owner=owner or GH_OWNER,
            repo=repo,
            workflow_name=WORKFLOW_NAME,
            artifact_name=ARTIFACT_NAME,
            branch=GH_BRANCH,
            token=GH_TOKEN,
        )
        sentinel.write_text(f"ok {datetime.utcnow().isoformat()}Z\n")
        info["artifact"] = meta
        info["note"] = "Artifact indirildi ve açıldı."
        return info

    info["note"] = f"Bilinmeyen DATA_SOURCE='{DATA_SOURCE}'. RAW gibi davranılacak."
    return info


def build_table(base_dir: Path) -> pd.DataFrame:
    rows = []
    missing = []
    for stage, fname, note in STAGES:
        p = base_dir / fname
        if p.exists():
            stats = calc_file_stats(p)
            rows.append({
                "Aşama": stage,
                "Dosya": fname,
                "Satır": stats.get("rows"),
                "Sütun": stats.get("cols"),
                "NaN hücre (%)": stats.get("nan_pct"),
                "Tam boş satır": stats.get("empty_rows"),
                "Not": note,
            })
        else:
            missing.append(fname)
            rows.append({
                "Aşama": stage,
                "Dosya": fname,
                "Satır": None,
                "Sütun": None,
                "NaN hücre (%)": None,
                "Tam boş satır": None,
                "Not": note,
            })

    df = pd.DataFrame(rows)

    # format (görüntü için)
    df_disp = df.copy()
    df_disp["Satır"] = df_disp["Satır"].apply(fmt_int)
    df_disp["Sütun"] = df_disp["Sütun"].apply(fmt_int)
    df_disp["NaN hücre (%)"] = df_disp["NaN hücre (%)"].apply(lambda x: "—" if x is None else f"{x:.3f}")
    df_disp["Tam boş satır"] = df_disp["Tam boş satır"].apply(fmt_int)

    return df_disp, missing


# =========================
# MAIN
# =========================
with st.expander("⚙️ Kaynak ayarları (okunuyor)", expanded=False):
    st.write({
        "DATA_SOURCE": DATA_SOURCE,
        "LOCAL_DATA_DIR": LOCAL_DATA_DIR,
        "GITHUB_REPO": GITHUB_REPO,
        "GH_OWNER": GH_OWNER,
        "GH_BRANCH": GH_BRANCH,
        "WORKFLOW_NAME": WORKFLOW_NAME,
        "ARTIFACT_NAME": ARTIFACT_NAME,
        "ALLOW_PIPELINE": ALLOW_PIPELINE,
        "RELEASE_ZIP_URL": RELEASE_ZIP_URL or "(boş)",
        "TOKEN_SET": bool(GH_TOKEN),
    })

base_dir = Path(LOCAL_DATA_DIR)

# 1) Veriyi hazırla (artifact ise indir-aç)
try:
    meta = maybe_fetch_data(base_dir)
except Exception as e:
    st.error("Veri kaynağı hazırlanamadı (uygulama çökmemesi için burada durduruldu).")
    st.exception(e)
    st.stop()

# 2) Tabloyu üret
df_table, missing_files = build_table(base_dir)

st.dataframe(df_table, use_container_width=True, hide_index=True)

# 3) Alt bilgi: güncellik + veri zaman aralığı (İlk–Son kayıt)
last_update, last_file = get_latest_file_mtime(base_dir)
now_utc = datetime.now(timezone.utc)

st.markdown("---")

# --- İlk/Son kayıt zamanını bulmak için en güncel suç dosyasını seçelim ---
# Öncelik: sf_crime_09 -> sf_crime_00
latest_csv = None
for _, fname, _ in STAGES[::-1]:
    p = base_dir / fname
    if p.exists() and p.stat().st_size > 0:
        latest_csv = p
        break

first_txt, last_txt = "—", "—"

if latest_csv is not None:
    try:
        df_latest = safe_read_csv(latest_csv)

        # Zaman sütunu isimleri (senin dosyalarda olabilecek varyantlar)
        time_candidates = [
            "event_datetime", "incident_datetime", "datetime",
            "event_time", "incident_time", "time",
            "date", "report_datetime", "created_at", "timestamp"
        ]
        time_col = next((c for c in time_candidates if c in df_latest.columns), None)

        if time_col:
            dt = pd.to_datetime(df_latest[time_col], errors="coerce", utc=False)
            dt = dt.dropna()

            if len(dt) > 0:
                first_dt = dt.min()
                last_dt  = dt.max()

                # Görsel format: 01.01.2021 00:00
                first_txt = first_dt.strftime("%d.%m.%Y %H:%M")
                last_txt  = last_dt.strftime("%d.%m.%Y %H:%M")
        else:
            # zaman sütunu yoksa sessizce "—" bırak
            pass
    except Exception:
        # okuma/parse patlarsa UI çökmesin
        pass

# Pipeline güncelleme: son görülen dosyanın mtime'ı (UTC)
pipeline_update_txt = "—"
source_file_txt = "—"
if last_update:
    pipeline_update_txt = last_update.strftime("%d.%m.%Y %H:%M UTC")
    source_file_txt = last_file or "—"

system_time_txt = now_utc.strftime("%d.%m.%Y %H:%M UTC")

st.markdown(
    f"""
    <div class="sutam-foot">
        <div>
            <strong>Suç Veri Zaman Aralığı (İlk – Son Kayıt)</strong><br>
            {first_txt} → {last_txt}
        </div>
        <br>
        <div>
            <strong>Pipeline Son Güncelleme</strong><br>
            {pipeline_update_txt} | Kaynak dosya: {source_file_txt}
        </div>
        <br>
        <div>
            <strong>Sistem Saati</strong><br>
            {system_time_txt}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# 4) Eksikler
if missing_files:
    st.warning(
        f"Eksik/erişilemeyen dosyalar ({len(missing_files)}): " + ", ".join(missing_files)
    )
    # Kullanıcıya net yönlendirme:
    if DATA_SOURCE == "artifact":
        st.info("DATA_SOURCE=artifact kullanıyorsun. Artifact'ın SUCCESS run’dan üretildiğinden ve ARTIFACT_NAME’in doğru olduğundan emin ol.")
        if not GH_TOKEN:
            st.error("GH_TOKEN / GITHUB_TOKEN set değil. Artifact indiremez.")
        owner, repo = repo_owner_and_name()
        if not owner or not repo:
            st.error("GITHUB_REPO='owner/repo' formatında set edilmeli (repo adı boş görünüyor).")
    else:
        st.info("DATA_SOURCE=raw/local ise dosyaların repoda 'crime_prediction_data/' altında commit edilmiş olması gerekir.")

# 5) Debug meta
with st.expander("🧾 Debug (kaynak çözümleme sonucu)", expanded=False):
    st.write(meta)
    st.write({"base_dir_exists": base_dir.exists(), "base_dir": str(base_dir)})
    if base_dir.exists():
        try:
            st.write({"dir_listing": sorted([p.name for p in base_dir.iterdir()])[:200]})
        except Exception as _:
            pass
