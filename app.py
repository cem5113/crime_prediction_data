# app.py
from __future__ import annotations

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple

import streamlit as st

try:
    import pytz
except Exception:
    pytz = None


# =============================================================================
# 0) CONFIG
# =============================================================================
REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = REPO_ROOT  # YAML: CRIME_DATA_DIR = github.workspace
DEFAULT_OUT_SUBDIR = "crime_prediction_data"

SF_TZ = "America/Los_Angeles"


# =============================================================================
# 1) HELPERS
# =============================================================================
def sf_now() -> datetime:
    if pytz is None:
        # pytz yoksa, gate'i devre dışı bırak (best effort)
        return datetime.now()
    return datetime.now(pytz.timezone(SF_TZ))


def gate_07(force: bool) -> Tuple[bool, str]:
    now = sf_now()
    if force:
        return True, f"FORCE açık → kapı bypass. SF time: {now}"
    if now.strftime("%H") == "07":
        return True, f"SF 07:xx → devam. SF time: {now}"
    return False, f"SF 07:00 kapısı nedeniyle durduruldu. SF time: {now}"


def run_cmd(cmd: List[str], cwd: Path, env: dict, label: str) -> None:
    st.write(f"**▶ {label}**")
    st.code(" ".join(cmd))
    p = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    out_lines = []
    assert p.stdout is not None
    for line in p.stdout:
        out_lines.append(line)
        # streamlit canlı log
        st.write(line.rstrip("\n"))
    rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"{label} failed (exit={rc}).")


def copy_first_existing(candidates: List[Path], dst: Path, mode: str = "copy") -> bool:
    for p in candidates:
        if p.exists() and p.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists() and p.samefile(dst):
                return True
            if mode == "move":
                shutil.move(str(p), str(dst))
            else:
                shutil.copy2(str(p), str(dst))
            return True
    return False


def head_file(p: Path, n: int = 5) -> str:
    if not p.exists():
        return f"(missing) {p}"
    try:
        lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()[:n]
        return "\n".join(lines)
    except Exception as e:
        return f"(head read error) {p}: {e}"


# =============================================================================
# 2) PIPELINE STEPS (YAML -> Python)
# =============================================================================
def normalize_911(crime_dir: Path) -> None:
    # YAML 02.fix
    cands = [
        crime_dir / "sf_911_last_5_year.csv",
        crime_dir / "sf_911_last_5_year_y.csv",
        REPO_ROOT / "sf_911_last_5_year.csv",
        REPO_ROOT / "sf_911_last_5_year_y.csv",
        REPO_ROOT / "_tmp_911.csv",
        REPO_ROOT / "sf_911_recent.csv",
        REPO_ROOT / "data" / "sf_911_last_5_year.csv",
        REPO_ROOT / "data" / "sf_911_last_5_year_y.csv",
        REPO_ROOT / "outputs" / "sf_911_last_5_year.csv",
        REPO_ROOT / "outputs" / "sf_911_last_5_year_y.csv",
        REPO_ROOT / "outputs" / "_tmp_911.csv",
        REPO_ROOT / "outputs" / "sf_911_recent.csv",
    ]

    # kopyala (var olanları crime_dir altına)
    for p in cands:
        if p.exists() and p.is_file():
            dst = crime_dir / p.name
            if dst.exists() and p.samefile(dst):
                continue
            shutil.copy2(str(p), str(dst))

    # isim normalize
    target = crime_dir / "sf_911_last_5_year.csv"
    if not target.exists():
        if (crime_dir / "_tmp_911.csv").exists():
            shutil.move(str(crime_dir / "_tmp_911.csv"), str(target))
        elif (crime_dir / "sf_911_recent.csv").exists():
            shutil.copy2(str(crime_dir / "sf_911_recent.csv"), str(target))


def normalize_311(crime_dir: Path) -> None:
    # YAML 03.fix
    cands = [
        REPO_ROOT / "sf_311_last_5_years.csv",
        REPO_ROOT / "sf_311_last_5_years_y.csv",
        REPO_ROOT / "sf_311_last_5_year.csv",
        REPO_ROOT / "sf_311_last_5_year_y.csv",
        REPO_ROOT / "data" / "sf_311_last_5_years.csv",
        REPO_ROOT / "data" / "sf_311_last_5_years_y.csv",
        REPO_ROOT / "outputs" / "sf_311_last_5_years.csv",
        REPO_ROOT / "outputs" / "sf_311_last_5_years_y.csv",
        crime_dir / "sf_311_last_5_years.csv",
        crime_dir / "sf_311_last_5_years_y.csv",
        crime_dir / "sf_311_last_5_year.csv",
        crime_dir / "sf_311_last_5_year_y.csv",
    ]

    for p in cands:
        if p.exists() and p.is_file():
            dst = crime_dir / p.name
            if dst.exists() and p.samefile(dst):
                continue
            shutil.copy2(str(p), str(dst))

    # tekil -> çoğul normalize
    if not (crime_dir / "sf_311_last_5_years.csv").exists() and (crime_dir / "sf_311_last_5_year.csv").exists():
        shutil.copy2(str(crime_dir / "sf_311_last_5_year.csv"), str(crime_dir / "sf_311_last_5_years.csv"))
    if not (crime_dir / "sf_311_last_5_years_y.csv").exists() and (crime_dir / "sf_311_last_5_year_y.csv").exists():
        shutil.copy2(str(crime_dir / "sf_311_last_5_year_y.csv"), str(crime_dir / "sf_311_last_5_years_y.csv"))


def normalize_grid(crime_dir: Path) -> None:
    dst = crime_dir / "sf_crime_grid_full_labeled.csv"
    candidates = [
        REPO_ROOT / "sf_crime_grid_full_labeled.csv",
        REPO_ROOT / "data" / "sf_crime_grid_full_labeled.csv",
        REPO_ROOT / "outputs" / "sf_crime_grid_full_labeled.csv",
    ]
    copy_first_existing(candidates, dst, mode="copy")


def quick_verify(crime_dir: Path) -> dict:
    expected = [
        "sf_crime.csv",
        "sf_crime_y.csv",
        "sf_population.csv",
        "sf_crime_01.csv", "sf_crime_02.csv", "sf_crime_03.csv", "sf_crime_04.csv", "sf_crime_05.csv",
        "sf_crime_06.csv", "sf_crime_07.csv", "sf_crime_08.csv", "sf_crime_09.csv",
        "sf_crime_grid_full_labeled.csv",
        "neighbors.csv",
        "sf_911_last_5_year.csv", "sf_911_last_5_year_y.csv",
        "sf_311_last_5_years.csv", "sf_311_last_5_years_y.csv",
        "sf_bus_stops_with_geoid.csv",
        "sf_train_stops_with_geoid.csv",
        "sf_pois_cleaned_with_geoid.csv",
        "sf_weather_5years.csv", "sf_weather_5years_y.csv",
        "week.csv",
    ]
    status = {}
    for f in expected:
        p = crime_dir / f
        status[f] = {
            "exists": p.exists(),
            "size": p.stat().st_size if p.exists() else 0,
        }
    return status


def run_pipeline(
    crime_dir: Path,
    persist: str,
    force_gate: bool,
    top_k: int,
    backfill_days: int,
    freshness_lag_days: int,
    wx_location: str,
    wx_unit: str,
) -> None:
    ok, msg = gate_07(force_gate)
    st.info(msg)
    if not ok:
        return

    env = os.environ.copy()
    env.update({
        "CRIME_DATA_DIR": str(crime_dir),
        "GEOID_LEN": env.get("GEOID_LEN", "11"),
        "BACKFILL_DAYS": str(backfill_days),
        "FRESHNESS_SF_MAX_LAG_DAYS": str(freshness_lag_days),
        "PATROL_TOP_K": str(top_k),
        "WX_LOCATION": wx_location,
        "WX_UNIT": wx_unit,
    })

    # 7-Day Forecast
    if (REPO_ROOT / "scripts" / "update_week_forecast.py").exists():
        run_cmd([sys.executable, "-u", "scripts/update_week_forecast.py"], REPO_ROOT, env, "7-Day Forecast → week.csv")

    # 00) Prefetch sf_crime_y.csv (yerelde: varsa kullan, yoksa geç)
    out_dir = crime_dir / DEFAULT_OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    if not (out_dir / "sf_crime_y.csv").exists():
        # repo root veya data/outputs'tan çekmeyi dene
        copy_first_existing(
            [REPO_ROOT / "sf_crime_y.csv", REPO_ROOT / "data" / "sf_crime_y.csv", REPO_ROOT / "outputs" / "sf_crime_y.csv"],
            out_dir / "sf_crime_y.csv",
            mode="copy"
        )

    # 01) update_crime.py
    run_cmd([sys.executable, "-u", "update_crime.py"], REPO_ROOT, env, "01) Crime base + grid")

    # 02) 911
    py_911 = "update_911.py" if (REPO_ROOT / "update_911.py").exists() else "scripts/update_911.py"
    if (REPO_ROOT / py_911).exists():
        run_cmd([sys.executable, "-u", py_911], REPO_ROOT, env, "02) 911")
        normalize_911(crime_dir)

    # 03) 311
    py_311 = "update_311.py" if (REPO_ROOT / "update_311.py").exists() else "scripts/update_311.py"
    if (REPO_ROOT / py_311).exists():
        run_cmd([sys.executable, "-u", py_311], REPO_ROOT, env, "03) 311")
        normalize_311(crime_dir)

    # 04) population
    if (REPO_ROOT / "update_population.py").exists():
        run_cmd([sys.executable, "-u", "update_population.py"], REPO_ROOT, env, "04) Population")

    # 05) bus
    if (REPO_ROOT / "update_bus.py").exists():
        run_cmd([sys.executable, "-u", "update_bus.py"], REPO_ROOT, env, "05) Bus")

    # 06) train
    if (REPO_ROOT / "update_train.py").exists():
        run_cmd([sys.executable, "-u", "update_train.py"], REPO_ROOT, env, "06) Train (BART)")

    # POI enrich (GEOID-only)
    if (REPO_ROOT / "update_poi.py").exists():
        run_cmd([sys.executable, "-u", "update_poi.py"], REPO_ROOT, env, "POI enrich → sf_crime_06.csv")
    elif (REPO_ROOT / "pipeline_make_sf_crime_06.py").exists():
        run_cmd([sys.executable, "-u", "pipeline_make_sf_crime_06.py"], REPO_ROOT, env, "POI enrich → sf_crime_06.csv")

    # 08) police & gov
    if (REPO_ROOT / "update_police_gov.py").exists():
        run_cmd([sys.executable, "-u", "update_police_gov.py"], REPO_ROOT, env, "08) Police & Gov")
    elif (REPO_ROOT / "scripts/enrich_police_gov_06_to_07.py").exists():
        run_cmd([sys.executable, "-u", "scripts/enrich_police_gov_06_to_07.py"], REPO_ROOT, env, "08) Police & Gov")

    # 09) weather
    if (REPO_ROOT / "update_weather.py").exists():
        run_cmd([sys.executable, "-u", "update_weather.py"], REPO_ROOT, env, "09) Weather")
    elif (REPO_ROOT / "scripts/update_weather.py").exists():
        run_cmd([sys.executable, "-u", "scripts/update_weather.py"], REPO_ROOT, env, "09) Weather")

    # grid normalize
    normalize_grid(crime_dir)

    # neighbors: varsa çalıştır
    if (REPO_ROOT / "scripts/make_neighbors.py").exists():
        run_cmd([sys.executable, "-u", "scripts/make_neighbors.py"], REPO_ROOT, env, "Neighbors: make_neighbors.py")
    elif (REPO_ROOT / "make_neighbors.py").exists():
        run_cmd([sys.executable, "-u", "make_neighbors.py"], REPO_ROOT, env, "Neighbors: make_neighbors.py")

    # quick verify
    st.subheader("✅ Quick verify")
    status = quick_verify(crime_dir)
    st.json(status)

    # persist
    if persist == "commit":
        st.warning("commit seçildi: Yerelde git commit/push yapmaya çalışacak.")
        run_cmd(["git", "status"], REPO_ROOT, env, "git status")
        run_cmd(["git", "add", "-A"], REPO_ROOT, env, "git add -A")
        run_cmd(["git", "commit", "-m", "Full pipeline output update"], REPO_ROOT, env, "git commit")
        run_cmd(["git", "push"], REPO_ROOT, env, "git push")
    else:
        st.info("persist=none (yerelde sadece üretim yapıldı).")


# =============================================================================
# 3) STREAMLIT UI
# =============================================================================
st.set_page_config(page_title="Full SF Crime Pipeline", layout="wide")

st.title("Full SF Crime Pipeline — app.py")

with st.sidebar:
    st.header("Run Options")
    persist = st.selectbox("persist", ["none", "commit"], index=0)
    force_gate = st.checkbox("force (07:00 kapısını bypass)", value=True)
    top_k = st.number_input("top_k (PATROL_TOP_K)", min_value=1, max_value=500, value=50, step=1)

    st.divider()
    st.subheader("Env overrides")
    crime_dir_str = st.text_input("CRIME_DATA_DIR", value=str(DEFAULT_DATA_DIR))
    backfill_days = st.number_input("BACKFILL_DAYS", min_value=0, max_value=365, value=0, step=1)
    freshness_lag_days = st.number_input("FRESHNESS_SF_MAX_LAG_DAYS", min_value=0, max_value=30, value=2, step=1)

    st.divider()
    st.subheader("Weather")
    wx_location = st.text_input("WX_LOCATION", value="San Francisco, CA")
    wx_unit = st.selectbox("WX_UNIT", ["us", "metric"], index=0)

crime_dir = Path(crime_dir_str).resolve()
st.caption(f"Repo root: `{REPO_ROOT}`")
st.caption(f"CRIME_DATA_DIR: `{crime_dir}`")
st.caption(f"SF time now: `{sf_now()}`")

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("🚀 Run pipeline", type="primary"):
        try:
            run_pipeline(
                crime_dir=crime_dir,
                persist=persist,
                force_gate=force_gate,
                top_k=int(top_k),
                backfill_days=int(backfill_days),
                freshness_lag_days=int(freshness_lag_days),
                wx_location=wx_location,
                wx_unit=wx_unit,
            )
            st.success("Pipeline finished.")
        except Exception as e:
            st.error(f"Pipeline failed: {e}")

with col2:
    if st.button("📂 Show CRIME_DATA_DIR listing"):
        if crime_dir.exists():
            files = sorted([p.name for p in crime_dir.glob("*")])
            st.write(files[:200])
        else:
            st.warning("CRIME_DATA_DIR yok.")

st.divider()
st.subheader("📄 Peek at key outputs (head)")
peek = [
    "sf_crime.csv", "sf_crime_y.csv",
    "sf_crime_01.csv", "sf_crime_02.csv", "sf_crime_03.csv",
    "sf_crime_04.csv", "sf_crime_05.csv", "sf_crime_06.csv",
    "sf_crime_07.csv", "sf_crime_08.csv", "sf_crime_09.csv",
    "sf_crime_grid_full_labeled.csv", "neighbors.csv",
    "week.csv",
]
for f in peek:
    p = crime_dir / f
    with st.expander(f"{f} — head"):
        st.code(head_file(p, n=5))
