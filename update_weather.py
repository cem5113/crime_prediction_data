# update_weather.py
from __future__ import annotations

from datetime import datetime, timedelta, date, timezone
from zoneinfo import ZoneInfo
import os
import pandas as pd
import numpy as np

# Opsiyonel: PyGithub ve Meteostat
try:
    from github import Github
except Exception:
    Github = None

try:
    from meteostat import Daily, Point
except Exception as e:
    print("⚠️ meteostat kurulu değilse yalnızca mevcut weather CSV'si ile devam ederiz:", e)
    Daily = None
    Point = None

pd.options.mode.copy_on_write = True

# =====================================================================================
# AYARLAR
# =====================================================================================
SF_TZ = ZoneInfo("America/Los_Angeles")

DATA_DIR = os.getenv("CRIME_DATA_DIR", "crime_prediction_data").rstrip("/")
WEATHER_CSV = os.getenv("WEATHER_CSV", os.path.join(DATA_DIR, "sf_weather_5years.csv"))

UPLOAD_WEATHER_TO_GH = os.getenv("UPLOAD_WEATHER_TO_GH", "0") in ("1", "true", "True")
PROBE_GH_STATUS = os.getenv("PROBE_GH_STATUS", "1") in ("1", "true", "True")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
REPO_NAME = os.getenv("REPO_NAME", "cem5113/crime_prediction_data")
WEATHER_TARGET_PATH = os.getenv("WEATHER_TARGET_PATH", f"{DATA_DIR}/sf_weather_5years.csv")

# Upload modu: force_update | skip_if_same
GH_UPLOAD_MODE = os.getenv("GH_UPLOAD_MODE", "skip_if_same").strip()

# Meteostat ayarları
LAT = float(os.getenv("WX_LAT", "37.7749"))
LON = float(os.getenv("WX_LON", "-122.4194"))
HOT_DAY_THRESHOLD_C = float(os.getenv("HOT_DAY_THRESHOLD_C", "30.0"))

ENRICH_CRIME_WITH_WEATHER = os.getenv("ENRICH_CRIME_WITH_WEATHER", "1") in ("1", "true", "True")

CRIME_IN_PATH = os.getenv("CRIME_IN_PATH", os.path.join(DATA_DIR, "sf_crime_07.parquet"))
CRIME_OUT_PATH = os.getenv("CRIME_OUT_PATH", os.path.join(DATA_DIR, "sf_crime_08.parquet"))
WRITE_CSV = os.getenv("WRITE_CSV", "0").strip().lower() in ("1", "true", "yes", "on")
CRIME_OUT_CSV = os.getenv("CRIME_OUT_CSV", os.path.join(DATA_DIR, "sf_crime_08.csv"))

CRIME_DATE_COL_CANDIDATES = [
    c.strip() for c in os.getenv(
        "CRIME_DATE_COL_CANDIDATES",
        "date,datetime,time"
    ).split(",") if c.strip()
]

# =====================================================================================
# TARİH PENCERESİ
# =====================================================================================
def five_year_window(today_: date) -> tuple[date, date]:
    try:
        start = today_.replace(year=today_.year - 5)
    except ValueError:
        start = today_ - timedelta(days=365 * 5 + 2)
    return start, today_

today = datetime.now(SF_TZ).date()
win_start, win_end = five_year_window(today)
print(f"📅 5Y Pencere: {win_start} → {win_end}")

# =====================================================================================
# YARDIMCILAR
# =====================================================================================
def to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.date

def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

def find_first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    low = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None

def nan_report(df: pd.DataFrame, title: str, only_cols: list[str] | None = None) -> None:
    x = df[only_cols].copy() if only_cols else df
    s = x.isna().sum()
    s = s[s > 0].sort_values(ascending=False)
    print(f"🔎 NaN sayıları ({title}):")
    if s.empty:
        print("✅ NaN yok.")
    else:
        print(s.to_string())

# =====================================================================================
# WEATHER NORMALIZATION + FEATURE ENGINEERING
# =====================================================================================
def normalize_weather_columns(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if d.empty:
        cols = [
            "date",
            "tavg", "tmin", "tmax", "prcp",
            "temp_range", "is_rainy", "is_hot_day",
            "prcp_lag1", "tavg_lag1", "tmax_lag1",
            "prcp_roll3", "prcp_roll7",
            "tavg_roll3", "tavg_roll7",
            "tmax_roll3", "tmax_roll7",
            "temp_anom_7d", "prcp_anom_7d",
            "rain_streak_3d", "hot_streak_3d"
        ]
        return pd.DataFrame(columns=cols)

    lmap = {c.lower(): c for c in d.columns}

    def has(c: str) -> bool:
        return c in lmap

    def col(c: str) -> str:
        return lmap[c]

    # tarih kolonu standardizasyonu
    if has("date"):
        d[col("date")] = to_date(d[col("date")])
    elif has("time"):
        d["date"] = to_date(d[col("time")])
    elif has("datetime"):
        d["date"] = to_date(d[col("datetime")])
    else:
        d["date"] = pd.NaT

    # kolon adlarını standardize et
    ren = {}
    if has("temp_min") and not has("tmin"):
        ren[col("temp_min")] = "tmin"
    if has("temp_max") and not has("tmax"):
        ren[col("temp_max")] = "tmax"
    if has("precipitation_mm") and not has("prcp"):
        ren[col("precipitation_mm")] = "prcp"
    if has("prcp_mm") and not has("prcp"):
        ren[col("prcp_mm")] = "prcp"
    if has("taverage") and not has("tavg"):
        ren[col("taverage")] = "tavg"
    d.rename(columns=ren, inplace=True)

    # numeric'e çevir
    for c in ["tavg", "tmin", "tmax", "prcp", "snow", "wspd", "pres"]:
        if c in d.columns:
            d[c] = safe_numeric(d[c])

    for c in ["tavg", "tmin", "tmax", "prcp"]:
        if c not in d.columns:
            d[c] = np.nan

    d["date"] = to_date(d["date"])
    d.dropna(subset=["date"], inplace=True)
    d = d.drop_duplicates(subset=["date"]).sort_values("date")
    d = d[(d["date"] >= win_start) & (d["date"] <= win_end)].copy()

    # temel feature'lar
    d["temp_range"] = (d["tmax"] - d["tmin"]).astype(float)
    d["is_rainy"] = (safe_numeric(d["prcp"]).fillna(0) > 0).astype("Int64")
    d["is_hot_day"] = (safe_numeric(d["tmax"]) > HOT_DAY_THRESHOLD_C).astype("Int64")

    d = d.sort_values("date").reset_index(drop=True)

    # lag
    d["prcp_lag1"] = d["prcp"].shift(1)
    d["tavg_lag1"] = d["tavg"].shift(1)
    d["tmax_lag1"] = d["tmax"].shift(1)

    # rolling
    d["prcp_roll3"] = d["prcp"].shift(1).rolling(3, min_periods=1).mean()
    d["prcp_roll7"] = d["prcp"].shift(1).rolling(7, min_periods=1).mean()
    
    d["tavg_roll3"] = d["tavg"].shift(1).rolling(3, min_periods=1).mean()
    d["tavg_roll7"] = d["tavg"].shift(1).rolling(7, min_periods=1).mean()
    
    d["tmax_roll3"] = d["tmax"].shift(1).rolling(3, min_periods=1).mean()
    d["tmax_roll7"] = d["tmax"].shift(1).rolling(7, min_periods=1).mean()
    
    d["temp_anom_7d"] = d["tavg"] - d["tavg_roll7"]
    d["prcp_anom_7d"] = d["prcp"] - d["prcp_roll7"]

    # streak
    d["rain_streak_3d"] = (
        d["is_rainy"].fillna(0).astype(int).rolling(3, min_periods=1).sum()
    ).astype(float)

    d["hot_streak_3d"] = (
        d["is_hot_day"].fillna(0).astype(int).rolling(3, min_periods=1).sum()
    ).astype(float)

    final_cols = [
        "date",
        "tavg", "tmin", "tmax", "prcp",
        "temp_range", "is_rainy", "is_hot_day",
        "prcp_lag1", "tavg_lag1", "tmax_lag1",
        "prcp_roll3", "prcp_roll7",
        "tavg_roll3", "tavg_roll7",
        "tmax_roll3", "tmax_roll7",
        "temp_anom_7d", "prcp_anom_7d",
        "rain_streak_3d", "hot_streak_3d"
    ]

    for c in final_cols:
        if c not in d.columns:
            d[c] = np.nan

    return d[final_cols]

def fetch_weather(lat: float, lon: float, start_d: date, end_d: date) -> pd.DataFrame:
    if Daily is None or Point is None:
        print("ℹ️ meteostat yok → boş DataFrame dönüyorum.")
        return normalize_weather_columns(pd.DataFrame())

    start_dt = datetime(start_d.year, start_d.month, start_d.day)
    end_dt = datetime(end_d.year, end_d.month, end_d.day)

    df = Daily(Point(lat, lon), start_dt, end_dt).fetch().reset_index()
    df.rename(columns={"time": "date"}, inplace=True)
    return normalize_weather_columns(df)

def read_existing_weather(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return normalize_weather_columns(pd.DataFrame())

    try:
        ex = pd.read_csv(path, low_memory=False)
        return normalize_weather_columns(ex)
    except Exception as e:
        print("⚠️ Mevcut weather dosyası okunamadı, baştan çekilecek:", e)
        return normalize_weather_columns(pd.DataFrame())

def fill_missing_prev_year_same_week(allw: pd.DataFrame) -> pd.DataFrame:
    """
    5Y pencere içinde tam tarih evreni kurar.
    Eksik günleri bir önceki yıl aynı ISO haftasının ortalamasıyla doldurur.
    Ardından edge durumda ffill/bfill uygular.
    """
    if allw.empty:
        return allw

    base_num_cols = ["tavg", "tmin", "tmax", "prcp"]

    full_dates = pd.date_range(pd.to_datetime(win_start), pd.to_datetime(win_end), freq="D")
    full_df = pd.DataFrame({"date": full_dates.date})

    out = full_df.merge(allw[["date"] + [c for c in allw.columns if c != "date"]], on="date", how="left")

    # mevcut veri üzerinden ISO haftalık ortalama
    base = allw[["date"] + base_num_cols].copy()
    base["date_ts"] = pd.to_datetime(base["date"])
    iso = base["date_ts"].dt.isocalendar()
    base["iso_year"] = iso.year.astype(int)
    base["iso_week"] = iso.week.astype(int)

    week_means = (
        base.groupby(["iso_year", "iso_week"], as_index=False)[base_num_cols]
        .mean()
        .rename(columns={c: f"mean_{c}" for c in base_num_cols})
    )

    out["date_ts"] = pd.to_datetime(out["date"])
    iso_out = out["date_ts"].dt.isocalendar()
    out["iso_year"] = iso_out.year.astype(int)
    out["iso_week"] = iso_out.week.astype(int)

    missing_mask = out[base_num_cols].isna().any(axis=1)
    out.loc[missing_mask, "prev_iso_year"] = out.loc[missing_mask, "iso_year"] - 1
    out.loc[missing_mask, "prev_iso_week"] = out.loc[missing_mask, "iso_week"]

    out = out.merge(
        week_means,
        left_on=["prev_iso_year", "prev_iso_week"],
        right_on=["iso_year", "iso_week"],
        how="left",
        suffixes=("", "_prev")
    )

    for c in base_num_cols:
        out[c] = out[c].fillna(out[f"mean_{c}"])

    remain = int(out[base_num_cols].isna().sum().sum())
    if remain > 0:
        print(f"⚠️ Prev-year same-week ile doldurulamayan NaN sayısı: {remain} → ffill/bfill uygulanacak")
        out = out.sort_values("date")
        out[base_num_cols] = out[base_num_cols].ffill().bfill()

        remain2 = int(out[base_num_cols].isna().sum().sum())
        if remain2 > 0:
            print(f"⚠️ ffill/bfill sonrası bile NaN kaldı: {remain2}")
        else:
            print("✅ Edge fallback sonrası NaN kalmadı.")
    else:
        print("✅ Tüm eksikler prev-year same-week ile dolduruldu.")

    # sadece temel kolonlarla normalize yeniden çalıştır
    out = out[["date"] + base_num_cols].copy()
    out = normalize_weather_columns(out)

    return out

# =====================================================================================
# CRIME MERGE
# =====================================================================================
def enrich_crime_with_weather(crime_path: str, out_path: str, weather_df: pd.DataFrame) -> None:
    if not os.path.exists(crime_path):
        print(f"⚠️ Crime dosyası yok, merge atlandı: {crime_path}")
        return

    if crime_path.lower().endswith(".parquet"):
        crime = pd.read_parquet(crime_path)
    else:
        crime = pd.read_csv(crime_path, low_memory=False)
    print(f"📊 CRIME (weather merge girdi): {crime.shape[0]} satır × {crime.shape[1]} sütun")

    dcol = find_first_existing_col(crime, CRIME_DATE_COL_CANDIDATES)
    if dcol is None:
        raise KeyError(f"❌ Crime içinde tarih kolonu bulunamadı. Denenenler: {CRIME_DATE_COL_CANDIDATES}")

    crime["_date_"] = pd.to_datetime(crime[dcol], errors="coerce").dt.date
    if crime["_date_"].isna().all():
        raise ValueError(f"❌ Crime tarih kolonu parse edilemedi: {dcol}")

    w = weather_df.copy()
    if "date" not in w.columns:
        raise KeyError("❌ Weather DF içinde 'date' yok.")

    w["date"] = pd.to_datetime(w["date"], errors="coerce").dt.date
    w = w.dropna(subset=["date"]).drop_duplicates("date").sort_values("date")

    weather_cols_all = [c for c in w.columns if c != "date"]
    wcols = ["date"] + weather_cols_all
    w = w[wcols].copy()

    before = crime.shape

    # eski weather kolonları varsa temizle
    to_drop = [c for c in weather_cols_all if c in crime.columns]
    if to_drop:
        crime = crime.drop(columns=to_drop, errors="ignore")

    w2 = w.rename(columns={"date": "wx_date"}).copy()
    out = crime.merge(w2, left_on="_date_", right_on="wx_date", how="left", validate="m:1")

    coverage_cols = [c for c in ["tavg", "tmax", "prcp", "temp_anom_7d", "prcp_roll7"] if c in out.columns]
    cov_parts = [f"{c}={out[c].notna().mean():.3%}" for c in coverage_cols]
    print("🧪 WX coverage: " + (" | ".join(cov_parts) if cov_parts else "raporlanacak kolon yok"))

    out.drop(columns=["wx_date"], errors="ignore", inplace=True)
    out.drop(columns=["_date_"], errors="ignore", inplace=True)

    print(
        f"🔗 CRIME ⨯ WEATHER (date-merge): "
        f"{before[0]}×{before[1]} → {out.shape[0]}×{out.shape[1]} "
        f"(Δr={out.shape[0] - before[0]}, Δc={out.shape[1] - before[1]})"
    )

    new_weather_cols = [c for c in weather_cols_all if c in out.columns]
    nan_report(out, "sf_crime_08 yazılmadan önce (tüm kolonlar)")
    if new_weather_cols:
        nan_report(out, "Weather kolonları (sf_crime_08 yazılmadan önce)", only_cols=new_weather_cols)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    tmp_parquet = out_path + ".tmp.parquet"
    out.to_parquet(
        tmp_parquet,
        index=False,
        engine="pyarrow",
        compression="snappy"
    )
    os.replace(tmp_parquet, out_path)
    print(f"✅ Weather eklendi → {out_path} | Satır: {len(out):,} | Sütun: {out.shape[1]}")
    
    if WRITE_CSV:
        out.to_csv(CRIME_OUT_CSV, index=False)
        print(f"✅ CSV de yazıldı → {CRIME_OUT_CSV}")
    else:
        print("ℹ️ CSV yazımı kapalı; yalnız parquet kaydedildi.")

# =====================================================================================
# GITHUB YARDIMCILARI
# =====================================================================================
def _get_repo():
    if Github is None or not GITHUB_TOKEN:
        return None
    try:
        return Github(GITHUB_TOKEN).get_repo(REPO_NAME)
    except Exception as e:
        print("⚠️ Repo erişimi başarısız:", e)
        return None

def github_file_status(path: str):
    repo = _get_repo()
    if repo is None:
        return {
            "exists": False,
            "size": None,
            "sha": None,
            "html_url": None,
            "last_commit_iso": None,
            "content": None
        }

    try:
        contents = repo.get_contents(path)
        commits = repo.get_commits(path=path)
        last_iso = None
        try:
            c = next(iter(commits))
            dt = getattr(c.commit.author, "date", None)
            if isinstance(dt, datetime):
                last_iso = dt.astimezone(timezone.utc).isoformat()
        except Exception:
            pass

        return {
            "exists": True,
            "size": getattr(contents, "size", None),
            "sha": getattr(contents, "sha", None),
            "html_url": getattr(contents, "html_url", None),
            "last_commit_iso": last_iso,
            "content": contents.decoded_content.decode("utf-8", errors="ignore"),
        }
    except Exception:
        return {
            "exists": False,
            "size": None,
            "sha": None,
            "html_url": None,
            "last_commit_iso": None,
            "content": None
        }

def upsert_github_csv_smart(df: pd.DataFrame, target_path: str):
    repo = _get_repo()
    if repo is None:
        print("ℹ️ GitHub upload atlandı (token veya PyGithub yok).")
        return

    csv_str = df.to_csv(index=False)
    status = github_file_status(target_path)

    if PROBE_GH_STATUS:
        if status["exists"]:
            print(f"🔎 GH Durum: VAR — {target_path} (boyut={status['size']}, son_commit={status['last_commit_iso']})")
        else:
            print(f"🔎 GH Durum: YOK — {target_path}")

    if not UPLOAD_WEATHER_TO_GH:
        return

    same = (status["content"] == csv_str) if status["exists"] and status["content"] is not None else False

    if status["exists"]:
        if GH_UPLOAD_MODE == "skip_if_same" and same:
            print("✅ GH güncel: içerik aynı, update atlandı.")
            return
        try:
            repo.update_file(
                status["html_url"].split("blob/")[-1].split("/", 1)[-1] if status["html_url"] else target_path,
                f"update {os.path.basename(target_path)}",
                csv_str,
                status["sha"],
                branch="main"
            )
            print(f"✅ GitHub güncellendi: {target_path}")
        except Exception:
            contents = repo.get_contents(target_path)
            repo.update_file(contents.path, f"update {os.path.basename(target_path)}", csv_str, contents.sha, branch="main")
            print(f"✅ GitHub güncellendi: {target_path}")
    else:
        repo.create_file(target_path, f"add {os.path.basename(target_path)}", csv_str, branch="main")
        print(f"🆕 GitHub oluşturuldu: {target_path}")

# =====================================================================================
# WEATHER DF CACHE
# =====================================================================================
_WEATHER_LATEST: pd.DataFrame | None = None

def get_weather_df() -> pd.DataFrame:
    global _WEATHER_LATEST
    if _WEATHER_LATEST is not None:
        return _WEATHER_LATEST

    if os.path.exists(WEATHER_CSV):
        try:
            df = pd.read_csv(WEATHER_CSV, low_memory=False)
            _WEATHER_LATEST = normalize_weather_columns(df)
            return _WEATHER_LATEST
        except Exception:
            pass

    _WEATHER_LATEST = normalize_weather_columns(pd.DataFrame())
    return _WEATHER_LATEST

# =====================================================================================
# WEATHER GÜNCELLE
# =====================================================================================
existing = read_existing_weather(WEATHER_CSV)
last_date = existing["date"].max() if not existing.empty else None
fetch_start = (last_date + timedelta(days=1)) if pd.notna(last_date) else win_start
fetch_end = win_end

if fetch_start <= fetch_end:
    print(f"📥 Meteostat Daily: {fetch_start} → {fetch_end}")
    neww = fetch_weather(LAT, LON, fetch_start, fetch_end)
    print(f"✅ Yeni gün sayısı: {len(neww)}")
    allw = pd.concat([existing, neww], ignore_index=True) if not existing.empty else neww.copy()
else:
    print("ℹ️ Weather güncel; indirilecek yeni gün yok.")
    allw = existing.copy()

allw = normalize_weather_columns(allw)
allw = allw.drop_duplicates(subset=["date"]).sort_values("date")
allw = allw[(allw["date"] >= win_start) & (allw["date"] <= win_end)].copy()

# eksik gün doldur
allw = fill_missing_prev_year_same_week(allw)

cov = allw["tavg"].notna().mean() if "tavg" in allw.columns else 0.0
print(f"🧪 WX table coverage (tavg notna): {cov:.3%}")

nan_counts = allw.isna().sum()
nan_counts = nan_counts[nan_counts > 0].sort_values(ascending=False)

print("🔎 NaN sayıları (sf_weather_5years yazılmadan önce):")
if nan_counts.empty:
    print("✅ NaN yok.")
else:
    print(nan_counts.to_string())

wx_cols = [c for c in allw.columns if c != "date"]
if wx_cols:
    print("🔎 Weather kolonları NaN sayıları:")
    print(allw[wx_cols].isna().sum().to_string())

os.makedirs(os.path.dirname(WEATHER_CSV), exist_ok=True)
allw.to_csv(WEATHER_CSV, index=False, encoding="utf-8", lineterminator="\n")
print(f"💾 Weather kaydedildi: {WEATHER_CSV} — {len(allw)} satır, {allw['date'].min()} → {allw['date'].max()}")

# cache
_WEATHER_LATEST = allw.copy()

# GitHub
if Github is not None and (PROBE_GH_STATUS or UPLOAD_WEATHER_TO_GH):
    upsert_github_csv_smart(allw, WEATHER_TARGET_PATH)

# Crime merge
if ENRICH_CRIME_WITH_WEATHER:
    try:
        enrich_crime_with_weather(CRIME_IN_PATH, CRIME_OUT_PATH, _WEATHER_LATEST)

        if os.path.exists(CRIME_OUT_PATH):
            print("sf_crime_08.parquet — ilk 5 satır:", flush=True)
            try:
                _h = pd.read_parquet(CRIME_OUT_PATH).head(5)
                print(_h.to_string(index=False), flush=True)
            except Exception as e:
                print(f"⚠️ sf_crime_08 head okunamadı: {e}", flush=True)

    except Exception as e:
        print(f"❌ Crime-weather merge hatası: {e}")
else:
    print("ℹ️ ENRICH_CRIME_WITH_WEATHER=0 → Crime merge atlandı.")
