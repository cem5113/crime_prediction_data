# scripts/update_311.py
import os, re, time, requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from datetime import datetime, timedelta
from urllib.parse import quote
from pathlib import Path

# ---- TZ (opsiyonel) ---------------------------------------------------------
try:
    import zoneinfo
    SF_TZ = zoneinfo.ZoneInfo("America/Los_Angeles")
except Exception:
    SF_TZ = None

def _to_date_series(x):
    """UTC -> SF yerel tarihe dönüştür (varsa); olmazsa naive tarihe düş."""
    try:
        s = pd.to_datetime(x, utc=True, errors="coerce")
        if SF_TZ is not None:
            s = s.tz_convert(SF_TZ)
        return s.dt.date.dropna()
    except Exception:
        return pd.to_datetime(x, errors="coerce").dt.date.dropna()

def log_shape(df, label):
    r, c = df.shape
    print(f"📊 {label}: {r} satır × {c} sütun")

def log_merge_delta(before_shape, after_shape, label):
    br, bc = before_shape
    ar, ac = after_shape
    print(f"🔗 {label}: {br}×{bc} → {ar}×{ac} (Δr={ar-br}, Δc={ac-bc})")

# ---- GEOID normalize ---------------------------------------------------------
DEFAULT_GEOID_LEN = int(os.getenv("GEOID_LEN", "11"))

def normalize_geoid(series, target_len: int | None = None):
    L = int(target_len or DEFAULT_GEOID_LEN)
    s = series.astype("string")
    s = s.str.extract(r"(\d+)", expand=False)
    s = s.str.slice(0, L)
    return s.str.zfill(L)

def normalize_geoid_11(x):
    if pd.isna(x):
        return pd.NA
    digits = re.sub(r"\D", "", str(x))
    return digits[:11] if len(digits) >= 11 else pd.NA

def save_atomic(df, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)

def is_lfs_pointer_file(p: Path) -> bool:
    """Git LFS pointer dosyası mı? (gerçek CSV değilse eler)"""
    try:
        return "git-lfs.github.com/spec/v1" in p.read_text(errors="ignore")[:200]
    except Exception:
        return False

# ================== AYARLAR ==================
SAVE_DIR = os.getenv("CRIME_DATA_DIR", "crime_prediction_data")
os.makedirs(SAVE_DIR, exist_ok=True)

# 🔴 Adlandırma standardı
# - Ham 5y kayıt:             sf_311_last_5_years_y.csv
# - 3 saatlik özet (3h bin):  sf_311_last_5_years.csv  (alias: sf_311_last_5_years_3h.csv)
RAW_311_NAME_Y = os.getenv("RAW_311_NAME_Y", "sf_311_last_5_years_y.csv")
AGG_BASENAME   = os.getenv("AGG_311_NAME",   "sf_311_last_5_years.csv")
AGG_ALIAS      = os.getenv("AGG_311_ALIAS",  "sf_311_last_5_years_3h.csv")

# Eski ad uyumluluğu (opsiyonel kopya)
LEGACY_311_Y = os.getenv("LEGACY_311_Y", "sf_311_last_5_year_y.csv")
LEGACY_311   = os.getenv("LEGACY_311",   "sf_311_last_5_year.csv")

# Socrata dataset
DATASET_BASE = os.getenv("SF311_DATASET", "https://data.sfgov.org/resource/vw6y-z8j6.json")
SOCRATA_APP_TOKEN = os.getenv("SOCS_APP_TOKEN", "").strip()

# GeoJSON adayları
GEOJSON_NAME = os.getenv("SF_BLOCKS_GEOJSON", "sf_census_blocks.geojson")
GEOJSON_CANDIDATES = [
    os.path.join(SAVE_DIR, GEOJSON_NAME),
    os.path.join("crime_prediction_data", GEOJSON_NAME),
    os.path.join(".", GEOJSON_NAME),
]

# İndirme/bölütleme ayarları
PAGE_LIMIT      = int(os.getenv("SF_SODA_PAGE_LIMIT", "50000"))
MAX_PAGES       = int(os.getenv("SF_SODA_MAX_PAGES", "100"))
SLEEP_SEC       = float(os.getenv("SF_SODA_THROTTLE_SEC", "0.25"))
SODA_TIMEOUT    = int(os.getenv("SF_SODA_TIMEOUT", "90"))
SODA_RETRIES    = int(os.getenv("SF_SODA_RETRIES", "5"))

# Chunk modu: tarih aralığına böl (timeout’lara karşı daha dayanıklı)
CHUNK_DAYS              = int(os.getenv("SF311_CHUNK_DAYS", "31"))    # ~aylık
MAX_PAGES_PER_CHUNK     = int(os.getenv("SF311_MAX_PAGES_PER_CHUNK", "40"))
MAX_CONSEC_EMPTY_CHUNKS = int(os.getenv("SF311_MAX_EMPTY_CHUNKS", "8"))  # çok boş geliyorsa erken çık

# Pencere: varsayılan 5 yıl veya BACKFILL_DAYS override
FIVE_YEARS     = 5 * 365
TODAY          = datetime.utcnow().date()
DEFAULT_START  = TODAY - timedelta(days=FIVE_YEARS)
BACKFILL_DAYS  = int(os.getenv("BACKFILL_DAYS", "0"))

# ================== SOCRATA ==================
def socrata_get(session: requests.Session, url, params):
    headers = {"Accept": "application/json"}
    if SOCRATA_APP_TOKEN:
        headers["X-App-Token"] = SOCRATA_APP_TOKEN

    last_err = None
    for i in range(SODA_RETRIES):
        try:
            r = session.get(url, params=params, headers=headers, timeout=SODA_TIMEOUT)
            if r.status_code in (408, 429) or 500 <= r.status_code < 600:
                raise requests.HTTPError(f"status={r.status_code}")
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            sleep_s = max(SLEEP_SEC, SLEEP_SEC * (2 ** i))
            print(f"⚠️ Socrata retry {i+1}/{SODA_RETRIES} ({e}); {sleep_s:.1f}s bekleme…")
            time.sleep(sleep_s)
    raise last_err

# ================== GEO ==================
def ensure_blocks_gdf():
    for cand in GEOJSON_CANDIDATES:
        if os.path.exists(cand):
            gdf = gpd.read_file(cand)
            if "GEOID" not in gdf.columns:
                possible = [c for c in gdf.columns if str(c).upper().startswith("GEOID")]
                if not possible:
                    continue
                gdf["GEOID"] = gdf[possible[0]].astype(str)
            gdf["TRACT11"] = gdf["GEOID"].apply(normalize_geoid_11)
            gdf = gdf[["TRACT11", "geometry"]].dropna(subset=["TRACT11"])
            if gdf.crs is None:
                gdf.set_crs("EPSG:4326", inplace=True)
            elif str(gdf.crs).lower() not in ("epsg:4326", "wgs84", "wgs 84"):
                gdf = gdf.to_crs(epsg=4326)
            print(f"🧭 GEOJSON kullanılıyor: {os.path.abspath(cand)}")
            return gdf
    print("⚠️ GEOJSON bulunamadı; GEOID eşleme yapılamayacak (stub).")
    return None

def geotag_to_geoid11(df_new):
    df_new = df_new.copy()
    if "latitude" not in df_new.columns and "lat" in df_new.columns:
        df_new["latitude"] = pd.to_numeric(df_new["lat"], errors="coerce")
    if "longitude" not in df_new.columns and "long" in df_new.columns:
        df_new["longitude"] = pd.to_numeric(df_new["long"], errors="coerce")

    df_new = df_new.dropna(subset=["latitude", "longitude"])
    if df_new.empty:
        df_new["GEOID"] = pd.NA
        return df_new

    gdf_blocks = ensure_blocks_gdf()
    if gdf_blocks is None:
        df_new["GEOID"] = pd.NA
        return df_new

    gdf_pts = gpd.GeoDataFrame(
        df_new,
        geometry=gpd.points_from_xy(df_new["longitude"], df_new["latitude"]),
        crs="EPSG:4326",
    )
    try:
        gdf_join = gpd.sjoin(gdf_pts, gdf_blocks, how="left", predicate="within")
    except Exception:
        try:
            gdf_join = gpd.sjoin_nearest(gdf_pts, gdf_blocks, how="left", max_distance=5)
        except Exception:
            gdf_pts["GEOID"] = pd.NA
            return pd.DataFrame(gdf_pts.drop(columns=["geometry"]))
    out = pd.DataFrame(gdf_join.drop(columns=["geometry"]))
    out.rename(columns={"TRACT11": "GEOID"}, inplace=True)
    out["GEOID"] = out["GEOID"].astype(str).str.extract(r"(\d+)")[0].str[:11]
    return out

# ================== YARDIMCI: şema tespiti & tohum yükleme ==================
def _looks_like_raw_311(cols: list[str]) -> bool:
    lc = {c.lower() for c in cols}
    # raw için karakteristik alanlar
    return any(x in lc for x in ["id", "service_request_id"]) and \
           any(x in lc for x in ["time", "requested_datetime"]) and \
           any(x in lc for x in ["latitude", "lat"]) and \
           "311_request_count" not in lc

def _load_raw_seed_from_base(base_csv_path: str) -> pd.DataFrame:
    """Repo'daki sf_311_last_5_years.csv ham ise seed olarak yükle."""
    try:
        df = pd.read_csv(base_csv_path, low_memory=False)
    except Exception as e:
        print(f"⚠️ Base CSV okunamadı ({base_csv_path}): {e}")
        return pd.DataFrame()
    if not _looks_like_raw_311(list(df.columns)):
        print(f"ℹ️ {base_csv_path} özet (3h) gibi görünüyor; ham seed olarak kullanılamaz.")
        return pd.DataFrame()

    # alan adlarını normalize et
    rename_map = {}
    if "service_request_id" in df.columns:
        rename_map["service_request_id"] = "id"
    if "service_name" in df.columns:
        rename_map["service_name"] = "category"
    if "service_subtype" in df.columns:
        rename_map["service_subtype"] = "subcategory"
    if rename_map:
        df = df.rename(columns=rename_map)

    # datetime / date / time kur
    if "datetime" not in df.columns:
        if "requested_datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["requested_datetime"], errors="coerce", utc=True)
        elif {"date","time"}.issubset(df.columns):
            df["datetime"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str), errors="coerce", utc=True)
        else:
            df["datetime"] = pd.NaT
    if "date" not in df.columns:
        df["date"] = pd.to_datetime(df["datetime"], errors="coerce").dt.date
    if "time" not in df.columns:
        df["time"] = pd.to_datetime(df["datetime"], errors="coerce").dt.time

    # kolon setini tamamla
    keep = ["id","datetime","date","time","lat","long","category","subcategory",
            "agency_responsible","latitude","longitude"]
    for c in keep:
        if c not in df.columns:
            df[c] = pd.NA

    log_shape(df, "Base CSV (ham seed)")
    return df[keep + ["GEOID"] if "GEOID" in df.columns else keep].copy()

# ================== DOSYA YOLLARI ==================
def resolve_existing_raw_path():
    """
    Actions/artifact klasör adı değişse bile _y ham 311 dosyasını bulur.
    Bulamazsa SAVE_DIR altında oluşturulacak yolu döndürür.
    """
    ARTIFACT_NAME = os.getenv("ARTIFACT_NAME", "sf-crime-pipeline-output").strip()
    target_names = [RAW_311_NAME_Y, LEGACY_311_Y]  # önce yeni isim, sonra legacy

    def _ok(p: Path) -> bool:
        if not p or not p.exists() or p.is_dir():
            return False
        if p.suffix.lower() != ".csv":
            return False
        if is_lfs_pointer_file(p):
            return False
        try:
            if p.stat().st_size < 200:  # aşırı küçükse şüpheli
                return False
        except Exception:
            return False
        return True

    # Aranacak kök klasörler
    roots = []
    try:
        roots.append(Path(SAVE_DIR).resolve())
    except Exception:
        roots.append(Path(SAVE_DIR))
    roots += [Path.cwd(), Path(".")]

    artifact_dir = Path(ARTIFACT_NAME)
    if artifact_dir.exists() and artifact_dir.is_dir():
        roots.append(artifact_dir)

    # sf-crime-pipeline-output* gibi açılan klasörleri de tara
    for r in [Path.cwd(), Path(".")]:
        try:
            for d in r.glob("sf-crime-pipeline-output*"):
                if d.is_dir():
                    roots.append(d)
        except Exception:
            pass

    # 1) Deterministik: direkt bakılacak yerler
    for nm in target_names:
        for rt in roots:
            for cand in [
                rt / nm,
                rt / "crime_prediction_data" / nm,
                rt / "outputs" / nm,
            ]:
                if _ok(cand):
                    print(f"🔎 Mevcut 311 _y CSV bulundu: {cand.resolve()}")
                    return str(cand)

    # 2) Son çare: recursive arama (artifact içinde farklı yere konmuşsa)
    for nm in target_names:
        for rt in roots:
            try:
                for found in rt.rglob(nm):
                    if _ok(found):
                        print(f"🔎 Mevcut 311 _y CSV bulundu (rglob): {found.resolve()}")
                        return str(found)
            except Exception:
                continue

    # 3) Bulunamadı → SAVE_DIR altında oluştur
    preferred = Path(SAVE_DIR) / RAW_311_NAME_Y
    print(f"ℹ️ Mevcut 311 ham CSV yok; oluşturulacak: {preferred.resolve()}")
    return str(preferred)

def _short(p: str) -> str:
    try:
        return os.path.relpath(p)
    except Exception:
        return p

def load_existing_raw(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, dtype={"GEOID": str}, low_memory=False)
    if "index_right" in df.columns:
        df = df.drop(columns=["index_right"])
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    elif "date" in df.columns and "time" in df.columns:
        df["datetime"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str), errors="coerce", utc=True)
    else:
        df["datetime"] = pd.NaT
    if "date" not in df.columns:
        df["date"] = pd.to_datetime(df["datetime"], errors="coerce").dt.date
    for c in ["id","lat","long","category","subcategory","agency_responsible","latitude","longitude","GEOID","time"]:
        if c not in df.columns:
            df[c] = pd.NA
    mx = pd.to_datetime(df["datetime"], errors="coerce").max()
    print(f"📁 Mevcut satır: {len(df):,} | max datetime={mx}")
    return df

def load_existing_raw_or_seed(raw_path: str) -> pd.DataFrame:
    # 1) artifact / mevcut ham dosya
    if raw_path and os.path.exists(raw_path):
        df = load_existing_raw(raw_path)
        if df is not None and not df.empty:
            return df

    # 2) base seed: ham olabilecek adaylar (öncelik: repo'daki sf_311_last_5_years.csv)
    seed_candidates = [
        os.path.join(SAVE_DIR, "sf_311_last_5_years.csv"),     # ✅ SENİN DEDİĞİN
        os.path.join(SAVE_DIR, "sf_311_last_5_years_3h.csv"),  # (varsa)
        os.path.join(SAVE_DIR, AGG_BASENAME),
        os.path.join(SAVE_DIR, AGG_ALIAS),
        os.path.join(SAVE_DIR, RAW_311_NAME_Y),
        os.path.join(SAVE_DIR, LEGACY_311_Y),
        os.path.join(SAVE_DIR, LEGACY_311),
    ]
    # boş/None'ları temizle
    seed_candidates = [p for p in seed_candidates if p and isinstance(p, str)]

    for cand in seed_candidates:
        if os.path.exists(cand):
            df_seed = _load_raw_seed_from_base(cand)
            if df_seed is not None and not df_seed.empty:
                print(f"🌱 Seed (base ham) kullanıldı: {os.path.abspath(cand)}")
                return df_seed

    # 3) hiçbir şey yok
    print("ℹ️ Ham 311 bulunamadı; boş seed ile başlanıyor.")
    return pd.DataFrame()

def decide_start_date(df_existing):
    if BACKFILL_DAYS > 0:
        start = TODAY - timedelta(days=BACKFILL_DAYS)
        print(f"📌 Mod: backfill | start={start}")
        return start, "backfill"
    if df_existing.empty or not df_existing["datetime"].notna().any():
        print(f"📌 Mod: full-5y (dosya yok/boş) | window ≥ {DEFAULT_START}")
        return DEFAULT_START, "full-5y"
    last = pd.to_datetime(df_existing["datetime"], errors="coerce").max().date()
    start = last + timedelta(days=1)
    print(f"📌 Mod: incremental | start={start} | window ≥ {DEFAULT_START}")
    return start, "incremental"

# ================== İNDİRME (TARİH CHUNK) ==================
def download_by_date_chunks(start_date):
    """
    5 yıl gibi geniş aralıkları offset yerine tarih parçalara bölerek indir.
    Her chunk’ta yine sayfalama var; chunk başarısızsa retry sonrası pas geçilir.
    """
    print(f"🧩 İndirme modu: DATE-CHUNKS ({CHUNK_DAYS}gün) + paging")
    session = requests.Session()
    police_filter = "(agency_responsible like '%Police%' OR agency_responsible like '%SFPD%')"
    cols = ",".join([
        "service_request_id","requested_datetime",
        "lat","long",
        "service_name","service_subtype","agency_responsible"
    ])

    all_chunks = []
    consec_empty = 0
    cur = start_date
    end = TODAY

    while cur <= end:
        chunk_end = min(cur + timedelta(days=CHUNK_DAYS - 1), end)
        start_iso = f"{cur.isoformat()}T00:00:00.000"
        end_iso   = f"{chunk_end.isoformat()}T23:59:59.999"
        where = f"$where=requested_datetime between '{start_iso}' and '{end_iso}' AND {police_filter}"
        print(f"⛏️  {cur} → {chunk_end} aralığı çekiliyor…")

        offset = 0
        pages = 0
        chunk_rows = []

        while True:
            params = {
                "$select": cols,
                "$where": f"requested_datetime between '{start_iso}' and '{end_iso}' AND {police_filter}",
                "$order": "requested_datetime ASC",
                "$limit": PAGE_LIMIT,
                "$offset": offset
            }
        
            try:
                data = socrata_get(session, DATASET_BASE, params)
            except Exception as e:
                print(f"❌ Chunk hata ( {cur}→{chunk_end} , offset={offset} ): {e} → chunk geçiliyor.")
                break
        
            df = pd.DataFrame(data)
            if df.empty:
                break

            if pages == 0:
                print("   • kolonlar:", list(df.columns))
            chunk_rows.append(df)
            offset += len(df)
            pages  += 1
            print(f"   + {offset} kayıt (sayfa={pages})")

            if len(df) < PAGE_LIMIT or pages >= MAX_PAGES_PER_CHUNK:
                if pages >= MAX_PAGES_PER_CHUNK:
                    print(f"   ↪️ MAX_PAGES_PER_CHUNK={MAX_PAGES_PER_CHUNK} doldu, chunk kesildi.")
                break

            time.sleep(SLEEP_SEC)

        if chunk_rows:
            consec_empty = 0
            all_chunks.append(pd.concat(chunk_rows, ignore_index=True))
            print(f"✅ Chunk bitti: satır={sum(len(x) for x in chunk_rows)}")
        else:
            consec_empty += 1
            print(f"ℹ️ Chunk boş döndü (ardışık boş={consec_empty}).")
            if consec_empty >= MAX_CONSEC_EMPTY_CHUNKS and cur > start_date:
                print("⏹️ Çok sayıda ardışık boş chunk; erken durdurma.")
                break

        cur = chunk_end + timedelta(days=1)
        time.sleep(SLEEP_SEC)

    return pd.concat(all_chunks, ignore_index=True) if all_chunks else pd.DataFrame()

# ================== ANA ==================
def main():
    print("🔎 CWD:", os.getcwd())
    print("🔎 Tercih edilen SAVE_DIR:", os.path.abspath(SAVE_DIR))

    # 1) Mevcut ham dosya (artifact’tan gelmiş olabilir) veya base’den seed
    raw_path = resolve_existing_raw_path()
    agg_path = os.path.join(os.path.dirname(raw_path) or ".", AGG_BASENAME)
    df_raw = load_existing_raw_or_seed(raw_path)

    # 2) Başlangıç tarihi
    start_date, _mode = decide_start_date(
        load_existing_raw(raw_path) if os.path.exists(raw_path) else df_raw
    )

    # 3) Yeni veriyi indir (tarih-chunk)
    df_new = download_by_date_chunks(start_date)
    if df_new.empty:
        print("ℹ️ Yeni 311 kaydı bulunamadı (veya erişilemedi).")
    else:
        print(f"➕ Yeni indirilen: {len(df_new):,}")
        df_new = df_new.rename(columns={
            "service_request_id": "id",
            "requested_datetime": "datetime",
            "service_name": "category",
            "service_subtype": "subcategory"
        })
        df_new["datetime"] = pd.to_datetime(df_new["datetime"], errors="coerce", utc=True)
        df_new["date"] = pd.to_datetime(df_new["datetime"]).dt.date
        df_new["time"] = pd.to_datetime(df_new["datetime"]).dt.time

        # GEOID (mümkünse)
        df_new_geo = geotag_to_geoid11(df_new)

        keep = ["id","datetime","date","time","lat","long","category","subcategory",
                "agency_responsible","latitude","longitude","GEOID"]
        for c in keep:
            if c not in df_new_geo.columns:
                df_new_geo[c] = pd.NA
        df_new_geo = df_new_geo[keep]
        df_new_geo["GEOID"] = normalize_geoid(df_new_geo["GEOID"], DEFAULT_GEOID_LEN)

        if df_raw is None or df_raw.empty:
            df_raw = df_new_geo
        else:
            df_raw = pd.concat([df_raw, df_new_geo], ignore_index=True)

    # 4) Tekilleştir + pencere + sırala + KAYDET (ham, *_y*)
    if not df_raw.empty:
        df_raw["GEOID"] = normalize_geoid(df_raw["GEOID"], DEFAULT_GEOID_LEN)
        df_raw["id"] = df_raw["id"].astype(str)
        df_raw.drop_duplicates(subset=["id"], keep="last", inplace=True)

        df_raw["date"] = pd.to_datetime(df_raw["date"], errors="coerce").dt.date
        min_date = start_date if BACKFILL_DAYS > 0 else DEFAULT_START
        df_raw = df_raw[df_raw["date"] >= min_date]

        df_raw["datetime"] = pd.to_datetime(df_raw["datetime"], errors="coerce", utc=True)
        df_raw.sort_values("datetime", inplace=True)

        save_atomic(df_raw, raw_path)  # << artifact adı
        print(f"✅ Ham (5y/chunk) kaydedildi: {os.path.abspath(raw_path)}")

        # Uyumluluk kopyaları (workflow eski adları arıyor olabilir)
        try:
            save_atomic(df_raw, os.path.join(SAVE_DIR, RAW_311_NAME_Y))
            save_atomic(df_raw, os.path.join(SAVE_DIR, LEGACY_311_Y))
            save_atomic(df_raw, os.path.join(SAVE_DIR, LEGACY_311))
        except Exception as e:
            print(f"⚠️ Legacy kopya yazım uyarısı: {e}")

        mx = pd.to_datetime(df_raw["datetime"], errors="coerce").max()
        print(f"🧪 Ham Satır: {len(df_raw):,} | Son tarih: {mx.date() if pd.notna(mx) else 'NA'}")
        try:
            print(df_raw.head(3).to_string(index=False))
        except Exception:
            pass
    else:
        print("⚠️ Ham veri boş.")
        # Şemalı boşlar (artifact uyumu)
        empty_raw_cols = ["id","datetime","date","time","lat","long",
                          "category","subcategory","agency_responsible","latitude","longitude","GEOID"]
        for p in [RAW_311_NAME_Y, LEGACY_311_Y, LEGACY_311]:
            save_atomic(pd.DataFrame(columns=empty_raw_cols), os.path.join(SAVE_DIR, p))
        empty_agg_cols = ["GEOID","date","hour_range","311_request_count"]
        for p in [AGG_BASENAME, AGG_ALIAS]:
            if p:
                save_atomic(pd.DataFrame(columns=empty_agg_cols), os.path.join(SAVE_DIR, p))
        print("ℹ️ Şemalı boş 311 ham/özet dosyaları yazıldı.")

    # 5) 3 SAATLİK ÖZET (sf_311_last_5_years.csv + alias)
    if not df_raw.empty:
        df_ok = df_raw.dropna(subset=["date"]).copy()
    
        # GEOID yoksa özet üretilemez; uyarı ver, boş şemalı özet yaz
        if "GEOID" not in df_ok.columns or df_ok["GEOID"].isna().all():
            print("⚠️ GEOID üretilemedi; özet boş yazılacak.")
            grouped = pd.DataFrame(columns=["GEOID","date","hour_range","311_request_count"])
        else:
            h = pd.to_datetime(df_ok["datetime"], errors="coerce", utc=True).dt.hour.fillna(0).astype(int)
            start_h = (h // 3) * 3
            end_h = start_h + 3
            end_h = end_h.where(end_h < 24, 24)  # 21-24 fix
    
            df_ok["hour_range"] = (
                start_h.astype(int).astype(str).str.zfill(2) + "-" +
                end_h.astype(int).astype(str).str.zfill(2)
            )
    
            grouped = (
                df_ok.dropna(subset=["GEOID"])
                     .groupby(["GEOID","date","hour_range"])
                     .size()
                     .reset_index(name="311_request_count")
            )
            grouped["GEOID"] = normalize_geoid(grouped["GEOID"], DEFAULT_GEOID_LEN)
    
        # 1) artifact/prev klasörüne (raw_path ile aynı yere)
        save_atomic(grouped, agg_path)
    
        # 2) repo root SAVE_DIR'e (merge burada arıyor)
        save_atomic(grouped, os.path.join(SAVE_DIR, AGG_BASENAME))
    
        # 3) alias (opsiyonel) — sadece SAVE_DIR'e yazmak yeter
        if AGG_ALIAS and AGG_ALIAS != AGG_BASENAME:
            save_atomic(grouped, os.path.join(SAVE_DIR, AGG_ALIAS))
    
        print(f"📁 Özet yazıldı (artifact): {os.path.abspath(agg_path)}")
        print(f"📁 Özet yazıldı (SAVE_DIR): {os.path.join(SAVE_DIR, AGG_BASENAME)}")
    
        try:
            print(grouped.head(5).to_string(index=False))
        except Exception:
            pass
    else:
        print("ℹ️ Özet adımı skip (ham veri yok).")

    # 6) 311 ÖZET + SUÇ (sf_crime_01.csv) → sf_crime_02.csv (fallback’lı)
    try:
        crime_01_path = os.path.join(SAVE_DIR, "sf_crime_01.csv")
        if not os.path.exists(crime_01_path):
            print(f"ℹ️ {crime_01_path} yok. 911 adımı üretilmeden 311 merge atlandı.")
            return

        print("🔗 sf_crime_01 ile birleştiriliyor...")
        crime = pd.read_csv(crime_01_path, dtype={"GEOID": str}, low_memory=False)

        # Özet dosyası adayları
        summary_path = None
        for name in (AGG_BASENAME, AGG_ALIAS, "sf_311_last_5_years_3h.csv", "sf_311_last_5_years.csv"):
            cand = os.path.join(SAVE_DIR, name)
            if os.path.exists(cand):
                summary_path = cand
                break

        if summary_path is None:
            print("⚠️ 311 özet bulunamadı → PASSTHROUGH: 311_request_count=0")
            crime["311_request_count"] = 0
            save_atomic(crime, os.path.join(SAVE_DIR, "sf_crime_02.csv"))
            print("✅ Passthrough yazıldı.")
            return

        summary = pd.read_csv(summary_path, dtype={"GEOID": str}, low_memory=False)
        summary.columns = (summary.columns.astype(str)
                           .str.replace("\ufeff", "", regex=False)
                           .str.strip())

        summary["hour_range"] = summary["hour_range"].astype(str).str.replace(r"^21-00$", "21-24", regex=True)
        
        need = ["GEOID","date","hour_range","311_request_count"]
        missing = [c for c in need if c not in summary.columns]
        if missing:
            raise ValueError(f"❌ 311 summary kolon eksik: {missing} | cols={list(summary.columns)}")

        # GEOID ortak uzunluk
        def _mode_len(s: pd.Series) -> int:
            s2 = s.dropna().astype(str).str.extract(r"(\d+)")[0]
            return int(s2.str.len().mode().iat[0]) if len(s2) else DEFAULT_GEOID_LEN
        tgt_len = min(_mode_len(crime["GEOID"]), _mode_len(summary["GEOID"]))
        def _left(series, n):
            s = series.astype(str).str.extract(r"(\d+)")[0]
            return s.str[:n]
        crime["GEOID"]   = _left(crime["GEOID"],   tgt_len)
        summary["GEOID"] = _left(summary["GEOID"], tgt_len)

        # summary tarafında yardımcı anahtarlar
        if "hour_range" not in summary.columns:
            if "hr_key" in summary.columns:
                hr = pd.to_numeric(summary["hr_key"], errors="coerce").fillna(0).astype(int)
                summary["hour_range"] = hr.astype(str).str.zfill(2) + "-" + (hr + 3).astype(str).str.zfill(2)
            else:
                summary["hour_range"] = "00-03"

        summary["date"] = pd.to_datetime(summary["date"], errors="coerce").dt.date
        hrp = summary["hour_range"].astype(str).str.extract(r"(\d{1,2})\s*-\s*(\d{1,2})")
        summary["hr_key"] = pd.to_numeric(hrp[0], errors="coerce").fillna(0).astype(int)
        _dt = pd.to_datetime(summary["date"], errors="coerce")
        summary["day_of_week"] = _dt.dt.weekday
        summary["month"] = _dt.dt.month
        _smap = {12:"Winter",1:"Winter",2:"Winter",
                 3:"Spring",4:"Spring",5:"Spring",
                 6:"Summer",7:"Summer",8:"Summer",
                 9:"Fall",10:"Fall",11:"Fall"}
        summary["season"] = summary["month"].map(_smap)

        # crime tarafında saat anahtarı
        if "hour_range" not in crime.columns:
            if "event_hour" not in crime.columns:
                raise ValueError("❌ sf_crime_01.csv için hour_range/event_hour bulunamadı.")
            hr = (pd.to_numeric(crime["event_hour"], errors="coerce").fillna(0).astype(int) // 3) * 3
            end = hr + 3
            end = end.where(end < 24, 24)
            crime["hour_range"] = hr.astype(str).str.zfill(2) + "-" + end.astype(str).str.zfill(2)

        hrp2 = crime["hour_range"].astype(str).str.extract(r"(\d{1,2})")
        crime["hr_key"] = pd.to_numeric(hrp2[0], errors="coerce").fillna(0).astype(int)

        # Join seçimi
        has_date = ("date" in crime.columns) or ("datetime" in crime.columns)
        if has_date:
            if "date" not in crime.columns:
                crime["date"] = pd.to_datetime(crime["datetime"], errors="coerce").dt.date
            else:
                crime["date"] = pd.to_datetime(crime["date"], errors="coerce").dt.date
            keys = ["GEOID", "date", "hour_range"]
            _sum = summary[["GEOID","date","hour_range","311_request_count"]].copy()
            _overlap = (set(crime.columns) & set(_sum.columns)) - set(keys)
            if _overlap:
                print(f"🧹 DATE-BASED overlap bulundu, summary'den düşürüldü: {sorted(_overlap)}")
                _sum = _sum.drop(columns=list(_overlap), errors="ignore")
            _before = crime.shape
            _sum = summary[["GEOID", "date", "hour_range", "311_request_count"]].copy()

            _overlap = (set(crime.columns) & set(_sum.columns)) - set(keys)
            if _overlap:
                print(f"🧹 DATE-BASED merge overlap bulundu, summary'den düşürüldü: {sorted(_overlap)}")
                _sum = _sum.drop(columns=list(_overlap), errors="ignore")
            
            merged = crime.merge(_sum, on=keys, how="left")
            
            log_merge_delta(_before, merged.shape, "crime ⨯ 311 (tarihli)")
            print("🔗 Join modu: DATE-BASED (GEOID, date, hour_range)")
        else:
            cal_keys = ["GEOID", "hr_key", "day_of_week", "season"]
            cal_agg = (summary.groupby(cal_keys, as_index=False)["311_request_count"].median())
            
            _overlap = (set(crime.columns) & set(cal_agg.columns)) - set(cal_keys)
            if _overlap:
                print(f"🧹 CALENDAR-BASED merge overlap bulundu, cal_agg'den düşürüldü: {sorted(_overlap)}")
                cal_agg = cal_agg.drop(columns=list(_overlap), errors="ignore")
            
            merged = crime.merge(cal_agg, on=cal_keys, how="left")
            log_merge_delta(_before, merged.shape, "crime ⨯ 311 (takvim)")
            print("🔗 Join modu: CALENDAR-BASED (GEOID, hr_key, day_of_week, season)")

        # NAs → 0 ve tip
        if "311_request_count" in merged.columns:
            merged["311_request_count"] = pd.to_numeric(merged["311_request_count"], errors="coerce").fillna(0).astype(int)
        else:
            merged["311_request_count"] = 0

        nan_counts = merged.isna().sum()
        nan_counts = nan_counts[nan_counts > 0].sort_values(ascending=False)
        
        print("🔎 Final NaN kontrolü (sf_crime_02 yazılmadan önce):")
        if len(nan_counts) == 0:
            print("✅ NaN yok.")
        else:
            print(nan_counts.to_string())

        log_shape(merged, "CRIME⨯311 (kayıt öncesi)")
        save_atomic(merged, os.path.join(SAVE_DIR, "sf_crime_02.csv"))
        print("✅ Suç + 311 birleştirmesi tamamlandı.")
        try:
            print(merged.head(5).to_string(index=False))
        except Exception:
            pass

    except Exception as e:
        print(f"⚠️ 311 merge aşamasında hata: {e}\n↪️ PASSTHROUGH uygulanıyor…")
        try:
            crime_01_path = os.path.join(SAVE_DIR, "sf_crime_01.csv")
            if os.path.exists(crime_01_path):
                crime = pd.read_csv(crime_01_path, dtype={"GEOID": str}, low_memory=False)
                crime["311_request_count"] = 0
                save_atomic(crime, os.path.join(SAVE_DIR, "sf_crime_02.csv"))
                print("✅ Passthrough yazıldı (exception fallback).")
        except Exception as ee:
            print(f"❌ Passthrough da başarısız: {ee}")

if __name__ == "__main__":
    main()
