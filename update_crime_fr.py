# update_crime_fr.py 
from __future__ import annotations
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import shutil
import hashlib

# =========================
# Ayarlar (ENV ile özelleştirilebilir)
# =========================
EVENTS_PATH = Path(os.getenv("FR_EVENTS_PATH", "sf_crime.csv")) 
LABEL_PATH  = Path(os.getenv("FR_LABEL_PATH", "sf_crime_L.csv"))    # grid etiket kaynağı
OUT_PATH    = Path(os.getenv("FR_OUT_PATH",   "fr_crime.csv"))      # hedef: olay bazlı çıktı
MIRROR_DIR  = Path(os.getenv("FR_MIRROR_DIR", "crime_prediction_data"))
MIRROR_ENABLED = os.getenv("FR_MIRROR_ENABLED", "0") == "1"
EVENTS_URL = os.getenv(
    "FR_EVENTS_URL",
    "https://github.com/cem5113/crime_prediction_data/releases/latest/download/sf_crime.csv"
)

# Otomatik label bulma açık/kapalı (1/0) – kapatmak isterseniz FR_AUTOFIND_LABEL=0
AUTOFIND_LABEL = os.getenv("FR_AUTOFIND_LABEL", "1") == "1"

# Potansiyel eşleşme anahtarları (öncelik sırasıyla)
FULL_KEYS = ["GEOID", "season", "day_of_week", "event_hour"]
GEOID_ONLY = ["GEOID"]

# Çekilecek etiket kolonu adı
YCOL = os.getenv("FR_YCOL", "Y_label")
SKIP_IF_NO_CHANGE = os.getenv("FR_SKIP_IF_NO_CHANGE", "1") == "1"

# =========================
# Yardımcılar
# =========================
def _abs(p: Path) -> Path:
    """Absolute & expanded path."""
    return p.expanduser().resolve()

def safe_read_csv(p) -> pd.DataFrame:
    # URL desteği: EVENTS_URL gibi durumlarda Path yerine url string geçmeyeceğiz,
    # ama burada güvenli olması için yine de kontrol edelim.
    try:
        ps = str(p)
        if ps.startswith("http://") or ps.startswith("https://"):
            df = pd.read_csv(ps, low_memory=False)
            print(f"🌐 Okundu (URL): {ps}  ({len(df):,} satır, {df.shape[1]} sütun)")
            return df
    except Exception as e:
        print(f"⚠️ URL okunamadı: {p} → {e}")
        return pd.DataFrame()

    p = _abs(p)
    if not p.exists():
        print(f"ℹ️ Bulunamadı: {p}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(p, low_memory=False)
        print(f"📖 Okundu: {p}  ({len(df):,} satır, {df.shape[1]} sütun)")
        return df
    except Exception as e:
        print(f"⚠️ Okunamadı: {p} → {e}")
        return pd.DataFrame()

def safe_save_csv(df: pd.DataFrame, p: Path) -> None:
    p = _abs(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp.csv")
    df.to_csv(tmp, index=False)
    tmp.replace(p)
    print(f"💾 Kaydedildi: {p}  ({len(df):,} satır, {df.shape[1]} sütun)")

def file_sha256(p: Path) -> str:
    p = _abs(p)
    if not p.exists():
        return ""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def df_sha256(df: pd.DataFrame) -> str:
    # deterministik hash için: index yok, satır sırası stabil olmalı
    b = df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(b).hexdigest()

def normalize_event_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "GEOID" in df:
        df["GEOID"] = (
            df["GEOID"].astype(str).str.extract(r"(\d+)")[0].str.zfill(11)
        )
    for c in ("day_of_week", "event_hour"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    return df

def normalize_label_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "GEOID" in df:
        df["GEOID"] = (
            df["GEOID"].astype(str).str.extract(r"(\d+)")[0].str.zfill(11)
        )
    for c in ("day_of_week", "event_hour", YCOL):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    return df

def dedupe_labels(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Aynı anahtar için birden fazla Y varsa 1'i tercih et; yoksa 0."""
    if not all(k in df.columns for k in keys + [YCOL]):
        # Y_label yoksa sadece anahtara göre tekilleştir
        return df.drop_duplicates(subset=[c for c in keys if c in df.columns])
    return df.groupby(keys, as_index=False)[YCOL].max()  # 1 > 0

def try_merge(events: pd.DataFrame, labels: pd.DataFrame, keys: list[str]) -> tuple[pd.DataFrame, list[str]]:
    keys = [k for k in keys if k in events.columns and k in labels.columns]
    if not keys:
        raise ValueError("Eşleşme için ortak anahtar bulunamadı.")
    L = dedupe_labels(labels, keys)
    merged = events.merge(L[keys + [YCOL]], on=keys, how="left")
    return merged, keys

def resolve_label_path(initial: Path) -> Path | None:
    """LABEL_PATH yoksa makul adayları dener (isteğe bağlı)."""
    p = _abs(initial)
    if p.exists():
        return p
    candidates = [
        Path("crime_prediction_data/sf_crime_L.csv"),
        Path("sf_crime_L.csv"),
        Path("crime_prediction_data/sf_crime_grid_full_labeled.csv"),
        Path("sf_crime_grid_full_labeled.csv"),
        Path("crime_prediction_data/sf_crime_52.csv"),
        Path("sf_crime_52.csv"),
        Path("crime_prediction_data/sf_crime_y.csv"),
        Path("sf_crime_y.csv"),
    ]
    for c in candidates:
        if _abs(c).exists():
            print(f"🔎 Etiket kaynağı otomatik bulundu: {_abs(c)}")
            return _abs(c)
    return None

# =========================
# Akış
# =========================
def main() -> int:
    print("📂 CWD:", Path.cwd())
    print("🔧 ENV → FR_EVENTS_PATH:", _abs(EVENTS_PATH))
    print("🔧 ENV → FR_EVENTS_URL :", EVENTS_URL)
    print("🔧 ENV → FR_LABEL_PATH :", _abs(LABEL_PATH))
    print("🔧 ENV → FR_OUT_PATH   :", _abs(OUT_PATH))
    print("🔧 ENV → FR_MIRROR_DIR :", _abs(MIRROR_DIR))
    print("🔧 ENV → FR_YCOL       :", YCOL)
    print("🔧 ENV → FR_AUTOFIND_LABEL:", AUTOFIND_LABEL)
    print("🔧 ENV → FR_SKIP_IF_NO_CHANGE:", SKIP_IF_NO_CHANGE)
    
    # 1) Olay verisini oku
    if EVENTS_URL.strip():
        events = safe_read_csv(EVENTS_URL.strip())
    else:
        events = safe_read_csv(EVENTS_PATH)
    if events.empty:
        src = EVENTS_URL.strip() if EVENTS_URL.strip() else str(_abs(EVENTS_PATH))
        print(f"❌ Olay verisi boş veya yok: {src}")
        return 0
    if "id" not in events.columns:
        print("⚠️ Uyarı: 'id' kolonu bulunamadı. Yine de olay bazlı devam edilecek.")
    events = normalize_event_df(events)
    base_len = len(events)
    print(f"🧮 Olay satır sayısı: {base_len:,}")

    # 2) Etiket gridini oku (gerekirse otomatik bul)
    lbl_path = LABEL_PATH
    labels = safe_read_csv(lbl_path)
    if labels.empty and AUTOFIND_LABEL:
        auto = resolve_label_path(lbl_path)
        if auto is not None and auto != _abs(lbl_path):
            lbl_path = auto
            labels = safe_read_csv(lbl_path)

    if labels.empty:
        print(f"❌ Etiket kaynağı boş veya yok: {_abs(lbl_path)}")
        return 0

    labels = normalize_label_df(labels)

    # 3) Önce tam anahtar, olmazsa yalnızca GEOID ile eşle
    used_keys: list[str] | None = None
    try:
        out, used_keys = try_merge(events, labels, FULL_KEYS)
        print(f"🔗 Eşleşme anahtarları: {used_keys} (tam anahtar)")
    except Exception:
        out, used_keys = try_merge(events, labels, GEOID_ONLY)
        print(f"🔗 Eşleşme anahtarları: {used_keys} (yalnızca GEOID)")

    # 4) Eşleşemeyen satırlar
    missing = int(out[YCOL].isna().sum()) if YCOL in out.columns else 0
    if missing > 0:
        rate = round(missing / base_len * 100, 2) if base_len else 0.0
        print(f"ℹ️ Eşleşemeyen olay satırı: {missing:,} (%{rate}) → {YCOL} NaN. NaN'ları 0'a çeviriyorum.")
        out[YCOL] = out[YCOL].fillna(0).astype(int)
        
    sort_cols = [c for c in ["id","incident_datetime","datetime","date","GEOID","season","day_of_week","event_hour"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    if SKIP_IF_NO_CHANGE:
        old_hash = file_sha256(OUT_PATH)
        new_hash = df_sha256(out)
        if old_hash and (old_hash == new_hash):
            print("✅ Çıktı değişmedi → dosya yazılmadı / mirror yapılmadı.")
            return 0
            
    # 5) Kaydet & mirror
    safe_save_csv(out, OUT_PATH)
    if MIRROR_ENABLED:
        try:
            _abs(MIRROR_DIR).mkdir(parents=True, exist_ok=True)
            shutil.copy2(_abs(OUT_PATH), _abs(MIRROR_DIR) / _abs(OUT_PATH).name)
            print(f"📦 Mirror kopya: {_abs(MIRROR_DIR) / _abs(OUT_PATH).name}")
        except Exception as e:
            print(f"ℹ️ Mirror kopya atlandı/başarısız: {e}")
    else:
        print("ℹ️ Mirror kapalı (FR_MIRROR_ENABLED=0).")

    # 6) Hızlı özet
    if YCOL in out.columns:
        vc = out[YCOL].value_counts(normalize=True, dropna=False).mul(100).round(2)
        print("\n📊 Y_label oranları (%):")
        try:
            # print(vc.to_string()) bazen geniş veri setinde uyarı verebiliyor
            for k, v in vc.items():
                print(f"  {k}: {v}%")
        except Exception:
            print(vc)

    # 7) Temel kalite kontrolleri
    if "id" in out.columns:
        dup = int(out["id"].duplicated().sum())
        if dup:
            print(f"⚠️ Uyarı: {dup} adet tekrar eden id var.")
        else:
            print("✅ id benzersiz görünüyor.")

    return 0


if __name__ == "__main__":
    try:
        code = main()
        raise SystemExit(code if isinstance(code, int) else 0)
    except Exception as e:
        print(f"⚠️ FR derleme sırasında yakalanmamış hata: {e}")
        raise SystemExit(0)
