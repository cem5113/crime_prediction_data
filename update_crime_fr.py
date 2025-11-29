# update_crime_fr.py 
from __future__ import annotations
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import shutil

# =========================
# Ayarlar (ENV ile özelleştirilebilir)
# =========================
EVENTS_PATH = Path(os.getenv("FR_EVENTS_PATH", "sf_crime_y.csv"))     # olay bazlı kaynak (suç_id içermeli)
LABEL_PATH  = Path(os.getenv("FR_LABEL_PATH", "sf_crime_L.csv"))    # grid etiket kaynağı
OUT_PATH    = Path(os.getenv("FR_OUT_PATH",   "fr_crime.csv"))      # hedef: olay bazlı çıktı
MIRROR_DIR  = Path(os.getenv("FR_MIRROR_DIR", "crime_prediction_data"))
AGG_DOW_HOUR_SEASON_PATH = Path(os.getenv("FR_AGG_DOW_HOUR_SEASON_PATH", "fr_crime_grid_dow_hour_season.csv"))
AGG_DOW_SEASON_PATH      = Path(os.getenv("FR_AGG_DOW_SEASON_PATH",      "fr_crime_grid_dow_season.csv"))

# Otomatik label bulma açık/kapalı (1/0) – kapatmak isterseniz FR_AUTOFIND_LABEL=0
AUTOFIND_LABEL = os.getenv("FR_AUTOFIND_LABEL", "1") == "1"

# Potansiyel eşleşme anahtarları (öncelik sırasıyla)
FULL_KEYS = ["GEOID", "season", "day_of_week", "event_hour"]
GEOID_ONLY = ["GEOID"]

# Çekilecek etiket kolonu adı
YCOL = os.getenv("FR_YCOL", "Y_label")


# =========================
# Yardımcılar
# =========================
def _abs(p: Path) -> Path:
    """Absolute & expanded path."""
    return p.expanduser().resolve()

def safe_read_csv(p: Path) -> pd.DataFrame:
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

def ensure_hour_range(df: pd.DataFrame, hour_col: str = "event_hour") -> pd.DataFrame:
    """event_hour'dan kategorik hour_range üretir (yoksa)."""
    df = df.copy()
    if "hour_range" in df.columns:
        return df

    if hour_col not in df.columns:
        print("ℹ️ hour_range üretilemedi: event_hour kolonu yok.")
        return df

    # 4 saatlik bin'ler örneği: 00-03, 04-07, ...
    bins = [0, 4, 8, 12, 16, 20, 24]
    labels = ["00-03", "04-07", "08-11", "12-15", "16-19", "20-23"]
    df["hour_range"] = pd.cut(df[hour_col], bins=bins, labels=labels, right=False)
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
    print("🔧 ENV → FR_LABEL_PATH :", _abs(LABEL_PATH))
    print("🔧 ENV → FR_OUT_PATH   :", _abs(OUT_PATH))
    print("🔧 ENV → FR_MIRROR_DIR :", _abs(MIRROR_DIR))
    print("🔧 ENV → FR_YCOL       :", YCOL)
    print("🔧 ENV → FR_AUTOFIND_LABEL:", AUTOFIND_LABEL)

    # 1) Olay verisini oku
    events = safe_read_csv(EVENTS_PATH)
    if events.empty:
        print(f"❌ Olay verisi boş veya yok: {_abs(EVENTS_PATH)}")
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

    # 5) İz bilgisi
    out["fr_snapshot_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    out["fr_label_keys"]  = "+".join(used_keys or [])

    # 5b) Ek aggregate dosyalar (GEOID, day_of_week, hour_range, season) ve (GEOID, day_of_week, season)
    if YCOL in out.columns:
        # hour_range kolonunu garanti et
        out = ensure_hour_range(out, hour_col="event_hour")

        snapshot_ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"

        # 5b-1) GEOID, day_of_week, hour_range, season bazlı aggregate
        group_cols1 = ["GEOID", "day_of_week", "hour_range", "season"]
        if all(c in out.columns for c in group_cols1):
            agg1 = (
                out.groupby(group_cols1, as_index=False)
                   .agg(
                       events_total    =(YCOL, "size"),
                       events_positive =(YCOL, "sum"),
                   )
            )
            agg1["events_rate"]    = np.where(agg1["events_total"] > 0,
                                              agg1["events_positive"] / agg1["events_total"],
                                              0.0)
            agg1["fr_snapshot_at"] = snapshot_ts

            safe_save_csv(agg1, AGG_DOW_HOUR_SEASON_PATH)
        else:
            print(f"ℹ️ GEOID/day_of_week/hour_range/season aggregate atlandı: eksik kolon(lar): "
                  f"{[c for c in group_cols1 if c not in out.columns]}")

        # 5b-2) GEOID, day_of_week, season bazlı aggregate
        group_cols2 = ["GEOID", "day_of_week", "season"]
        if all(c in out.columns for c in group_cols2):
            agg2 = (
                out.groupby(group_cols2, as_index=False)
                   .agg(
                       events_total    =(YCOL, "size"),
                       events_positive =(YCOL, "sum"),
                   )
            )
            agg2["events_rate"]    = np.where(agg2["events_total"] > 0,
                                              agg2["events_positive"] / agg2["events_total"],
                                              0.0)
            agg2["fr_snapshot_at"] = snapshot_ts

            safe_save_csv(agg2, AGG_DOW_SEASON_PATH)
        else:
            print(f"ℹ️ GEOID/day_of_week/season aggregate atlandı: eksik kolon(lar): "
                  f"{[c for c in group_cols2 if c not in out.columns]}")
    else:
        print(f"ℹ️ {YCOL} kolonu yok, aggregate dosyalar üretilmedi.")
    
    # 6) Kaydet & mirror
    safe_save_csv(out, OUT_PATH)
    try:
        _abs(MIRROR_DIR).mkdir(parents=True, exist_ok=True)
        shutil.copy2(_abs(OUT_PATH), _abs(MIRROR_DIR) / _abs(OUT_PATH).name)
        print(f"📦 Mirror kopya: {_abs(MIRROR_DIR) / _abs(OUT_PATH).name}")
    except Exception as e:
        print(f"ℹ️ Mirror kopya atlandı/başarısız: {e}")

    # 7) Hızlı özet
    if YCOL in out.columns:
        vc = out[YCOL].value_counts(normalize=True, dropna=False).mul(100).round(2)
        print("\n📊 Y_label oranları (%):")
        try:
            # print(vc.to_string()) bazen geniş veri setinde uyarı verebiliyor
            for k, v in vc.items():
                print(f"  {k}: {v}%")
        except Exception:
            print(vc)

    # 8) Temel kalite kontrolleri
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
