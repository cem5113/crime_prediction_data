# scripts/update_week_forecast.py

import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# ======================
# ENV
# ======================
CRIME_DATA_DIR = os.getenv("CRIME_DATA_DIR", ".")
API_KEY = os.getenv("VISUAL_CROSSING_API_KEY")
WX_LOCATION = os.getenv("WX_LOCATION", "San Francisco, CA")
WX_UNIT = os.getenv("WX_UNIT", "us")

if not API_KEY:
    raise ValueError("❌ VISUAL_CROSSING_API_KEY env değişkeni gerekli.")

out_path = os.path.join(CRIME_DATA_DIR, "week.csv")
os.makedirs(CRIME_DATA_DIR, exist_ok=True)

# ======================
# SF timezone
# ======================
SF_TZ = ZoneInfo("America/Los_Angeles")
today = datetime.now(SF_TZ).date()

# ======================
# WINDOW
# 72saat geriden başlasın istiyorsan:
start_date = today - timedelta(days=3)
end_date = today + timedelta(days=8)

# ======================
# API request
# tarih aralığını URL içine VER
# ======================
url = (
    "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/"
    f"timeline/{WX_LOCATION}/{start_date}/{end_date}"
    f"?unitGroup={WX_UNIT}&key={API_KEY}&contentType=json"
)

r = requests.get(url, timeout=30)
r.raise_for_status()
data = r.json()

days = data.get("days", [])
if not days:
    raise ValueError("API yanıtında days boş.")

df = pd.DataFrame(days)

rename_map = {
    "datetime": "date",
    "temp": "tavg",
    "tempmin": "tmin",
    "tempmax": "tmax",
    "precip": "prcp",
}
for k, v in rename_map.items():
    if k in df.columns and v not in df.columns:
        df.rename(columns={k: v}, inplace=True)

df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
df = df.dropna(subset=["date"]).sort_values("date")

for c in ["tavg", "tmin", "tmax", "prcp"]:
    df[c] = pd.to_numeric(df.get(c), errors="coerce")

df["temp_range"] = df["tmax"] - df["tmin"]
df["day"] = pd.to_datetime(df["date"]).dt.day_name()
df["is_rainy"] = (df["prcp"].fillna(0) > 0).astype(int)

HOT_C = 25.0
HOT_F = HOT_C * 9 / 5 + 32
hot_thr = HOT_F if WX_UNIT.lower() == "us" else HOT_C
df["is_hot"] = (df["tmax"] > hot_thr).astype(int)

cols = [
    "date", "tavg", "tmin", "tmax", "prcp",
    "temp_range", "day", "is_rainy", "is_hot"
]

df_week = df[cols].copy()

df_week.to_csv(out_path, index=False)

print(f"✅ week.csv üretildi → {out_path}")
print(f"📅 Aralık: {start_date} → {end_date} | Gün sayısı: {len(df_week)}")
print(df_week.head(10))
