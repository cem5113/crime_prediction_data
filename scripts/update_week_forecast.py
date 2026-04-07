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
now = datetime.now(SF_TZ)
today = now.date()

# ======================
# WINDOW
# -2 günden başlayıp +6 güne kadar = toplam 9 gün
# ======================
start_date = today - timedelta(days=2)
end_date = today + timedelta(days=6)

# ======================
# API request
# ======================
url = (
    "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/"
    f"timeline/{WX_LOCATION}"
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

df["tavg"] = pd.to_numeric(df.get("tavg"), errors="coerce")
df["tmin"] = pd.to_numeric(df.get("tmin"), errors="coerce")
df["tmax"] = pd.to_numeric(df.get("tmax"), errors="coerce")
df["prcp"] = pd.to_numeric(df.get("prcp"), errors="coerce")

df["temp_range"] = df["tmax"] - df["tmin"]
df["day"] = pd.to_datetime(df["date"]).dt.day_name()
df["is_rainy"] = (df["prcp"].fillna(0) > 0).astype(int)

HOT_C = 25.0
HOT_F = HOT_C * 9 / 5 + 32
hot_thr = HOT_F if WX_UNIT.lower() == "us" else HOT_C
df["is_hot"] = (df["tmax"] > hot_thr).astype(int)

cols = ["date", "tavg", "tmin", "tmax", "prcp", "temp_range", "day", "is_rainy", "is_hot"]

# -2 gün ... +6 gün
df_week = df[(df["date"] >= start_date) & (df["date"] <= end_date)][cols].copy()

df_week.to_csv(out_path, index=False)

print(f"✅ week.csv üretildi → {out_path}")
print(f"📅 Aralık: {start_date} → {end_date} | Gün sayısı: {len(df_week)}")
print(df_week.head(10))
