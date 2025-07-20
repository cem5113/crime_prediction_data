# San Francisco Suç Verisi Zenginleştirme ve Zamansal Özellik Üretimi

import pandas as pd
import numpy as np
import holidays
from datetime import datetime, timedelta

# === 1. Veri Yükleme ===
df = pd.read_csv("sf_crime.csv", parse_dates=["date"])
print(f"🔹 Yüklenen veri boyutu: {df.shape}")

# === 2. GEOID Temizleme ===
def fix_geoid(geoid):
    if pd.isna(geoid): return np.nan
    return str(geoid).split('.')[0].strip().zfill(11)

df["GEOID"] = df["GEOID"].apply(fix_geoid)

# === 3. Temel Temizlik ===
df = df.dropna(subset=["latitude", "longitude", "date", "time", "GEOID"])
df["datetime"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str), errors="coerce")
df = df.dropna(subset=["datetime"])
df["datetime"] = df["datetime"].dt.floor("H")
df["event_hour"] = df["datetime"].dt.hour
df["date"] = df["datetime"].dt.date
df["month"] = df["datetime"].dt.month
df["year"] = df["datetime"].dt.year
df["day_of_week"] = df["datetime"].dt.dayofweek

# === 4. Zaman Etiketleri ===
df["is_night"] = df["event_hour"].apply(lambda x: 1 if (x >= 20 or x < 4) else 0)
df["is_weekend"] = df["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)

# === 5. Resmi Tatil & Durumsal Etiketler ===
years = df["year"].unique()
us_holidays = pd.to_datetime(list(holidays.US(years=years).keys()))
df["is_holiday"] = df["date"].isin(us_holidays).astype(int)
df["latlon"] = df["latitude"].round(5).astype(str) + "_" + df["longitude"].round(5).astype(str)
df["is_repeat_location"] = df.duplicated("latlon").astype(int)
df.drop(columns=["latlon"], inplace=True)
df["is_school_hour"] = df["event_hour"].apply(lambda x: 1 if 7 <= x <= 16 else 0)
df["is_business_hour"] = df.apply(lambda x: 1 if (9 <= x["event_hour"] < 18 and x["day_of_week"] < 5) else 0, axis=1)

# === 6. Mevsim Etiketi ===
season_map = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Fall", 10: "Fall", 11: "Fall"
}
df["season"] = df["month"].map(season_map)

# === 7. Zamansal Kriminal Yoğunluk ===
df = df.sort_values(by=["GEOID", "datetime"]).reset_index(drop=True)
df["past_7d_crimes"] = 0
df["crime_count_past_24h"] = 0
df["crime_count_past_48h"] = 0
df["crime_trend_score"] = 0
df["prev_crime_1h"] = 0
df["prev_crime_2h"] = 0
df["prev_crime_3h"] = 0

for geoid, group in df.groupby("GEOID"):
    times = pd.to_datetime(group["datetime"]).values.astype("datetime64[ns]")
    event_hours = group["event_hour"].values
    idx = group.index
    time_deltas = times[:, None] - times[None, :]

    df.loc[idx, "past_7d_crimes"] = ((time_deltas > np.timedelta64(0, 'ns')) & (time_deltas <= np.timedelta64(7, 'D'))).sum(axis=1)
    df.loc[idx, "crime_count_past_24h"] = ((time_deltas > np.timedelta64(0, 'ns')) & (time_deltas <= np.timedelta64(1, 'D'))).sum(axis=1)
    df.loc[idx, "crime_count_past_48h"] = ((time_deltas > np.timedelta64(0, 'ns')) & (time_deltas <= np.timedelta64(2, 'D'))).sum(axis=1)

    # Trend skoru: aynı saatte geçmiş 7 günde kaç olay olmuş?
    trend_score = []
    for i in range(len(times)):
        current_time = times[i]
        current_hour = event_hours[i]
        mask = (times < current_time) & (times >= current_time - np.timedelta64(7, 'D')) & (event_hours == current_hour)
        trend_score.append(mask.sum())
    df.loc[idx, "crime_trend_score"] = trend_score

    # Önceki saatlerde olay var mı?
    for lag in [1, 2, 3]:
        lag_col = f"prev_crime_{lag}h"
        has_prev = []
        for i in range(len(times)):
            current_time = times[i]
            mask = (times < current_time) & (times >= current_time - np.timedelta64(lag, 'h'))
            has_prev.append(1 if mask.sum() > 0 else 0)
        df.loc[idx, lag_col] = has_prev

# === 8. Kaydet ===
output_path = "sf_crime_49.csv"
df.to_csv(output_path, index=False)
print(f"✅ Zenginleştirilmiş veri kaydedildi: {output_path}")
