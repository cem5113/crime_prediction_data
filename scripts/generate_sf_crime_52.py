# Zaman-Mekân Kombinasyonlarına Göre Suç Verisi Özetleme ve Y_label Etiketleme

import pandas as pd
import numpy as np
import itertools

# 1. Veriyi oku
input_path = "sf_crime_49.csv"
df = pd.read_csv(input_path, low_memory=False)
df["GEOID"] = df["GEOID"].astype(str).str.extract(r"(\d+)")[0].str.zfill(11)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date", "event_hour", "GEOID"])

# 2. Zaman etiketleri
df["event_hour"] = df["event_hour"].astype(int)
df["day_of_week"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month

def month_to_season(month):
    if month in [12, 1, 2]: return "Winter"
    elif month in [3, 4, 5]: return "Spring"
    elif month in [6, 7, 8]: return "Summer"
    elif month in [9, 10, 11]: return "Fall"
    return "Unknown"

df["season"] = df["month"].apply(month_to_season)

# 3. Grup ve öznitelik tanımları
group_cols = ["GEOID", "season", "day_of_week", "event_hour"]

mean_cols = [
    "latitude", "longitude", "past_7d_crimes", "crime_count_past_24h",
    "crime_count_past_48h", "crime_trend_score", "prev_crime_1h",
    "prev_crime_2h", "prev_crime_3h"
]

mode_cols = [
    "is_weekend", "is_night", "is_holiday", "is_repeat_location",
    "is_school_hour", "is_business_hour", "year", "month"
]

def safe_mode(x):
    try: return x.mode().iloc[0]
    except: return np.nan

agg_dict = {col: "mean" for col in mean_cols}
agg_dict.update({col: safe_mode for col in mode_cols})
agg_dict.update({"date": "min", "id": "count"})

grouped = df.groupby(group_cols).agg(agg_dict).reset_index()
grouped = grouped.rename(columns={"id": "crime_count"})
grouped["Y_label"] = (grouped["crime_count"] >= 2).astype(int)

# 4. Tüm kombinasyonlar
geoids = df["GEOID"].unique()
seasons = ["Winter", "Spring", "Summer", "Fall"]
days = list(range(7))
hours = list(range(24))

full_grid = pd.DataFrame(
    itertools.product(geoids, seasons, days, hours),
    columns=["GEOID", "season", "day_of_week", "event_hour"]
)

df_final = full_grid.merge(grouped, on=group_cols, how="left")
df_final["crime_count"] = df_final["crime_count"].fillna(0).astype(int)
df_final["Y_label"] = df_final["Y_label"].fillna(0).astype(int)

# 5. Etiketleri tekrar üret

df_final["is_weekend"] = df_final["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)
df_final["is_night"] = df_final["event_hour"].apply(lambda x: 1 if (x >= 20 or x < 4) else 0)
df_final["is_school_hour"] = df_final.apply(
    lambda x: 1 if (x["day_of_week"] < 5 and 7 <= x["event_hour"] <= 16) else 0, axis=1
)
df_final["is_business_hour"] = df_final.apply(
    lambda x: 1 if (x["day_of_week"] < 6 and 9 <= x["event_hour"] < 18) else 0, axis=1
)

# 6. NaN temizliği
columns_with_nan = [
    "latitude", "longitude", "past_7d_crimes", "crime_count_past_24h", "crime_count_past_48h",
    "crime_trend_score", "prev_crime_1h", "prev_crime_2h", "prev_crime_3h",
    "is_holiday", "is_repeat_location", "year", "month", "date"
]

before_rows = df_final.shape[0]
df_final = df_final.dropna(subset=columns_with_nan)
after_rows = df_final.shape[0]
print(f"🧹 {before_rows - after_rows} satır silindi (NaN içeriyor)")

# 7. Kaydet (sf_crime_50)
df_final.to_csv("sf_crime_50.csv", index=False)
print("✅ sf_crime_50.csv kaydedildi.")

# 8. Eksik kombinasyonlar
expected_grid = pd.DataFrame(itertools.product(geoids, seasons, days, hours),
                             columns=["GEOID", "season", "day_of_week", "event_hour"])
existing_combinations = df_final[["GEOID", "season", "day_of_week", "event_hour"]]

missing = expected_grid.merge(existing_combinations.drop_duplicates(),
                              on=["GEOID", "season", "day_of_week", "event_hour"],
                              how="left", indicator=True)

missing = missing[missing["_merge"] == "left_only"].drop(columns=["_merge"])
missing["crime_count"] = 0
missing["Y_label"] = 0

df_full_52 = pd.concat([df_final, missing], ignore_index=True)
df_full_52.to_csv("sf_crime_52.csv", index=False)

print("✅ sf_crime_52.csv kaydedildi.")
print(f"Toplam kombinasyon: {expected_grid.shape[0]}, Eksik olanlar: {missing.shape[0]}")
