# update_weather_data.py
import pandas as pd
import datetime
import requests
import io
import os

def update_weather_data(save_path="sf_weather_5years.csv"):
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=5 * 365)
    station_id = "USW00023234"  # San Francisco Havaalanı

    url = (
        "https://www.ncei.noaa.gov/access/services/data/v1"
        f"?dataset=daily-summaries"
        f"&stations={station_id}"
        f"&startDate={start_date}"
        f"&endDate={end_date}"
        f"&dataTypes=TMAX,TMIN,PRCP"
        f"&format=csv"
    )

    try:
        response = requests.get(url)
        if response.status_code == 200:
            df_new = pd.read_csv(io.StringIO(response.text))
            df_new["DATE"] = pd.to_datetime(df_new["DATE"])

            if os.path.exists(save_path):
                df_old = pd.read_csv(save_path)
                df_old["DATE"] = pd.to_datetime(df_old["DATE"])
                df_combined = pd.concat([df_old, df_new]).drop_duplicates(subset=["DATE"])
            else:
                df_combined = df_new

            df_filtered = df_combined[df_combined["DATE"] >= pd.to_datetime(start_date)]
            df_filtered.to_csv(save_path, index=False)
            print(f"✅ Güncellendi: {start_date} → {end_date}")
            print(f"📁 Kaydedildi: {save_path}")
        else:
            print(f"❌ Veri çekilemedi: {response.status_code}")
    except Exception as e:
        print(f"❌ Hata oluştu: {e}")
