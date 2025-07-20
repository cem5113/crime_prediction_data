
import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import holidays
import itertools
from datetime import datetime, timedelta
from fpdf import FPDF

st.set_page_config(page_title="Veri Güncelleme", layout="wide")
st.title("📦 Günlük Suç Verisi İşleme ve Özetleme Paneli")

DOWNLOAD_URL = "https://github.com/cem5113/crime_prediction_data/releases/download/latest/sf_crime.csv"

def create_pdf_report(file_name, row_count_before, nan_cols, row_count_after, removed_rows):
    now = datetime.now()
    timestamp = now.strftime("%d.%m.%Y %H:%M:%S")

    if not nan_cols.empty:
        nan_parts = [f"- {col}: {count}" for col, count in nan_cols.items()]
        nan_text = " ".join(nan_parts)
    else:
        nan_text = "Yok"

    summary = (
        f"- Tarih/Saat: {timestamp}; "
        f"Dosya: {file_name} ; "
        f"Toplam satir sayisi: {row_count_before:,}; "
        f"NaN iceren sutunlar: {nan_text}; "
        f"Revize satir sayisi: {row_count_after:,}; "
        f"Silinen eski tarihli satir sayisi: {removed_rows}"
    )

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=summary.encode("latin1", "replace").decode("latin1"))

    output_name = f"report_{now.strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf.output(output_name)
    return output_name

if st.button("📥 sf_crime.csv indir, zenginleştir ve özetle"):
    with st.spinner("⏳ İşlem devam ediyor... Lütfen bekleyin. Bu birkaç dakika sürebilir."):
        try:
            response = requests.get(DOWNLOAD_URL)
            if response.status_code == 200:
                with open("sf_crime.csv", "wb") as f:
                    f.write(response.content)
                st.success("✅ sf_crime.csv başarıyla indirildi.")

                df = pd.read_csv("sf_crime.csv", low_memory=False)
                original_row_count = len(df)

                # Örnek: NaN sütunları bul ve rapor hazırla (bu satırları işlem sonrası yerleştir)
                nan_summary = df.isna().sum()
                nan_cols = nan_summary[nan_summary > 0]
                removed_rows = 0  # Henüz işlem yapılmadığı için başlangıçta sıfır

                # 📄 PDF rapor oluştur ve indirme butonu ekle
                report_path = create_pdf_report("sf_crime.csv", original_row_count, nan_cols, len(df), removed_rows)
                with open(report_path, "rb") as f:
                    st.download_button("📄 PDF Raporu İndir", f, file_name=report_path, mime="application/pdf")
            else:
                st.error(f"❌ Indirme hatası: {response.status_code}")
        except Exception as e:
            st.error(f"❌ Hata oluştu: {e}")
            
            df["GEOID"] = df["GEOID"].astype(str).str.extract(r"(\d+)")[0].str.zfill(11)
            nan_summary = df.isna().sum()
            nan_cols = nan_summary[nan_summary > 0]
            df = df.dropna()

            removed_rows = 0
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            five_years_ago = datetime.now() - timedelta(days=5*365)
            before_filter = len(df)
            df = df[df["date"] >= five_years_ago]
            removed_rows = before_filter - len(df)

            # Enrichment
            df["datetime"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str), errors="coerce")
            df = df.dropna(subset=["datetime"])
            df["datetime"] = df["datetime"].dt.floor("H")
            df["event_hour"] = df["datetime"].dt.hour
            df["date"] = df["datetime"].dt.date
            df["month"] = df["datetime"].dt.month
            df["year"] = df["datetime"].dt.year
            df["day_of_week"] = df["datetime"].dt.dayofweek
            df["is_night"] = df["event_hour"].apply(lambda x: 1 if (x >= 20 or x < 4) else 0)
            df["is_weekend"] = df["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)
            years = df["year"].dropna().astype(int).unique()
            us_holidays = pd.to_datetime(list(holidays.US(years=years).keys()))
            df["is_holiday"] = df["date"].isin(us_holidays).astype(int)
            df["latlon"] = df["latitude"].round(5).astype(str) + "_" + df["longitude"].round(5).astype(str)
            df["is_repeat_location"] = df.duplicated("latlon").astype(int)
            df.drop(columns=["latlon"], inplace=True)
            df["is_school_hour"] = df["event_hour"].apply(lambda x: 1 if 7 <= x <= 16 else 0)
            df["is_business_hour"] = df.apply(lambda x: 1 if (9 <= x["event_hour"] < 18 and x["day_of_week"] < 5) else 0, axis=1)
            season_map = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring", 6: "Summer", 7: "Summer", 8: "Summer", 9: "Fall", 10: "Fall", 11: "Fall"}
            df["season"] = df["month"].map(season_map)

            df = df.sort_values(by=["GEOID", "datetime"]).reset_index(drop=True)
            for col in ["past_7d_crimes", "crime_count_past_24h", "crime_count_past_48h", "crime_trend_score", "prev_crime_1h", "prev_crime_2h", "prev_crime_3h"]:
                df[col] = 0

            for geoid, group in df.groupby("GEOID"):
                times = pd.to_datetime(group["datetime"]).values.astype("datetime64[ns]")
                event_hours = group["event_hour"].values
                idx = group.index
                deltas = times[:, None] - times[None, :]

                df.loc[idx, "past_7d_crimes"] = ((deltas > np.timedelta64(0, 'ns')) & (deltas <= np.timedelta64(7, 'D'))).sum(axis=1)
                df.loc[idx, "crime_count_past_24h"] = ((deltas > np.timedelta64(0, 'ns')) & (deltas <= np.timedelta64(1, 'D'))).sum(axis=1)
                df.loc[idx, "crime_count_past_48h"] = ((deltas > np.timedelta64(0, 'ns')) & (deltas <= np.timedelta64(2, 'D'))).sum(axis=1)
                df.loc[idx, "crime_trend_score"] = [((times[:i] >= t - np.timedelta64(7, 'D')) & (event_hours[:i] == h)).sum() for i, (t, h) in enumerate(zip(times, event_hours))]

                for lag in [1, 2, 3]:
                    lag_col = f"prev_crime_{lag}h"
                    df.loc[idx, lag_col] = [1 if ((times[:i] >= t - np.timedelta64(lag, 'h')) & (times[:i] < t)).sum() > 0 else 0 for i, t in enumerate(times)]

            # === Özetleme ===
            df["event_hour"] = df["event_hour"].astype(int)
            df["day_of_week"] = df["datetime"].dt.dayofweek
            df["month"] = df["datetime"].dt.month
            df["season"] = df["month"].map(season_map)

            group_cols = ["GEOID", "season", "day_of_week", "event_hour"]
            mean_cols = ["latitude", "longitude", "past_7d_crimes", "crime_count_past_24h", "crime_count_past_48h", "crime_trend_score", "prev_crime_1h", "prev_crime_2h", "prev_crime_3h"]
            mode_cols = ["is_weekend", "is_night", "is_holiday", "is_repeat_location", "is_school_hour", "is_business_hour", "year", "month"]

            def safe_mode(x):
                try: return x.mode().iloc[0]
                except: return np.nan

            agg_dict = {col: "mean" for col in mean_cols}
            agg_dict.update({col: safe_mode for col in mode_cols})
            agg_dict.update({"date": "min", "id": "count"})

            df["id"] = 1
            grouped = df.groupby(group_cols).agg(agg_dict).reset_index()
            grouped = grouped.rename(columns={"id": "crime_count"})
            grouped["Y_label"] = (grouped["crime_count"] >= 2).astype(int)

            geoids = df["GEOID"].unique()
            seasons = ["Winter", "Spring", "Summer", "Fall"]
            days = list(range(7))
            hours = list(range(24))
            expected_grid = pd.DataFrame(itertools.product(geoids, seasons, days, hours), columns=group_cols)

            df_final = expected_grid.merge(grouped, on=group_cols, how="left")
            df_final["crime_count"] = df_final["crime_count"].fillna(0).astype(int)
            df_final["Y_label"] = df_final["Y_label"].fillna(0).astype(int)

            df_final["is_weekend"] = df_final["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)
            df_final["is_night"] = df_final["event_hour"].apply(lambda x: 1 if (x >= 20 or x < 4) else 0)
            df_final["is_school_hour"] = df_final.apply(lambda x: 1 if (x["day_of_week"] < 5 and 7 <= x["event_hour"] <= 16) else 0, axis=1)
            df_final["is_business_hour"] = df_final.apply(lambda x: 1 if (x["day_of_week"] < 6 and 9 <= x["event_hour"] < 18) else 0, axis=1)

            columns_with_nan = ["latitude", "longitude", "past_7d_crimes", "crime_count_past_24h", "crime_count_past_48h", "crime_trend_score", "prev_crime_1h", "prev_crime_2h", "prev_crime_3h", "is_holiday", "is_repeat_location", "year", "month", "date"]
            df_final = df_final.dropna(subset=columns_with_nan)

            existing_combinations = df_final[group_cols]
            missing = expected_grid.merge(existing_combinations.drop_duplicates(), on=group_cols, how="left", indicator=True)
            missing = missing[missing["_merge"] == "left_only"].drop(columns=["_merge"])
            missing["crime_count"] = 0
            missing["Y_label"] = 0

            df_full_52 = pd.concat([df_final, missing], ignore_index=True)

            df_final.to_csv("sf_crime_50.csv", index=False)
            df_full_52.to_csv("sf_crime_52.csv", index=False)
            df.to_csv("sf_crime.csv", index=False)
            st.success("✅ Tüm dosyalar başarıyla kaydedildi: sf_crime.csv, sf_crime_50.csv, sf_crime_52.csv")

    except Exception as e:
        st.error(f"❌ Hata oluştu: {e}")
