
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
DOWNLOAD_911_URL = "https://github.com/cem5113/crime_prediction_data/releases/download/v1.0.1/sf_911_last_5_year.csv"
DOWNLOAD_311_URL = "https://github.com/cem5113/crime_prediction_data/releases/download/v1.0.2/sf_311_last_5_years.csv"
DOWNLOAD_311_PATH = "311_requests_range.csv"

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

                # 911 verisini indir
                df_911 = None  # ön tanım
                try:
                    response_911 = requests.get(DOWNLOAD_911_URL)
                    if response_911.status_code == 200:
                        with open("sf_911_last_5_year.csv", "wb") as f:
                            f.write(response_911.content)
                        st.success("✅ sf_911_last_5_year.csv başarıyla indirildi.")
                        
                        # 👁‍🗨 İçeriği göster
                        df_911 = pd.read_csv("sf_911_last_5_year.csv")
                        st.write("📟 911 Verisi İlk 5 Satır")
                        st.dataframe(df_911.head())
                        st.write("📌 911 Sütunları:")
                        st.write(df_911.columns.tolist())
                    else:
                        st.warning(f"⚠️ sf_911_last_5_year.csv indirilemedi: {response_911.status_code}")
                except Exception as e:
                    st.error(f"❌ 911 verisi indirilemedi: {e}")

                # 311 verisini oku 
                df_311 = None
                if os.path.exists(DOWNLOAD_311_PATH):
                    try:
                        df_311 = pd.read_csv(DOWNLOAD_311_PATH)
                        st.success("✅ 311 verisi yüklendi.")
                        st.write("📟 311 Verisi İlk 5 Satır")
                        st.dataframe(df_311.head())
                        st.write("📌 311 Sütunları:")
                        st.write(df_311.columns.tolist())
                    except Exception as e:
                        st.warning(f"⚠️ 311 verisi yüklenemedi: {e}")
                else:
                    st.warning("⚠️ 311_requests_range.csv bulunamadı.")


                # Suç verisini oku
                df = pd.read_csv("sf_crime.csv", low_memory=False)
                original_row_count = len(df)

                # NaN özetle
                nan_summary = df.isna().sum()
                nan_cols = nan_summary[nan_summary > 0]
                removed_rows = 0  # Henüz satır silinmedi

                # PDF rapor oluştur
                report_path = create_pdf_report("sf_crime.csv", original_row_count, nan_cols, len(df), removed_rows)
                with open(report_path, "rb") as f:
                    st.download_button("📄 PDF Raporu İndir", f, file_name=report_path, mime="application/pdf")

            else:
                st.error(f"❌ sf_crime.csv indirilemedi, HTTP kodu: {response.status_code}")
                st.stop()  # Hatalı indirme varsa durdur
        except Exception as e:
            st.error(f"❌ Hata oluştu: {e}")

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

            # 911 verilerini yükle ve birleştir
            if os.path.exists("sf_911_last_5_year.csv"):
                df_911 = pd.read_csv("sf_911_last_5_year.csv")
                df_911["date"] = pd.to_datetime(df_911["date"]).dt.date
                df["hour_range"] = (df["event_hour"] // 3) * 3
                df["hour_range"] = df["hour_range"].astype(str) + "-" + (df["event_hour"] // 3 * 3 + 3).astype(str)
                
                # Birleştir
                df = pd.merge(df, df_911, on=["GEOID", "date", "hour_range"], how="left")
                
                # Yeni sütunları gözlemle
                cols_911 = [col for col in df.columns if "911" in col or "request" in col]
                st.write("🔍 911 Sütunları:")
                st.write(cols_911)
                st.write("🧯 911 NaN Sayıları:")
                st.write(df[cols_911].isna().sum())
            
                # Eksik olanları 0 yap
                for col in cols_911:
                    df[col] = df[col].fillna(0)
                
                # 311 verisini birleştir
                if df_311 is not None:
                    df_311["date"] = pd.to_datetime(df_311["date"]).dt.date
                    df = pd.merge(df, df_311, on=["GEOID", "date", "hour_range"], how="left")
                
                    cols_311 = [col for col in df.columns if "311" in col]
                    st.write("🔍 311 Sütunları:")
                    st.write(cols_311)
                    st.write("🧯 311 NaN Sayıları:")
                    st.write(df[cols_311].isna().sum())
                
                    for col in cols_311:
                        df[col] = df[col].fillna(0)

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
            mean_cols.extend([col for col in df.columns if "911" in col or "request" in col])
            mean_cols.extend([col for col in df.columns if "311" in col])

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

            # NaN raporu ve PDF
            nan_summary = df.isna().sum()
            nan_cols = nan_summary[nan_summary > 0]
            report_path = create_pdf_report("sf_crime.csv", original_row_count, nan_cols, len(df), removed_rows)
            with open(report_path, "rb") as f:
                st.download_button("📄 PDF Raporu İndir", f, file_name=report_path, mime="application/pdf")
    
            # İlk 5 satır, sütunlar, NaN sayıları
            st.write("### 📈 sf_crime.csv İlk 5 Satır")
            st.dataframe(df.head())
            st.write("### 🔢 Sütunlar")
            st.write(df.columns.tolist())
            st.write("### 🔔 NaN Sayıları")
            st.write(nan_cols)
            st.write("📦 sf_crime.csv Dosyasındaki 911 Sütunları ve İlk Satırlar:")
            st.dataframe(df[cols_911 + ["GEOID", "datetime"]].head())
    
            df.to_csv("sf_crime.csv", index=False)
            st.success("✅ sf_crime.csv dosyası zenginleştirildi ve kaydedildi.")
