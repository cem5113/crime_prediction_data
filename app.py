import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import holidays
import itertools
from datetime import datetime, timedelta
from fpdf import FPDF
import subprocess
import geopandas as gpd
import json
import requests
import io
import json
from shapely.geometry import Point
from scipy.spatial import cKDTree

st.set_page_config(page_title="Veri Güncelleme", layout="wide")
st.title("📦 Günlük Suç Verisi İşleme ve Özetleme Paneli")

DOWNLOAD_URL = "https://github.com/cem5113/crime_prediction_data/releases/download/latest/sf_crime.csv"
DOWNLOAD_911_URL = "https://github.com/cem5113/crime_prediction_data/releases/download/v1.0.1/sf_911_last_5_year.csv"
DOWNLOAD_311_URL = "https://github.com/cem5113/crime_prediction_data/releases/download/v1.0.2/sf_311_last_5_years.csv"
POPULATION_PATH = "sf_population.csv"
DOWNLOAD_BUS_URL = "https://github.com/cem5113/crime_prediction_data/raw/main/sf_bus_stops_with_geoid.csv"
DOWNLOAD_TRAIN_URL = "https://github.com/cem5113/crime_prediction_data/raw/main/sf_train_stops_with_geoid.csv"
DOWNLOAD_POIS_URL = "https://github.com/cem5113/crime_prediction_data/raw/main/sf_pois.geojson"
RISKY_POIS_JSON_PATH = "risky_pois_dynamic.json"
DOWNLOAD_POLICE_URL = "https://github.com/cem5113/crime_prediction_data/raw/main/sf_police_stations.csv"
DOWNLOAD_GOV_URL = "https://github.com/cem5113/crime_prediction_data/raw/main/sf_government_buildings.csv"
DOWNLOAD_WEATHER_URL = "https://github.com/cem5113/crime_prediction_data/releases/download/latest/sf_weather_5years.csv"

def update_bus_data_if_needed():
    import geopandas as gpd
    from shapely.geometry import Point
    import os

    api_url = "https://data.sfgov.org/resource/i28k-bkz6.json"
    timestamp_file = "bus_stops_last_update.txt"

    def is_month_passed(file):
        if os.path.exists(file):
            with open(file, "r") as f:
                last = f.read().strip()
            try:
                last_date = datetime.strptime(last, "%Y-%m-%d")
                return (datetime.today() - last_date).days >= 30
            except:
                return True
        return True

    if is_month_passed(timestamp_file):
        try:
            response = requests.get(api_url)
            if response.status_code == 200:
                bus_data = response.json()
                df = pd.DataFrame(bus_data)

                # latitude / longitude sütunlarını float olarak al
                df = df.dropna(subset=["latitude", "longitude"])
                df["stop_lat"] = df["latitude"].astype(float)
                df["stop_lon"] = df["longitude"].astype(float)

                # GeoDataFrame'e çevir
                gdf_stops = gpd.GeoDataFrame(
                    df,
                    geometry=gpd.points_from_xy(df["stop_lon"], df["stop_lat"]),
                    crs="EPSG:4326"
                )

                # Census bloklarını oku
                census_path = "sf_census_blocks_with_population.geojson"
                gdf_blocks = gpd.read_file(census_path)[["GEOID", "geometry"]].to_crs("EPSG:4326")

                # Spatial join
                gdf_joined = gpd.sjoin(gdf_stops, gdf_blocks, how="left", predicate="within")
                gdf_joined["GEOID"] = gdf_joined["GEOID"].astype(str).str.zfill(11)
                gdf_joined.drop(columns=["geometry", "index_right"], errors="ignore").to_csv("sf_bus_stops_with_geoid.csv", index=False)

                # Tarih kaydet
                with open(timestamp_file, "w") as f:
                    f.write(datetime.today().strftime("%Y-%m-%d"))

                st.success("🚌 Otobüs durakları Socrata API'den indirildi ve GEOID ile eşleştirildi.")
            else:
                st.warning(f"⚠️ Otobüs verisi indirilemedi: {response.status_code}")
        except Exception as e:
            st.error(f"❌ Otobüs verisi güncellenemedi: {e}")
    else:
        st.info("📅 Otobüs verisi bu ay zaten güncellendi.")

def update_train_data_if_needed():
    import zipfile
    import geopandas as gpd
    import pandas as pd
    import os
    from datetime import datetime

    zip_path = "bart_gtfs.zip"
    timestamp_file = "train_stops_last_update.txt"

    def is_month_passed(file):
        if os.path.exists(file):
            with open(file, "r") as f:
                last = f.read().strip()
            try:
                last_date = datetime.strptime(last, "%Y-%m-%d")
                return (datetime.today() - last_date).days >= 30
            except:
                return True
        return True

    if is_month_passed(timestamp_file):
        try:
            response = requests.get(DOWNLOAD_TRAIN_URL)
            if response.status_code == 200:
                with open(zip_path, "wb") as f:
                    f.write(response.content)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extract("stops.txt", ".")
                os.rename("stops.txt", "sf_train_stops.csv")
                with open(timestamp_file, "w") as f:
                    f.write(datetime.today().strftime("%Y-%m-%d"))
                st.success("🚆 BART tren durakları güncellendi (sf_train_stops.csv)")

                # === GEOID EŞLEME ===
                train_df = pd.read_csv("sf_train_stops.csv")
                gdf_stops = gpd.GeoDataFrame(
                    train_df,
                    geometry=gpd.points_from_xy(train_df["stop_lon"], train_df["stop_lat"]),
                    crs="EPSG:4326"
                )

                census_path = "https://github.com/cem5113/crime_prediction_data/raw/main/sf_pois.geojson"
                gdf_blocks = gpd.read_file(census_path)[["GEOID", "geometry"]].to_crs("EPSG:4326")

                gdf_joined = gpd.sjoin(gdf_stops, gdf_blocks, how="left", predicate="within")
                gdf_joined["GEOID"] = gdf_joined["GEOID"].astype(str).str.zfill(11)
                gdf_joined.drop(columns=["geometry", "index_right"], errors="ignore").to_csv("sf_train_stops_with_geoid.csv", index=False)

                st.success("📌 GEOID ile eşleştirilmiş tren durakları oluşturuldu (sf_train_stops_with_geoid.csv)")

            else:
                st.warning(f"⚠️ Tren verisi indirilemedi: {response.status_code}")
        except Exception as e:
            st.error(f"❌ Tren verisi indirme hatası: {e}")
    else:
        st.info("📅 Tren verisi bu ay zaten güncellenmiş.")

def update_pois_if_needed():
    import os
    import pandas as pd
    import streamlit as st
    from datetime import datetime
    import update_pois  # 🔁 SUBPROCESS YERİNE DİREKT MODÜL GİBİ KULLAN

    timestamp_file = "poi_last_update.txt"

    def is_month_passed(file):
        if os.path.exists(file):
            with open(file, "r") as f:
                last = f.read().strip()
            try:
                last_date = datetime.strptime(last, "%Y-%m-%d")
                return (datetime.today() - last_date).days >= 30
            except:
                return True
        return True

    if is_month_passed(timestamp_file):
        try:
            st.info("📥 POI verisi güncelleniyor...")
            update_pois.process_pois()
            update_pois.calculate_dynamic_risk()

            with open(timestamp_file, "w") as f:
                f.write(datetime.today().strftime("%Y-%m-%d"))

            st.success("✅ POI verisi başarıyla güncellendi.")
        except Exception as e:
            st.error(f"❌ POI güncelleme hatası: {e}")
    else:
        st.info("📅 POI verisi bu ay zaten güncellendi.")

def update_police_and_gov_buildings_if_needed():
    import requests
    import geopandas as gpd
    import pandas as pd
    import os
    from datetime import datetime
    from shapely.geometry import Point

    timestamp_file = "police_gov_last_update.txt"
    overpass_url = "http://overpass-api.de/api/interpreter"

    def is_month_passed(file):
        if os.path.exists(file):
            with open(file, "r") as f:
                last = f.read().strip()
            try:
                last_date = datetime.strptime(last, "%Y-%m-%d")
                return (datetime.today() - last_date).days >= 30
            except:
                return True
        return True

    if is_month_passed(timestamp_file):
        try:
            st.write("🌐 Overpass API'den veri çekiliyor...")

            # === Overpass Sorguları ===
            queries = {
                "police": """
                [out:json][timeout:60];
                (
                  node["amenity"="police"](37.70,-123.00,37.83,-122.35);
                  way["amenity"="police"](37.70,-123.00,37.83,-122.35);
                );
                out center;
                """,
                "government": """
                [out:json][timeout:60];
                (
                  node["amenity"="townhall"](37.70,-123.00,37.83,-122.35);
                  node["office"="government"](37.70,-123.00,37.83,-122.35);
                );
                out center;
                """
            }

            def fetch_pois(name, query):
                response = requests.post(overpass_url, data={"data": query})
                data = response.json()["elements"]

                rows = []
                for el in data:
                    lat = el.get("lat") or el.get("center", {}).get("lat")
                    lon = el.get("lon") or el.get("center", {}).get("lon")
                    if lat and lon:
                        tags = el.get("tags", {})
                        rows.append({
                            "id": el["id"],
                            "lat": lat,
                            "lon": lon,
                            "name": tags.get("name", ""),
                            "type": tags.get("amenity") or tags.get("office", ""),
                        })

                df = pd.DataFrame(rows)
                df["latitude"] = df["lat"]
                df["longitude"] = df["lon"]
                df = df.drop(columns=["lat", "lon"])
                gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs="EPSG:4326")
                return gdf

            gdf_police = fetch_pois("police", queries["police"])
            gdf_police.to_csv("sf_police_stations.csv", index=False)
            st.success("✅ sf_police_stations.csv indirildi ve kaydedildi.")

            gdf_gov = fetch_pois("government", queries["government"])
            gdf_gov.to_csv("sf_government_buildings.csv", index=False)
            st.success("✅ sf_government_buildings.csv indirildi ve kaydedildi.")

            with open(timestamp_file, "w") as f:
                f.write(datetime.today().strftime("%Y-%m-%d"))

        except Exception as e:
            st.error(f"❌ Polis/kamu binası güncelleme hatası: {e}")
    else:
        st.info("📅 Polis ve kamu binası verisi bu ay zaten güncellendi.")

def update_weather_data():
    st.info("🌦️ Hava durumu verisi kontrol ediliyor...")
    try:
        save_path = "sf_weather_5years.csv"
        station_id = "USW00023234"
        end_date = datetime.today().date()
        start_date = end_date - pd.Timedelta(days=5*365)

        url = (
            "https://www.ncei.noaa.gov/access/services/data/v1"
            f"?dataset=daily-summaries"
            f"&stations={station_id}"
            f"&startDate={start_date}"
            f"&endDate={end_date}"
            f"&dataTypes=TMAX,TMIN,PRCP"
            f"&format=csv"
        )

        response = requests.get(url)
        if response.status_code == 200:
            df_new = pd.read_csv(io.StringIO(response.text))
            df_new["DATE"] = pd.to_datetime(df_new["DATE"])

            if os.path.exists(save_path):
                df_old = pd.read_csv(save_path)
                df_old["DATE"] = pd.to_datetime(df_old["DATE"])
                df_combined = pd.concat([df_old, df_new])
                df_combined = df_combined.drop_duplicates(subset=["DATE"])
            else:
                df_combined = df_new

            df_filtered = df_combined[df_combined["DATE"] >= pd.to_datetime(start_date)]
            df_filtered.to_csv(save_path, index=False)

            st.success(f"✅ Hava durumu güncellendi: {start_date} → {end_date}")
        else:
            st.warning(f"❌ NOAA'dan veri çekilemedi: {response.status_code}")
    except Exception as e:
        st.error(f"❌ Hava durumu güncellenemedi: {e}")

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

                update_train_data_if_needed()
                update_bus_data_if_needed() 
                update_pois_if_needed()
                update_pois_if_needed()
                update_weather_data()
                update_police_and_gov_buildings_if_needed()
                if os.path.exists("sf_pois_cleaned_with_geoid.csv"):
                    st.success("✅ POI CSV dosyası başarıyla oluşturuldu.")
                else:
                    st.error("❌ POI CSV dosyası eksik!")
            
                if os.path.exists("risky_pois_dynamic.json"):
                    st.success("✅ POI risk skoru dosyası oluşturuldu.")
                else:
                    st.error("❌ Risk skoru JSON dosyası bulunamadı.")

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
                try:
                    response_311 = requests.get(DOWNLOAD_311_URL)
                    if response_311.status_code == 200:
                        with open("sf_311_last_5_years.csv", "wb") as f:
                            f.write(response_311.content)
                        st.success("✅ sf_311_last_5_years.csv başarıyla indirildi.")
                
                        df_311 = pd.read_csv("sf_311_last_5_years.csv")
                        df_311["date"] = pd.to_datetime(df_311["date"]).dt.date
                
                        st.write("📟 311 Verisi İlk 5 Satır")
                        st.dataframe(df_311.head())
                        st.write("📌 311 Sütunları:")
                        st.write(df_311.columns.tolist())
                
                    else:
                        st.warning(f"⚠️ sf_311_last_5_years.csv indirilemedi: {response_311.status_code}")
                except Exception as e:
                    st.error(f"❌ 311 verisi yüklenemedi: {e}")
                    
                # Suç verisini oku
                df = pd.read_csv("sf_crime.csv", low_memory=False)
                original_row_count = len(df)

                # Suç verisini oku
                df = pd.read_csv("sf_crime.csv", low_memory=False)
                original_row_count = len(df)

                # 🔁 POI Risk ve Yoğunluk Özelliklerini Ekle
                try:
                    df_poi = pd.read_csv("sf_pois_cleaned_with_geoid.csv")
                    
                    with open("risky_pois_dynamic.json") as f:
                        risk_dict = json.load(f)
                    
                    df_poi["risk_score"] = df_poi["poi_subcategory"].map(risk_dict).fillna(0)

                    poi_features = df_poi.groupby("GEOID").agg(
                        poi_total_count=("id", "count"),
                        risky_poi_score=("risk_score", "mean")
                    ).reset_index()

                    df["GEOID"] = df["GEOID"].astype(str).str.zfill(11)
                    poi_features["GEOID"] = poi_features["GEOID"].astype(str).str.zfill(11)
                    df = df.merge(poi_features, on="GEOID", how="left")

                    st.success("✅ POI yoğunluğu ve risk skoru başarıyla eklendi.")
                    st.write("📍 Örnek POI verisi:")
                    st.dataframe(df[["GEOID", "poi_total_count", "risky_poi_score"]].drop_duplicates().head())

                except Exception as e:
                    st.warning(f"⚠️ POI verisi eklenemedi: {e}")
                
                try:
                    df_poi = pd.read_csv("sf_pois_cleaned_with_geoid.csv")
                    df_poi["risk_score"] = df_poi["poi_subcategory"].map(risk_dict).fillna(0)
                    
                    gdf_crime = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs="EPSG:4326").to_crs(3857)
                    gdf_poi = gpd.GeoDataFrame(df_poi, geometry=gpd.points_from_xy(df_poi["lon"], df_poi["lat"]), crs="EPSG:4326").to_crs(3857)
                
                    # Genel POI mesafesi
                    poi_coords = np.vstack([gdf_poi.geometry.x, gdf_poi.geometry.y]).T
                    crime_coords = np.vstack([gdf_crime.geometry.x, gdf_crime.geometry.y]).T
                    poi_tree = cKDTree(poi_coords)
                    df["distance_to_poi"], _ = poi_tree.query(crime_coords, k=1)
                
                    # Riskli POI’lere mesafe
                    risky_poi = gdf_poi[gdf_poi["risk_score"] > 0]
                    if not risky_poi.empty:
                        risky_coords = np.vstack([risky_poi.geometry.x, risky_poi.geometry.y]).T
                        risky_tree = cKDTree(risky_coords)
                        df["distance_to_high_risk_poi"], _ = risky_tree.query(crime_coords, k=1)
                    else:
                        df["distance_to_high_risk_poi"] = np.nan
                
                    # POI Risk yoğunluğu (GEOID bazlı)
                    risk_density = df_poi.groupby("GEOID")["risk_score"].mean().reset_index(name="poi_risk_density")
                    df["GEOID"] = df["GEOID"].astype(str).str.zfill(11)
                    risk_density["GEOID"] = risk_density["GEOID"].astype(str).str.zfill(11)
                
                    st.success("✅ POI mesafe ve risk yoğunluğu eklendi.")
                except Exception as e:
                    st.error(f"❌ POI mesafe/risk hesaplama hatası: {e}")

                # Nüfus verisini oku
                if os.path.exists(POPULATION_PATH):
                    df_pop = pd.read_csv(POPULATION_PATH)
                    df["GEOID"] = df["GEOID"].astype(str).str.extract(r'(\d+)')[0].str.zfill(11)
                    df_pop["GEOID"] = df_pop["GEOID"].astype(str).str.zfill(11)
                    df = pd.merge(df, df_pop, on="GEOID", how="left")
                    df["population"] = df["population"].fillna(0).astype(int)
                    st.success("✅ Nüfus verisi eklendi.")
                    st.write("👥 Nüfus örnek verisi:")
                    st.dataframe(df[["GEOID", "population"]].drop_duplicates().head())
                else:
                    st.warning("⚠️ Nüfus verisi (sf_population.csv) bulunamadı.")

                # Otobüs durak verisini indir
                df_bus = None
                try:
                    response_bus = requests.get(DOWNLOAD_BUS_URL)
                    if response_bus.status_code == 200:
                        with open("sf_bus_stops.csv", "wb") as f:
                            f.write(response_bus.content)
                        st.success("✅ sf_bus_stops.csv başarıyla indirildi.")
                        df_bus = pd.read_csv("sf_bus_stops.csv").dropna(subset=["stop_lat", "stop_lon"])
                        st.write("🚌 Otobüs Verisi İlk 5 Satır:")
                        st.dataframe(df_bus.head())
                    else:
                        st.warning(f"⚠️ sf_bus_stops.csv indirilemedi: {response_bus.status_code}")
                except Exception as e:
                    st.error(f"❌ Otobüs verisi indirilemedi: {e}")

                # NaN özetle
                nan_summary = df.isna().sum()
                nan_cols = nan_summary[nan_summary > 0]
                removed_rows = 0  # Henüz satır silinmedi

                # PDF rapor oluştur
                removed_rows = original_row_count - len(df)
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
            df["datetime"] = df["datetime"].dt.floor("h")
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
                    if "hour_range" not in df_311.columns and "time" in df_311.columns:
                        df_311["datetime"] = pd.to_datetime(df_311["date"].astype(str) + " " + df_311["time"].astype(str), errors="coerce")
                        df_311["hour"] = df_311["datetime"].dt.hour
                        df_311["hour_range"] = (df_311["hour"] // 3) * 3
                        df_311["hour_range"] = df_311["hour_range"].astype(str) + "-" + (df_311["hour_range"] + 3).astype(str)
                
                    # Merge öncesi tip düzeltmeleri
                    df["GEOID"] = df["GEOID"].astype(str).str.extract(r"(\d+)")[0].str.zfill(11)
                    df["GEOID"] = df["GEOID"].apply(lambda x: str(int(x)).zfill(11) if pd.notna(x) else None)
                    df_311["GEOID"] = df_311["GEOID"].apply(lambda x: str(int(float(x))).zfill(11) if pd.notna(x) else None)
                    df["date"] = pd.to_datetime(df["date"]).dt.date
                    df_311["date"] = pd.to_datetime(df_311["date"]).dt.date
                    df["hour_range"] = df["hour_range"].astype(str)
                    df_311["hour_range"] = df_311["hour_range"].astype(str)
                
                    # Aggregate: saat aralığı başına toplam çağrı
                    agg_311 = df_311.groupby(["GEOID", "date", "hour_range"]).size().reset_index(name="311_request_count")
                    df = pd.merge(df, agg_311, on=["GEOID", "date", "hour_range"], how="left")
                    df["311_request_count"] = df["311_request_count"].fillna(0)
                
                    # Ek sütunları (örneğin category vs.) merge et (örnek kayıt üzerinden)
                    meta_cols = ["GEOID", "date", "hour_range", "category", "subcategory"]
                    df_311_meta = df_311[meta_cols].drop_duplicates()
                    df = pd.merge(df, df_311_meta, on=["GEOID", "date", "hour_range"], how="left")
                
                    # Göstermek için:
                    cols_311 = [col for col in df.columns if "311" in col or col in ["category", "subcategory"]]
                    st.write("🔍 311 Sütunları:")
                    st.write(cols_311)
                    st.write("🧯 311 NaN Sayıları:")
                    st.write(df[cols_311].isna().sum())
                
                    for col in cols_311:
                        df[col] = df[col].fillna(0) if df[col].dtype != 'object' else df[col].fillna("Unknown")
                        
            df = df.sort_values(by=["GEOID", "datetime"]).reset_index(drop=True)

            # En yakın otobüs durağına mesafe ve durak sayısı
            if df_bus is not None:
                try:
                    import geopandas as gpd
                    from shapely.geometry import Point
                    from scipy.spatial import cKDTree
                    import numpy as np
            
                    gdf_crime = gpd.GeoDataFrame(
                        df,
                        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
                        crs="EPSG:4326"
                    ).to_crs(epsg=3857)
            
                    gdf_bus = gpd.GeoDataFrame(
                        df_bus,
                        geometry=gpd.points_from_xy(df_bus["stop_lon"], df_bus["stop_lat"]),
                        crs="EPSG:4326"
                    ).to_crs(epsg=3857)
            
                    crime_coords = np.vstack([gdf_crime.geometry.x, gdf_crime.geometry.y]).T
                    bus_coords = np.vstack([gdf_bus.geometry.x, gdf_bus.geometry.y]).T
                    tree = cKDTree(bus_coords)
                    distances, _ = tree.query(crime_coords, k=1)
                    df["distance_to_bus"] = distances
            
                    dynamic_radius = np.percentile(distances, 75)
                    def count_stops(pt):
                        return gdf_bus.distance(pt).lt(dynamic_radius).sum()
            
                    df["bus_stop_count"] = gdf_crime.geometry.apply(count_stops)
            
                    st.success("✅ Otobüs mesafesi ve durak sayısı eklendi.")
                    st.write(df[["GEOID", "distance_to_bus", "bus_stop_count"]].head())
                except Exception as e:
                    st.error(f"❌ Otobüs entegrasyon hatası: {e}")

                # === 🚆 En Yakın Tren Durağı ve Durağa Uzaklık ===
                try:
                    import geopandas as gpd
                    from shapely.geometry import Point
                    from scipy.spatial import cKDTree
                    import numpy as np
                
                    # 1. Suç verisini GeoDataFrame'e çevir
                    gdf_crime = gpd.GeoDataFrame(
                        df,
                        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
                        crs="EPSG:4326"
                    ).to_crs(epsg=3857)
                
                    # 2. Tren duraklarını oku ve GeoDataFrame'e çevir
                    if os.path.exists("sf_train_stops_with_geoid.csv"):
                        df_train = pd.read_csv("sf_train_stops_with_geoid.csv").dropna(subset=["stop_lat", "stop_lon"])
                        gdf_train = gpd.GeoDataFrame(
                            df_train,
                            geometry=gpd.points_from_xy(df_train["stop_lon"], df_train["stop_lat"]),
                            crs="EPSG:4326"
                        ).to_crs(epsg=3857)
                
                        # 3. KDTree ile mesafe hesapla
                        crime_coords = np.vstack([gdf_crime.geometry.x, gdf_crime.geometry.y]).T
                        train_coords = np.vstack([gdf_train.geometry.x, gdf_train.geometry.y]).T
                        tree = cKDTree(train_coords)
                        distances, _ = tree.query(crime_coords, k=1)
                        df["distance_to_train"] = distances
                
                        # 4. Belirli bir yarıçapta (örn. 500m) tren durağı sayısını hesapla
                        radius = 500  # metre
                        df["train_stop_count"] = gdf_crime.geometry.apply(lambda pt: gdf_train.distance(pt).lt(radius).sum())
                
                        st.success("🚆 Tren mesafesi ve durak sayısı eklendi.")
                        st.write(df[["GEOID", "distance_to_train", "train_stop_count"]].head())
                    else:
                        st.warning("⚠️ sf_train_stops_with_geoid.csv bulunamadı. Tren durakları eklenmedi.")
                
                except Exception as e:
                    st.error(f"❌ Tren entegrasyon hatası: {e}")

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
            mode_cols = [
                "is_weekend", "is_night", "is_holiday", "is_repeat_location",
                "is_school_hour", "is_business_hour", "year", "month",
                "distance_to_police_range", "distance_to_government_building_range",
                "is_near_police", "is_near_government"
            ]
            
            mean_cols.extend([
                col for col in df.columns if "911" in col or "request" in col
            ])
            mean_cols.extend(["distance_to_bus", "bus_stop_count"])
            mean_cols.extend([
                col for col in df.columns if "311" in col
            ])
            mean_cols.extend([
                "poi_total_count", "risky_poi_score", "distance_to_high_risk_poi",
                "distance_to_poi", "poi_risk_density",
                "distance_to_police", "distance_to_government_building"
            ])
            mean_cols.extend([
                "temp_max",               # Maksimum sıcaklık
                "temp_min",               # Minimum sıcaklık
                "precipitation_mm",       # Yağış miktarı
                "temp_range",             # Sıcaklık aralığı
                "precipitation_range"     # Yağış aralığı (varsa)
            ])
            if "population" in df.columns:
                mean_cols.append("population")
                
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

            st.subheader("📊 Zenginleştirilmiş Suç Verisi (Örnek)")
            st.write("🧩 Sütunlar:")
            st.write(df.columns.tolist())
            
            st.write("🔍 İlk 5 Satır:")
            st.dataframe(df.head())

# === 5 FONKSİYON: Veri zenginleştirme ===
def enrich_with_poi(df):
    try:
        df_poi = pd.read_csv("sf_pois_cleaned_with_geoid.csv")
        with open("risky_pois_dynamic.json") as f:
            risk_dict = json.load(f)
        df_poi["risk_score"] = df_poi["poi_subcategory"].map(risk_dict).fillna(0)

        poi_features = df_poi.groupby("GEOID").agg(
            poi_total_count=("id", "count"),
            risky_poi_score=("risk_score", "mean")
        ).reset_index()

        df["GEOID"] = df["GEOID"].astype(str).str.zfill(11)
        poi_features["GEOID"] = poi_features["GEOID"].astype(str).str.zfill(11)
        df = df.merge(poi_features, on="GEOID", how="left")

        gdf_crime = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs="EPSG:4326").to_crs(3857)
        gdf_poi = gpd.GeoDataFrame(df_poi, geometry=gpd.points_from_xy(df_poi["lon"], df_poi["lat"]), crs="EPSG:4326").to_crs(3857)

        poi_coords = np.vstack([gdf_poi.geometry.x, gdf_poi.geometry.y]).T
        crime_coords = np.vstack([gdf_crime.geometry.x, gdf_crime.geometry.y]).T
        poi_tree = cKDTree(poi_coords)
        df["distance_to_poi"], _ = poi_tree.query(crime_coords, k=1)

        risky_poi = gdf_poi[gdf_poi["risk_score"] > 0]
        if not risky_poi.empty:
            risky_coords = np.vstack([risky_poi.geometry.x, risky_poi.geometry.y]).T
            risky_tree = cKDTree(risky_coords)
            df["distance_to_high_risk_poi"], _ = risky_tree.query(crime_coords, k=1)
        else:
            df["distance_to_high_risk_poi"] = np.nan

        risk_density = df_poi.groupby("GEOID")["risk_score"].mean().reset_index(name="poi_risk_density")
        risk_density["GEOID"] = risk_density["GEOID"].astype(str).str.zfill(11)
        df = df.merge(risk_density, on="GEOID", how="left")

        return df
    except Exception as e:
        st.error(f"POI zenginleştirme hatası: {e}")
        return df

def enrich_with_911(df):
    try:
        df_911 = pd.read_csv("sf_911_last_5_year.csv")
        df_911["date"] = pd.to_datetime(df_911["date"])
        df["date"] = pd.to_datetime(df["date"])
        df = df.merge(df_911, on=["GEOID", "date", "event_hour"], how="left")
        return df
    except Exception as e:
        st.error(f"911 verisi eklenemedi: {e}")
        return df

def enrich_with_311(df):
    try:
        df_311 = pd.read_csv("sf_311_last_5_years.csv")
        df_311["date"] = pd.to_datetime(df_311["date"])
        df["date"] = pd.to_datetime(df["date"])
        df = df.merge(df_311, on=["GEOID", "date", "event_hour"], how="left")
        return df
    except Exception as e:
        st.error(f"311 verisi eklenemedi: {e}")
        return df

def enrich_with_weather(df):
    try:
        df_weather = pd.read_csv("sf_weather_5years.csv")
        df_weather["DATE"] = pd.to_datetime(df_weather["DATE"])
        df["date"] = pd.to_datetime(df["date"])
        df = df.merge(df_weather, left_on="date", right_on="DATE", how="left")
        df.drop(columns=["DATE"], inplace=True)
        return df
    except Exception as e:
        st.error(f"Hava durumu verisi eklenemedi: {e}")
        return df

def enrich_with_police_and_gov(df):
    try:
        gdf_crime = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs="EPSG:4326").to_crs(3857)

        police_df = pd.read_csv("sf_police_stations.csv")
        gov_df = pd.read_csv("sf_government_buildings.csv")

        gdf_police = gpd.GeoDataFrame(police_df, geometry=gpd.points_from_xy(police_df["longitude"], police_df["latitude"]), crs="EPSG:4326").to_crs(3857)
        gdf_gov = gpd.GeoDataFrame(gov_df, geometry=gpd.points_from_xy(gov_df["longitude"], gov_df["latitude"]), crs="EPSG:4326").to_crs(3857)

        crime_coords = np.vstack([gdf_crime.geometry.x, gdf_crime.geometry.y]).T

        police_tree = cKDTree(np.vstack([gdf_police.geometry.x, gdf_police.geometry.y]).T)
        gov_tree = cKDTree(np.vstack([gdf_gov.geometry.x, gdf_gov.geometry.y]).T)

        df["distance_to_police"], _ = police_tree.query(crime_coords, k=1)
        df["distance_to_government_building"], _ = gov_tree.query(crime_coords, k=1)

        df["is_near_police"] = (df["distance_to_police"] < 200).astype(int)
        df["is_near_government"] = (df["distance_to_government_building"] < 200).astype(int)

        df["distance_to_police_range"] = pd.cut(df["distance_to_police"], bins=[0, 100, 200, 500, 1000, np.inf], labels=["0-100", "100-200", "200-500", "500-1000", ">1000"])
        df["distance_to_government_building_range"] = pd.cut(df["distance_to_government_building"], bins=[0, 100, 200, 500, 1000, np.inf], labels=["0-100", "100-200", "200-500", "500-1000", ">1000"])

        return df
    except Exception as e:
        st.error(f"Polis ve devlet binası hesaplama hatası: {e}")
        return df

# (app.py'deki ana kodlar burada devam eder)

# Örnek test butonu:
if st.button("🧪 Veriyi Göster (Test)"):
    df = pd.read_csv("sf_crime.csv")
    df["datetime"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str), errors="coerce")
    df["event_hour"] = df["datetime"].dt.hour
    df["date"] = df["datetime"].dt.date
    df = enrich_with_poi(df)
    df = enrich_with_911(df)
    df = enrich_with_311(df)
    df = enrich_with_weather(df)
    df = enrich_with_police_and_gov(df)
    df.to_csv("sf_crime.csv", index=False)

    st.success("✅ sf_crime.csv başarıyla zenginleştirildi.")
    st.write("📌 Sütunlar:", df.columns.tolist())
    st.dataframe(df.head())


