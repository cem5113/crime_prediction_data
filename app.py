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

                # ✅ Önizleme göster
                st.write("📌 [sf_bus_stops_with_geoid.csv] sütunlar:")
                st.write(gdf_joined.columns.tolist())
                st.write("📋 İlk 3 satır:")
                st.dataframe(gdf_joined.head(3))

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
                st.write("📋 [Tren Durakları] Sütunlar:")
                st.write(train_df.columns.tolist())
                st.write("🚉 [Tren Durakları] İlk 3 Satır:")
                st.dataframe(train_df.head(3))

                gdf_stops = gpd.GeoDataFrame(
                    train_df.dropna(subset=["stop_lat", "stop_lon"]),
                    geometry=gpd.points_from_xy(train_df["stop_lon"], train_df["stop_lat"]),
                    crs="EPSG:4326"
                )

                census_path = "https://github.com/cem5113/crime_prediction_data/raw/main/sf_pois.geojson"
                gdf_blocks = gpd.read_file(census_path)[["GEOID", "geometry"]].to_crs("EPSG:4326")

                gdf_joined = gpd.sjoin(gdf_stops, gdf_blocks, how="left", predicate="within")
                gdf_joined["GEOID"] = gdf_joined["GEOID"].astype(str).str.zfill(11)

                final_df = gdf_joined.drop(columns=["geometry", "index_right"], errors="ignore")
                final_df.to_csv("sf_train_stops_with_geoid.csv", index=False)

                st.success("📌 GEOID ile eşleştirilmiş tren durakları oluşturuldu (sf_train_stops_with_geoid.csv)")
                st.write("📋 [Tren + GEOID] Sütunlar:")
                st.write(final_df.columns.tolist())
                st.write("🚉 [Tren + GEOID] İlk 3 Satır:")
                st.dataframe(final_df.head(3))

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

            # 📦 POI işleme (temizleme ve risk hesaplama)
            update_pois.process_pois()
            update_pois.calculate_dynamic_risk()

            # 🕒 Güncelleme zamanını kaydet
            with open(timestamp_file, "w") as f:
                f.write(datetime.today().strftime("%Y-%m-%d"))

            st.success("✅ POI verisi başarıyla güncellendi.")

            # 📄 Sonuç dosyasını göster
            poi_path = "sf_pois_cleaned_with_geoid.csv"
            if os.path.exists(poi_path):
                df_poi = pd.read_csv(poi_path)
                st.write("📌 [sf_pois_cleaned_with_geoid.csv] sütunlar:")
                st.write(df_poi.columns.tolist())
                st.write("📋 İlk 3 satır:")
                st.dataframe(df_poi.head(3))
            else:
                st.warning("⚠️ POI dosyası bulunamadı (sf_pois_cleaned_with_geoid.csv)")

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
                df["latitude"] = pd.to_numeric(df["lat"], errors="coerce")
                df["longitude"] = pd.to_numeric(df["lon"], errors="coerce")
                df = df.drop(columns=["lat", "lon"])
                gdf = gpd.GeoDataFrame(df.dropna(subset=["latitude", "longitude"]),
                                       geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
                                       crs="EPSG:4326")
                return gdf

            # 🔹 Polis istasyonlarını al
            gdf_police = fetch_pois("police", queries["police"])
            gdf_police.to_csv("sf_police_stations.csv", index=False)
            st.success("✅ sf_police_stations.csv indirildi ve kaydedildi.")
            st.write("📌 Polis verisi sütunları:")
            st.write(gdf_police.columns.tolist())
            st.write("📋 İlk 3 satır:")
            st.dataframe(gdf_police.head(3))

            # 🔹 Devlet binalarını al
            gdf_gov = fetch_pois("government", queries["government"])
            gdf_gov.to_csv("sf_government_buildings.csv", index=False)
            st.success("✅ sf_government_buildings.csv indirildi ve kaydedildi.")
            st.write("📌 Devlet verisi sütunları:")
            st.write(gdf_gov.columns.tolist())
            st.write("📋 İlk 3 satır:")
            st.dataframe(gdf_gov.head(3))

            # 🔄 Güncelleme zamanını yaz
            with open(timestamp_file, "w") as f:
                f.write(datetime.today().strftime("%Y-%m-%d"))

        except Exception as e:
            st.error(f"❌ Polis/kamu binası güncelleme hatası: {e}")
    else:
        st.info("📅 Polis ve kamu binası verisi bu ay zaten güncellendi.")

def update_weather_data():
    import pandas as pd
    import streamlit as st
    import requests
    import os
    import io
    from datetime import datetime

    st.info("🌦️ Hava durumu verisi kontrol ediliyor...")
    try:
        save_path = "sf_weather_5years.csv"
        station_id = "USW00023234"  # San Francisco Hava İstasyonu
        end_date = datetime.today().date()
        start_date = end_date - pd.Timedelta(days=5 * 365)

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

            # ✅ Sonuç göster
            st.write("📌 [sf_weather_5years.csv] sütunlar:")
            st.write(df_filtered.columns.tolist())
            st.write("📋 İlk 3 satır:")
            st.dataframe(df_filtered.head(3))

        else:
            st.warning(f"❌ NOAA'dan veri çekilemedi: {response.status_code}")
    except Exception as e:
        st.error(f"❌ Hava durumu güncellenemedi: {e}")

def create_pdf_report(file_name, row_count_before, nan_cols, row_count_after, removed_rows):
    """Veri temizleme/işleme sonrası özet PDF raporu oluşturur."""
    now = datetime.now()
    timestamp = now.strftime("%d.%m.%Y %H:%M:%S")

    # NaN sütunlarını metin olarak derle
    if not nan_cols.empty:
        nan_parts = [f"- {col}: {count}" for col, count in nan_cols.items()]
        nan_text = "\n".join(nan_parts)
    else:
        nan_text = "Yok"

    # Rapor metni
    summary = (
        f"🕒 Tarih/Saat: {timestamp}\n"
        f"📄 Dosya: {file_name}\n"
        f"📊 Toplam satır (önce): {row_count_before:,}\n"
        f"📉 Toplam satır (sonra): {row_count_after:,}\n"
        f"🗑️ Silinen eski tarihli satır sayısı: {removed_rows}\n"
        f"⚠️ NaN içeren sütunlar:\n{nan_text}"
    )

    # PDF oluştur
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, txt=summary.encode("latin1", "replace").decode("latin1"))

    # Dosya adını tarihli oluştur
    output_name = f"report_{now.strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf.output(output_name)

    return output_name

if st.button("📥 sf_crime.csv indir, zenginleştir ve özetle"):
    with st.spinner("⏳ İşlem devam ediyor..."):
        try:
            response = requests.get(DOWNLOAD_URL)
            if response.status_code == 200:
                with open("sf_crime.csv", "wb") as f:
                    f.write(response.content)
                st.success("✅ sf_crime.csv başarıyla indirildi.")
                
                # 🔍 Dosya önizlemesini göster
                try:
                    df_preview = pd.read_csv("sf_crime.csv")
                    st.write("📌 [sf_crime.csv] sütunlar:")
                    st.write(df_preview.columns.tolist())
                    st.write("📋 İlk 3 satır:")
                    st.dataframe(df_preview.head(3))
                except Exception as e:
                    st.warning(f"⚠️ sf_crime.csv önizleme hatası: {e}")

                if os.path.exists("sf_pois_cleaned_with_geoid.csv"):
                    st.success("✅ POI CSV dosyası mevcut.")
                    try:
                        df_poi_prev = pd.read_csv("sf_pois_cleaned_with_geoid.csv")
                        st.write("📌 [POI] Sütunlar:", df_poi_prev.columns.tolist())
                        st.dataframe(df_poi_prev.head(3))
                    except Exception as e:
                        st.warning(f"⚠️ POI dosyası okunamadı: {e}")
                else:
                    st.error("❌ POI CSV dosyası eksik!")

                if os.path.exists("risky_pois_dynamic.json"):
                    st.success("✅ Risk skoru dosyası mevcut.")
                    try:
                        with open("risky_pois_dynamic.json") as f:
                            risk_data = json.load(f)
                        st.write("📌 [Risk Skoru JSON] İlk 3 kayıt:")
                        preview_risk = dict(list(risk_data.items())[:3])
                        st.json(preview_risk)
                    except Exception as e:
                        st.warning(f"⚠️ Risk skoru JSON okunamadı: {e}")
                else:
                    st.error("❌ Risk skoru JSON dosyası eksik!")

                try:
                    df_911 = None
                    response_911 = requests.get(DOWNLOAD_911_URL)
                    if response_911.status_code == 200:
                        with open("sf_911_last_5_year.csv", "wb") as f:
                            f.write(response_911.content)
                        st.success("✅ 911 verisi indirildi.")
                        df_911 = pd.read_csv("sf_911_last_5_year.csv")

                        if "GEOID" in df_911.columns:
                            df_911["GEOID"] = df_911["GEOID"].astype(str).str.zfill(11)

                        if "event_hour" not in df_911.columns:
                            if "time" in df_911.columns:
                                df_911["event_hour"] = pd.to_datetime(df_911["time"], errors="coerce").dt.hour
                            elif "datetime" in df_911.columns:
                                df_911["event_hour"] = pd.to_datetime(df_911["datetime"], errors="coerce").dt.hour
                            else:
                                st.warning("⚠️ 'event_hour' üretilemedi.")

                        if "date" in df_911.columns:
                            df_911["date"] = pd.to_datetime(df_911["date"], errors="coerce").dt.date

                        st.dataframe(df_911.head())
                    else:
                        st.warning(f"⚠️ 911 verisi indirilemedi: {response_911.status_code}")
                except Exception as e:
                    st.error(f"❌ 911 verisi işlenemedi: {e}")

                try:
                    df_311 = None
                    response_311 = requests.get(DOWNLOAD_311_URL)
                    if response_311.status_code == 200:
                        with open("sf_311_last_5_years.csv", "wb") as f:
                            f.write(response_311.content)
                        st.success("✅ 311 verisi indirildi.")
                
                        # 📋 Veriyi oku ve göster
                        try:
                            df_311 = pd.read_csv("sf_311_last_5_years.csv")
                            st.write("📋 [311] Sütunlar:", df_311.columns.tolist())
                            st.dataframe(df_311.head(3))
                        except Exception as e:
                            st.warning(f"⚠️ 311 verisi okunamadı: {e}")
                    else:
                        st.warning(f"⚠️ 311 verisi indirilemedi: {response_311.status_code}")
                except Exception as e:
                    st.error(f"❌ 311 verisi işlenemedi: {e}")

                df = pd.read_csv("sf_crime.csv", low_memory=False)
                original_row_count = len(df)
                
                try:
                    df_poi = pd.read_csv("sf_pois_cleaned_with_geoid.csv")
                    st.write("📋 [POI] Sütunlar:", df_poi.columns.tolist())
                    st.dataframe(df_poi.head(3))
                
                    with open("risky_pois_dynamic.json") as f:
                        risk_dict = json.load(f)
                
                    df_poi["risk_score"] = df_poi["poi_subcategory"].map(risk_dict).fillna(0)
                
                    # Suç geometrisi
                    gdf_crime = gpd.GeoDataFrame(
                        df,
                        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
                        crs="EPSG:4326"
                    ).to_crs(3857)
                
                    # POI geometrisi
                    df_poi = df_poi.rename(columns={"lat": "poi_lat", "lon": "poi_lon"})
                    gdf_poi = gpd.GeoDataFrame(
                        df_poi,
                        geometry=gpd.points_from_xy(df_poi["poi_lon"], df_poi["poi_lat"]),
                        crs="EPSG:4326"
                    ).to_crs(3857)
                
                    # Mesafe hesapla
                    poi_coords = np.vstack([gdf_poi.geometry.x, gdf_poi.geometry.y]).T
                    crime_coords = np.vstack([gdf_crime.geometry.x, gdf_crime.geometry.y]).T
                    poi_tree = cKDTree(poi_coords)
                    df["distance_to_poi"], _ = poi_tree.query(crime_coords, k=1)
                
                    # Riskli POI mesafesi
                    risky_poi = gdf_poi[gdf_poi["risk_score"] > 0]
                    if not risky_poi.empty:
                        risky_coords = np.vstack([risky_poi.geometry.x, risky_poi.geometry.y]).T
                        risky_tree = cKDTree(risky_coords)
                        df["distance_to_high_risk_poi"], _ = risky_tree.query(crime_coords, k=1)
                    else:
                        df["distance_to_high_risk_poi"] = np.nan
                
                    # GEOID'e göre risk yoğunluğu
                    risk_density = df_poi.groupby("GEOID")["risk_score"].mean().reset_index(name="poi_risk_density")
                    df = df.merge(risk_density, on="GEOID", how="left")
                
                    st.success("✅ POI mesafe ve risk yoğunluğu eklendi.")
                    st.write("📌 Yeni Sütunlar:", ["distance_to_poi", "distance_to_high_risk_poi", "poi_risk_density"])
                    st.dataframe(df[["distance_to_poi", "distance_to_high_risk_poi", "poi_risk_density"]].head(3))
                
                except Exception as e:
                    st.error(f"❌ POI mesafe/risk hesaplama hatası: {e}")

                if os.path.exists(POPULATION_PATH):
                    try:
                        df_pop = pd.read_csv(POPULATION_PATH)
                        st.write("📋 [Nüfus] Sütunlar:", df_pop.columns.tolist())
                        st.dataframe(df_pop.head(3))
                
                        df["GEOID"] = df["GEOID"].astype(str).str.extract(r'(\d+)')[0].str.zfill(11)
                        df_pop["GEOID"] = df_pop["GEOID"].astype(str).str.zfill(11)
                
                        df = pd.merge(df, df_pop, on="GEOID", how="left")
                        df["population"] = df["population"].fillna(0).astype(int)
                
                        st.success("✅ Nüfus verisi eklendi.")
                        st.write("👥 Nüfus örnek verisi (ilk 3 satır):")
                        st.dataframe(df[["GEOID", "population"]].drop_duplicates().head(3))
                    except Exception as e:
                        st.error(f"❌ Nüfus verisi işlenemedi: {e}")
                else:
                    st.warning("⚠️ Nüfus verisi (sf_population.csv) bulunamadı.")

                try:
                    # 🚌 Otobüs verisi indir
                    response_bus = requests.get(DOWNLOAD_BUS_URL)
                    if response_bus.status_code == 200:
                        with open("sf_bus_stops.csv", "wb") as f:
                            f.write(response_bus.content)
                        st.success("✅ sf_bus_stops.csv başarıyla indirildi.")
                
                        try:
                            df_bus = pd.read_csv("sf_bus_stops.csv").dropna(subset=["stop_lat", "stop_lon"])
                            st.write("📋 [Otobüs] Sütunlar:", df_bus.columns.tolist())
                            st.write("🚌 Otobüs verisi (ilk 3 satır):")
                            st.dataframe(df_bus.head(3))
                        except Exception as e:
                            st.warning(f"⚠️ Otobüs CSV okunurken hata oluştu: {e}")
                    else:
                        st.warning(f"⚠️ Otobüs verisi indirilemedi: {response_bus.status_code}")
                except Exception as e:
                    st.error(f"❌ Otobüs verisi indirilemedi: {e}")
                
                try:
                    # 🚆 Tren verisi oku
                    if os.path.exists("sf_train_stops_with_geoid.csv"):
                        df_train = pd.read_csv("sf_train_stops_with_geoid.csv").dropna(subset=["stop_lat", "stop_lon"])
                        st.success("✅ sf_train_stops_with_geoid.csv dosyası mevcut.")
                        st.write("📋 [Tren] Sütunlar:", df_train.columns.tolist())
                        st.write("🚆 Tren verisi (ilk 3 satır):")
                        st.dataframe(df_train.head(3))
                    else:
                        st.warning("⚠️ sf_train_stops_with_geoid.csv bulunamadı.")
                except Exception as e:
                    st.error(f"❌ Tren verisi okunamadı: {e}")


                nan_summary = df.isna().sum()
                nan_cols = nan_summary[nan_summary > 0]
                removed_rows = 0
                removed_rows = original_row_count - len(df)
                report_path = create_pdf_report("sf_crime.csv", original_row_count, nan_cols, len(df), removed_rows)
                with open(report_path, "rb") as f:
                    st.download_button("📄 PDF Raporu İndir", f, file_name=report_path, mime="application/pdf")

            else:
                st.error(f"❌ sf_crime.csv indirilemedi, HTTP kodu: {response.status_code}")
                st.stop()

        except Exception as e:
            st.error(f"❌ Genel hata oluştu: {e}")
            
# === Yardimci Fonksiyonlar ===
def check_and_fix_coordinates(df, context=""):
    """Koordinat sütunlarını kontrol eder, dönüştürür ve geçersiz değerleri temizler"""
    rename_map = {}
    if "latitude" not in df.columns:
        for alt in ["lat", "enlem"]:
            if alt in df.columns:
                rename_map[alt] = "latitude"
    if "longitude" not in df.columns:
        for alt in ["lon", "long", "lng", "boylam"]:
            if alt in df.columns:
                rename_map[alt] = "longitude"

    if rename_map:
        df.rename(columns=rename_map, inplace=True)
        st.warning(f"⚠️ {context}: {rename_map} olarak yeniden adlandırıldı.")

    # Eksik sütun varsa durdur
    if "latitude" not in df.columns or "longitude" not in df.columns:
        st.error(f"❌ {context}: 'latitude' veya 'longitude' eksik.")
        return df.iloc[0:0]  # Boş DataFrame

    # Sayısal dönüşüm + filtre
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    # Geçersiz koordinatları at (örneğin: lat < 30, long > -100 gibi)
    df = df.dropna(subset=["latitude", "longitude"])
    df = df[
        (df["latitude"].between(37.5, 37.9)) & 
        (df["longitude"].between(-123, -122))
    ].copy()

    if df.empty:
        st.warning(f"⚠️ {context}: Geçerli koordinat içeren satır yok.")
    return df
    
def enrich_with_911(df):
    try:
        df_911 = pd.read_csv("sf_911_last_5_year.csv")
        df_911["GEOID"] = df_911["GEOID"].astype(str).str.zfill(11)
        df["GEOID"] = df["GEOID"].astype(str).str.zfill(11)
        df_911["date"] = pd.to_datetime(df_911["date"], errors="coerce")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        df = df.merge(
            df_911.rename(columns={
                "time": "call_time",
                "latitude": "call_lat",
                "longitude": "call_lon"
            }),
            on=["GEOID", "date", "event_hour"],
            suffixes=("", "_911")
        )
        return df
        
    except Exception as e:
        st.error(f"❌ 911 verisi eklenemedi: {e}")
        return df

def enrich_with_311(df):
    try:
        df_311 = pd.read_csv("sf_311_last_5_years.csv")
        df_311["datetime"] = pd.to_datetime(df_311["date"] + " " + df_311["time"], errors="coerce")
        df_311["event_hour"] = df_311["datetime"].dt.hour
        df_311["date"] = pd.to_datetime(df_311["date"]).dt.date
        df["date"] = pd.to_datetime(df["date"]).dt.date

        df_311["GEOID"] = df_311["GEOID"].astype(str).str.zfill(11)
        df["GEOID"] = df["GEOID"].astype(str).str.zfill(11)

        df = df.merge(
            df_311.rename(columns={
                "time": "request_time",
                "category": "service_category"
            }),
            on=["GEOID", "date", "event_hour"],
            suffixes=("", "_311")
        )
        return df

    except Exception as e:
        st.error(f"❌ 311 verisi eklenemedi: {e}")
        return df

def enrich_with_weather(df):
    try:
        if not os.path.exists("sf_weather_5years.csv"):
            st.warning("⚠️ Hava durumu verisi bulunamadı")
            return df

        weather = pd.read_csv("sf_weather_5years.csv")
        weather.columns = weather.columns.str.lower()
        date_col = next((col for col in weather.columns if 'date' in col), None)
        if not date_col:
            st.error("❌ Hava durumu verisinde tarih sütunu bulunamadı")
            return df

        weather['date'] = pd.to_datetime(weather[date_col]).dt.date
        df['date'] = pd.to_datetime(df['date']).dt.date

        df = df.merge(
            weather.rename(columns={"date": "weather_date"}),
            left_on="date",
            right_on="weather_date",
            suffixes=("", "_weather")
        ).drop(columns=["weather_date"])
        st.success("✅ Hava durumu verisi başarıyla eklendi")
        return df

    except Exception as e:
        st.error(f"❌ Hava durumu zenginleştirme hatası: {str(e)}")
        return df

def enrich_with_police(df):
    try:
        # 1. Suç verisinde koordinatları kontrol et
        df_checked = check_and_fix_coordinates(df.copy(), "Polis istasyonu entegrasyonu")
        if df_checked.empty:
            st.warning("⚠️ Polis: Geçerli suç koordinatı bulunamadı.")
            return df

        df_valid = df_checked.dropna(subset=["longitude", "latitude"]).copy()

        # 2. Polis verisi dosyasını kontrol et
        if not os.path.exists("sf_police_stations.csv"):
            st.error("❌ Polis istasyonu verisi bulunamadı (sf_police_stations.csv)")
            return df

        df_police = pd.read_csv("sf_police_stations.csv")

        # 3. Polis verisinde koordinatları kontrol et
        df_police_checked = check_and_fix_coordinates(df_police.copy(), "Polis istasyonu verisi")
        if df_police_checked.empty:
            st.warning("⚠️ Polis: Geçerli istasyon koordinatı yok.")
            return df

        # 4. GeoDataFrame oluştur
        gdf_crime = gpd.GeoDataFrame(
            df_valid,
            geometry=gpd.points_from_xy(df_valid["longitude"], df_valid["latitude"]),
            crs="EPSG:4326"
        ).to_crs(epsg=3857)

        gdf_police = gpd.GeoDataFrame(
            df_police_checked,
            geometry=gpd.points_from_xy(df_police_checked["longitude"], df_police_checked["latitude"]),
            crs="EPSG:4326"
        ).to_crs(epsg=3857)

        # 5. Mesafe hesapla
        crime_coords = np.vstack([gdf_crime.geometry.x, gdf_crime.geometry.y]).T
        police_coords = np.vstack([gdf_police.geometry.x, gdf_police.geometry.y]).T
        police_tree = cKDTree(police_coords)

        df_valid["distance_to_police"], _ = police_tree.query(crime_coords, k=1)
        df_valid["is_near_police"] = (df_valid["distance_to_police"] < 200).astype(int)
        df_valid["distance_to_police_range"] = pd.cut(
            df_valid["distance_to_police"],
            bins=[0, 100, 200, 500, 1000, np.inf],
            labels=["0-100", "100-200", "200-500", "500-1000", ">1000"]
        )

        # 6. Ana df ile güncelle
        df.update(df_valid)
        st.success("✅ Polis istasyonu bilgileri başarıyla eklendi")
        return df

    except Exception as e:
        st.error(f"❌ Polis istasyonu zenginleştirme hatası: {str(e)}")
        return df

def enrich_with_government(df):
    try:
        # 1. Suç verisi koordinatlarını kontrol et
        df_checked = check_and_fix_coordinates(df, "Devlet binaları entegrasyonu")
        if df_checked.empty:
            st.warning("⚠️ Devlet binaları: Geçerli suç koordinatı bulunamadı.")
            return df

        df_valid = df_checked.dropna(subset=["longitude", "latitude"]).copy()

        # 2. Devlet binası dosyasını kontrol et
        if not os.path.exists("sf_government_buildings.csv"):
            st.error("❌ Devlet binaları verisi bulunamadı")
            return df

        df_gov = pd.read_csv("sf_government_buildings.csv")

        # 3. Devlet binası koordinatlarını kontrol et
        df_gov_checked = check_and_fix_coordinates(df_gov, "Devlet binaları verisi")
        if df_gov_checked.empty:
            st.warning("⚠️ Devlet binaları: Geçerli istasyon koordinatı yok.")
            return df

        # 4. GeoDataFrame dönüşümleri
        gdf_crime = gpd.GeoDataFrame(
            df_valid,
            geometry=gpd.points_from_xy(df_valid["longitude"], df_valid["latitude"]),
            crs="EPSG:4326"
        ).to_crs(epsg=3857)

        gdf_gov = gpd.GeoDataFrame(
            df_gov_checked,
            geometry=gpd.points_from_xy(df_gov_checked["longitude"], df_gov_checked["latitude"]),
            crs="EPSG:4326"
        ).to_crs(epsg=3857)

        # 5. Mesafe hesapla
        crime_coords = np.vstack([gdf_crime.geometry.x, gdf_crime.geometry.y]).T
        gov_coords = np.vstack([gdf_gov.geometry.x, gdf_gov.geometry.y]).T
        gov_tree = cKDTree(gov_coords)

        df_valid["distance_to_government"], _ = gov_tree.query(crime_coords, k=1)
        df_valid["is_near_government"] = (df_valid["distance_to_government"] < 200).astype(int)
        df_valid["distance_to_government_range"] = pd.cut(
            df_valid["distance_to_government"],
            bins=[0, 100, 200, 500, 1000, np.inf],
            labels=["0-100m", "100-200m", "200-500m", "500-1000m", ">1000m"]
        )

        # 6. Ana df'e geri yaz
        df.update(df_valid)
        st.success("✅ Devlet binası bilgileri başarıyla eklendi")
        return df

    except Exception as e:
        st.error(f"❌ Devlet binası zenginleştirme hatası: {str(e)}")
        return df


# Veri zenginleştirme 

def check_and_fix_coordinates(df, context=""):
    rename_map = {}
    if "latitude" not in df.columns:
        for alt in ["lat", "enlem"]:
            if alt in df.columns:
                rename_map[alt] = "latitude"
    if "longitude" not in df.columns:
        for alt in ["lon", "long", "lng", "boylam"]:
            if alt in df.columns:
                rename_map[alt] = "longitude"

    if rename_map:
        df.rename(columns=rename_map, inplace=True)
        st.warning(f"⚠️ {context}: {rename_map} olarak yeniden adlandırıldı.")

    if "latitude" not in df.columns or "longitude" not in df.columns:
        st.error(f"❌ {context}: 'latitude' veya 'longitude' eksik.")
        return df.iloc[0:0]

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])

    # Daha geniş aralık
    df = df[
        (df["latitude"].between(37.7, 37.84)) & 
        (df["longitude"].between(-123.2, -122.3))
    ].copy()

    if df.empty:
        st.warning(f"⚠️ {context}: Geçerli koordinat içeren satır yok.")
    return df


def enrich_with_poi(df):
    """
    Suç verisini POI (Point of Interest) verileriyle zenginleştirir.
    - En yakın POI'ya ve en yakın riskli POI'ya olan mesafeleri hesaplar.
    - Her coğrafi bölge (GEOID) için POI risk yoğunluğunu hesaplar.
    """
    try:
        # Gerekli dosyaları yükle
        df_poi = pd.read_csv("sf_pois_cleaned_with_geoid.csv")
        with open("risky_pois_dynamic.json") as f:
            risk_dict = json.load(f)
        
        # Risk skorunu ata
        df_poi["risk_score"] = df_poi["poi_subcategory"].map(risk_dict).fillna(0)

        # GeoDataFrame'leri oluştur ve projeksiyonu ayarla (hesaplama için)
        gdf_crime = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs="EPSG:4326").to_crs(3857)
        gdf_poi = gpd.GeoDataFrame(df_poi, geometry=gpd.points_from_xy(df_poi["lon"], df_poi["lat"]), crs="EPSG:4326").to_crs(3857)

        # KDTree ile en yakın POI mesafesini hesapla
        poi_coords = np.vstack([gdf_poi.geometry.x, gdf_poi.geometry.y]).T
        crime_coords = np.vstack([gdf_crime.geometry.x, gdf_crime.geometry.y]).T
        poi_tree = cKDTree(poi_coords)
        df["distance_to_poi"], _ = poi_tree.query(crime_coords, k=1)

        # Yüksek riskli POI'lar için mesafeyi hesapla
        risky_poi = gdf_poi[gdf_poi["risk_score"] > 0]
        if not risky_poi.empty:
            risky_coords = np.vstack([risky_poi.geometry.x, risky_poi.geometry.y]).T
            risky_tree = cKDTree(risky_coords)
            df["distance_to_high_risk_poi"], _ = risky_tree.query(crime_coords, k=1)
        else:
            df["distance_to_high_risk_poi"] = np.nan
        
        # GEOID bazında risk yoğunluğunu hesapla ve ana dataframe'e ekle
        risk_density = df_poi.groupby("GEOID")["risk_score"].mean().reset_index(name="poi_risk_density")
        df = df.merge(risk_density, on="GEOID", how="left")
        
        st.success("✅ POI mesafesi ve risk yoğunluğu başarıyla eklendi.")
        return df

    except Exception as e:
        st.error(f"❌ POI zenginleştirme sırasında hata oluştu: {e}")
        return df # Hata durumunda bile orijinal df'i geri döndür

def enrich_with_police(df):
    st.write("📌 Sütunlar:", df.columns.tolist())
    st.write("📋 İlk 5 satır (koordinatlar):")
    st.write(df[["latitude", "longitude"]].head())
    st.write("❗ Eksik latitude sayısı:", df["latitude"].isna().sum())
    st.write("❗ Eksik longitude sayısı:", df["longitude"].isna().sum())
    try:
        # 1. Suç verisinde koordinatları kontrol et
        df_checked = check_and_fix_coordinates(df.copy(), "Polis istasyonu entegrasyonu")
        if df_checked.empty:
            st.warning("⚠️ Polis: Geçerli suç koordinatı bulunamadı.")
            return df

        df_valid = df_checked.dropna(subset=["longitude", "latitude"]).copy()

        # 2. Polis verisi dosyasını kontrol et
        if not os.path.exists("sf_police_stations.csv"):
            st.error("❌ Polis istasyonu verisi bulunamadı (sf_police_stations.csv)")
            return df

        df_police = pd.read_csv("sf_police_stations.csv")

        # 3. Polis verisinde koordinatları kontrol et
        df_police_checked = check_and_fix_coordinates(df_police.copy(), "Polis istasyonu verisi")
        if df_police_checked.empty:
            st.warning("⚠️ Polis: Geçerli istasyon koordinatı yok.")
            return df

        # 4. GeoDataFrame oluştur
        gdf_crime = gpd.GeoDataFrame(
            df_valid,
            geometry=gpd.points_from_xy(df_valid["longitude"], df_valid["latitude"]),
            crs="EPSG:4326"
        ).to_crs(epsg=3857)

        gdf_police = gpd.GeoDataFrame(
            df_police_checked,
            geometry=gpd.points_from_xy(df_police_checked["longitude"], df_police_checked["latitude"]),
            crs="EPSG:4326"
        ).to_crs(epsg=3857)

        # 5. Mesafe hesapla
        crime_coords = np.vstack([gdf_crime.geometry.x, gdf_crime.geometry.y]).T
        police_coords = np.vstack([gdf_police.geometry.x, gdf_police.geometry.y]).T
        police_tree = cKDTree(police_coords)

        df_valid["distance_to_police"], _ = police_tree.query(crime_coords, k=1)
        df_valid["is_near_police"] = (df_valid["distance_to_police"] < 200).astype(int)
        df_valid["distance_to_police_range"] = pd.cut(
            df_valid["distance_to_police"],
            bins=[0, 100, 200, 500, 1000, np.inf],
            labels=["0-100", "100-200", "200-500", "500-1000", ">1000"]
        )

        # 6. Orijinal df ile güncelle
        df.update(df_valid)
        st.success("✅ Polis istasyonu bilgileri başarıyla eklendi")
        return df

    except Exception as e:
        st.error(f"❌ Polis istasyonu zenginleştirme hatası: {str(e)}")
        return df
        
def enrich_with_government(df):
    """Suç verisini devlet binaları verileriyle zenginleştirir"""
    try:
        # 1. Suç verisi koordinatlarını kontrol et
        df_checked = check_and_fix_coordinates(df.copy(), "Devlet binaları entegrasyonu")
        if df_checked.empty:
            st.warning("⚠️ Devlet binaları: Geçerli suç koordinatı bulunamadı.")
            return df

        df_valid = df_checked.dropna(subset=["longitude", "latitude"]).copy()

        # 2. Devlet binası dosyasını kontrol et
        if not os.path.exists("sf_government_buildings.csv"):
            st.error("❌ Devlet binaları verisi bulunamadı")
            return df

        df_gov = pd.read_csv("sf_government_buildings.csv")

        # 3. Devlet binası koordinatlarını kontrol et
        df_gov_checked = check_and_fix_coordinates(df_gov.copy(), "Devlet binaları verisi")
        if df_gov_checked.empty:
            st.warning("⚠️ Devlet binaları: Geçerli istasyon koordinatı yok.")
            return df

        # 4. GeoDataFrame dönüşümleri
        gdf_crime = gpd.GeoDataFrame(
            df_valid,
            geometry=gpd.points_from_xy(df_valid["longitude"], df_valid["latitude"]),
            crs="EPSG:4326"
        ).to_crs(epsg=3857)

        gdf_gov = gpd.GeoDataFrame(
            df_gov_checked,
            geometry=gpd.points_from_xy(df_gov_checked["longitude"], df_gov_checked["latitude"]),
            crs="EPSG:4326"
        ).to_crs(epsg=3857)

        # 5. Mesafe hesapla
        crime_coords = np.vstack([gdf_crime.geometry.x, gdf_crime.geometry.y]).T
        gov_coords = np.vstack([gdf_gov.geometry.x, gdf_gov.geometry.y]).T
        gov_tree = cKDTree(gov_coords)

        df_valid["distance_to_government"], _ = gov_tree.query(crime_coords, k=1)
        df_valid["is_near_government"] = (df_valid["distance_to_government"] < 200).astype(int)
        df_valid["distance_to_government_range"] = pd.cut(
            df_valid["distance_to_government"],
            bins=[0, 100, 200, 500, 1000, np.inf],
            labels=["0-100m", "100-200m", "200-500m", "500-1000m", ">1000m"]
        )

        # 6. Ana df'e geri yaz
        df.update(df_valid)
        st.success("✅ Devlet binası bilgileri başarıyla eklendi")
        return df

    except Exception as e:
        st.error(f"❌ Devlet binası zenginleştirme hatası: {str(e)}")
        return df

def enrich_with_911(df):
    try:
        df_911 = pd.read_csv("sf_911_last_5_year.csv")

        # GEOID'leri string ve 11 haneli olarak ayarla
        df_911["GEOID"] = df_911["GEOID"].astype(str).str.zfill(11)
        df["GEOID"] = df["GEOID"].astype(str).str.zfill(11)

        # Tarih formatı düzelt
        df_911["date"] = pd.to_datetime(df_911["date"], errors="coerce")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # Merge işlemi
        df = df.merge(
            df_911.rename(columns={
                "time": "call_time",
                "latitude": "call_lat",
                "longitude": "call_lon"
            }),
            on=["GEOID", "date", "event_hour"],
            suffixes=("", "_911")
        )
        return df

    except Exception as e:
        st.error(f"❌ 911 verisi eklenemedi: {e}")
        return df

def enrich_with_311(df):
    try:
        df_311 = pd.read_csv("sf_311_last_5_years.csv")

        # datetime birleştirme ve saat çıkarımı
        df_311["datetime"] = pd.to_datetime(df_311["date"] + " " + df_311["time"], errors="coerce")
        df_311["event_hour"] = df_311["datetime"].dt.hour
        df_311["hour_range"] = (df_311["event_hour"] // 3) * 3
        df_311["hour_range"] = df_311["hour_range"].astype(str) + "-" + (df_311["hour_range"] + 3).astype(str)

        # tarih formatı düzelt
        df_311["date"] = pd.to_datetime(df_311["date"]).dt.date
        df["date"] = pd.to_datetime(df["date"]).dt.date

        # GEOID tipleri uyuşmalı
        df_311["GEOID"] = df_311["GEOID"].astype(str).str.zfill(11)
        df["GEOID"] = df["GEOID"].astype(str).str.zfill(11)

        # saat eşleşmesiyle birleştir
        df = df.merge(
            df_311.rename(columns={
                "time": "request_time",
                "category": "service_category"
            }),
            on=["GEOID", "date", "event_hour"],
            suffixes=("", "_311")
        )
        return df

    except Exception as e:
        st.error(f"❌ 311 verisi eklenemedi: {e}")
        return df

def enrich_with_weather(df):
    try:
        if not os.path.exists("sf_weather_5years.csv"):
            st.warning("⚠️ Hava durumu verisi bulunamadı")
            return df

        weather = pd.read_csv("sf_weather_5years.csv")
        weather.columns = weather.columns.str.lower()
        
        # Tarih sütununu bulmak için esnek yaklaşım
        date_col = next((col for col in weather.columns if 'date' in col), None)
        if not date_col:
            st.error("❌ Hava durumu verisinde tarih sütunu bulunamadı")
            return df
            
        weather['date'] = pd.to_datetime(weather[date_col]).dt.date
        
        # Ana veride tarih sütununu bul
        main_date_col = 'date' if 'date' in df.columns else \
                       next((col for col in df.columns if 'date' in col), None)
        
        if not main_date_col:
            st.error("❌ Ana veride tarih sütunu bulunamadı")
            return df
            
        # datetime sütunu yoksa oluştur
        if 'datetime' not in df.columns and 'date' in df.columns and 'time' in df.columns:
            try:
                df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str))
            except:
                df['datetime'] = pd.to_datetime(df['date'].astype(str))
        
        df['date'] = pd.to_datetime(df[main_date_col]).dt.date
        
        # Birleştirme
        df = df.merge(
            weather.rename(columns={"date": "weather_date"}),
            left_on="date",
            right_on="weather_date",
            suffixes=("", "_weather")
        ).drop(columns=["weather_date"])
        st.success("✅ Hava durumu verisi başarıyla eklendi")
        return df

    except Exception as e:
        st.error(f"❌ Hava durumu zenginleştirme hatası: {str(e)}")
        return df
        
if st.button("🧪 Veriyi Göster (Test)"):
    try:
        # Veri yükleme
        if not os.path.exists("sf_crime.csv"):
            st.error("❌ sf_crime.csv bulunamadı!")
            st.stop()
            
        df = pd.read_csv("sf_crime.csv", low_memory=False)
        
        # Koordinat kontrolü
        df = check_and_fix_coordinates(df, "Ana veri")
        if df.empty:
            st.warning("⚠️ Ana veri: Geçerli koordinat içeren satır yok.")
            st.stop()

        # Tarih/saat işlemleri
        if 'date' in df.columns and 'time' in df.columns:
            try:
                df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str))
                df['date'] = df['datetime'].dt.date
                df['event_hour'] = df['datetime'].dt.hour
            except Exception as e:
                st.error(f"❌ Tarih dönüşüm hatası: {str(e)}")
        
        # Zenginleştirme adımları
        enrichment_functions = [
            ("POI", enrich_with_poi),
            ("911", enrich_with_911),
            ("311", enrich_with_311),
            ("Hava Durumu", enrich_with_weather),
            ("Polis İstasyonları", enrich_with_police),
            ("Devlet Binaları", enrich_with_government)
        ]
        
        for name, func in enrichment_functions:
            try:
                df = func(df)
            except Exception as e:
                st.error(f"❌ {name} zenginleştirme hatası: {str(e)}")
        
        # Sonuçları göster
        st.write("### Sonuçlar")
        st.dataframe(df.head(3))
        st.write("Sütunlar:", df.columns.tolist())
        
    except Exception as e:
        st.error(f"❌ Genel hata: {str(e)}")
