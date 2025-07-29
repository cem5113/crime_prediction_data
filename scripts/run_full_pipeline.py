import os
import shutil
import traceback
import subprocess

# === 1. Yol Ayarları ===
BASE_DIR = "data"  # veya "./data" veya os.environ.get("BASE_DIR", "data")
LOG_PATH = os.path.join(BASE_DIR, "pipeline_log.txt")
STEPS = [
    ("step_01_add_311.py",       "sf_crime.csv",     "sf_crime_01.csv"),
    ("step_02_add_911.py",       "sf_crime_01.csv",  "sf_crime_02.csv"),
    ("step_03_add_population.py","sf_crime_02.csv",  "sf_crime_03.csv"),
    ("step_04_add_bus.py",       "sf_crime_03.csv",  "sf_crime_04.csv"),
    ("step_05_add_train.py",     "sf_crime_04.csv",  "sf_crime_05.csv"),
    ("step_06_add_pois.py",      "sf_crime_05.csv",  "sf_crime_06.csv"),
    ("step_07_add_police_gov.py","sf_crime_06.csv",  "sf_crime_07.csv"),
    ("step_08_add_weather.py",   "sf_crime_07.csv",  "sf_crime_08.csv"),
]

os.makedirs(BASE_DIR, exist_ok=True)

def run_step(script, input_file, output_file):
    print(f"\n🚀 Adım: {script} ({input_file} → {output_file})")
    log_lines = []
    
    # Geçici kopya (backup) dosyası
    backup_path = os.path.join(BASE_DIR, f"backup_{output_file}")
    output_path = os.path.join(BASE_DIR, output_file)
    input_path = os.path.join(BASE_DIR, input_file)

    # Eski dosya varsa yedekle
    if os.path.exists(output_path):
        shutil.copy(output_path, backup_path)

    try:
        result = subprocess.run(["python", script], capture_output=True, text=True, check=True)
        log_lines.append(f"✅ {script} başarıyla çalıştı.\n")
        log_lines.append(result.stdout)
    except Exception as e:
        log_lines.append(f"❌ {script} çalışırken hata oluştu: {str(e)}\n")
        log_lines.append(traceback.format_exc())

        # Hata varsa backup'ı geri yükle
        if os.path.exists(backup_path):
            shutil.copy(backup_path, output_path)
            log_lines.append(f"⚠️ Hata nedeniyle eski {output_file} dosyası geri yüklendi.\n")
        else:
            log_lines.append(f"⚠️ Hata ama yedek dosya bulunamadı: {backup_path}\n")

    # Log kaydı
    with open(LOG_PATH, "a", encoding="utf-8") as log:
        log.write("\n".join(log_lines))
        log.write("\n" + "="*80 + "\n")

    print("📝 Log güncellendi.")

# === 2. Tüm adımları sırayla çalıştır ===
if __name__ == "__main__":
    print("🔁 Tam zenginleştirme süreci başlatılıyor...")
    for script, input_file, output_file in STEPS:
        run_step(script, input_file, output_file)
    print("✅ Tüm adımlar tamamlandı. Son dosya: sf_crime_08.csv")
