import pandas as pd

# 1. File paths
crime_02_path = "/content/drive/MyDrive/crime_data/sf_crime_02.csv"
population_path = "/content/drive/MyDrive/crime_data/sf_population.csv"
output_path = "/content/drive/MyDrive/crime_data/sf_crime_03.csv"  

# 2. Load data
df_crime = pd.read_csv(crime_02_path)
df_population = pd.read_csv(population_path)

# 3. Fix GEOID formats
df_crime["GEOID"] = df_crime["GEOID"].apply(lambda x: str(int(x)).zfill(11) if pd.notna(x) else x)
df_population["GEOID"] = df_population["GEOID"].astype(str).str.zfill(11)

# 4. Merge: add population to crime data
df_merged = pd.merge(
    df_crime,
    df_population,
    on="GEOID",
    how="left"
)

# 5. Replace missing population values with 0 and correct type
df_merged["population"] = df_merged["population"].fillna(0).astype("Int64")

# 6. Preview
print("🔍 First 5 rows:")
print(df_merged[["GEOID", "population"]].head())

print(f"\n📊 Number of rows: {df_merged.shape[0]}")
print(f"📊 Number of columns: {df_merged.shape[1]}")

# 7. Save as sf_crime_03.csv
df_merged.to_csv(output_path, index=False)
print(f"\n✅ Final merged dataset saved as: {output_path}")
