"""
Build an embedded SQLite database (data/kerala_health.db) from all cleaned & master datasets.
Zero installation required — SQLite is built directly into standard Python.

Tables created:
  1. master_district_yearly  - Integrated ML training table (14 districts x 15 years = 210 rows)
  2. master_state_monthly    - Integrated temporal table (15 years x 12 months = 180 rows)
  3. water_quality_stations  - Raw 100 CPCB river monitoring stations (2024)
  4. demographics            - Census 2011 district populations & density
  5. climate_weekly          - Continuous weekly weather observations (10,976 rows)
  6. user_predictions_log    - Log table for when website users run predictions

Run: python build_database.py
"""
import os
import sqlite3
import pandas as pd

DB_PATH = "data/kerala_health.db"
os.makedirs("data", exist_ok=True)

# Connect to SQLite database (creates the file if it does not exist)
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

print(f"Creating / Updating SQLite Database at: {DB_PATH}")

# 1. Master District-Yearly Table
df_dist_yr = pd.read_csv("data/processed/master_district_yearly.csv")
df_dist_yr.to_sql("master_district_yearly", conn, if_exists="replace", index=False)
print(f"  [OK] Table 'master_district_yearly' populated ({len(df_dist_yr)} rows)")

# 2. Master State-Monthly Table
df_state_mo = pd.read_csv("data/processed/master_state_monthly.csv")
df_state_mo.to_sql("master_state_monthly", conn, if_exists="replace", index=False)
print(f"  [OK] Table 'master_state_monthly' populated ({len(df_state_mo)} rows)")

# 3. Water Quality Stations Table (100 CPCB stations)
df_wq = pd.read_csv("data/raw/water_quality/kerala_water_quality_2024.csv")
df_wq.to_sql("water_quality_stations", conn, if_exists="replace", index=False)
print(f"  [OK] Table 'water_quality_stations' populated ({len(df_wq)} rows)")

# 4. Demographics Table
df_pop = pd.read_csv("data/raw/demographics/kerala_population_census2011.csv")
df_pop.to_sql("demographics", conn, if_exists="replace", index=False)
print(f"  [OK] Table 'demographics' populated ({len(df_pop)} rows)")

# 5. Climate Weekly Table (10,976 rows)
df_clim = pd.read_csv("data/raw/climate/kerala_climate_weekly.csv")
df_clim.to_sql("climate_weekly", conn, if_exists="replace", index=False)
print(f"  [OK] Table 'climate_weekly' populated ({len(df_clim)} rows)")

# 6. User Predictions Log Table (for the website)
cursor.execute("""
CREATE TABLE IF NOT EXISTS user_predictions_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    district TEXT,
    predicted_disease TEXT,
    predicted_cases REAL,
    risk_level TEXT,
    user_rainfall_input REAL,
    user_temperature_input REAL,
    user_water_quality_input REAL
)
""")
print("  [OK] Table 'user_predictions_log' initialized (ready for website interactions)")

# Add indices for fast website querying
cursor.execute("CREATE INDEX IF NOT EXISTS idx_dist_yr ON master_district_yearly(district, year)")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_state_mo ON master_state_monthly(year, month)")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_clim_dist ON climate_weekly(district, year, week)")

conn.commit()
conn.close()

print("\n" + "=" * 60)
print(f"Database build complete! File: {DB_PATH}")
print(f"Database size: {os.path.getsize(DB_PATH) / 1024:.1f} KB")
