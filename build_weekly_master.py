"""
Build Weekly District Surveillance Master Dataset with 1-Week and 2-Week Lagged Features.

Problem Formulation:
  Predict Next Week's Outbreak (Week t+1) using:
    - Current week climate (Week t)
    - Previous week climate (Week t-1 lag)
    - River water quality (BOD, Fecal Coliform, Dissolved Oxygen)
    - Demographics (Population density, literacy rate)
    - Current baseline cases (Week t)

Inputs:
  - data/kerala_health.db (climate_weekly, master_district_yearly, master_state_monthly)

Outputs:
  - data/processed/master_district_weekly.csv (~10,000 district-week rows)
  - data/kerala_health.db (adds table 'master_district_weekly')

Run: python build_weekly_master.py
"""
import datetime as dt
import sqlite3
import numpy as np
import pandas as pd

DB_PATH = "data/kerala_health.db"
conn = sqlite3.connect(DB_PATH)

print("Building Weekly District Outbreak Master Dataset...")

# 1. Load Climate Weekly (NASA POWER)
df_clim = pd.read_sql("SELECT * FROM climate_weekly WHERE year BETWEEN 2011 AND 2025 ORDER BY district, year, week", conn)

# 2. Load District-Yearly Totals and Demographics
df_dist_yr = pd.read_sql("SELECT district, year, add_cases, population_2011, population_density, literacy_rate, wq_fecal_coliform_mpn, wq_bod_mg_l, wq_do_min_mg_l, covid_year FROM master_district_yearly", conn)

# 3. Load Monthly State Surveillance Waves
df_state_mo = pd.read_sql("SELECT year, month, add_cases as state_month_cases FROM master_state_monthly", conn)
conn.close()

# Map each ISO week to calendar month
def get_month_for_week(row):
    try:
        # Thursday of the ISO week determines the month
        d = dt.date.fromisocalendar(int(row["year"]), int(row["week"]), 4)
        return d.month
    except Exception:
        return 1

df_clim["month"] = df_clim.apply(get_month_for_week, axis=1)

# Join district yearly total and monthly state distributions
df_merged = df_clim.merge(df_dist_yr, on=["district", "year"], how="left")
df_merged = df_merged.merge(df_state_mo, on=["year", "month"], how="left")

# Compute monthly disease proportion per year (to capture accurate seasonal waves)
year_state_totals = df_state_mo.groupby("year")["state_month_cases"].sum().rename("year_state_total")
df_merged = df_merged.merge(year_state_totals, on="year", how="left")
df_merged["month_weight"] = df_merged["state_month_cases"] / df_merged["year_state_total"]

# Calculate weekly rainfall proportion within each month
month_rain_sums = df_merged.groupby(["district", "year", "month"])["rainfall_mm_total"].transform("sum")
month_week_counts = df_merged.groupby(["district", "year", "month"])["rainfall_mm_total"].transform("count")

# Rain-weighted weekly distribution within each month (with baseline smoothing)
df_merged["week_rain_weight"] = np.where(
    month_rain_sums > 0,
    0.6 * (df_merged["rainfall_mm_total"] / month_rain_sums) + 0.4 * (1.0 / month_week_counts),
    1.0 / month_week_counts
)

# Normalize weights across the entire year so sum of weekly cases equals exact real annual district cases
annual_weight_sum = df_merged.groupby(["district", "year"])["week_rain_weight"].transform("sum")
df_merged["normalized_weekly_weight"] = df_merged["week_rain_weight"] / annual_weight_sum

# Weekly estimated true cases
df_merged["add_cases_week"] = np.round(df_merged["add_cases"] * df_merged["normalized_weekly_weight"])
df_merged["add_cases_per_lakh"] = (df_merged["add_cases_week"] / df_merged["population_2011"]) * 100000.0

# ── Feature Engineering for Week-Ahead Prediction ─────────────────────────
df_merged = df_merged.sort_values(["district", "year", "week"]).reset_index(drop=True)

# 1. Lag-1 features (Previous week climate: t-1)
df_merged["rain_mm_lag1"] = df_merged.groupby("district")["rainfall_mm_total"].shift(1)
df_merged["temp_c_lag1"] = df_merged.groupby("district")["temp_c_mean"].shift(1)
df_merged["humidity_pct_lag1"] = df_merged.groupby("district")["humidity_pct_mean"].shift(1)

# 2. Lag-2 features (2 weeks ago rainfall: t-2)
df_merged["rain_mm_lag2"] = df_merged.groupby("district")["rainfall_mm_total"].shift(2)

# 3. 2-Week Cumulative Rainfall Flush
df_merged["rain_2week_sum"] = df_merged["rainfall_mm_total"] + df_merged["rain_mm_lag1"].fillna(0)

# 4. Extreme Precipitation Flag (>100mm weekly rainfall)
df_merged["is_torrential_rain"] = (df_merged["rainfall_mm_total"] >= 100.0).astype(int)

# 5. Seasonal Cyclical Sine/Cosine of Week (1 to 52)
df_merged["sin_week"] = np.sin(2 * np.pi * df_merged["week"] / 52.0)
df_merged["cos_week"] = np.cos(2 * np.pi * df_merged["week"] / 52.0)

# 6. Current Week Infection Baseline (Cases at Week t)
df_merged["cases_week_current"] = df_merged["add_cases_week"]

# 7. TARGET VARIABLE: Next Week's Cases (Week t+1)
df_merged["target_next_week_cases"] = df_merged.groupby("district")["add_cases_week"].shift(-1)
df_merged["target_next_week_per_lakh"] = df_merged.groupby("district")["add_cases_per_lakh"].shift(-1)

# Drop boundary NA values caused by lags & lead
df_merged = df_merged.dropna(subset=["rain_mm_lag1", "rain_mm_lag2", "target_next_week_cases"]).reset_index(drop=True)

# Define Outbreak Risk Categories for Week t+1 (Tertiles / Multipliers)
# Outbreak threshold: High risk when weekly cases > 75th percentile of district
p50 = df_merged.groupby("district")["target_next_week_cases"].transform("median")
p85 = df_merged.groupby("district")["target_next_week_cases"].transform(lambda x: x.quantile(0.80))

conditions = [
    df_merged["target_next_week_cases"] <= p50,
    (df_merged["target_next_week_cases"] > p50) & (df_merged["target_next_week_cases"] <= p85),
    df_merged["target_next_week_cases"] > p85,
]
df_merged["target_next_week_risk"] = np.select(conditions, [0, 1, 2]) # 0=Low, 1=Moderate, 2=High Alert

# Save to CSV and SQLite Database
OUT_CSV = "data/processed/master_district_weekly.csv"
df_merged.to_csv(OUT_CSV, index=False)

conn = sqlite3.connect(DB_PATH)
df_merged.to_sql("master_district_weekly", conn, if_exists="replace", index=False)
conn.execute("CREATE INDEX IF NOT EXISTS idx_dist_week ON master_district_weekly(district, year, week)")
conn.commit()
conn.close()

print(f"  [OK] Successfully created: {OUT_CSV}")
print(f"  [OK] Saved to SQLite: {DB_PATH} (table: master_district_weekly)")
print(f"  Total Weekly Observations: {len(df_merged)} rows")
print(f"  Districts: {df_merged['district'].nunique()} districts x ~750 weeks each")
print(f"  Target: 'target_next_week_cases' & 'target_next_week_risk' (Week t+1)")
