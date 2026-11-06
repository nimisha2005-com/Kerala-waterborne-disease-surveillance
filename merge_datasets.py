"""
Merge all 4 REAL datasets into model-ready tables.

Inputs
  data/processed/disease/kerala_disease_district_yearly.csv   (DHS Kerala / IDSP, 2011-2025)
  data/processed/disease/kerala_disease_state_monthly.csv
  data/raw/climate/kerala_climate_weekly.csv                  (NASA POWER)
  data/raw/water_quality/kerala_water_quality_2024.csv        (CPCB 2024)
  data/raw/demographics/kerala_population_census2011.csv      (Census 2011)

Outputs (data/processed/)
  master_district_yearly.csv   - one row per district per year  (14 x 15 = 210 rows)
  master_state_monthly.csv     - one row per month, whole Kerala (15 x 12 = 180 rows)

Run:  python merge_datasets.py
"""
import datetime as dt
import os

import pandas as pd

OUT = "data/processed"
YEARS = range(2011, 2026)
COVID_YEARS = {2020, 2021}

# ── 1. Load ───────────────────────────────────────────────────────────────
disease = pd.read_csv(f"{OUT}/disease/kerala_disease_district_yearly.csv")
disease_m = pd.read_csv(f"{OUT}/disease/kerala_disease_state_monthly.csv")
climate = pd.read_csv("data/raw/climate/kerala_climate_weekly.csv")
water = pd.read_csv("data/raw/water_quality/kerala_water_quality_2024.csv")
pop = pd.read_csv("data/raw/demographics/kerala_population_census2011.csv")

# ── 2. Climate: ISO week -> calendar year + month ─────────────────────────
# Use the Thursday of each ISO week to decide which year/month it belongs to
# (this also places the "2010 week 52" / "2026 week 1" edge weeks correctly).
thursday = [dt.date.fromisocalendar(int(y), int(w), 4) for y, w in zip(climate.year, climate.week)]
climate["cal_year"] = [d.year for d in thursday]
climate["month"] = [d.month for d in thursday]
climate = climate[climate["cal_year"].isin(YEARS)]


def season_sum(months):
    return lambda g: g.loc[g["month"].isin(months), "rainfall_mm_total"].sum()


rows = []
for (district, year), g in climate.groupby(["district", "cal_year"]):
    rows.append({
        "district": district,
        "year": year,
        "rain_annual_mm": g["rainfall_mm_total"].sum(),
        "rain_premonsoon_mm": season_sum([3, 4, 5])(g),          # Mar-May
        "rain_sw_monsoon_mm": season_sum([6, 7, 8, 9])(g),       # Jun-Sep (main monsoon)
        "rain_ne_monsoon_mm": season_sum([10, 11, 12])(g),       # Oct-Dec
        "rain_max_week_mm": g["rainfall_mm_total"].max(),
        "heavy_rain_weeks": int((g["rainfall_mm_total"] > 150).sum()),  # weeks > 150 mm
        "temp_mean_c": g["temp_c_mean"].mean(),
        "humidity_mean_pct": g["humidity_pct_mean"].mean(),
    })
climate_y = pd.DataFrame(rows)

# ── 3. Water quality: one summary per district (median across stations) ───
wq_cols = {
    "fecal_coliform_max_mpn": "wq_fecal_coliform_mpn",
    "total_coliform_max_mpn": "wq_total_coliform_mpn",
    "bod_max_mg_l": "wq_bod_mg_l",
    "do_min_mg_l": "wq_do_min_mg_l",
    "ph_max": "wq_ph_max",
    "nitrate_max_mg_l": "wq_nitrate_mg_l",
}
water_d = (water.groupby("district")[list(wq_cols)].median()
           .rename(columns=wq_cols).reset_index())
water_d["wq_n_stations"] = water.groupby("district").size().values

# ── 4. Merge district-year master ─────────────────────────────────────────
master = (disease
          .merge(climate_y, on=["district", "year"], how="left")
          .merge(water_d, on="district", how="left")
          .merge(pop[["district", "population_2011", "area_km2",
                      "population_density", "literacy_rate"]], on="district", how="left"))

# Incidence per 1 lakh (100,000) people - fair comparison across districts
per_lakh = {
    "add_cases": "add_per_lakh",
    "typhoid_cases": "typhoid_per_lakh",
    "hepatitis_a_total_cases": "hepatitis_a_per_lakh",
    "leptospirosis_total_cases": "leptospirosis_per_lakh",
    "cholera_cases": "cholera_per_lakh",
}
for src, dst in per_lakh.items():
    master[dst] = master[src].fillna(0) / master["population_2011"] * 1e5

master["covid_year"] = master["year"].isin(COVID_YEARS)
master = master.sort_values(["district", "year"]).reset_index(drop=True)
master.to_csv(f"{OUT}/master_district_yearly.csv", index=False)

# ── 5. State-month master ─────────────────────────────────────────────────
climate_m = (climate.groupby(["cal_year", "month", "district"])
             .agg(rain=("rainfall_mm_total", "sum"),
                  temp=("temp_c_mean", "mean"),
                  hum=("humidity_pct_mean", "mean"))
             .groupby(["cal_year", "month"]).mean()       # average over 14 districts
             .rename(columns={"rain": "rain_mm", "temp": "temp_mean_c", "hum": "humidity_mean_pct"})
             .reset_index().rename(columns={"cal_year": "year"}))
monthly = disease_m.merge(climate_m, on=["year", "month"], how="left")
monthly = monthly.sort_values(["year", "month"])
monthly["rain_prev_month_mm"] = monthly["rain_mm"].shift(1)   # lag: rain last month
monthly["covid_year"] = monthly["year"].isin(COVID_YEARS)
monthly.to_csv(f"{OUT}/master_state_monthly.csv", index=False)

# ── 6. Report ─────────────────────────────────────────────────────────────
print("=" * 60)
print(f"master_district_yearly.csv : {master.shape[0]} rows x {master.shape[1]} columns")
print(f"master_state_monthly.csv   : {monthly.shape[0]} rows x {monthly.shape[1]} columns")
key = ["rain_annual_mm", "wq_fecal_coliform_mpn", "population_2011", "add_cases"]
print("\nMissing values in key columns (should be 0):")
print(master[key].isna().sum().to_string())

print("\nAverage ADD per lakh by district (2011-2025, excl. COVID years):")
avg = (master[~master.covid_year].groupby("district")["add_per_lakh"].mean()
       .sort_values(ascending=False))
for d, v in avg.items():
    print(f"  {d:20s} {v:8.0f}  {'#' * int(v / 100)}")

print("\nCorrelation with ADD per lakh (district-year, excl. COVID):")
num = master[~master.covid_year].select_dtypes("number")
corr = num.corr()["add_per_lakh"].drop(list(per_lakh.values()) + list(per_lakh), errors="ignore")
for c in ["rain_annual_mm", "rain_sw_monsoon_mm", "heavy_rain_weeks", "temp_mean_c",
          "humidity_mean_pct", "wq_fecal_coliform_mpn", "wq_bod_mg_l",
          "population_density", "literacy_rate"]:
    print(f"  {c:25s} {corr[c]:+.2f}")

print("\nMonthly (state): correlation of rainfall with ADD cases:")
mm = monthly[~monthly.covid_year]
print(f"  same month rain  : {mm['rain_mm'].corr(mm['add_cases']):+.2f}")
print(f"  previous month   : {mm['rain_prev_month_mm'].corr(mm['add_cases']):+.2f}")
