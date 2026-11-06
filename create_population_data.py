import pandas as pd
import os

# ── Kerala district population data (Census 2011) ─────────────────────────
# Source: Census of India 2011, Primary Census Abstract, Kerala
# URL: https://censusindia.gov.in
#
# Columns:
#   district          : District name (matches our GeoJSON and other datasets)
#   population_2011   : Total population (Census 2011)
#   area_km2          : Geographic area in square kilometres
#   population_density: Persons per square kilometre (Census 2011)
#   urban_pct         : Percentage of population living in urban areas
#   literacy_rate     : Literacy rate percentage (Census 2011)
#
# NOTE: Population figures are from the official 2011 Census of India.
# Kerala's Census 2021 data has not been fully released as of 2024.

KERALA_DEMOGRAPHICS = [
    {
        "district":           "Thiruvananthapuram",
        "population_2011":    3301427,
        "area_km2":           2192,
        "population_density": 1509,
        "urban_pct":          53.8,
        "literacy_rate":      93.0,
    },
    {
        "district":           "Kollam",
        "population_2011":    2635375,
        "area_km2":           2491,
        "population_density": 1058,
        "urban_pct":          38.4,
        "literacy_rate":      94.1,
    },
    {
        "district":           "Pathanamthitta",
        "population_2011":    1197412,
        "area_km2":           2642,
        "population_density": 453,
        "urban_pct":          16.0,
        "literacy_rate":      96.7,
    },
    {
        "district":           "Alappuzha",
        "population_2011":    2127789,
        "area_km2":           1414,
        "population_density": 1505,
        "urban_pct":          26.9,
        "literacy_rate":      95.8,
    },
    {
        "district":           "Kottayam",
        "population_2011":    1979384,
        "area_km2":           2208,
        "population_density": 896,
        "urban_pct":          27.3,
        "literacy_rate":      97.2,
    },
    {
        "district":           "Idukki",
        "population_2011":    1107453,
        "area_km2":           4358,
        "population_density": 254,
        "urban_pct":          7.9,
        "literacy_rate":      95.0,
    },
    {
        "district":           "Ernakulam",
        "population_2011":    3282388,
        "area_km2":           3068,
        "population_density": 1070,
        "urban_pct":          67.1,
        "literacy_rate":      95.9,
    },
    {
        "district":           "Thrissur",
        "population_2011":    3121200,
        "area_km2":           3032,
        "population_density": 1030,
        "urban_pct":          44.0,
        "literacy_rate":      95.3,
    },
    {
        "district":           "Palakkad",
        "population_2011":    2809934,
        "area_km2":           4480,
        "population_density": 627,
        "urban_pct":          27.5,
        "literacy_rate":      89.3,
    },
    {
        "district":           "Malappuram",
        "population_2011":    4112920,
        "area_km2":           3550,
        "population_density": 1158,
        "urban_pct":          27.1,
        "literacy_rate":      93.6,
    },
    {
        "district":           "Kozhikode",
        "population_2011":    3086293,
        "area_km2":           2345,
        "population_density": 1316,
        "urban_pct":          54.5,
        "literacy_rate":      95.3,
    },
    {
        "district":           "Wayanad",
        "population_2011":    817420,
        "area_km2":           2132,
        "population_density": 383,
        "urban_pct":          13.0,
        "literacy_rate":      89.0,
    },
    {
        "district":           "Kannur",
        "population_2011":    2523003,
        "area_km2":           2966,
        "population_density": 851,
        "urban_pct":          29.8,
        "literacy_rate":      95.4,
    },
    {
        "district":           "Kasaragod",
        "population_2011":    1307375,
        "area_km2":           1992,
        "population_density": 657,
        "urban_pct":          23.3,
        "literacy_rate":      90.1,
    },
]

# ── Create DataFrame ───────────────────────────────────────────────────────
df = pd.DataFrame(KERALA_DEMOGRAPHICS)

# Add a calculated column: population per 1 lakh (100,000)
# This is used to normalize disease case counts for fair district comparison
df["pop_per_lakh"] = (df["population_2011"] / 100000).round(2)

# ── Save to CSV ───────────────────────────────────────────────────────────
os.makedirs("data/raw/demographics", exist_ok=True)
output_path = "data/raw/demographics/kerala_population_census2011.csv"
df.to_csv(output_path, index=False)

# ── Print summary ─────────────────────────────────────────────────────────
print("Kerala District Population — Census 2011")
print("=" * 60)
print(df[["district", "population_2011", "area_km2",
          "population_density", "literacy_rate"]].to_string(index=False))
print()
print(f"Total Kerala population: {df['population_2011'].sum():,}")
print(f"Most populous district:  {df.loc[df['population_2011'].idxmax(), 'district']}")
print(f"Least populous district: {df.loc[df['population_2011'].idxmin(), 'district']}")
print(f"Most densely populated:  {df.loc[df['population_density'].idxmax(), 'district']}")
print()
print(f"Saved to: {output_path}")
