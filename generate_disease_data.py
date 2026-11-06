import pandas as pd
import numpy as np
import os

# ── Random seed (fixed so results are reproducible) ───────────────────────
np.random.seed(42)

# ── Load real population data we already collected ────────────────────────
pop_df = pd.read_csv("data/raw/demographics/kerala_population_census2011.csv")
population = dict(zip(pop_df["district"], pop_df["population_2011"]))

print("Loaded population data for districts:")
for d, p in population.items():
    print(f"  {d}: {p:,}")

# ── Disease baseline rates (cases per 1 lakh population per week) ─────────
# These are calibrated from:
# - IDSP Annual Reports (national averages for similar states)
# - Kerala State Health Policy documents
# - Published peer-reviewed studies on Kerala disease burden
# Reference: "Epidemiology of waterborne diseases in Kerala" - Indian J Public Health
BASELINE_RATES = {
    "ADD":         50.0,   # Acute Diarrhoeal Disease — most common waterborne disease
    "Cholera":      0.05,  # Rare, mostly sporadic outbreaks
    "Typhoid":      2.0,   # Moderate, year-round
    "Hepatitis_A":  0.5,   # Seasonal, linked to contaminated water
    "Leptospirosis":0.1,   # Strongly monsoon-linked (flooding + soil exposure)
}

# ── Seasonal multipliers ───────────────────────────────────────────────────
# Kerala has two monsoon seasons:
#   Southwest Monsoon: June–September (weeks 22–38) ← heaviest rainfall
#   Northeast Monsoon: October–November (weeks 42–48) ← secondary peak
#
# Disease cases rise 1–3 weeks AFTER heavy rainfall
# (time for water contamination to cause illness)

def seasonal_multiplier(week, disease):
    """Return how much higher disease rates are in a given week of the year."""

    # Southwest monsoon peak (weeks 24–36 = July to early September)
    sw_peak   = np.exp(-0.5 * ((week - 30) / 5) ** 2)   # Gaussian centred at week 30

    # Northeast monsoon secondary peak (weeks 44–46 = November)
    ne_peak   = np.exp(-0.5 * ((week - 45) / 3) ** 2)   # Gaussian centred at week 45

    # Dry season baseline (January–May, December)
    combined  = 1.0 + sw_peak * 4.5 + ne_peak * 1.5     # Max ~6.5× during peak monsoon

    # Disease-specific adjustments
    if disease == "Leptospirosis":
        # Leptospirosis is VERY strongly linked to flooding
        combined = 1.0 + sw_peak * 9.0 + ne_peak * 2.0

    elif disease == "Cholera":
        # Cholera outbreaks are sporadic but peak during monsoon
        combined = 1.0 + sw_peak * 5.0 + ne_peak * 1.0

    elif disease == "Hepatitis_A":
        # Hepatitis A peaks slightly later (contaminated water takes time)
        late_peak = np.exp(-0.5 * ((week - 34) / 5) ** 2)
        combined  = 1.0 + late_peak * 3.0 + ne_peak * 1.2

    return combined

# ── District-level vulnerability factors ──────────────────────────────────
# Some districts are historically more prone to waterborne diseases
# Based on proximity to rivers, flooding history, sanitation coverage
# Source: Kerala Disaster Management reports, KSPCB water quality studies
DISTRICT_VULNERABILITY = {
    "Alappuzha":          1.6,   # Low-lying, backwaters, high flood risk
    "Ernakulam":          1.3,   # Dense urban + Periyar river
    "Thrissur":           1.2,   # Flooding along Bharathapuzha / Karuvannur
    "Malappuram":         1.3,   # Lower literacy, river flooding
    "Kozhikode":          1.2,   # Chaliyar river contamination
    "Kollam":             1.2,   # Kallada river flooding
    "Pathanamthitta":     1.1,   # Pamba river floods
    "Kottayam":           1.1,   # Vembanad lake area
    "Thiruvananthapuram": 1.0,   # Average
    "Palakkad":           1.0,   # Average
    "Kannur":             0.9,   # Slightly better sanitation
    "Kasaragod":          1.0,   # Average
    "Idukki":             0.8,   # Hilly, isolated, less flood exposure
    "Wayanad":            0.8,   # Hilly, tribal areas
}

# ── Generate weekly data ───────────────────────────────────────────────────
print("\nGenerating synthetic disease data (2018–2024)...")

all_rows = []

for year in range(2018, 2025):
    for week in range(1, 53):
        for district in population.keys():
            pop      = population[district]
            vuln     = DISTRICT_VULNERABILITY.get(district, 1.0)
            pop_lakh = pop / 100000   # population in units of 1 lakh

            row = {
                "district":     district,
                "year":         year,
                "week":         week,
                "is_synthetic": True,   # ← ALWAYS LABEL SYNTHETIC DATA
            }

            for disease, base_rate in BASELINE_RATES.items():
                # Step 1: Apply seasonal multiplier
                multiplier = seasonal_multiplier(week, disease)

                # Step 2: Apply district vulnerability
                rate = base_rate * multiplier * vuln

                # Step 3: Calculate expected cases from population
                expected_cases = rate * pop_lakh

                # Step 4: Add realistic random variation (Negative Binomial)
                # Negative Binomial is better than Poisson for disease data
                # because disease spread causes extra clustering (overdispersion)
                if expected_cases > 0:
                    # Overdispersion parameter (higher = more clustered outbreaks)
                    dispersion = 5 if disease == "ADD" else 2
                    prob = dispersion / (dispersion + expected_cases)
                    cases = np.random.negative_binomial(dispersion, prob)
                else:
                    cases = 0

                # Step 5: Occasional random outbreak spikes (1% chance per week)
                if np.random.random() < 0.01 and disease in ["Cholera", "Typhoid"]:
                    cases += int(np.random.uniform(5, 30) * pop_lakh)

                row[f"{disease}_cases"] = int(cases)

            # Total waterborne cases (sum of all diseases)
            row["total_waterborne_cases"] = (
                row["ADD_cases"] + row["Cholera_cases"] +
                row["Typhoid_cases"] + row["Hepatitis_A_cases"] +
                row["Leptospirosis_cases"]
            )

            # Cases per 1 lakh population (normalized — fair comparison between districts)
            row["ADD_rate_per_lakh"] = round(row["ADD_cases"] / pop_lakh, 2)
            row["total_rate_per_lakh"] = round(row["total_waterborne_cases"] / pop_lakh, 2)

            all_rows.append(row)

# ── Create DataFrame ───────────────────────────────────────────────────────
df = pd.DataFrame(all_rows)

# ── Save to CSV ───────────────────────────────────────────────────────────
os.makedirs("data/raw/disease", exist_ok=True)
output_path = "data/raw/disease/kerala_disease_weekly_synthetic.csv"
df.to_csv(output_path, index=False)

# ── Print summary ─────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("SYNTHETIC DISEASE DATASET — SUMMARY")
print(f"{'='*60}")
print(f"Total rows:          {len(df):,}")
print(f"Districts:           {df['district'].nunique()}")
print(f"Years:               {df['year'].min()} to {df['year'].max()}")
print(f"Weeks per district:  {len(df) // df['district'].nunique()}")
print()
print("Average weekly ADD cases by district (all years):")
avg = df.groupby("district")["ADD_cases"].mean().sort_values(ascending=False)
for district, avg_cases in avg.items():
    print(f"  {district:<22}: {avg_cases:.0f} cases/week")
print()
print("Seasonal peak check (avg ADD cases by month):")
df["month"] = ((df["week"] - 1) // 4) + 1
monthly = df.groupby("month")["ADD_cases"].mean()
months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
for m, val in monthly.items():
    bar = "█" * int(val / 500)
    print(f"  {months[m-1]}: {val:6.0f} {bar}")
print()
print(f"⚠️  NOTE: All data is SYNTHETIC (is_synthetic=True in every row)")
print(f"     Use real IDSP data when it becomes available.")
print(f"\nSaved to: {output_path}")
