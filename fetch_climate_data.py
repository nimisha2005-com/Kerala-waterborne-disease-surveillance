import requests
import pandas as pd
import os
import time

# ── Kerala district centroids (latitude, longitude) ───────────────────────
# These are the centre points of each district, used to fetch weather data
DISTRICTS = {
    "Thiruvananthapuram": (8.5074,  76.9720),
    "Kollam":             (8.8932,  76.6141),
    "Pathanamthitta":     (9.2648,  76.7870),
    "Alappuzha":          (9.4981,  76.3388),
    "Kottayam":           (9.5916,  76.5222),
    "Idukki":             (9.9189,  77.1025),
    "Ernakulam":          (10.0161, 76.3780),
    "Thrissur":           (10.5276, 76.2144),
    "Palakkad":           (10.7867, 76.6548),
    "Malappuram":         (11.0730, 76.0740),
    "Kozhikode":          (11.2588, 75.7804),
    "Wayanad":            (11.6854, 76.1320),
    "Kannur":             (11.8745, 75.3704),
    "Kasaragod":          (12.4996, 74.9869),
}

# ── Settings ──────────────────────────────────────────────────────────────
START_DATE = "20110101"   # matches disease data (2011-2025)
END_DATE   = "20251231"
NASA_URL   = "https://power.larc.nasa.gov/api/temporal/daily/point"

# Parameters we want:
# PRECTOTCORR = Rainfall (mm/day)
# T2M         = Temperature at 2 metres (°C)
# RH2M        = Relative Humidity at 2 metres (%)
PARAMETERS = "PRECTOTCORR,T2M,RH2M"

# ── Output folder ─────────────────────────────────────────────────────────
os.makedirs("data/raw/climate", exist_ok=True)
output_path = "data/raw/climate/kerala_climate_weekly.csv"

# ── Fetch data for each district ──────────────────────────────────────────
all_data = []

# Keep districts already downloaded for the full period (so a re-run only
# fetches the ones that failed, e.g. after a "429 Too Many Requests" error)
done = set()
if os.path.exists(output_path):
    old = pd.read_csv(output_path)
    yr = old.groupby("district")["year"].agg(["min", "max"])
    done = set(yr[(yr["min"] <= int(START_DATE[:4])) & (yr["max"] >= int(END_DATE[:4]))].index)
    if done:
        all_data.append(old[old["district"].isin(done)])
        print(f"Already downloaded (skipping): {len(done)} districts")

for district, (lat, lon) in DISTRICTS.items():
    if district in done:
        continue
    print(f"Fetching data for {district}...")

    params = {
        "parameters": PARAMETERS,
        "community":  "RE",
        "longitude":  lon,
        "latitude":   lat,
        "start":      START_DATE,
        "end":        END_DATE,
        "format":     "JSON",
    }

    try:
        # Retry up to 4 times, waiting longer each time (NASA rate limit = error 429)
        for attempt in range(4):
            response = requests.get(NASA_URL, params=params, timeout=120)
            if response.status_code == 429 and attempt < 3:
                wait = 30 * (attempt + 1)
                print(f"  NASA says 'too many requests' - waiting {wait}s and retrying...")
                time.sleep(wait)
                continue
            break
        response.raise_for_status()
        data = response.json()

        # Extract the daily values
        daily = data["properties"]["parameter"]
        rainfall    = daily["PRECTOTCORR"]   # mm/day
        temperature = daily["T2M"]           # °C
        humidity    = daily["RH2M"]          # %

        # Build a DataFrame from daily data
        df = pd.DataFrame({
            "date":        list(rainfall.keys()),
            "rainfall_mm": list(rainfall.values()),
            "temp_c":      list(temperature.values()),
            "humidity_pct":list(humidity.values()),
        })

        # Convert date column to proper date format
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")

        # Add district name
        df["district"] = district

        # Replace NASA's fill value (-999) with NaN (missing)
        df.replace(-999, float("nan"), inplace=True)

        # Aggregate from daily to weekly
        df["year"] = df["date"].dt.isocalendar().year
        df["week"] = df["date"].dt.isocalendar().week

        weekly = df.groupby(["district", "year", "week"]).agg(
            rainfall_mm_total = ("rainfall_mm", "sum"),
            temp_c_mean       = ("temp_c",       "mean"),
            humidity_pct_mean = ("humidity_pct", "mean"),
        ).reset_index()

        all_data.append(weekly)
        print(f"  ✓ {len(weekly)} weeks of data collected for {district}")

    except Exception as e:
        print(f"  ✗ Error fetching {district}: {e}")

    # Wait 2 seconds between requests to be polite to NASA's server
    time.sleep(2)

# ── Combine all districts and save ───────────────────────────────────────
if all_data:
    final_df = pd.concat(all_data, ignore_index=True)
    final_df.sort_values(["district", "year", "week"], inplace=True)
    final_df.to_csv(output_path, index=False)

    print()
    print("=" * 50)
    print(f"Saved to: {output_path}")
    print(f"Total rows: {len(final_df)}")
    print(f"Districts: {final_df['district'].nunique()}")
    print(f"Date range: {final_df['year'].min()} to {final_df['year'].max()}")
    print()
    print("Preview (first 5 rows):")
    print(final_df.head().to_string(index=False))
else:
    print("No data was collected. Check your internet connection.")
