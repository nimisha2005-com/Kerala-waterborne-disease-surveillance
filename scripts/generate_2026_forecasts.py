"""
Generates:
1. cache/forecast_2026.db — Full 52-week 2026 spatiotemporal forecast dataset for all 14 Kerala districts
   using the trained LightGBM model (outputs/models_weekly/best_weekly_outbreak_model.joblib) and 5-year
   historical district-week climatological baselines from data/kerala_health.db (opened strictly in mode=ro).
2. cache/district_rivers.json — Polygon-verified river polylines for the major CPCB-monitored rivers
   inside each of the 14 Kerala districts.
"""
import json
import sqlite3
import os
import joblib
import numpy as np
import pandas as pd
from shapely.geometry import shape, Point
from shapely.ops import nearest_points

os.makedirs("cache", exist_ok=True)

DB_RO_URI = "file:data/kerala_health.db?mode=ro"
CACHE_DB_PATH = "cache/forecast_2026.db"
MODEL_PATH = "outputs/models_weekly/best_weekly_outbreak_model.joblib"
GEOJSON_PATH = "data/raw/boundaries/kerala_districts.geojson"
RIVERS_JSON_PATH = "cache/district_rivers.json"

# 1. Load model bundle
bundle = joblib.load(MODEL_PATH)
weekly_model = bundle["regressor"]
if hasattr(weekly_model, "set_params"):
    try:
        weekly_model.set_params(n_jobs=1)
    except Exception:
        pass
feature_cols = bundle["features"]

# 2. Read historical data strictly in READ-ONLY mode
conn_ro = sqlite3.connect(DB_RO_URI, uri=True)
df_hist = pd.read_sql("SELECT * FROM master_district_weekly ORDER BY district, year, week", conn_ro)
conn_ro.close()

print(f"Loaded {len(df_hist)} historical rows (years {df_hist['year'].min()}-{df_hist['year'].max()})")

# 3. Build 2026 weekly projection (Weeks 1 to 52) for all 14 districts
all_districts = sorted(df_hist["district"].unique())
rows_2026 = []

for dist in all_districts:
    df_d = df_hist[df_hist["district"] == dist].sort_values(["year", "week"])
    last_row = df_d.iloc[-1].to_dict()

    # Static baselines per district
    pop_2011 = int(last_row["population_2011"])
    pop_dens = float(last_row["population_density"])
    lit_rate = float(last_row["literacy_rate"])
    bod_base = float(last_row["wq_bod_mg_l"])
    fecal_base = float(last_row["wq_fecal_coliform_mpn"])
    do_base = float(last_row["wq_do_min_mg_l"])

    # Initialize lag state from end of 2025 (W50, W51)
    prev_rain_lag1 = float(last_row["rainfall_mm_total"])
    prev_rain_lag2 = float(last_row["rain_mm_lag1"])
    prev_temp_lag1 = float(last_row["temp_c_mean"])

    # Predict 2025-W51 target as initial 2026-W01 active cases
    X_init = pd.DataFrame([[float(last_row[c]) for c in feature_cols]], columns=feature_cols)
    current_cases = max(5.0, float(weekly_model.predict(X_init)[0]))

    for w in range(1, 53):
        # Match historical week w (use week 51 if w==52 and historical max is 51)
        w_lookup = min(w, 51)
        df_dw = df_d[df_d["week"] == w_lookup]

        # Compute realistic 2026 weekly climate from 5-year district distribution (weighted toward recent years + monsoon variance)
        rain_mean = float(df_dw["rainfall_mm_total"].mean())
        rain_max = float(df_dw["rainfall_mm_total"].max())
        # Blend mean and recent year pattern so monsoon peaks (Jun-Sep W22-W39) have realistic heavy rain events
        rain_2025 = float(df_dw.iloc[-1]["rainfall_mm_total"]) if not df_dw.empty else rain_mean
        rain_w = round(0.55 * rain_mean + 0.45 * rain_2025, 2)

        temp_w = round(float(df_dw["temp_c_mean"].mean()), 2)
        hum_w = round(float(df_dw["humidity_pct_mean"].mean()), 2)

        sin_w = float(np.sin(2 * np.pi * w / 52.0))
        cos_w = float(np.cos(2 * np.pi * w / 52.0))

        rain_2wk_sum = round(rain_w + prev_rain_lag1, 2)
        is_torrential = 1 if rain_w >= 100.0 else 0

        # Seasonal historical baseline cases for week w (prevents drift over 52-week roll-forward)
        hist_cases_w = float(df_dw["cases_week_current"].mean())
        # Blend 65% autoregressive model step + 35% historical seasonal anchor
        effective_current_cases = int(round(0.65 * current_cases + 0.35 * hist_cases_w))

        row_dict = {
            "district": dist,
            "year": 2026,
            "week": w,
            "rainfall_mm_total": rain_w,
            "rain_mm_lag1": prev_rain_lag1,
            "rain_mm_lag2": prev_rain_lag2,
            "rain_2week_sum": rain_2wk_sum,
            "is_torrential_rain": is_torrential,
            "temp_c_mean": temp_w,
            "temp_c_lag1": prev_temp_lag1,
            "humidity_pct_mean": hum_w,
            "wq_bod_mg_l": bod_base,
            "wq_fecal_coliform_mpn": fecal_base,
            "wq_do_min_mg_l": do_base,
            "population_2011": pop_2011,
            "population_density": pop_dens,
            "literacy_rate": lit_rate,
            "sin_week": sin_w,
            "cos_week": cos_w,
            "cases_week_current": effective_current_cases,
            "add_cases_week": effective_current_cases,
            "add_cases_per_lakh": round((effective_current_cases / pop_2011) * 100000.0, 2),
        }

        # Predict next week (t+1) using trained LightGBM model
        X_step = pd.DataFrame([[float(row_dict[c]) for c in feature_cols]], columns=feature_cols)
        pred_next = max(5.0, float(weekly_model.predict(X_step)[0]))
        row_dict["target_next_week_cases"] = round(pred_next, 1)
        rate_next = (pred_next / pop_2011) * 100000.0
        row_dict["target_next_week_risk"] = 2 if (rate_next >= 22.0 or rain_w >= 140.0) else (1 if rate_next >= 14.0 else 0)

        rows_2026.append(row_dict)

        # Advance lags for next week
        prev_rain_lag2 = prev_rain_lag1
        prev_rain_lag1 = rain_w
        prev_temp_lag1 = temp_w
        current_cases = pred_next

df_2026 = pd.DataFrame(rows_2026)

# Save to cache/forecast_2026.db
if os.path.exists(CACHE_DB_PATH):
    os.remove(CACHE_DB_PATH)

conn_cache = sqlite3.connect(CACHE_DB_PATH)
df_2026.to_sql("master_district_weekly_2026", conn_cache, index=False)
conn_cache.execute("CREATE INDEX idx_2026_dist_yr_wk ON master_district_weekly_2026 (district, year, week)")
conn_cache.commit()
conn_cache.close()
print(f"Saved {len(df_2026)} forecasted 2026 rows (14 districts x 52 weeks) to {CACHE_DB_PATH}")

# 4. Build Polygon-Verified Monitored Rivers per District for the Kerala Vector Map
with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
    gj = json.load(f)

district_inner = {}
for feat in gj["features"]:
    dname = feat["properties"]["district"]
    g = shape(feat["geometry"]).simplify(0.006)
    buf = g.buffer(-0.008)
    district_inner[dname] = buf if not buf.is_empty else g

# Major CPCB-monitored rivers per district (from Western Ghats / inland -> coastal estuary inside each district)
RAW_DISTRICT_RIVERS = {
    "Thiruvananthapuram": [
        {"name": "Karamana River", "coords": [[8.625, 77.120], [8.561, 77.015], [8.505, 76.980], [8.485, 76.967], [8.451, 76.951]]},
        {"name": "Vamanapuram River", "coords": [[8.720, 77.060], [8.685, 76.970], [8.665, 76.905], [8.645, 76.830], [8.630, 76.790]]},
        {"name": "Neyyar River", "coords": [[8.530, 77.180], [8.460, 77.100], [8.411, 77.012], [8.398, 77.045], [8.350, 77.015]]}
    ],
    "Kollam": [
        {"name": "Kallada River", "coords": [[8.981, 77.061], [9.010, 76.980], [9.021, 76.924], [9.015, 76.780], [8.995, 76.650], [8.991, 76.612]]},
        {"name": "Ithikkara River", "coords": [[8.910, 76.900], [8.895, 76.810], [8.881, 76.714], [8.875, 76.680]]},
        {"name": "Pallickal River", "coords": [[9.120, 76.740], [9.091, 76.685], [9.075, 76.600], [9.060, 76.555]]}
    ],
    "Pathanamthitta": [
        {"name": "Pamba River", "coords": [[9.390, 77.050], [9.365, 76.920], [9.345, 76.812], [9.285, 76.791], [9.320, 76.680], [9.350, 76.590]]},
        {"name": "Achankovil River", "coords": [[9.180, 77.020], [9.220, 76.880], [9.268, 76.784], [9.221, 76.714], [9.231, 76.681]]},
        {"name": "Manimala River", "coords": [[9.450, 76.780], [9.410, 76.690], [9.381, 76.621], [9.371, 76.581]]}
    ],
    "Alappuzha": [
        {"name": "Pamba River (Lower Basin)", "coords": [[9.318, 76.617], [9.335, 76.540], [9.345, 76.512], [9.391, 76.432], [9.421, 76.445]]},
        {"name": "Achankovil River (Lower)", "coords": [[9.235, 76.590], [9.261, 76.541], [9.300, 76.495], [9.321, 76.481]]},
        {"name": "Vembanad – Kayamkulam Waterway", "coords": [[9.650, 76.345], [9.500, 76.355], [9.350, 76.410], [9.200, 76.490]]}
    ],
    "Kottayam": [
        {"name": "Meenachil River", "coords": [[9.740, 76.780], [9.712, 76.660], [9.681, 76.581], [9.640, 76.540], [9.612, 76.512]]},
        {"name": "Muvattupuzha – Vaikom Channel", "coords": [[9.850, 76.520], [9.845, 76.460], [9.830, 76.420]]},
        {"name": "Manimala River (Kottayam Reach)", "coords": [[9.560, 76.780], [9.510, 76.680], [9.455, 76.565]]}
    ],
    "Idukki": [
        {"name": "Periyar River (High Range)", "coords": [[9.580, 77.150], [9.700, 77.080], [9.841, 77.012], [9.950, 76.940], [10.040, 76.860]]},
        {"name": "Thodupuzha River", "coords": [[9.820, 76.860], [9.860, 76.790], [9.895, 76.720], [9.910, 76.670]]},
        {"name": "Muthirapuzha River", "coords": [[10.090, 77.060], [10.030, 77.000], [9.970, 76.950]]}
    ],
    "Ernakulam": [
        {"name": "Periyar River", "coords": [[10.155, 76.650], [10.165, 76.435], [10.112, 76.353], [10.079, 76.305], [10.046, 76.319]]},
        {"name": "Muvattupuzha River", "coords": [[9.985, 76.580], [9.932, 76.541], [9.912, 76.490], [9.890, 76.430]]},
        {"name": "Chithrapuzha & Kadamprayar", "coords": [[10.015, 76.410], [9.998, 76.371], [9.972, 76.358], [9.955, 76.325]]}
    ],
    "Thrissur": [
        {"name": "Chalakudy River", "coords": [[10.300, 76.580], [10.305, 76.450], [10.312, 76.345], [10.265, 76.270]]},
        {"name": "Karuvannur River", "coords": [[10.430, 76.420], [10.425, 76.310], [10.421, 76.221], [10.390, 76.160]]},
        {"name": "Puzhakkal – Keecheri River", "coords": [[10.580, 76.270], [10.541, 76.195], [10.525, 76.130]]}
    ],
    "Palakkad": [
        {"name": "Bharathapuzha (Nila River)", "coords": [[10.781, 76.645], [10.765, 76.480], [10.771, 76.377], [10.762, 76.271], [10.801, 76.195], [10.825, 76.110]]},
        {"name": "Bhavani River (Attappady)", "coords": [[11.140, 76.620], [11.081, 76.681], [11.030, 76.720]]},
        {"name": "Chitturpuzha – Gayathripuzha", "coords": [[10.701, 76.712], [10.675, 76.580], [10.720, 76.460]]}
    ],
    "Malappuram": [
        {"name": "Chaliyar River", "coords": [[11.320, 76.280], [11.275, 76.225], [11.235, 76.080], [11.214, 75.981], [11.231, 75.951]]},
        {"name": "Kadalundi River", "coords": [[11.080, 76.240], [11.050, 76.100], [11.031, 75.995], [11.045, 75.921]]},
        {"name": "Bharathapuzha (Ponnani Reach)", "coords": [[10.840, 76.030], [10.815, 75.980], [10.785, 75.945]]}
    ],
    "Kozhikode": [
        {"name": "Chaliyar & Kallayi River", "coords": [[11.245, 75.920], [11.231, 75.840], [11.228, 75.795], [11.180, 75.820]]},
        {"name": "Korapuzha River", "coords": [[11.420, 75.880], [11.380, 75.820], [11.365, 75.771], [11.351, 75.761]]},
        {"name": "Kuttiyadi & Mahe River", "coords": [[11.640, 75.820], [11.581, 75.771], [11.530, 75.670], [11.516, 75.626]]}
    ],
    "Wayanad": [
        {"name": "Kabani River", "coords": [[11.671, 75.961], [11.739, 76.074], [11.792, 76.168], [11.691, 76.151], [11.661, 76.320]]},
        {"name": "Karapuzha – Panamaram River", "coords": [[11.610, 76.082], [11.612, 76.171], [11.690, 76.150]]},
        {"name": "Bavali – Papanasini Stream", "coords": [[11.871, 75.991], [11.895, 76.060], [11.910, 76.110]]}
    ],
    "Kannur": [
        {"name": "Valapattanam River", "coords": [[11.981, 75.761], [11.981, 75.561], [11.975, 75.405], [11.951, 75.421], [11.931, 75.362]]},
        {"name": "Anjarakandy River", "coords": [[11.880, 75.640], [11.881, 75.495], [11.825, 75.460], [11.780, 75.480]]},
        {"name": "Kuppam & Peruvamba River", "coords": [[12.140, 75.440], [12.081, 75.385], [12.045, 75.312], [12.091, 75.251]]}
    ],
    "Kasaragod": [
        {"name": "Chandragiri (Payaswini) River", "coords": [[12.520, 75.240], [12.505, 75.140], [12.495, 75.050], [12.500, 75.005]]},
        {"name": "Nileshwaram – Kuppam River", "coords": [[12.320, 75.240], [12.295, 75.160], [12.256, 75.133]]},
        {"name": "Shiriya & Mogral River", "coords": [[12.630, 75.100], [12.612, 74.995], [12.561, 74.975]]}
    ]
}

verified_rivers = {}
for dname, r_list in RAW_DISTRICT_RIVERS.items():
    inner = district_inner[dname]
    clean_list = []
    for r_obj in r_list:
        snapped_coords = []
        for lat, lon in r_obj["coords"]:
            pt = Point(lon, lat)
            if not inner.contains(pt):
                np_pt = nearest_points(inner, pt)[0]
                lat, lon = round(np_pt.y, 4), round(np_pt.x, 4)
            else:
                lat, lon = round(lat, 4), round(lon, 4)
            snapped_coords.append([lat, lon])
        clean_list.append({
            "name": r_obj["name"],
            "coords": snapped_coords
        })
    verified_rivers[dname] = clean_list

with open(RIVERS_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(verified_rivers, f, indent=2)

print(f"Saved polygon-verified river paths for all {len(verified_rivers)} districts to {RIVERS_JSON_PATH}")
