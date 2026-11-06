"""
Compute Global and Local SHAP (SHapley Additive exPlanations) for the Winning LightGBM Outbreak Model.
Produces:
1. outputs/models_weekly/shap_summary_beeswarm.png
2. outputs/models_weekly/shap_feature_importance_bar.png
3. outputs/models_weekly/shap_rainfall_dependence.png
4. outputs/models_weekly/shap_global_importance.json
5. outputs/models_weekly/shap_district_explanations.json
"""
import os
import json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

# Set cache dir to avoid Windows temp permission warnings
os.environ["MPLCONFIGDIR"] = "outputs/.mplcache"
os.makedirs("outputs/.mplcache", exist_ok=True)
os.makedirs("outputs/models_weekly", exist_ok=True)

MODEL_PATH = "outputs/models_weekly/best_weekly_outbreak_model.joblib"
DATA_PATH = "data/processed/master_district_weekly.csv"

print("Loading model and weekly dataset...")
bundle = joblib.load(MODEL_PATH)
model = bundle["regressor"]
features = bundle["features"]

df = pd.read_csv(DATA_PATH)
df_clean = df.dropna(subset=features + [bundle["target"]])

X = df_clean[features]
y = df_clean[bundle["target"]]

# Friendly feature names for clean presentation
FEATURE_LABELS = {
    "cases_week_current": "Current Week Cases (Baseline)",
    "rain_mm_lag1": "Preceding Week Rain (Lag-1)",
    "rainfall_mm_total": "Current Week Rain (t)",
    "rain_2week_sum": "2-Week Cumulative Rain",
    "rain_mm_lag2": "2 Weeks Prior Rain (Lag-2)",
    "wq_bod_mg_l": "River BOD (Pollution)",
    "wq_fecal_coliform_mpn": "Fecal Coliform Count",
    "wq_do_min_mg_l": "Dissolved Oxygen (DO)",
    "temp_c_mean": "Mean Temperature (°C)",
    "temp_c_lag1": "Preceding Week Temp",
    "humidity_pct_mean": "Relative Humidity (%)",
    "population_density": "Population Density",
    "literacy_rate": "Literacy Rate (%)",
    "is_torrential_rain": "Torrential Rain Shock",
    "sin_week": "Monsoon Seasonality (Sin)",
    "cos_week": "Monsoon Seasonality (Cos)",
}

X_renamed = X.rename(columns=FEATURE_LABELS)

print(f"Initializing SHAP TreeExplainer on LightGBM ({len(X)} rows)...")
explainer = shap.TreeExplainer(model)

# Compute SHAP values on a stratified sample of 2,000 rows for swift, accurate global distribution
sample_size = min(2000, len(X))
X_sample = X.sample(n=sample_size, random_state=42)
X_sample_renamed = X_sample.rename(columns=FEATURE_LABELS)

print("Calculating SHAP values...")
shap_values = explainer(X_sample)
shap_vals_matrix = shap_values.values

# 1. SHAP Beeswarm Summary Plot (Dark Theme styling)
print("Generating SHAP Beeswarm Summary Plot...")
plt.figure(figsize=(10, 6.5))
plt.style.use("dark_background")
fig = plt.gcf()
fig.patch.set_facecolor("#000000")
ax = plt.gca()
ax.set_facecolor("#000000")

shap.summary_plot(
    shap_vals_matrix,
    X_sample_renamed,
    show=False,
    plot_size=(10, 6.5),
    color_bar_label="Feature Value (Low → High)"
)
plt.title("SHAP Global Feature Impact on Next-Week Outbreak Cases (LightGBM)", fontsize=13, fontweight="bold", pad=15, color="#ffffff")
plt.xlabel("SHAP Value (Impact on Predicted Disease Cases)", fontsize=11, color="#e2e8f0")
plt.tight_layout()
plt.savefig("outputs/models_weekly/shap_summary_beeswarm.png", dpi=200, facecolor="#000000")
plt.close()

# 2. SHAP Mean Absolute Importance Bar Plot
print("Generating SHAP Feature Importance Bar Plot...")
mean_abs_shap = np.abs(shap_vals_matrix).mean(axis=0)
df_shap_imp = pd.DataFrame({
    "feature": [FEATURE_LABELS.get(f, f) for f in features],
    "raw_feature": features,
    "mean_abs_shap": mean_abs_shap
}).sort_values("mean_abs_shap", ascending=True)

plt.figure(figsize=(9, 6))
plt.style.use("dark_background")
fig = plt.gcf()
fig.patch.set_facecolor("#000000")
ax = plt.gca()
ax.set_facecolor("#000000")

bars = ax.barh(df_shap_imp["feature"], df_shap_imp["mean_abs_shap"], color="#0284c7", edgecolor="#38bdf8", alpha=0.85)
ax.set_xlabel("Mean |SHAP Value| (Average Cases / Week Impact)", fontsize=11, color="#e2e8f0")
ax.set_title("Global Feature Importance (SHAP Cooperative Game Theory)", fontsize=12, fontweight="bold", color="#ffffff", pad=12)
ax.grid(axis="x", color="#ffffff", alpha=0.1, linestyle="--")

for bar in bars:
    w = bar.get_width()
    ax.text(w + 1.5, bar.get_y() + bar.get_height()/2, f"{w:.1f}", va="center", ha="left", fontsize=9, color="#ffffff", fontfamily="sans-serif")

plt.tight_layout()
plt.savefig("outputs/models_weekly/shap_feature_importance_bar.png", dpi=200, facecolor="#000000")
plt.close()

# 3. SHAP Partial Dependence: Lag-1 Rainfall vs Outbreak Cases
print("Generating SHAP Rainfall Dependence Plot...")
plt.figure(figsize=(8.5, 5))
fig = plt.gcf()
fig.patch.set_facecolor("#000000")
ax = plt.gca()
ax.set_facecolor("#000000")

idx_rain_lag = features.index("rain_mm_lag1")
rain_vals = X_sample["rain_mm_lag1"].values
shap_rain = shap_vals_matrix[:, idx_rain_lag]

scatter = ax.scatter(rain_vals, shap_rain, c=X_sample["wq_bod_mg_l"].values, cmap="plasma", alpha=0.75, s=25)
cb = plt.colorbar(scatter, ax=ax)
cb.set_label("River BOD (mg/L)", color="#e2e8f0")
cb.ax.yaxis.set_tick_params(color="#e2e8f0")
plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="#e2e8f0")

ax.set_title("SHAP Partial Dependence: Preceding-Week Rain (Lag-1) Impact", fontsize=11, fontweight="bold", color="#ffffff")
ax.set_xlabel("Preceding Week Rainfall (mm)", fontsize=10, color="#cbd5e1")
ax.set_ylabel("SHAP Impact on Next-Week Cases", fontsize=10, color="#cbd5e1")
ax.axhline(0, color="#ffffff", alpha=0.3, linestyle="--", linewidth=1)
ax.grid(color="#ffffff", alpha=0.08, linestyle=":")

plt.tight_layout()
plt.savefig("outputs/models_weekly/shap_rainfall_dependence.png", dpi=200, facecolor="#000000")
plt.close()

# 4. Save Global Importance to JSON
global_imp_dict = df_shap_imp.sort_values("mean_abs_shap", ascending=False).to_dict(orient="records")
with open("outputs/models_weekly/shap_global_importance.json", "w") as f:
    json.dump(global_imp_dict, f, indent=2)

# 5. Local SHAP Decomposition per District for the Latest Week
print("Computing Local SHAP decomposition for all 14 districts...")
district_explanations = {}
base_val = float(explainer.expected_value)

for dist in df["district"].unique():
    dist_row = df[(df["district"] == dist) & (df["year"] == 2025)].sort_values("week", ascending=False).head(1)
    if dist_row.empty:
        dist_row = df[df["district"] == dist].sort_values(["year", "week"], ascending=False).head(1)
    
    x_dist = dist_row[features]
    shap_dist = explainer(x_dist)
    values = shap_dist.values[0]
    
    # Pack feature contributions
    contributions = []
    for f_name, f_val, s_val in zip(features, x_dist.iloc[0].values, values):
        contributions.append({
            "feature": f_name,
            "label": FEATURE_LABELS.get(f_name, f_name),
            "actual_value": round(float(f_val), 2),
            "shap_impact_cases": round(float(s_val), 1),
            "direction": "Pushes Risk Up" if s_val > 0 else "Lowers Risk"
        })
    
    contributions.sort(key=lambda item: abs(item["shap_impact_cases"]), reverse=True)
    
    district_explanations[dist] = {
        "district": dist,
        "base_expected_cases": round(base_val, 1),
        "predicted_cases": round(float(model.predict(x_dist)[0]), 1),
        "top_drivers": contributions[:6],
        "all_contributions": contributions
    }

with open("outputs/models_weekly/shap_district_explanations.json", "w") as f:
    json.dump(district_explanations, f, indent=2)

print("\n[SHAP SUCCESS] All SHAP plots and JSON attributions generated successfully!")
print("  - outputs/models_weekly/shap_summary_beeswarm.png")
print("  - outputs/models_weekly/shap_feature_importance_bar.png")
print("  - outputs/models_weekly/shap_rainfall_dependence.png")
print("  - outputs/models_weekly/shap_district_explanations.json")
