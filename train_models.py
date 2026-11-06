"""
Comprehensive 6-Model Machine Learning Benchmark Pipeline for Kerala Waterborne Disease Surveillance.

6 Distinct Model Paradigms Evaluated:
  1. Ridge Regression              - Linear Regularized Baseline
  2. k-Nearest Neighbors (k-NN)    - Instance/Distance-based Learning
  3. Support Vector Machine (SVR)  - Kernel & Margin-based Non-linear Learning
  4. Random Forest Regressor       - Bagging Decision Tree Ensemble
  5. Gradient Boosting Machine     - Sequential Residual Boosting
  6. LightGBM Regressor            - High-Performance Leaf-wise Tree Boosting

Tasks Evaluated:
  - Task A: Spatial District Risk & Incidence Prediction (master_district_yearly)
  - Task B: Temporal Statewide Monthly Outbreak Surge Forecast (master_state_monthly)

Artifacts Exported (outputs/models/):
  - best_district_spatial_model.joblib
  - best_monthly_temporal_model.joblib
  - 6_model_comparison_metrics.csv
  - model_metadata.json
  - 01_spatial_model_comparison.png
  - 02_temporal_forecast_vs_actual.png
  - 03_feature_importance.png

Run: python train_models.py
"""
import json
import os
import sqlite3

os.environ.setdefault("MPLCONFIGDIR", os.path.abspath("outputs/.mplcache"))

import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

OUT_DIR = "outputs/models"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs("outputs/.mplcache", exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
})

# ── Load from SQLite Database ─────────────────────────────────────────────
DB_PATH = "data/kerala_health.db"
conn = sqlite3.connect(DB_PATH)
df_dist = pd.read_sql("SELECT * FROM master_district_yearly", conn)
df_mon = pd.read_sql("SELECT * FROM master_state_monthly", conn)
conn.close()

# Exclude COVID-19 pandemic anomaly years (2020-2021)
df_dist_train = df_dist[df_dist["covid_year"] == 0].copy().reset_index(drop=True)
df_mon_train = df_mon[df_mon["covid_year"] == 0].copy().reset_index(drop=True)

print("=" * 70)
print("BENCHMARKING 6 MACHINE LEARNING ALGORITHMS ON KERALA HEALTH SURVEILLANCE")
print("=" * 70)
print(f"Database Source: {DB_PATH}")
print(f"Spatial training records (2011-2025, excl. COVID): {len(df_dist_train)}")
print(f"Temporal monthly records (2011-2025, excl. COVID): {len(df_mon_train)}")

# ═══════════════════════════════════════════════════════════════════════════
# 1. SPATIAL DISTRICT RISK & INCIDENCE BENCHMARK (6 MODELS)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "-" * 60)
print("TASK 1: SPATIAL DISTRICT INCIDENCE PREDICTION (6 MODELS)")
print("-" * 60)

FEATURE_COLS_DIST = [
    "rain_annual_mm",
    "rain_sw_monsoon_mm",
    "rain_premonsoon_mm",
    "heavy_rain_weeks",
    "temp_mean_c",
    "humidity_mean_pct",
    "wq_fecal_coliform_mpn",
    "wq_bod_mg_l",
    "wq_do_min_mg_l",
    "population_density",
    "literacy_rate",
]
TARGET_DIST = "add_per_lakh"

X_dist = df_dist_train[FEATURE_COLS_DIST].copy()
y_dist = df_dist_train[TARGET_DIST].copy()

# Outbreak Risk Categories (Tertiles: Low, Moderate, High)
q_low = y_dist.quantile(0.333)
q_high = y_dist.quantile(0.666)


def categorize_risk(val):
    if val <= q_low:
        return 0  # Low Risk
    elif val <= q_high:
        return 1  # Moderate Risk
    else:
        return 2  # High Risk


y_dist_cat = y_dist.apply(categorize_risk)
RISK_LABELS = {0: "Low Risk", 1: "Moderate Risk", 2: "High Risk"}

# Define the 6 distinct models for spatial task
models_dist = {
    "1. Ridge Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=10.0)),
    ]),
    "2. k-Nearest Neighbors": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsRegressor(n_neighbors=5, weights="distance")),
    ]),
    "3. Support Vector (SVR)": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVR(C=100.0, epsilon=10.0, kernel="rbf")),
    ]),
    "4. Random Forest": RandomForestRegressor(
        n_estimators=100, max_depth=6, random_state=42
    ),
    "5. Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.08, max_depth=3, random_state=42
    ),
    "6. LightGBM": lgb.LGBMRegressor(
        n_estimators=100, learning_rate=0.08, num_leaves=15, random_state=42, verbose=-1
    ),
}

results_dist = {}
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models_dist.items():
    r2_scores, mae_scores, rmse_scores = [], [], []

    for train_idx, val_idx in kf.split(X_dist):
        X_tr, X_val = X_dist.iloc[train_idx], X_dist.iloc[val_idx]
        y_tr, y_val = y_dist.iloc[train_idx], y_dist.iloc[val_idx]

        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)

        r2_scores.append(r2_score(y_val, preds))
        mae_scores.append(mean_absolute_error(y_val, preds))
        rmse_scores.append(np.sqrt(mean_squared_error(y_val, preds)))

    results_dist[name] = {
        "R2": float(np.mean(r2_scores)),
        "MAE": float(np.mean(mae_scores)),
        "RMSE": float(np.mean(rmse_scores)),
    }
    print(f"  {name:<25}: R² = {np.mean(r2_scores):.3f} | MAE = {np.mean(mae_scores):.1f} cases/lakh | RMSE = {np.mean(rmse_scores):.1f}")

# Train the winning spatial model on the full training set
best_dist_name = max(results_dist, key=lambda k: results_dist[k]["R2"])
best_dist_model = models_dist[best_dist_name]
best_dist_model.fit(X_dist, y_dist)
y_dist_pred_all = best_dist_model.predict(X_dist)

# Train Classifier for Categorical Outbreak Risk
clf_risk = GradientBoostingClassifier(n_estimators=80, max_depth=3, random_state=42)
clf_risk.fit(X_dist, y_dist_cat)
acc_risk = accuracy_score(y_dist_cat, clf_risk.predict(X_dist))
print(f"\n  [BEST SPATIAL MODEL]: {best_dist_name} (R² = {results_dist[best_dist_name]['R2']:.3f})")
print(f"  [OUTBREAK RISK CLASSIFIER ACCURACY]: {acc_risk * 100:.1f}%")

# ═══════════════════════════════════════════════════════════════════════════
# 2. TEMPORAL STATEWIDE MONTHLY SURGE BENCHMARK (6 MODELS)
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "-" * 60)
print("TASK 2: TEMPORAL STATEWIDE MONTHLY SURGE FORECAST (6 MODELS)")
print("-" * 60)

df_mon_train["sin_month"] = np.sin(2 * np.pi * df_mon_train["month"] / 12)
df_mon_train["cos_month"] = np.cos(2 * np.pi * df_mon_train["month"] / 12)
df_mon_train["rain_prev_month_mm"] = df_mon_train["rain_prev_month_mm"].fillna(df_mon_train["rain_mm"])

FEATURE_COLS_MON = [
    "rain_mm",
    "rain_prev_month_mm",
    "temp_mean_c",
    "humidity_mean_pct",
    "sin_month",
    "cos_month",
]
TARGET_MON = "add_cases"

X_mon = df_mon_train[FEATURE_COLS_MON].copy()
y_mon = df_mon_train[TARGET_MON].copy()

models_mon = {
    "1. Ridge Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=5.0)),
    ]),
    "2. k-Nearest Neighbors": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsRegressor(n_neighbors=5, weights="distance")),
    ]),
    "3. Support Vector (SVR)": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVR(C=1000.0, epsilon=50.0, kernel="rbf")),
    ]),
    "4. Random Forest": RandomForestRegressor(
        n_estimators=100, max_depth=5, random_state=42
    ),
    "5. Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.06, max_depth=3, random_state=42
    ),
    "6. LightGBM": lgb.LGBMRegressor(
        n_estimators=100, learning_rate=0.06, num_leaves=12, random_state=42, verbose=-1
    ),
}

results_mon = {}
kf_mon = KFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models_mon.items():
    r2_m, mae_m, rmse_m = [], [], []

    for train_idx, val_idx in kf_mon.split(X_mon):
        X_tr, X_val = X_mon.iloc[train_idx], X_mon.iloc[val_idx]
        y_tr, y_val = y_mon.iloc[train_idx], y_mon.iloc[val_idx]

        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)

        r2_m.append(r2_score(y_val, preds))
        mae_m.append(mean_absolute_error(y_val, preds))
        rmse_m.append(np.sqrt(mean_squared_error(y_val, preds)))

    results_mon[name] = {
        "R2": float(np.mean(r2_m)),
        "MAE": float(np.mean(mae_m)),
        "RMSE": float(np.mean(rmse_m)),
    }
    print(f"  {name:<25}: R² = {np.mean(r2_m):.3f} | MAE = {np.mean(mae_m):.0f} cases | RMSE = {np.mean(rmse_m):.0f}")

best_mon_name = max(results_mon, key=lambda k: results_mon[k]["R2"])
best_mon_model = models_mon[best_mon_name]
best_mon_model.fit(X_mon, y_mon)
print(f"\n  [BEST TEMPORAL MODEL]: {best_mon_name} (R² = {results_mon[best_mon_name]['R2']:.3f})")

# ═══════════════════════════════════════════════════════════════════════════
# 3. EXPORT MODEL ARTIFACTS & METADATA
# ═══════════════════════════════════════════════════════════════════════════
spatial_bundle = {
    "model_name": best_dist_name,
    "regressor": best_dist_model,
    "classifier": clf_risk,
    "features": FEATURE_COLS_DIST,
    "target": TARGET_DIST,
    "metrics": results_dist[best_dist_name],
    "all_model_metrics": results_dist,
    "risk_thresholds": {"low_cutoff": float(q_low), "high_cutoff": float(q_high)},
    "risk_labels": RISK_LABELS,
}
joblib.dump(spatial_bundle, f"{OUT_DIR}/best_district_spatial_model.joblib")

temporal_bundle = {
    "model_name": best_mon_name,
    "regressor": best_mon_model,
    "features": FEATURE_COLS_MON,
    "target": TARGET_MON,
    "metrics": results_mon[best_mon_name],
    "all_model_metrics": results_mon,
}
joblib.dump(temporal_bundle, f"{OUT_DIR}/best_monthly_temporal_model.joblib")

# Save CSV of all 6 models comparison
df_metrics = pd.DataFrame([
    {
        "Model": k,
        "Spatial_R2": results_dist[k]["R2"],
        "Spatial_MAE": results_dist[k]["MAE"],
        "Spatial_RMSE": results_dist[k]["RMSE"],
        "Temporal_R2": results_mon[k]["R2"],
        "Temporal_MAE": results_mon[k]["MAE"],
        "Temporal_RMSE": results_mon[k]["RMSE"],
    }
    for k in models_dist.keys()
])
df_metrics.to_csv(f"{OUT_DIR}/6_model_comparison_metrics.csv", index=False)

metadata = {
    "models_evaluated": list(models_dist.keys()),
    "best_spatial_model": best_dist_name,
    "best_temporal_model": best_mon_name,
    "spatial_metrics": results_dist,
    "temporal_metrics": results_mon,
}
with open(f"{OUT_DIR}/model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

# ═══════════════════════════════════════════════════════════════════════════
# 4. VISUALIZATION & CHARTS FOR ALL 6 MODELS
# ═══════════════════════════════════════════════════════════════════════════
print("\nGenerating updated charts across all 6 models in outputs/models/...")

# Chart 1: 6-Model Spatial Performance Comparison Bar Chart & Scatter Fit
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))

# Scatter of Best Model Actual vs Predicted
ax1.scatter(y_dist, y_dist_pred_all, color="#02818a", alpha=0.75, edgecolors="white", s=55)
lims = [0, max(y_dist.max(), y_dist_pred_all.max()) * 1.05]
ax1.plot(lims, lims, color="#e41a1c", linestyle="--", linewidth=1.5, label="Ideal 100% Fit")
ax1.set_xlim(lims)
ax1.set_ylim(lims)
ax1.set_title(f"Best Model Fit: {best_dist_name}\n(Actual vs Predicted ADD Cases / 1 Lakh)")
ax1.set_xlabel("Actual Historical Cases / 1 Lakh")
ax1.set_ylabel("Predicted Cases / 1 Lakh")
ax1.legend(loc="upper left")
ax1.grid(True, linestyle=":", alpha=0.6)

# 6-Model Bar Chart Comparison
model_labels = [k.split(". ")[-1] for k in results_dist.keys()]
r2_vals = [results_dist[k]["R2"] for k in results_dist.keys()]
colors = ["#74a9cf" if k != best_dist_name else "#016450" for k in results_dist.keys()]
bars = ax2.barh(model_labels, r2_vals, color=colors, edgecolor="black", height=0.65)
for bar in bars:
    xval = bar.get_width()
    ax2.text(xval + 0.015, bar.get_y() + bar.get_height() / 2, f"R² = {xval:.3f}",
             ha="left", va="center", fontweight="bold", fontsize=8.5)
ax2.set_xlim(0, max(r2_vals) * 1.25)
ax2.set_title("Spatial Outbreak Risk: 6-Model Cross-Validation (R²)")
ax2.set_xlabel("Cross-Validation R² Score (Higher is Better)")
ax2.grid(True, axis="x", linestyle=":", alpha=0.6)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/01_spatial_model_comparison.png")
plt.close()

# Chart 2: Temporal 15-Year Timeline
fig, ax = plt.subplots(figsize=(13, 5))
y_mon_pred = best_mon_model.predict(X_mon)
dates_mon = pd.to_datetime(df_mon_train["year"].astype(str) + "-" + df_mon_train["month"].astype(str).str.zfill(2) + "-01")
ax.plot(dates_mon, y_mon / 1000, label="Actual Monthly Cases (Thousands)", color="#08519c", linewidth=2.0)
ax.plot(dates_mon, y_mon_pred / 1000, label=f"Predicted ({best_mon_name})", color="#e6550d", linewidth=1.8, linestyle="--")
ax.set_title(f"Statewide Monthly Surge Forecast vs Actuals (Best Temporal Model: {best_mon_name})")
ax.set_ylabel("Monthly ADD Cases (in Thousands)")
ax.set_xlabel("Year")
ax.legend(loc="upper right", frameon=True)
ax.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/02_temporal_forecast_vs_actual.png")
plt.close()

# Chart 3: Feature Importances
feat_importances = None
if hasattr(best_dist_model, "feature_importances_"):
    feat_importances = best_dist_model.feature_importances_
elif hasattr(best_dist_model, "named_steps") and hasattr(best_dist_model.named_steps["model"], "feature_importances_"):
    feat_importances = best_dist_model.named_steps["model"].feature_importances_

if feat_importances is not None:
    feat_names = [f.replace("_", " ").title() for f in FEATURE_COLS_DIST]
    indices = np.argsort(feat_importances)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(range(len(indices)), feat_importances[indices], color="#41b6c4", edgecolor="black", align="center")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feat_names[i] for i in indices])
    ax.set_xlabel("Relative Importance Weight")
    ax.set_title(f"Feature Importance in Best Model ({best_dist_name})")
    ax.grid(True, axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/03_feature_importance.png")
    plt.close()

print("\n" + "=" * 70)
print(f"6-Model Evaluation Metrics exported to: {OUT_DIR}/6_model_comparison_metrics.csv")
print(f"Models and visual charts saved successfully!")
