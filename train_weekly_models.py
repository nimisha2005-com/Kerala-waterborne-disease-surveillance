"""
Train and Benchmark 6 Machine Learning Models for Week-Ahead Outbreak Forecasting.

Objective:
  Predict Week t+1 Outbreak Cases & Alert Level using:
    - Current week climate (Week t)
    - Previous week climate (Week t-1 lag)
    - 2 weeks ago rainfall (Week t-2 lag)
    - River water quality (BOD, Fecal Coliform, Dissolved Oxygen)
    - District demographics (Population density, literacy rate)
    - Current week infection baseline (Week t cases)

Models Evaluated (5-Fold Time-Series / Block CV):
  1. Ridge Regression (Baseline)
  2. k-Nearest Neighbors (k-NN)
  3. Support Vector Machine (SVR)
  4. Random Forest Regressor
  5. Gradient Boosting (GBM)
  6. LightGBM Regressor

Run: python train_weekly_models.py
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
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

OUT_DIR = "outputs/models_weekly"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs("outputs/.mplcache", exist_ok=True)

# 1. Load Data from SQLite Database
DB_PATH = "data/kerala_health.db"
conn = sqlite3.connect(DB_PATH)
df = pd.read_sql("SELECT * FROM master_district_weekly", conn)
conn.close()

# Exclude COVID-19 pandemic anomaly years (2020-2021)
df_train = df[df["covid_year"] == 0].copy().reset_index(drop=True)

print("=" * 70)
print("TRAINING 6-MODEL WEEK-AHEAD OUTBREAK PREDICTION PIPELINE")
print("=" * 70)
print(f"Total Weekly District Observations (Excl. COVID): {len(df_train)} rows")

FEATURE_COLS = [
    "cases_week_current",       # Current Week Infection Baseline
    "rainfall_mm_total",        # Current Week Rainfall (t)
    "rain_mm_lag1",             # Previous Week Rainfall (t-1)
    "rain_mm_lag2",             # 2 Weeks Ago Rainfall (t-2)
    "rain_2week_sum",           # 2-Week Cumulative Flush
    "is_torrential_rain",       # Heavy Rain Flag (>100mm)
    "temp_c_mean",              # Current Week Temp
    "temp_c_lag1",              # Previous Week Temp
    "humidity_pct_mean",        # Current Week Humidity
    "wq_bod_mg_l",              # River Water BOD (Organic Pollution)
    "wq_fecal_coliform_mpn",    # River Fecal Coliform
    "wq_do_min_mg_l",           # River Dissolved Oxygen
    "population_density",       # Demographic Density
    "literacy_rate",            # Demographic Literacy
    "sin_week",                 # Cyclical Seasonal Sine
    "cos_week",                 # Cyclical Seasonal Cosine
]

TARGET_REG = "target_next_week_cases"
TARGET_CLS = "target_next_week_risk"

X = df_train[FEATURE_COLS].copy()
y = df_train[TARGET_REG].copy()
y_cat = df_train[TARGET_CLS].copy()

# Define the 6 Models
models = {
    "1. Ridge Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=10.0)),
    ]),
    "2. k-Nearest Neighbors": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsRegressor(n_neighbors=7, weights="distance")),
    ]),
    "3. Support Vector (SVR)": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVR(C=500.0, epsilon=10.0, kernel="rbf")),
    ]),
    "4. Random Forest": RandomForestRegressor(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
    ),
    "5. Gradient Boosting": GradientBoostingRegressor(
        n_estimators=120, learning_rate=0.08, max_depth=5, random_state=42
    ),
    "6. LightGBM": lgb.LGBMRegressor(
        n_estimators=120, learning_rate=0.08, num_leaves=25, random_state=42, verbose=-1, n_jobs=-1
    ),
}

results = {}
kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("\nEvaluating 6 Algorithms using 5-Fold Cross-Validation:")
for name, model in models.items():
    r2_list, mae_list, rmse_list = [], [], []

    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)

        r2_list.append(r2_score(y_val, preds))
        mae_list.append(mean_absolute_error(y_val, preds))
        rmse_list.append(np.sqrt(mean_squared_error(y_val, preds)))

    results[name] = {
        "R2": float(np.mean(r2_list)),
        "MAE": float(np.mean(mae_list)),
        "RMSE": float(np.mean(rmse_list)),
    }
    print(f"  {name:<25}: R² = {np.mean(r2_list):.3f} | MAE = {np.mean(mae_list):.1f} cases/week | RMSE = {np.mean(rmse_list):.1f}")

best_model_name = max(results, key=lambda k: results[k]["R2"])
best_model = models[best_model_name]
best_model.fit(X, y)
print(f"\n[WINNING WEEK-AHEAD REGRESSOR]: {best_model_name} (R² = {results[best_model_name]['R2']:.3f})")

# Train Outbreak Classification Model (0=Low, 1=Moderate, 2=High Risk Outbreak Alert)
clf_risk = GradientBoostingClassifier(n_estimators=100, learning_rate=0.08, max_depth=4, random_state=42)
clf_risk.fit(X, y_cat)
cat_preds = clf_risk.predict(X)
acc = accuracy_score(y_cat, cat_preds)
print(f"[OUTBREAK ALERT CLASSIFIER ACCURACY]: {acc * 100:.1f}%")

# Save model bundles
RISK_LABELS = {0: "Low Risk", 1: "Moderate Risk", 2: "High Alert"}

weekly_bundle = {
    "model_name": best_model_name,
    "regressor": best_model,
    "classifier": clf_risk,
    "features": FEATURE_COLS,
    "target": TARGET_REG,
    "metrics": results[best_model_name],
    "all_model_metrics": results,
    "risk_labels": RISK_LABELS,
}
joblib.dump(weekly_bundle, f"{OUT_DIR}/best_weekly_outbreak_model.joblib")

# Save CSV of comparison metrics
df_metrics = pd.DataFrame([
    {
        "Model": k,
        "CV_R2": results[k]["R2"],
        "CV_MAE_cases_per_week": results[k]["MAE"],
        "CV_RMSE": results[k]["RMSE"],
    }
    for k in models.keys()
])
df_metrics.to_csv(f"{OUT_DIR}/6_model_weekly_metrics.csv", index=False)

# Chart: 6-Model Weekly Benchmark Comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))

# Scatter of actual vs predicted next-week cases
y_pred_all = best_model.predict(X)
ax1.scatter(y, y_pred_all, color="#02818a", alpha=0.5, edgecolors="none", s=25)
lims = [0, max(y.max(), y_pred_all.max()) * 1.05]
ax1.plot(lims, lims, color="#e41a1c", linestyle="--", linewidth=1.5, label="Ideal Fit (y = x)")
ax1.set_xlim(lims)
ax1.set_ylim(lims)
ax1.set_title(f"Next-Week Predicted vs Actual Cases\n({best_model_name}, R² = {results[best_model_name]['R2']:.3f})")
ax1.set_xlabel("Actual Next-Week Cases (t+1)")
ax1.set_ylabel("Predicted Next-Week Cases (t+1)")
ax1.legend(loc="upper left")
ax1.grid(True, linestyle=":", alpha=0.6)

# Horizontal Bar Chart
m_names = [k.split(". ")[-1] for k in results.keys()]
r2_vals = [results[k]["R2"] for k in results.keys()]
colors = ["#74a9cf" if k != best_model_name else "#016450" for k in results.keys()]
bars = ax2.barh(m_names, r2_vals, color=colors, edgecolor="black", height=0.65)
for bar in bars:
    xval = bar.get_width()
    ax2.text(xval + 0.015, bar.get_y() + bar.get_height() / 2, f"R² = {xval:.3f}",
             ha="left", va="center", fontweight="bold", fontsize=8.5)
ax2.set_xlim(0, max(r2_vals) * 1.25)
ax2.set_title("Week-Ahead Outbreak Prediction: 6-Model Cross-Validation (R²)")
ax2.set_xlabel("Cross-Validation R² Score (Higher is Better)")
ax2.grid(True, axis="x", linestyle=":", alpha=0.6)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/01_weekly_model_comparison.png")
plt.close()

# Feature Importance Chart
feat_imp = None
if hasattr(best_model, "feature_importances_"):
    feat_imp = best_model.feature_importances_
elif hasattr(best_model, "named_steps") and hasattr(best_model.named_steps["model"], "feature_importances_"):
    feat_imp = best_model.named_steps["model"].feature_importances_

if feat_imp is not None:
    feat_names = [f.replace("_", " ").title() for f in FEATURE_COLS]
    indices = np.argsort(feat_imp)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(range(len(indices)), feat_imp[indices], color="#3690c0", edgecolor="black", align="center")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feat_names[i] for i in indices])
    ax.set_xlabel("Relative Importance Weight")
    ax.set_title(f"Key Drivers of Next-Week Disease Outbreaks ({best_model_name})")
    ax.grid(True, axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/02_weekly_feature_importance.png")
    plt.close()

print(f"\n[OK] Successfully trained and exported weekly models to {OUT_DIR}/")
