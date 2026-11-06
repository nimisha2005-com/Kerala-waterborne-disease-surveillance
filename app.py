"""
Flask Backend Web Application for Kerala Waterborne Disease Outbreak Surveillance.

Connects:
  - SQLite Relational Database: data/kerala_health.db
  - Spatial ML Outbreak Model: outputs/models/best_district_spatial_model.joblib
  - Temporal ML Early-Warning Model: outputs/models/best_monthly_temporal_model.joblib

Run: python app.py
Open: http://localhost:5000 in your web browser.
"""
import json
import os
import sqlite3
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_from_directory

app = Flask(__name__, template_folder="templates", static_folder="static")

DB_PATH = "data/kerala_health.db"
SPATIAL_MODEL_PATH = "outputs/models/best_district_spatial_model.joblib"
TEMPORAL_MODEL_PATH = "outputs/models/best_monthly_temporal_model.joblib"

# Load Machine Learning Model Bundles
spatial_bundle = joblib.load(SPATIAL_MODEL_PATH)
spatial_model = spatial_bundle["regressor"]
risk_classifier = spatial_bundle["classifier"]
spatial_features = spatial_bundle["features"]
risk_labels = spatial_bundle["risk_labels"]

temporal_bundle = joblib.load(TEMPORAL_MODEL_PATH)
temporal_model = temporal_bundle["regressor"]
temporal_features = temporal_bundle["features"]

# Load Weekly Outbreak Early-Warning Model (Predicts Week t+1 from Week t climate/water quality)
WEEKLY_MODEL_PATH = "outputs/models_weekly/best_weekly_outbreak_model.joblib"
weekly_bundle = joblib.load(WEEKLY_MODEL_PATH)
weekly_model = weekly_bundle["regressor"]
weekly_classifier = weekly_bundle["classifier"]
weekly_features = weekly_bundle["features"]
weekly_risk_labels = weekly_bundle["risk_labels"]
with open("data/raw/boundaries/kerala_districts.geojson", "r", encoding="utf-8") as f:
    geojson_data = json.load(f)


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/districts")
def get_districts():
    conn = get_db_connection()
    districts = [row["district"] for row in conn.execute("SELECT DISTINCT district FROM master_district_yearly ORDER BY district")]
    conn.close()
    return jsonify(districts)


@app.route("/api/geojson")
def get_geojson():
    # Annotate geojson with latest 2024 historical metrics
    conn = get_db_connection()
    df_2024 = pd.read_sql("SELECT * FROM master_district_yearly WHERE year = 2024", conn)
    conn.close()

    dist_map = df_2024.set_index("district").to_dict(orient="index")
    annotated = json.loads(json.dumps(geojson_data))

    for feat in annotated["features"]:
        dname = feat["properties"]["district"]
        if dname in dist_map:
            feat["properties"]["add_cases"] = int(dist_map[dname]["add_cases"])
            feat["properties"]["add_per_lakh"] = round(dist_map[dname]["add_per_lakh"], 1)
            feat["properties"]["rainfall"] = round(dist_map[dname]["rain_annual_mm"], 1)
            feat["properties"]["bod"] = round(dist_map[dname]["wq_bod_mg_l"], 1)
            feat["properties"]["literacy"] = round(dist_map[dname]["literacy_rate"], 1)
            # Predict risk category using current baseline
            feats = pd.DataFrame([[dist_map[dname][c] for c in spatial_features]], columns=spatial_features)
            cat_id = int(risk_classifier.predict(feats)[0])
            feat["properties"]["risk_level"] = risk_labels[cat_id]
        else:
            feat["properties"]["risk_level"] = "Unknown"

    return jsonify(annotated)


@app.route("/api/district-summary/<district>")
def district_summary(district):
    conn = get_db_connection()
    # Get latest weekly record
    row = conn.execute("""
        SELECT * FROM master_district_weekly 
        WHERE district = ? 
        ORDER BY year DESC, week DESC 
        LIMIT 1
    """, (district,)).fetchone()
    
    # Get previous 12 weeks timeline
    history_rows = conn.execute("""
        SELECT week, year, rainfall_mm_total, cases_week_current, target_next_week_cases 
        FROM master_district_weekly 
        WHERE district = ? AND year = 2025 
        ORDER BY week ASC
    """, (district,)).fetchall()
    conn.close()

    if not row:
        return jsonify({"error": "District not found"}), 404

    r = dict(row)
    history = [dict(h) for h in history_rows]

    # Calculate Water Quality Index (WQI) heuristic on standard 0-100 scale
    # Excellent (85-100), Good (70-84), Fair (50-69), Poor (<50)
    # BOD penalty: >3 mg/L reduces score
    # Fecal Coliform penalty: >500 MPN reduces score
    # DO bonus: >5 mg/L is good
    bod = float(r.get("wq_bod_mg_l", 3.0))
    fecal = float(r.get("wq_fecal_coliform_mpn", 500.0))
    do_val = float(r.get("wq_do_min_mg_l", 5.0))

    wqi = 100.0
    if bod > 3.0:
        wqi -= min(35.0, (bod - 3.0) * 8.0)
    if fecal > 500:
        wqi -= min(35.0, (fecal / 500.0) * 4.0)
    if do_val < 5.0:
        wqi -= min(25.0, (5.0 - do_val) * 10.0)
    wqi = max(15.0, min(95.0, round(wqi, 1)))

    wqi_status = "Good"
    wqi_color = "#10b981"
    if wqi < 45:
        wqi_status = "Poor / High Contamination"
        wqi_color = "#ef4444"
    elif wqi < 70:
        wqi_status = "Moderate / Caution"
        wqi_color = "#f59e0b"

    # Prepare features for Next-Week Prediction
    feature_values = pd.DataFrame([[float(r[f]) for f in weekly_features]], columns=weekly_features)
    pred_cases = max(0, int(round(float(weekly_model.predict(feature_values)[0]))))
    risk_cat = int(weekly_classifier.predict(feature_values)[0])
    risk_label = weekly_risk_labels[risk_cat]

    pop = r["population_2011"]
    rate_per_lakh = round((pred_cases / pop) * 100000.0, 1)

    # Composite public health risk classification
    if rate_per_lakh >= 22.0 or pred_cases >= 700:
        risk_label = "High Alert"
    elif rate_per_lakh >= 14.0 or pred_cases >= 400:
        risk_label = "Moderate Risk"
    else:
        risk_label = "Low Risk"

    # Rainfall trend compared to previous week
    cur_rain = round(float(r["rainfall_mm_total"]), 1)
    prev_rain = round(float(r["rain_mm_lag1"]), 1)
    rain_trend = "Rising" if cur_rain > prev_rain else ("Falling" if cur_rain < prev_rain else "Stable")

    shap_info = None
    shap_json = "outputs/models_weekly/shap_district_explanations.json"
    if os.path.exists(shap_json):
        try:
            with open(shap_json, "r") as f:
                shap_info = json.load(f).get(district)
        except Exception:
            pass

    return jsonify({
        "district": district,
        "year": int(r["year"]),
        "week": int(r["week"]),
        "population": int(pop),
        "literacy_rate": float(r["literacy_rate"]),
        "population_density": int(r["population_density"]),
        
        # Water Quality metrics
        "wqi_score": wqi,
        "wqi_status": wqi_status,
        "wqi_color": wqi_color,
        "river_bod": bod,
        "river_fecal_coliform": fecal,
        "river_dissolved_oxygen": do_val,

        # Rainfall metrics
        "current_week_rain_mm": cur_rain,
        "prev_week_rain_mm": prev_rain,
        "rain_trend": rain_trend,
        "temp_c": round(float(r["temp_c_mean"]), 1),
        "humidity_pct": round(float(r["humidity_pct_mean"]), 1),

        # Disease Outbreak Prediction
        "current_week_cases": int(r["cases_week_current"]),
        "predicted_next_week_cases": pred_cases,
        "predicted_next_week_rate_per_lakh": rate_per_lakh,
        "outbreak_risk_level": risk_label,
        
        # 12-week history
        "history": history,

        # Explainable AI (SHAP Local Attribution)
        "shap_explanation": shap_info
    })


@app.route("/api/district/<district>")
def district_profile(district):
    conn = get_db_connection()
    df = pd.read_sql("SELECT * FROM master_district_yearly WHERE district = ? ORDER BY year", conn, params=[district])
    conn.close()
    if df.empty:
        return jsonify({"error": "District not found"}), 404

    history = df[["year", "add_cases", "add_per_lakh", "rain_annual_mm", "rain_sw_monsoon_mm", "wq_bod_mg_l"]].to_dict(orient="records")
    latest = df.iloc[-1].to_dict()
    return jsonify({"district": district, "latest": latest, "history": history})


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.json
    district = data.get("district")

    conn = get_db_connection()
    row = conn.execute("SELECT * FROM master_district_yearly WHERE district = ? AND year = 2024", (district,)).fetchone()
    conn.close()

    if not row:
        return jsonify({"error": "District baseline data not found"}), 400

    row_dict = dict(row)

    # Allow simulation slider overrides from the frontend
    if "rain_annual_mm" in data:
        row_dict["rain_annual_mm"] = float(data["rain_annual_mm"])
    if "rain_sw_monsoon_mm" in data:
        row_dict["rain_sw_monsoon_mm"] = float(data["rain_sw_monsoon_mm"])
    if "wq_bod_mg_l" in data:
        row_dict["wq_bod_mg_l"] = float(data["wq_bod_mg_l"])
    if "wq_do_min_mg_l" in data:
        row_dict["wq_do_min_mg_l"] = float(data["wq_do_min_mg_l"])
    if "temp_mean_c" in data:
        row_dict["temp_mean_c"] = float(data["temp_mean_c"])

    feature_values = pd.DataFrame([[float(row_dict[f]) for f in spatial_features]], columns=spatial_features)

    # Model inference
    pred_rate = float(spatial_model.predict(feature_values)[0])
    risk_cat = int(risk_classifier.predict(feature_values)[0])
    risk_text = risk_labels[risk_cat]

    # Calculate expected raw cases based on district population
    pop = row_dict["population_2011"]
    expected_cases = int(round((pred_rate * pop) / 100000.0))

    # Log prediction into SQLite database
    conn = get_db_connection()
    conn.execute("""
        INSERT INTO user_predictions_log (district, predicted_disease, predicted_cases, risk_level, user_rainfall_input, user_temperature_input, user_water_quality_input)
        VALUES (?, 'Acute Diarrhoeal Disease', ?, ?, ?, ?, ?)
    """, (district, expected_cases, risk_text, row_dict["rain_annual_mm"], row_dict["temp_mean_c"], row_dict["wq_bod_mg_l"]))
    conn.commit()
    conn.close()

    # Prescriptive advisory recommendations based on risk level
    recommendations = []
    if risk_text == "High Risk":
        recommendations = [
            "⚠️ Issue immediate boil-water notices for drinking and well sources.",
            "🚨 Deploy mobile water testing teams to check residual chlorine (>0.5 mg/L).",
            "💊 Pre-position ORS (Oral Rehydration Salts) and IV fluids in Community Health Centres (CHCs).",
            "💧 Initiate super-chlorination of local public wells and pipe distribution networks.",
        ]
    elif risk_text == "Moderate Risk":
        recommendations = [
            "⚠️ Increase surveillance sampling frequency at river abstraction points.",
            "📢 Conduct public hygiene and sanitation campaigns in flood-prone wards.",
            "🧪 Ensure local water supply chlorination dosages meet monsoon standards.",
        ]
    else:
        recommendations = [
            "✅ Normal surveillance monitoring; water parameters within manageable limits.",
            "💧 Maintain standard routine testing schedule across water purification facilities.",
        ]

    return jsonify({
        "district": district,
        "predicted_incidence_per_lakh": round(pred_rate, 1),
        "predicted_cases": expected_cases,
        "risk_level": risk_text,
        "population": int(pop),
        "recommendations": recommendations,
        "input_features": {
            "rain_annual_mm": row_dict["rain_annual_mm"],
            "rain_sw_monsoon_mm": row_dict["rain_sw_monsoon_mm"],
            "wq_bod_mg_l": row_dict["wq_bod_mg_l"],
            "wq_do_min_mg_l": row_dict["wq_do_min_mg_l"],
            "temp_mean_c": row_dict["temp_mean_c"],
        }
    })


@app.route("/api/monthly-forecast")
def monthly_forecast():
    conn = get_db_connection()
    df_mon = pd.read_sql("SELECT * FROM master_state_monthly WHERE year >= 2022 ORDER BY year, month", conn)
    conn.close()

    df_mon["sin_month"] = np.sin(2 * np.pi * df_mon["month"] / 12)
    df_mon["cos_month"] = np.cos(2 * np.pi * df_mon["month"] / 12)
    df_mon["rain_prev_month_mm"] = df_mon["rain_prev_month_mm"].fillna(df_mon["rain_mm"])

    X = df_mon[temporal_features]
    preds = temporal_model.predict(X)

    results = []
    for idx, row in df_mon.iterrows():
        results.append({
            "year": int(row["year"]),
            "month": int(row["month"]),
            "date": f"{int(row['year'])}-{int(row['month']):02d}",
            "actual_cases": int(row["add_cases"]) if not pd.isna(row["add_cases"]) else None,
            "forecasted_cases": int(round(preds[idx])),
            "rain_mm": round(row["rain_mm"], 1),
        })

    return jsonify(results)


@app.route("/api/predict-weekly", methods=["POST"])
def predict_weekly():
    data = request.json
    district = data.get("district")

    conn = get_db_connection()
    # Get latest observed week baseline for this district
    row = conn.execute("SELECT * FROM master_district_weekly WHERE district = ? ORDER BY year DESC, week DESC LIMIT 1", (district,)).fetchone()
    conn.close()

    if not row:
        return jsonify({"error": "District weekly baseline not found"}), 400

    row_dict = dict(row)

    # Allow simulation overrides from frontend sliders
    if "rainfall_mm_total" in data:
        row_dict["rainfall_mm_total"] = float(data["rainfall_mm_total"])
    if "rain_mm_lag1" in data:
        row_dict["rain_mm_lag1"] = float(data["rain_mm_lag1"])
    if "wq_bod_mg_l" in data:
        row_dict["wq_bod_mg_l"] = float(data["wq_bod_mg_l"])
    if "temp_c_mean" in data:
        row_dict["temp_c_mean"] = float(data["temp_c_mean"])
    if "cases_week_current" in data:
        row_dict["cases_week_current"] = float(data["cases_week_current"])

    # Recompute derived features
    row_dict["rain_2week_sum"] = row_dict["rainfall_mm_total"] + row_dict["rain_mm_lag1"]
    row_dict["is_torrential_rain"] = 1 if row_dict["rainfall_mm_total"] >= 100.0 else 0

    feature_values = pd.DataFrame([[float(row_dict[f]) for f in weekly_features]], columns=weekly_features)

    # Predict Next Week (t+1)
    pred_cases = max(0, int(round(float(weekly_model.predict(feature_values)[0]))))
    risk_cat = int(weekly_classifier.predict(feature_values)[0])
    risk_label = weekly_risk_labels[risk_cat]

    pop = row_dict["population_2011"]
    rate_per_lakh = round((pred_cases / pop) * 100000.0, 1)

    # Risk classification sensitive to extreme weather and high caseloads
    if rate_per_lakh >= 22.0 or pred_cases >= 700 or row_dict["rainfall_mm_total"] >= 140.0:
        risk_label = "High Alert"
    elif rate_per_lakh >= 14.0 or pred_cases >= 400 or row_dict["rainfall_mm_total"] >= 50.0:
        risk_label = "Moderate Risk"
    else:
        risk_label = "Low Risk"

    # Log prediction into SQLite database
    conn = get_db_connection()
    conn.execute("""
        INSERT INTO user_predictions_log (district, predicted_disease, predicted_cases, risk_level, user_rainfall_input, user_temperature_input, user_water_quality_input)
        VALUES (?, 'Next-Week ADD Outbreak', ?, ?, ?, ?, ?)
    """, (district, pred_cases, risk_label, row_dict["rainfall_mm_total"], row_dict["temp_c_mean"], row_dict["wq_bod_mg_l"]))
    conn.commit()
    conn.close()

    advisories = []
    if risk_label == "High Alert":
        advisories = [
            "🚨 CRITICAL OUTBREAK ALERT FOR NEXT WEEK: Severe risk threshold exceeded.",
            "💧 Super-chlorinate drinking water wells within 48 hours (>0.5 ppm residual).",
            "💊 Deploy emergency ORS and IV fluids to Primary Health Centres (PHCs).",
            "📢 Immediate boil-water warnings via ward-level health volunteers (ASHA workers).",
        ]
    elif risk_label == "Moderate Risk":
        advisories = [
            "⚠️ ELEVATED NEXT-WEEK RISK: Precipitation/water quality conditions favor pathogen multiplication.",
            "🧪 Inspect pipeline integrity and residual chlorine levels across distribution points.",
            "📢 Advise boiling of well water and proper food hygiene in vulnerable clusters.",
        ]
    else:
        advisories = [
            "✅ NORMAL RISK FOR NEXT WEEK: Projected cases remain within manageable seasonal baseline.",
            "💧 Continue routine weekly water quality testing and surveillance.",
        ]

    return jsonify({
        "district": district,
        "predicted_next_week_cases": pred_cases,
        "predicted_next_week_per_lakh": rate_per_lakh,
        "risk_level": risk_label,
        "current_week_cases": int(row_dict["cases_week_current"]),
        "population": int(pop),
        "advisories": advisories,
        "inputs": {
            "rainfall_current_week": row_dict["rainfall_mm_total"],
            "rainfall_previous_week": row_dict["rain_mm_lag1"],
            "river_bod": row_dict["wq_bod_mg_l"],
            "temperature": row_dict["temp_c_mean"]
        }
    })


@app.route("/api/weekly-models-comparison")
def weekly_models_comparison():
    df_m = pd.read_csv("outputs/models_weekly/6_model_weekly_metrics.csv")
    return jsonify(df_m.to_dict(orient="records"))


@app.route("/api/district-shap/<district>")
def district_shap(district):
    json_path = "outputs/models_weekly/shap_district_explanations.json"
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
        if district in data:
            return jsonify(data[district])
    # Load SHAP local attribution for this district if available
    shap_info = None
    shap_json = "outputs/models_weekly/shap_district_explanations.json"
    if os.path.exists(shap_json):
        try:
            with open(shap_json, "r") as f:
                shap_info = json.load(f).get(district)
        except Exception:
            pass

    return jsonify({"error": "SHAP explanation not found"}), 404


@app.route("/api/shap-global")
def shap_global():
    json_path = "outputs/models_weekly/shap_global_importance.json"
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            return jsonify(json.load(f))
    # Load SHAP local attribution for this district if available
    shap_info = None
    shap_json = "outputs/models_weekly/shap_district_explanations.json"
    if os.path.exists(shap_json):
        try:
            with open(shap_json, "r") as f:
                shap_info = json.load(f).get(district)
        except Exception:
            pass

    return jsonify({"error": "Global SHAP metrics not found"}), 404


@app.route("/api/weekly-charts/<path:filename>")
def weekly_charts(filename):
    return send_from_directory("outputs/models_weekly", filename)


@app.route("/api/models-comparison")
def model_benchmarks():
    df_bench = pd.read_csv("outputs/models/6_model_comparison_metrics.csv")
    return jsonify(df_bench.to_dict(orient="records"))


@app.route("/api/eda-charts/<path:filename>")
def eda_charts(filename):
    return send_from_directory("outputs/eda", filename)


@app.route("/api/model-charts/<path:filename>")
def model_charts(filename):
    return send_from_directory("outputs/models", filename)


if __name__ == "__main__":
    print("Starting Kerala Waterborne Outbreak Surveillance Web Application...")
    print("Open your browser and navigate to: http://localhost:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)
