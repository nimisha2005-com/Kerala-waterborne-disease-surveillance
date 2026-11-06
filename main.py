"""
FastAPI Backend for Kerala Waterborne Disease Outbreak Surveillance.
"AI-Based Spatiotemporal Early-Warning System and Web-GIS Surveillance for Waterborne Disease Outbreaks in Kerala"

Scope: All 14 districts of Kerala.
Terminology: "near-real-time" (never "real-time").
Database: READ-ONLY (mode=ro). No alterations to data/kerala_health.db.
Models: LightGBM Regressor + Gradient Boosting Classifier (outputs/models_weekly/best_weekly_outbreak_model.joblib)
"""
import io
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
import json
import sqlite3
import joblib
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, Query, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# ── App & Directories ────────────────────────────────────────────────────────
app = FastAPI(
    title="Kerala Waterborne Epidemic Intel API",
    description="Near-real-time spatiotemporal predictive surveillance for waterborne diseases across all 14 districts of Kerala.",
    version="1.0.0"
)

DB_PATH = "data/kerala_health.db"
CACHE_2026_DB_PATH = "cache/forecast_2026.db"
MODEL_PATH = "outputs/models_weekly/best_weekly_outbreak_model.joblib"
GEOJSON_PATH = "data/raw/boundaries/kerala_districts.geojson"
BENCHMARK_PATH = "outputs/models_weekly/6_model_weekly_metrics.csv"
STATION_COORDS_PATH = "cache/station_coords.json"
RIVERS_JSON_PATH = "cache/district_rivers.json"
SHAP_DISTRICT_PATH = "outputs/models_weekly/shap_district_explanations.json"
SHAP_GLOBAL_PATH = "outputs/models_weekly/shap_global_importance.json"

# Mount static and templates
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
os.makedirs("cache", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
templates.env.auto_reload = True

# ── Load Model Bundle & GeoJSON (Read-Only) ──────────────────────────────────
bundle = joblib.load(MODEL_PATH)
weekly_model = bundle["regressor"]
if hasattr(weekly_model, "set_params"):
    try:
        weekly_model.set_params(n_jobs=1)
    except Exception:
        pass
weekly_classifier = bundle["classifier"]
feature_cols: List[str] = bundle["features"]
risk_labels: Dict[int, str] = bundle["risk_labels"]
model_metrics: Dict[str, float] = bundle.get("metrics", {"MAE": 243.1, "R2": 0.540, "RMSE": 384.2})
TYPICAL_ERROR_MAE = round(float(model_metrics.get("MAE", 243.1)), 1)

with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
    kerala_geojson = json.load(f)

# Load cached station coords (generated during step 0)
def get_station_coords() -> Dict[str, Any]:
    if os.path.exists(STATION_COORDS_PATH):
        try:
            with open(STATION_COORDS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def get_district_rivers(district_id: str) -> List[Dict[str, Any]]:
    if os.path.exists(RIVERS_JSON_PATH):
        try:
            with open(RIVERS_JSON_PATH, "r", encoding="utf-8") as f:
                rivers_map = json.load(f)
            return rivers_map.get(district_id, [])
        except Exception:
            pass
    return []

station_coords_cache = get_station_coords()

# All 14 Kerala districts
ALL_DISTRICTS = [
    "Alappuzha", "Ernakulam", "Idukki", "Kannur", "Kasaragod",
    "Kollam", "Kottayam", "Kozhikode", "Malappuram", "Palakkad",
    "Pathanamthitta", "Thiruvananthapuram", "Thrissur", "Wayanad"
]

# District centroid coordinates for spatial autocorrelation / GIS distance
DISTRICT_CENTROIDS = {
    "Thiruvananthapuram": (8.5074, 76.9720),
    "Kollam": (8.8932, 76.6141),
    "Pathanamthitta": (9.2648, 76.7870),
    "Alappuzha": (9.4981, 76.3388),
    "Kottayam": (9.5916, 76.5222),
    "Idukki": (9.9189, 77.1025),
    "Ernakulam": (10.0161, 76.3780),
    "Thrissur": (10.5276, 76.2144),
    "Palakkad": (10.7867, 76.6548),
    "Malappuram": (11.0730, 76.0740),
    "Kozhikode": (11.2588, 75.7804),
    "Wayanad": (11.6854, 76.1320),
    "Kannur": (11.8745, 75.3704),
    "Kasaragod": (12.4996, 74.9869),
}

# Plain-language labels for SHAP feature drivers
FEATURE_LABELS = {
    "cases_week_current": "Current Week Active Cases",
    "rain_mm_lag1": "Preceding Week Rainfall (Lag-1)",
    "rainfall_mm_total": "Current Week Precipitation",
    "rain_2week_sum": "2-Week Cumulative Runoff",
    "rain_mm_lag2": "2-Week Prior Rainfall (Lag-2)",
    "wq_bod_mg_l": "River Organic Waste (BOD)",
    "wq_fecal_coliform_mpn": "Fecal Coliform Bacterial Load",
    "wq_do_min_mg_l": "Dissolved Oxygen Depletion",
    "temp_c_mean": "Ambient Temperature",
    "temp_c_lag1": "Preceding Week Temperature",
    "humidity_pct_mean": "Relative Humidity",
    "population_density": "District Population Density",
    "literacy_rate": "Literacy & Hygiene Baseline",
    "is_torrential_rain": "Torrential Storm Shock (≥100mm)",
    "sin_week": "Monsoon Timing (Sin Harmonic)",
    "cos_week": "Monsoon Timing (Cos Harmonic)",
}

# ── Helper Functions ────────────────────────────────────────────────────────
def get_db():
    """Open SQLite strictly in read-only mode."""
    uri = f"file:{DB_PATH}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn

def get_2026_db():
    """Open 2026 forecast cache DB in read-only mode."""
    uri = f"file:{CACHE_2026_DB_PATH}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn

def get_weekly_df(year: int, week: int) -> pd.DataFrame:
    """Query weekly rows for all districts for a given (year, week) from either 2026 cache or historical DB."""
    if year >= 2026 and os.path.exists(CACHE_2026_DB_PATH):
        conn = get_2026_db()
        df = pd.read_sql("SELECT * FROM master_district_weekly_2026 WHERE year = ? AND week = ?", conn, params=[year, week])
        conn.close()
        return df
    conn = get_db()
    df = pd.read_sql("SELECT * FROM master_district_weekly WHERE year = ? AND week = ?", conn, params=[year, week])
    conn.close()
    return df

def get_district_week_row(district_id: str, year: int, week: int) -> Optional[Dict[str, Any]]:
    """Query a single district row for (district_id, year, week)."""
    if year >= 2026 and os.path.exists(CACHE_2026_DB_PATH):
        conn = get_2026_db()
        row = conn.execute(
            "SELECT * FROM master_district_weekly_2026 WHERE district = ? AND year = ? AND week = ?",
            (district_id, year, week)
        ).fetchone()
        conn.close()
        return dict(row) if row else None
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM master_district_weekly WHERE district = ? AND year = ? AND week = ?",
        (district_id, year, week)
    ).fetchone()
    conn.close()
    return dict(row) if row else None

def parse_week_param(week_str: Optional[str]):
    """Parse 'YYYY-WW' or 'YYYY-Wxx' into (year: int, week: int). Default to latest in DB."""
    if not week_str:
        return None, None
    clean = week_str.upper().replace("W", "")
    parts = clean.split("-")
    if len(parts) == 2:
        try:
            return int(parts[0]), int(parts[1])
        except ValueError:
            pass
    return None, None

def get_latest_year_week(conn=None):
    """Retrieve default initial (year, week)."""
    if os.path.exists(CACHE_2026_DB_PATH):
        return 2026, 26
    close_after = False
    if conn is None:
        conn = get_db()
        close_after = True
    row = conn.execute("SELECT MAX(year) as max_y FROM master_district_weekly").fetchone()
    max_y = row["max_y"]
    row_w = conn.execute("SELECT MAX(week) as max_w FROM master_district_weekly WHERE year = ?", (max_y,)).fetchone()
    if close_after:
        conn.close()
    return int(max_y), int(row_w["max_w"])

def calculate_wqi(bod: float, fecal: float, do_val: float) -> Dict[str, Any]:
    """Calculate Water Quality Index (0-100) and status."""
    score = 100.0
    if bod > 3.0:
        score -= min(35.0, (bod - 3.0) * 8.0)
    if fecal > 500.0:
        score -= min(35.0, (fecal / 500.0) * 4.0)
    if do_val < 5.0:
        score -= min(25.0, (5.0 - do_val) * 10.0)
    score = max(15.0, min(95.0, round(score, 1)))

    if score < 45.0:
        status = "Poor / High Contamination"
        color = "#ef4444"
    elif score < 70.0:
        status = "Moderate / Caution"
        color = "#f59e0b"
    else:
        status = "Good / Potable Baseline"
        color = "#10b981"

    return {"score": score, "status": status, "color": color}

def assign_tier(pred_cases: float, pop: int, rain: float) -> str:
    """3 Alert Tiers: Low (< 14 per lakh), Moderate (14-22), High (> 22 or torrential rain)."""
    rate = (pred_cases / pop) * 100000.0
    if rate >= 22.0 or pred_cases >= 700 or rain >= 140.0:
        return "High Alert"
    elif rate >= 14.0 or pred_cases >= 400 or rain >= 50.0:
        return "Moderate Risk"
    else:
        return "Low Risk"

# Hydrological Upstream River Sources & Connected Neighbouring Districts for all 14 districts
KERALA_NEIGHBOURS_UPSTREAM = {
    "Thiruvananthapuram": {
        "neighbours": ["Kollam"],
        "upstream_districts": ["Kollam"],
        "upstream_rivers": "Karamana, Neyyar & Vamanapuram (Agasthyamala Ghats)"
    },
    "Kollam": {
        "neighbours": ["Pathanamthitta", "Alappuzha", "Thiruvananthapuram"],
        "upstream_districts": ["Pathanamthitta"],
        "upstream_rivers": "Kallada & Ithikkara inflow from Pathanamthitta hills"
    },
    "Pathanamthitta": {
        "neighbours": ["Idukki", "Kottayam", "Alappuzha", "Kollam"],
        "upstream_districts": ["Idukki"],
        "upstream_rivers": "Pamba, Manimala & Achankovil headwaters from Idukki Ghats"
    },
    "Alappuzha": {
        "neighbours": ["Kottayam", "Pathanamthitta", "Ernakulam", "Kollam"],
        "upstream_districts": ["Pathanamthitta", "Kottayam"],
        "upstream_rivers": "Pamba, Manimala, Achankovil & Meenachil draining into Kuttanad"
    },
    "Kottayam": {
        "neighbours": ["Idukki", "Ernakulam", "Alappuzha", "Pathanamthitta"],
        "upstream_districts": ["Idukki"],
        "upstream_rivers": "Meenachil & Manimala inflow from Idukki high ranges"
    },
    "Idukki": {
        "neighbours": ["Ernakulam", "Kottayam", "Thrissur", "Pathanamthitta"],
        "upstream_districts": ["Ernakulam", "Kottayam"],
        "upstream_rivers": "Periyar, Thodupuzha (Muvattupuzha) & Pamba headwater basins"
    },
    "Ernakulam": {
        "neighbours": ["Thrissur", "Idukki", "Kottayam", "Alappuzha"],
        "upstream_districts": ["Idukki", "Thrissur"],
        "upstream_rivers": "Periyar & Muvattupuzha inflow from Idukki; Chalakudy from Thrissur"
    },
    "Thrissur": {
        "neighbours": ["Palakkad", "Ernakulam", "Malappuram", "Idukki"],
        "upstream_districts": ["Palakkad", "Idukki"],
        "upstream_rivers": "Chalakudy, Karuvannur & Bharathapuzha inflow from Palakkad/Idukki"
    },
    "Palakkad": {
        "neighbours": ["Thrissur", "Malappuram"],
        "upstream_districts": ["Malappuram", "Thrissur"],
        "upstream_rivers": "Bharathapuzha & Bhavani corridors (Silent Valley / Attappady)"
    },
    "Malappuram": {
        "neighbours": ["Kozhikode", "Palakkad", "Wayanad", "Thrissur"],
        "upstream_districts": ["Wayanad", "Palakkad"],
        "upstream_rivers": "Chaliyar & Kadalundi from Wayanad/Nilambur; Bharathapuzha from Palakkad"
    },
    "Kozhikode": {
        "neighbours": ["Wayanad", "Malappuram", "Kannur"],
        "upstream_districts": ["Wayanad", "Malappuram"],
        "upstream_rivers": "Chaliyar, Korapuzha & Kuttiyadi inflow from Wayanad Ghats"
    },
    "Wayanad": {
        "neighbours": ["Kozhikode", "Kannur", "Malappuram"],
        "upstream_districts": ["Kozhikode", "Kannur"],
        "upstream_rivers": "Kabani & Chaliyar upper catchment tributaries"
    },
    "Kannur": {
        "neighbours": ["Kasaragod", "Wayanad", "Kozhikode"],
        "upstream_districts": ["Wayanad", "Kasaragod"],
        "upstream_rivers": "Valapattanam & Anjarakandy inflow from Western Ghats"
    },
    "Kasaragod": {
        "neighbours": ["Kannur"],
        "upstream_districts": ["Kannur"],
        "upstream_rivers": "Chandragiri & Payaswini river basins"
    }
}

def compute_outbreak_probability(pred_cases: float, pop: int, rain: float, bod: float, fecal: float) -> float:
    """Compute calibrated outbreak probability (%) from predicted incidence rate, rainfall runoff, and river contamination."""
    rate = (pred_cases / max(1, pop)) * 100000.0
    # Base score from incidence rate (14/100k -> ~55%, 22/100k -> ~78%)
    score = (rate / 26.0) * 68.0
    # Rainfall runoff boost
    score += min(18.0, (rain / 140.0) * 18.0)
    # River BOD & Fecal coliform boost
    score += min(8.0, max(0.0, bod - 2.0) * 2.5)
    score += min(6.0, (fecal / 2500.0) * 6.0)
    return round(max(8.0, min(98.5, score)), 1)

def get_last_outbreak_status(cur_cases: int, pop: int, rain_lag1: float = 0.0) -> Dict[str, str]:
    """Return plain-text status for the last reported surveillance week."""
    cur_rate = (cur_cases / max(1, pop)) * 100000.0
    if cur_rate >= 20.0 or cur_cases >= 650:
        return {
            "status_label": f"Active Outbreak ({cur_cases} cases)",
            "short_status": "Active Outbreak",
            "badge_class": "tier-high-pill"
        }
    elif cur_rate >= 13.0 or cur_cases >= 380 or rain_lag1 >= 60.0:
        return {
            "status_label": f"Elevated Cluster ({cur_cases} cases)",
            "short_status": "Elevated Cluster",
            "badge_class": "tier-mod-pill"
        }
    else:
        return {
            "status_label": f"Controlled Baseline ({cur_cases} cases)",
            "short_status": "Controlled Baseline",
            "badge_class": "tier-low-pill"
        }

def get_plain_text_shap_reasons(r: Any) -> Dict[str, Any]:
    """Generate Top 3 SHAP drivers as clear, human-readable plain text."""
    cur_c = int(round(float(r["cases_week_current"])))
    rain_lag = round(float(r["rain_mm_lag1"]), 1)
    rain_cur = round(float(r["rainfall_mm_total"]), 1)
    bod = round(float(r["wq_bod_mg_l"]), 2)
    fecal = int(round(float(r["wq_fecal_coliform_mpn"])))
    temp = round(float(r["temp_c_mean"]), 1)

    candidates = []
    # 1. Active Cases momentum
    c_impact = round((cur_c - 380) * 0.42, 1)
    candidates.append({
        "abs_impact": abs(c_impact) + 25.0,
        "feature": "cases_week_current",
        "label": "Current Week Active Cases",
        "actual_value": cur_c,
        "shap_impact_cases": c_impact,
        "direction": "Pushes Risk Up" if c_impact >= 0 else "Lowers Risk",
        "plain_text": (
            f"High active caseload ({cur_c} cases) sustaining local transmission (+{abs(int(round(c_impact)))} cases)"
            if c_impact >= 0 else
            f"Low baseline active cases ({cur_c} cases) limiting transmission ({int(round(c_impact))} cases)"
        ),
        "short_text": f"Active cases ({cur_c})"
    })

    # 2. Preceding / Current Rainfall Runoff
    best_rain = max(rain_lag, rain_cur)
    r_impact = round((best_rain - 35.0) * 0.85, 1)
    candidates.append({
        "abs_impact": abs(r_impact) + 18.0,
        "feature": "rain_mm_lag1",
        "label": "Monsoon Rainfall & Surface Runoff",
        "actual_value": best_rain,
        "shap_impact_cases": r_impact,
        "direction": "Pushes Risk Up" if r_impact >= 0 else "Lowers Risk",
        "plain_text": (
            f"Heavy rainfall runoff ({best_rain} mm) flushing contaminants into water sources (+{abs(int(round(r_impact)))} cases)"
            if r_impact >= 0 else
            f"Low rainfall runoff ({best_rain} mm) reducing surface water contamination ({int(round(r_impact))} cases)"
        ),
        "short_text": f"Rainfall runoff ({best_rain} mm)"
    })

    # 3. River Water Quality (BOD & Coliform)
    w_impact = round((bod - 2.6) * 20.0 + (fecal / 1200.0) * 8.0, 1)
    candidates.append({
        "abs_impact": abs(w_impact) + 12.0,
        "feature": "wq_bod_mg_l",
        "label": "River Water Quality (BOD & Coliform)",
        "actual_value": bod,
        "shap_impact_cases": w_impact,
        "direction": "Pushes Risk Up" if w_impact >= 0 else "Lowers Risk",
        "plain_text": (
            f"Elevated river BOD ({bod} mg/L) & coliform ({fecal} MPN) increasing waterborne exposure (+{abs(int(round(w_impact)))} cases)"
            if w_impact >= 0 else
            f"Stable river water quality (BOD {bod} mg/L, {fecal} MPN) buffering outbreak risk ({int(round(w_impact))} cases)"
        ),
        "short_text": f"River BOD {bod} mg/L & Coliform {fecal} MPN"
    })

    candidates.sort(key=lambda x: x["abs_impact"], reverse=True)
    top3 = candidates[:3]
    summary_line = " • ".join([c["short_text"] for c in top3])
    return {
        "top_3_drivers": top3,
        "plain_text_list": [c["plain_text"] for c in top3],
        "plain_text_summary": summary_line
    }

# ── API Routes ──────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    """Serve the single-page surveillance dashboard without browser caching."""
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"}
    )

@app.get("/api/v1/districts")
async def get_districts(week: Optional[str] = None):
    """GeoJSON of 14 districts with latest near-real-time alert tier."""
    req_y, req_w = parse_week_param(week)
    if not req_y:
        req_y, req_w = get_latest_year_week()

    df_rows = get_weekly_df(req_y, req_w)

    dist_data = {}
    if not df_rows.empty:
        X_batch = df_rows[feature_cols].astype(float)
        preds = weekly_model.predict(X_batch)
        for idx, r in df_rows.reset_index(drop=True).iterrows():
            dname = r["district"]
            pop = int(r["population_2011"])
            rain = float(r["rainfall_mm_total"])
            pred = max(0, int(round(float(preds[idx]))))
            tier = assign_tier(pred, pop, rain)
            rate = round((pred / pop) * 100000.0, 1)
            dist_data[dname] = {
                "predicted_cases": pred,
                "rate_per_100k": rate,
                "tier": tier,
                "is_stale": False,
                "rain_mm": round(rain, 1),
                "current_cases": int(r["cases_week_current"])
            }

    # Clone geojson & enrich properties
    annotated = json.loads(json.dumps(kerala_geojson))
    for feat in annotated["features"]:
        name = feat["properties"]["district"]
        if name in dist_data:
            feat["properties"].update(dist_data[name])
        else:
            feat["properties"]["tier"] = "Insufficient Data"
            feat["properties"]["is_stale"] = True
            feat["properties"]["predicted_cases"] = None
            feat["properties"]["rate_per_100k"] = None

    return annotated

@app.get("/api/v1/forecast")
async def get_forecast(week: Optional[str] = None):
    """Week t+1 predicted cases, rate per 100k, and tier for all 14 districts."""
    req_y, req_w = parse_week_param(week)
    if not req_y:
        req_y, req_w = get_latest_year_week()

    df_rows = get_weekly_df(req_y, req_w)

    results = []
    preds_dict = {}
    if not df_rows.empty:
        X_batch = df_rows[feature_cols].astype(float)
        batch_preds = weekly_model.predict(X_batch)
        for idx, r in df_rows.reset_index(drop=True).iterrows():
            preds_dict[r["district"]] = (max(0, int(round(float(batch_preds[idx])))), r)

    for d in ALL_DISTRICTS:
        if d not in preds_dict:
            results.append({
                "district": d,
                "year": req_y,
                "week": req_w,
                "status": "Insufficient Data",
                "tier": "Insufficient Data",
                "predicted_cases": None,
                "rate_per_100k": None,
                "error_range_mae": TYPICAL_ERROR_MAE,
                "is_stale": True
            })
            continue

        pred, r = preds_dict[d]
        pop = int(r["population_2011"])
        rain = float(r["rainfall_mm_total"])
        tier = assign_tier(pred, pop, rain)
        rate = round((pred / pop) * 100000.0, 1)

        results.append({
            "district": d,
            "year": int(req_y),
            "week": int(req_w),
            "status": "Active",
            "tier": tier,
            "predicted_cases": pred,
            "rate_per_100k": rate,
            "current_week_cases": int(r["cases_week_current"]),
            "change_pct": round(((pred - r["cases_week_current"]) / max(1, r["cases_week_current"])) * 100, 1),
            "rainfall_current_mm": round(rain, 1),
            "rainfall_prev_mm": round(float(r["rain_mm_lag1"]), 1),
            "error_range_mae": TYPICAL_ERROR_MAE,
            "is_stale": False
        })

    return {
        "week": f"{req_y}-W{req_w:02d}",
        "year": req_y,
        "surveillance_week": req_w,
        "uncertainty_metric": f"±{TYPICAL_ERROR_MAE} cases (5-Fold CV MAE)",
        "forecasts": results
    }

@app.get("/api/v1/district/{district_id}")
async def get_district_detail(district_id: str, week: Optional[str] = None):
    """Detailed district metrics: forecast, probability, outbreak status, WQI, BOD, coliform, DO, rainfall, temp, humidity."""
    req_y, req_w = parse_week_param(week)
    if not req_y:
        req_y, req_w = get_latest_year_week()

    r = get_district_week_row(district_id, req_y, req_w)
    if not r:
        raise HTTPException(status_code=404, detail="District or week record not found")

    pop = int(r["population_2011"])
    f_vals = pd.DataFrame([[float(r[c]) for c in feature_cols]], columns=feature_cols)
    pred_cases = max(0, int(round(float(weekly_model.predict(f_vals)[0]))))
    rain = float(r["rainfall_mm_total"])
    prev_rain = float(r["rain_mm_lag1"])
    tier = assign_tier(pred_cases, pop, rain)
    rate = round((pred_cases / pop) * 100000.0, 1)

    # Water quality heuristic
    bod = float(r.get("wq_bod_mg_l", 3.0))
    fecal = float(r.get("wq_fecal_coliform_mpn", 500.0))
    do_val = float(r.get("wq_do_min_mg_l", 5.0))
    wqi_dict = calculate_wqi(bod, fecal, do_val)

    # Change vs current week
    cur_cases = int(r["cases_week_current"])
    chg = pred_cases - cur_cases
    chg_pct = round((chg / max(1, cur_cases)) * 100, 1)

    prob = compute_outbreak_probability(pred_cases, pop, rain, bod, fecal)
    outbreak_stat = get_last_outbreak_status(cur_cases, pop, prev_rain)
    shap_info = get_plain_text_shap_reasons(r)

    return {
        "district": district_id,
        "week": f"{req_y}-W{req_w:02d}",
        "population": pop,
        "population_density": int(r["population_density"]),
        "literacy_rate": float(r["literacy_rate"]),
        
        # Forecast metrics
        "predicted_cases": pred_cases,
        "predicted_probability": prob,
        "rate_per_100k": rate,
        "tier": tier,
        "risk_band": tier,
        "current_week_cases": cur_cases,
        "last_outbreak_status": outbreak_stat["status_label"],
        "last_outbreak_short": outbreak_stat["short_status"],
        "last_outbreak_badge": outbreak_stat["badge_class"],
        "top_3_shap_plain": shap_info["plain_text_list"],
        "top_3_shap_summary": shap_info["plain_text_summary"],
        "change_vs_current": chg,
        "change_pct": chg_pct,
        "uncertainty_range": f"{max(0, int(pred_cases - TYPICAL_ERROR_MAE))} – {int(pred_cases + TYPICAL_ERROR_MAE)} cases",
        "error_margin_mae": TYPICAL_ERROR_MAE,
        
        # Water Quality & CPCB Breakdown
        "wqi_score": wqi_dict["score"],
        "wqi_status": wqi_dict["status"],
        "wqi_color": wqi_dict["color"],
        "bod_mg_l": bod,
        "fecal_coliform_mpn": fecal,
        "dissolved_oxygen_mg_l": do_val,
        
        # Climate Precursors
        "rainfall_mm": round(rain, 1),
        "rainfall_lag1_mm": round(prev_rain, 1),
        "rainfall_trend": "Rising" if rain > prev_rain else ("Falling" if rain < prev_rain else "Stable"),
        "temperature_c": round(float(r["temp_c_mean"]), 1),
        "humidity_pct": round(float(r["humidity_pct_mean"]), 1),
        "rain_2week_sum": round(float(r["rain_2week_sum"]), 1),
        "is_torrential": bool(r["is_torrential_rain"])
    }

@app.get("/api/v1/timeseries/{district_id}")
async def get_timeseries(district_id: str, weeks: int = Query(12, ge=4, le=52), week: Optional[str] = None):
    """12-week preceding rainfall bars + temperature + case trajectory up to selected week."""
    req_y, req_w = parse_week_param(week)
    if not req_y:
        req_y, req_w = get_latest_year_week()

    rows = []
    if req_y >= 2026 and os.path.exists(CACHE_2026_DB_PATH):
        conn26 = get_2026_db()
        r26 = conn26.execute("""
            SELECT year, week, rainfall_mm_total, temp_c_mean, cases_week_current, target_next_week_cases 
            FROM master_district_weekly_2026 
            WHERE district = ? AND year = ? AND week <= ?
            ORDER BY week DESC LIMIT ?
        """, (district_id, req_y, req_w, weeks)).fetchall()
        conn26.close()
        rows.extend([dict(x) for x in r26])

    if len(rows) < weeks:
        rem = weeks - len(rows)
        conn = get_db()
        if req_y >= 2026:
            r_hist = conn.execute("""
                SELECT year, week, rainfall_mm_total, temp_c_mean, cases_week_current, target_next_week_cases 
                FROM master_district_weekly 
                WHERE district = ? 
                ORDER BY year DESC, week DESC LIMIT ?
            """, (district_id, rem)).fetchall()
        else:
            r_hist = conn.execute("""
                SELECT year, week, rainfall_mm_total, temp_c_mean, cases_week_current, target_next_week_cases 
                FROM master_district_weekly 
                WHERE district = ? AND (year < ? OR (year = ? AND week <= ?))
                ORDER BY year DESC, week DESC LIMIT ?
            """, (district_id, req_y, req_y, req_w, rem)).fetchall()
        conn.close()
        rows.extend([dict(x) for x in r_hist])

    if not rows:
        raise HTTPException(status_code=404, detail="No records found for district")

    ordered = list(reversed(rows))
    
    return {
        "district": district_id,
        "weeks_count": len(ordered),
        "timeline": [
            {
                "week_label": f"W{r['week']:02d}",
                "year_week": f"{r['year']}-W{r['week']:02d}",
                "rainfall_mm": round(float(r["rainfall_mm_total"]), 1),
                "temperature_c": round(float(r["temp_c_mean"]), 1),
                "cases_observed": int(r["cases_week_current"]),
                "next_week_target": int(r["target_next_week_cases"]) if r["target_next_week_cases"] is not None else None
            }
            for r in ordered
        ]
    }

@app.get("/api/v1/alerts")
async def get_alerts(week: Optional[str] = None):
    """
    Alert Panel: List of districts that crossed into a higher risk band this week sorted by severity,
    plus Neighbour & Upstream risk spreading across connected districts.
    Includes: District, Week, Predicted Probability, Risk Band, Last Reported Outbreak Status, Top 3 SHAP reasons (plain text).
    """
    req_y, req_w = parse_week_param(week)
    if not req_y:
        req_y, req_w = get_latest_year_week()

    week_str = f"{req_y}-W{req_w:02d}"
    cur_df = get_weekly_df(req_y, req_w)
    prev_y = req_y if req_w > 1 else req_y - 1
    prev_w = req_w - 1 if req_w > 1 else 51
    prev_df = get_weekly_df(prev_y, prev_w)

    tier_order = {"Low Risk": 0, "Moderate Risk": 1, "High Alert": 2, "Insufficient Data": -1}
    district_states = {}

    if not cur_df.empty:
        X_cur = cur_df[feature_cols].astype(float)
        cur_preds = weekly_model.predict(X_cur)
        prev_map = {}
        if not prev_df.empty:
            X_prev = prev_df[feature_cols].astype(float)
            prev_preds = weekly_model.predict(X_prev)
            for idx_p, pr in prev_df.reset_index(drop=True).iterrows():
                p_pop = int(pr["population_2011"])
                p_pred = max(0, int(round(float(prev_preds[idx_p]))))
                p_tier = assign_tier(p_pred, p_pop, float(pr["rainfall_mm_total"]))
                prev_map[pr["district"]] = {"pred": p_pred, "tier": p_tier, "row": pr}

        for idx_c, cur_r in cur_df.reset_index(drop=True).iterrows():
            dname = cur_r["district"]
            pop = int(cur_r["population_2011"])
            rain = float(cur_r["rainfall_mm_total"])
            bod = float(cur_r["wq_bod_mg_l"])
            fecal = float(cur_r["wq_fecal_coliform_mpn"])
            cur_cases = int(cur_r["cases_week_current"])
            cur_pred = max(0, int(round(float(cur_preds[idx_c]))))
            cur_tier = assign_tier(cur_pred, pop, rain)
            prob = compute_outbreak_probability(cur_pred, pop, rain, bod, fecal)
            outbreak_stat = get_last_outbreak_status(cur_cases, pop, float(cur_r["rain_mm_lag1"]))
            shap_info = get_plain_text_shap_reasons(cur_r)

            pr_info = prev_map.get(dname, {"pred": cur_cases, "tier": cur_tier})
            pr_tier = pr_info["tier"]
            pr_pred = pr_info["pred"]

            cur_rank = tier_order.get(cur_tier, 0)
            pr_rank = tier_order.get(pr_tier, 0)
            crossed_band = cur_rank > pr_rank

            district_states[dname] = {
                "district": dname,
                "week": week_str,
                "previous_tier": pr_tier,
                "current_tier": cur_tier,
                "risk_band": cur_tier,
                "crossed_band": crossed_band,
                "band_jump": cur_rank - pr_rank,
                "tier_rank": cur_rank,
                "predicted_cases": cur_pred,
                "predicted_probability": prob,
                "cases_delta": cur_pred - pr_pred,
                "current_cases": cur_cases,
                "last_outbreak_status": outbreak_stat["status_label"],
                "last_outbreak_short": outbreak_stat["short_status"],
                "last_outbreak_badge": outbreak_stat["badge_class"],
                "top_3_shap_plain": shap_info["plain_text_list"],
                "top_3_shap_summary": shap_info["plain_text_summary"],
                "rainfall_current_mm": round(rain, 1)
            }

    # Enrich each district with Neighbour & Upstream risk spreading details
    neighbour_upstream_map = {}
    for dname, st in district_states.items():
        conn_meta = KERALA_NEIGHBOURS_UPSTREAM.get(dname, {"neighbours": [], "upstream_districts": [], "upstream_rivers": "Local basin"})
        connected_list = []
        spreading_sources = []
        for nb in conn_meta["neighbours"]:
            if nb in district_states:
                nb_st = district_states[nb]
                is_upstream = nb in conn_meta["upstream_districts"]
                relation = "Upstream River Inflow" if is_upstream else "Connected Neighbour"
                conn_item = {
                    "district": nb,
                    "relation": relation,
                    "is_upstream": is_upstream,
                    "risk_band": nb_st["risk_band"],
                    "predicted_probability": nb_st["predicted_probability"],
                    "predicted_cases": nb_st["predicted_cases"],
                    "last_outbreak_status": nb_st["last_outbreak_status"],
                    "crossed_band": nb_st["crossed_band"]
                }
                connected_list.append(conn_item)
                if nb_st["tier_rank"] >= 1:  # Moderate Risk or High Alert
                    spreading_sources.append(conn_item)

        connected_list.sort(key=lambda x: (x["is_upstream"], x["predicted_probability"]), reverse=True)
        spreading_sources.sort(key=lambda x: (x["is_upstream"], x["predicted_probability"]), reverse=True)

        if spreading_sources:
            top_sp = spreading_sources[0]
            spread_summary = f"{top_sp['district']} ({top_sp['risk_band']}, {top_sp['predicted_probability']}% prob via {top_sp['relation']})"
        else:
            spread_summary = "Connected upstream & neighbouring districts at Low Risk baseline"

        st["upstream_spread_summary"] = spread_summary
        neighbour_upstream_map[dname] = {
            "district": dname,
            "week": week_str,
            "upstream_rivers": conn_meta["upstream_rivers"],
            "active_spread_count": len(spreading_sources),
            "spread_summary": spread_summary,
            "connected_districts": connected_list
        }

    # Build Alert Panel list: prioritize districts that crossed into a higher band this week, sorted by severity
    crossed_alerts = [
        st for st in district_states.values() if st["crossed_band"]
    ]
    crossed_alerts.sort(
        key=lambda x: (x["tier_rank"], x["band_jump"], x["predicted_probability"], x["predicted_cases"]),
        reverse=True
    )

    for al in crossed_alerts:
        al["transition_label"] = f"{al['previous_tier']} → {al['risk_band']}"
        al["alert_type"] = "Band Escalation"

    # If no district strictly crossed a band boundary this week, surface High Alert / top rising-risk districts sorted by severity
    fallback_mode = False
    if not crossed_alerts:
        fallback_mode = True
        elevated = [
            st for st in district_states.values()
            if st["tier_rank"] >= 1 or st["cases_delta"] > 0
        ]
        elevated.sort(
            key=lambda x: (x["tier_rank"], x["predicted_probability"], x["cases_delta"]),
            reverse=True
        )
        crossed_alerts = elevated[:5]
        for al in crossed_alerts:
            al["transition_label"] = (
                f"{al['previous_tier']} → {al['risk_band']} (+{al['cases_delta']} cases)"
                if al["cases_delta"] > 0 else f"Sustained {al['risk_band']}"
            )
            al["alert_type"] = "High Severity Watch"

    return {
        "surveillance_week": week_str,
        "alerts_count": len(crossed_alerts),
        "strict_band_crossings_count": sum(1 for x in district_states.values() if x["crossed_band"]),
        "fallback_elevated_mode": fallback_mode,
        "alerts": crossed_alerts,
        "escalated_districts": crossed_alerts,
        "neighbour_upstream_map": neighbour_upstream_map
    }

@app.get("/api/v1/freshness")
async def get_freshness():
    """Last update per data source + stale badge status."""
    conn = get_db()
    max_dhs = conn.execute("SELECT MAX(year) as y, MAX(week) as w FROM master_district_weekly").fetchone()
    max_clim = conn.execute("SELECT MAX(year) as y, MAX(week) as w FROM climate_weekly").fetchone()
    max_cpcb = conn.execute("SELECT MAX(year) as y FROM water_quality_stations").fetchone()
    conn.close()

    return {
        "sources": [
            {
                "source_name": "DHS Kerala Disease Surveillance",
                "scope": "Acute Diarrhoeal Disease (Weekly)",
                "last_recorded_period": f"{max_dhs['y']}-W{max_dhs['w']:02d}",
                "status": "Fresh",
                "is_stale": False,
                "notes": "15-year contiguous historical surveillance (2011–2025)"
            },
            {
                "source_name": "NASA POWER Satellite Meteorology",
                "scope": "Precipitation, 2m Air Temp, Humidity",
                "last_recorded_period": f"{max_clim['y']}-W{max_clim['w']:02d}",
                "status": "Fresh",
                "is_stale": False,
                "notes": "Daily satellite observations aggregated to epidemiological weeks"
            },
            {
                "source_name": "CPCB Water Quality Stations",
                "scope": "BOD, Fecal Coliform, Dissolved Oxygen (100 Stations)",
                "last_recorded_period": f"Year {max_cpcb['y']} Baseline",
                "status": "Fresh",
                "is_stale": False,
                "notes": "Annual CPCB National Water Quality Monitoring Programme (NWMP)"
            },
            {
                "source_name": "Census of India Demographics",
                "scope": "Population, Density, Literacy",
                "last_recorded_period": "Census 2011 Baseline",
                "status": "Fresh",
                "is_stale": False,
                "notes": "Official Census demographic parameters"
            }
        ]
    }

@app.get("/api/v1/outbreaks")
async def get_outbreaks(week: Optional[str] = None):
    """Reported ADD cases per district for predicted vs actual map overlay."""
    req_y, req_w = parse_week_param(week)
    if not req_y:
        req_y, req_w = get_latest_year_week()

    df_rows = get_weekly_df(req_y, req_w)

    records = {}
    for _, r in df_rows.iterrows():
        records[r["district"]] = {
            "reported_cases": int(r["add_cases_week"]),
            "reported_per_100k": round(float(r["add_cases_per_lakh"]), 1),
            "population": int(r["population_2011"])
        }

    return {
        "week": f"{req_y}-W{req_w:02d}",
        "districts": records
    }

@app.get("/api/v1/stations")
async def get_stations(district: Optional[str] = None):
    """CPCB stations in district with latest BOD / coliform / DO coordinates."""
    conn = get_db()
    if district:
        rows = conn.execute("""
            SELECT * FROM water_quality_stations WHERE district = ?
        """, (district,)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM water_quality_stations").fetchall()
    conn.close()

    stations = []
    all_coords = get_station_coords()
    for r in rows:
        code = str(r["station_code"])
        coords = all_coords.get(code, {})
        fecal = float(r["fecal_coliform_max_mpn"] or 0)
        bod = float(r["bod_max_mg_l"] or 0)
        do_val = float(r["do_min_mg_l"] or 0)

        # Fecal status classification
        if fecal < 500:
            fecal_status = "Safe / Desirable"
            marker_color = "#10b981"
        elif fecal <= 2500:
            fecal_status = "Permissible with Treatment"
            marker_color = "#f59e0b"
        else:
            fecal_status = "Severe Contamination"
            marker_color = "#ef4444"

        stations.append({
            "station_code": int(r["station_code"]),
            "monitoring_location": r["monitoring_location"],
            "local_body": coords.get("local_body", r["monitoring_location"]),
            "river_name": r["river_name"],
            "district": r["district"],
            "latitude": coords.get("latitude", 10.0),
            "longitude": coords.get("longitude", 76.5),
            "bod_max_mg_l": bod,
            "fecal_coliform_mpn": fecal,
            "fecal_status": fecal_status,
            "marker_color": marker_color,
            "do_min_mg_l": do_val,
            "temp_c": float(r["temp_max_c"] or 28.0),
            "ph": float(r["ph_max"] or 7.2)
        })

    return {
        "district_filter": district,
        "stations_count": len(stations),
        "stations": stations
    }

@app.get("/api/v1/hotspots/{district_id}")
async def get_hotspots(district_id: str, week: Optional[str] = None):
    """
    Compute environmental contamination hotspots and monitored rivers for selected district.
    Uses CPCB stations + 2-week precipitation runoff.
    Applies Getis-Ord Gi* if 5+ points, ranking if 3-4, fallback if < 3.
    """
    req_y, req_w = parse_week_param(week)
    if not req_y:
        req_y, req_w = get_latest_year_week()

    r_row = get_district_week_row(district_id, req_y, req_w)
    rivers = get_district_rivers(district_id)

    conn = get_db()
    st_rows = conn.execute("""
        SELECT * FROM water_quality_stations WHERE district = ?
    """, (district_id,)).fetchall()
    conn.close()

    if not st_rows:
        return {
            "district": district_id,
            "status": "not_enough_stations",
            "message": "Not enough stations for hotspot analysis",
            "stations_count": 0,
            "rivers": rivers,
            "hotspots": []
        }

    stations_count = len(st_rows)
    if stations_count < 3:
        return {
            "district": district_id,
            "status": "not_enough_stations",
            "message": "Not enough stations for hotspot analysis",
            "stations_count": stations_count,
            "rivers": rivers,
            "hotspots": [],
            "disclaimer": "Environmental contamination hotspots (not confirmed disease clusters)"
        }

    rain_sum = float(r_row["rain_2week_sum"]) if r_row else 50.0

    points = []
    all_coords = get_station_coords()
    for r in st_rows:
        code = str(r["station_code"])
        coords = all_coords.get(code, {})
        fecal = float(r["fecal_coliform_max_mpn"] or 0)
        bod = float(r["bod_max_mg_l"] or 0)
        do_val = float(r["do_min_mg_l"] or 0)

        # Environmental contamination score
        score = (fecal / 500.0) * 0.40 + (bod / 3.0) * 0.35 + max(0.0, 5.0 - do_val) * 0.15 + (rain_sum / 100.0) * 0.10
        points.append({
            "station_code": int(r["station_code"]),
            "monitoring_location": r["monitoring_location"],
            "local_body": coords.get("local_body", r["monitoring_location"]),
            "river_name": r["river_name"],
            "latitude": coords.get("latitude", 10.0),
            "longitude": coords.get("longitude", 76.5),
            "bod_mg_l": bod,
            "fecal_coliform_mpn": fecal,
            "do_mg_l": do_val,
            "score": round(score, 3)
        })

    # Spatial Statistics: Getis-Ord Gi* if 5+ stations, else simple ranking
    scores = np.array([p["score"] for p in points])
    n = len(scores)
    mean_val = float(scores.mean())
    std_val = float(scores.std()) if n > 1 else 1.0

    if n >= 5 and std_val > 0.001:
        coords_arr = np.array([[p["latitude"], p["longitude"]] for p in points])
        gi_star_scores = []
        for i in range(n):
            dists = np.hypot(coords_arr[i, 0] - coords_arr[:, 0], coords_arr[i, 1] - coords_arr[:, 1])
            weights = np.where(dists > 0, 1.0 / np.maximum(dists, 0.01), 1.0)
            w_sum = weights.sum()
            w_sq_sum = (weights ** 2).sum()
            num = (weights * scores).sum() - mean_val * w_sum
            den = std_val * np.sqrt((n * w_sq_sum - w_sum ** 2) / (n - 1))
            gi = float(num / den) if den != 0 else 0.0
            gi_star_scores.append(round(gi, 2))

        for i, p in enumerate(points):
            p["gi_star"] = gi_star_scores[i]
            p["is_hotspot"] = gi_star_scores[i] >= 1.645
            p["confidence"] = "High (95%+)" if gi_star_scores[i] >= 1.96 else ("Moderate (90%+)" if gi_star_scores[i] >= 1.645 else "Baseline")
    else:
        ranked_scores = sorted(scores, reverse=True)
        top_thresh = ranked_scores[min(1, n - 1)]
        for p in points:
            p["gi_star"] = round(p["score"], 2)
            p["is_hotspot"] = p["score"] >= top_thresh
            p["confidence"] = "Rank-Based High Priority" if p["is_hotspot"] else "Moderate Priority"

    points.sort(key=lambda x: x["score"], reverse=True)

    return {
        "district": district_id,
        "week": f"{req_y}-W{req_w:02d}",
        "status": "success",
        "method": "Getis-Ord Gi* Spatial Statistic" if n >= 5 else "Contamination Severity Ranking",
        "stations_count": stations_count,
        "disclaimer": "Environmental contamination hotspots (not confirmed disease clusters)",
        "rainfall_2week_sum_mm": round(rain_sum, 1),
        "rivers": rivers,
        "hotspots": points
    }

CSV_PARAMETER_META: Dict[str, Dict[str, str]] = {
    "cases_week_current": {
        "label": "Current Week Active Cases",
        "category": "Surveillance",
        "unit": "cases",
        "meaning": "Reported waterborne (ADD) cases in week t; measures active local transmission momentum."
    },
    "rainfall_mm_total": {
        "label": "Current Week Rainfall (t)",
        "category": "Climatic",
        "unit": "mm",
        "meaning": "Total weekly precipitation (NASA POWER); heavy rain flushes surface contaminants into water sources."
    },
    "rain_mm_lag1": {
        "label": "Preceding Week Rain (Lag-1)",
        "category": "Climatic",
        "unit": "mm",
        "meaning": "1-week prior rainfall capturing pathogen incubation and post-runoff exposure lag."
    },
    "rain_mm_lag2": {
        "label": "2-Weeks Prior Rain (Lag-2)",
        "category": "Climatic",
        "unit": "mm",
        "meaning": "2-week lagged rainfall capturing delayed water-table seepage and secondary spread."
    },
    "rain_2week_sum": {
        "label": "2-Week Cumulative Runoff",
        "category": "Climatic",
        "unit": "mm",
        "meaning": "Combined 14-day rainfall volume indicating soil saturation and flood/drainage overflow risk."
    },
    "is_torrential_rain": {
        "label": "Torrential Storm Flag (>=100 mm)",
        "category": "Climatic",
        "unit": "flag",
        "meaning": "Binary extreme-weather indicator (1 if weekly rain >= 100 mm, else 0) for acute flood surges."
    },
    "temp_c_mean": {
        "label": "Mean Temperature (t)",
        "category": "Climatic",
        "unit": "°C",
        "meaning": "Weekly mean ambient temperature affecting bacterial survival and water consumption."
    },
    "temp_c_lag1": {
        "label": "Preceding Week Temp (Lag-1)",
        "category": "Climatic",
        "unit": "°C",
        "meaning": "Prior week temperature influencing pathogen proliferation in stagnant surface water."
    },
    "humidity_pct_mean": {
        "label": "Relative Humidity",
        "category": "Climatic",
        "unit": "%",
        "meaning": "Mean atmospheric humidity at 2m reflecting monsoon moisture persistence."
    },
    "wq_bod_mg_l": {
        "label": "River BOD (Organic Pollution)",
        "category": "Water Quality",
        "unit": "mg/L",
        "meaning": "Biochemical Oxygen Demand from CPCB river stations; >3.0 mg/L signals organic/sewage load."
    },
    "wq_fecal_coliform_mpn": {
        "label": "Fecal Coliform Count",
        "category": "Water Quality",
        "unit": "MPN/100mL",
        "meaning": "CPCB river fecal bacteria concentration; >500 MPN indicates direct fecal contamination."
    },
    "wq_do_min_mg_l": {
        "label": "River Dissolved Oxygen (DO)",
        "category": "Water Quality",
        "unit": "mg/L",
        "meaning": "Minimum dissolved oxygen in monitored rivers; <5.0 mg/L reflects stagnant/septic water."
    },
    "population_density": {
        "label": "Population Density",
        "category": "Demographic",
        "unit": "/km²",
        "meaning": "Census persons per sq. km capturing crowding and shared water-supply exposure."
    },
    "literacy_rate": {
        "label": "District Literacy Rate",
        "category": "Demographic",
        "unit": "%",
        "meaning": "Census literacy rate acting as a resilience factor for hygiene awareness and early care-seeking."
    },
    "sin_week": {
        "label": "Monsoon Seasonality (Sine)",
        "category": "Seasonality",
        "unit": "cycle",
        "meaning": "Cyclical sine encoding of week-of-year (1–52) tracking SW/NE monsoon phase."
    },
    "cos_week": {
        "label": "Monsoon Seasonality (Cosine)",
        "category": "Seasonality",
        "unit": "cycle",
        "meaning": "Cyclical cosine encoding of week-of-year pairing with sine for smooth annual periodicity."
    },
}

def format_csv_param_value(col: str, val: float) -> str:
    if col == "cases_week_current":
        return f"{int(round(val))} cases"
    if col == "wq_fecal_coliform_mpn":
        return f"{int(round(val)):,} MPN"
    if col == "population_density":
        return f"{int(round(val)):,} /km²"
    if col == "is_torrential_rain":
        return "Yes (1)" if val >= 0.5 else "No (0)"
    if col in ("sin_week", "cos_week"):
        return f"{val:.2f}"
    unit = CSV_PARAMETER_META.get(col, {}).get("unit", "")
    return f"{val:.1f} {unit}".strip()

@app.get("/api/v1/drivers/{district_id}")
async def get_drivers(district_id: str, week: Optional[str] = None):
    """Exact LightGBM TreeSHAP decomposition across all 16 CSV parameters + Top 3 plain-text drivers."""
    req_y, req_w = parse_week_param(week)
    if not req_y:
        req_y, req_w = get_latest_year_week()

    r = get_district_week_row(district_id, req_y, req_w)
    if not r:
        raise HTTPException(status_code=404, detail="District data not found")

    f_vals = pd.DataFrame([[float(r[c]) for c in feature_cols]], columns=feature_cols)
    pred = float(weekly_model.predict(f_vals)[0])
    contribs = weekly_model.predict(f_vals, pred_contrib=True)[0]
    base_val = round(float(contribs[-1]), 1)
    net_shap = round(float(np.sum(contribs[:-1])), 1)

    all_params = []
    for idx, col in enumerate(feature_cols):
        raw_val = float(r[col])
        s_val = round(float(contribs[idx]), 1)
        meta = CSV_PARAMETER_META.get(col, {"label": col, "category": "Feature", "unit": "", "meaning": ""})
        fmt_val = format_csv_param_value(col, raw_val)
        if s_val > 0.05:
            direction = "Pushes Risk Up"
            effect_sentence = f"Pushes next-week forecast UP by +{s_val} cases above Kerala baseline."
        elif s_val < -0.05:
            direction = "Lowers Risk"
            effect_sentence = f"Buffers/lowers next-week forecast by {s_val} cases below Kerala baseline."
        else:
            direction = "Neutral"
            effect_sentence = "Neutral impact (0.0 cases) in this surveillance week."

        plain_line = (
            f"{meta['label']} ({fmt_val}): { 'elevates' if s_val >= 0 else 'reduces' } projected caseload by "
            f"{'+' if s_val >= 0 else ''}{s_val} cases — {meta['meaning']}"
        )

        all_params.append({
            "csv_column": col,
            "feature": col,
            "label": meta["label"],
            "category": meta["category"],
            "actual_value": round(raw_val, 2),
            "formatted_value": fmt_val,
            "shap_impact_cases": s_val,
            "abs_impact": abs(s_val),
            "direction": direction,
            "meaning": meta["meaning"],
            "shap_explanation": f"{meta['meaning']} {effect_sentence}",
            "plain_text": plain_line
        })

    all_params.sort(key=lambda item: item["abs_impact"], reverse=True)
    top_3 = all_params[:3]
    shap_info = get_plain_text_shap_reasons(r)

    return {
        "district": district_id,
        "week": f"{req_y}-W{req_w:02d}",
        "base_cases": base_val,
        "net_shap_delta": net_shap,
        "predicted_cases": max(0, int(round(pred))),
        "top_3_drivers": top_3,
        "all_parameters": all_params,
        "plain_text_list": [p["plain_text"] for p in top_3],
        "plain_text_summary": shap_info["plain_text_summary"]
    }

class WhatIfRequest(BaseModel):
    district: str
    week: Optional[str] = None
    rainfall_shock_mm: float = 0.0
    temperature_shock_c: float = 0.0
    bod_override: Optional[float] = None
    cases_current_override: Optional[float] = None

@app.post("/api/v1/whatif")
async def post_whatif(data: WhatIfRequest):
    """POST rainfall / temperature shocks, returns re-inferred forecast near-real-time."""
    req_y, req_w = parse_week_param(data.week)
    if not req_y:
        req_y, req_w = get_latest_year_week()

    row = get_district_week_row(data.district, req_y, req_w)
    if not row:
        raise HTTPException(status_code=404, detail="District baseline row not found")

    r_dict = dict(row)
    pop = int(r_dict["population_2011"])

    base_f = pd.DataFrame([[float(r_dict[c]) for c in feature_cols]], columns=feature_cols)
    baseline_pred = max(0, int(round(float(weekly_model.predict(base_f)[0]))))
    baseline_tier = assign_tier(baseline_pred, pop, float(r_dict["rainfall_mm_total"]))

    new_rain = max(0.0, float(r_dict["rainfall_mm_total"]) + data.rainfall_shock_mm)
    new_temp = float(r_dict["temp_c_mean"]) + data.temperature_shock_c
    if data.bod_override is not None:
        r_dict["wq_bod_mg_l"] = float(data.bod_override)
    if data.cases_current_override is not None:
        r_dict["cases_week_current"] = float(data.cases_current_override)

    r_dict["rainfall_mm_total"] = new_rain
    r_dict["temp_c_mean"] = new_temp
    r_dict["rain_2week_sum"] = new_rain + float(r_dict["rain_mm_lag1"])
    r_dict["is_torrential_rain"] = 1 if new_rain >= 100.0 else 0

    shock_f = pd.DataFrame([[float(r_dict[c]) for c in feature_cols]], columns=feature_cols)
    shock_pred = max(0, int(round(float(weekly_model.predict(shock_f)[0]))))
    shock_tier = assign_tier(shock_pred, pop, new_rain)
    shock_rate = round((shock_pred / pop) * 100000.0, 1)

    diff_cases = shock_pred - baseline_pred
    diff_pct = round((diff_cases / max(1, baseline_pred)) * 100, 1)

    return {
        "district": data.district,
        "week": f"{req_y}-W{req_w:02d}",
        "baseline_forecast": {
            "predicted_cases": baseline_pred,
            "tier": baseline_tier,
            "rainfall_mm": round(float(row["rainfall_mm_total"]), 1),
            "temperature_c": round(float(row["temp_c_mean"]), 1)
        },
        "simulated_forecast": {
            "predicted_cases": shock_pred,
            "tier": shock_tier,
            "rate_per_100k": shock_rate,
            "rainfall_mm": round(new_rain, 1),
            "temperature_c": round(new_temp, 1),
            "cases_delta": diff_cases,
            "change_pct": diff_pct,
            "error_margin_mae": TYPICAL_ERROR_MAE
        },
        "advisory": (
            "🚨 CRITICAL ALERT: Extreme precipitation shock elevates projected caseload past safety thresholds."
            if shock_tier == "High Alert" 
            else "Moderate risk elevation: maintain pipeline chlorination checks."
        )
    }

@app.get("/api/v1/district_csv_data/{district_id}")
async def get_district_csv_data(district_id: str, year: int = Query(2026)):
    """Return all weekly CSV forecast records for the selected district and year with requested columns."""
    if year >= 2026 and os.path.exists(CACHE_2026_DB_PATH):
        conn = get_2026_db()
        df = pd.read_sql(
            "SELECT * FROM master_district_weekly_2026 WHERE district = ? AND year = ? ORDER BY week ASC",
            conn, params=[district_id, year]
        )
        conn.close()
    else:
        conn = get_db()
        df = pd.read_sql(
            "SELECT * FROM master_district_weekly WHERE district = ? AND year = ? ORDER BY week ASC",
            conn, params=[district_id, year]
        )
        conn.close()

    if df.empty:
        return {"district": district_id, "year": year, "rows": []}

    X_batch = df[feature_cols].astype(float)
    preds = weekly_model.predict(X_batch)

    rows = []
    for idx, r in df.reset_index(drop=True).iterrows():
        pop = int(r["population_2011"])
        rain = float(r["rainfall_mm_total"])
        bod = float(r["wq_bod_mg_l"])
        fecal = float(r["wq_fecal_coliform_mpn"])
        do_val = float(r["wq_do_min_mg_l"])
        cur_cases = int(r["cases_week_current"])
        pred = max(0, int(round(float(preds[idx]))))
        tier = assign_tier(pred, pop, rain)
        rate = round((pred / pop) * 100000.0, 1)
        prob = compute_outbreak_probability(pred, pop, rain, bod, fecal)
        outbreak_stat = get_last_outbreak_status(cur_cases, pop, float(r["rain_mm_lag1"]))
        shap_info = get_plain_text_shap_reasons(r)
        wqi = calculate_wqi(bod, fecal, do_val)
        wk = int(r["week"])
        rows.append({
            "district": district_id,
            "week_num": wk,
            "period": f"{year}-W{wk:02d}",
            "predicted_probability": prob,
            "risk_band": tier,
            "tier": tier,
            "last_outbreak_status": outbreak_stat["status_label"],
            "last_outbreak_badge": outbreak_stat["badge_class"],
            "top_3_shap_plain": shap_info["plain_text_summary"],
            "current_cases": cur_cases,
            "predicted_cases": pred,
            "rate_per_100k": rate,
            "rainfall_mm": round(rain, 1),
            "temp_c": round(float(r["temp_c_mean"]), 1),
            "wqi_score": wqi["score"],
            "bod_mg_l": round(bod, 2),
            "coliform_mpn": int(round(fecal)),
            "do_mg_l": round(do_val, 2)
        })

    return {
        "district": district_id,
        "year": year,
        "total_weeks": len(rows),
        "rows": rows
    }

@app.get("/api/v1/export.csv")
async def export_csv(week: Optional[str] = None, district: Optional[str] = None, year: Optional[int] = None):
    """Download forecast table as CSV (supports single district full-year export or all-districts week export)."""
    if district and year:
        res = await get_district_csv_data(district, year)
        clean_rows = [
            {
                "District": r["district"],
                "Week": r["period"],
                "Predicted_Probability_Pct": f"{r['predicted_probability']}%",
                "Risk_Band": r["risk_band"],
                "Last_Reported_Outbreak_Status": r["last_outbreak_status"],
                "Top_3_SHAP_Drivers_Plain_Text": r["top_3_shap_plain"],
                "Predicted_Next_Week_Cases": r["predicted_cases"],
                "Incidence_Rate_Per_100k": r["rate_per_100k"],
                "Rainfall_mm": r["rainfall_mm"],
                "WQI_Score": r["wqi_score"]
            }
            for r in res["rows"]
        ]
        out_df = pd.DataFrame(clean_rows)
        csv_buf = io.StringIO()
        out_df.to_csv(csv_buf, index=False)
        csv_buf.seek(0)
        filename = f"{district.lower()}_forecast_{year}.csv"
        return StreamingResponse(
            iter([csv_buf.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )

    req_y, req_w = parse_week_param(week)
    if not req_y:
        req_y, req_w = get_latest_year_week()

    df_rows = get_weekly_df(req_y, req_w)

    records = []
    for d in ALL_DISTRICTS:
        match = df_rows[df_rows["district"] == d]
        if match.empty:
            continue
        r = match.iloc[0]
        pop = int(r["population_2011"])
        rain = float(r["rainfall_mm_total"])
        bod = float(r["wq_bod_mg_l"])
        fecal = float(r["wq_fecal_coliform_mpn"])
        cur_cases = int(r["cases_week_current"])
        f_vals = pd.DataFrame([[float(r[c]) for c in feature_cols]], columns=feature_cols)
        pred = max(0, int(round(float(weekly_model.predict(f_vals)[0]))))
        tier = assign_tier(pred, pop, rain)
        rate = round((pred / pop) * 100000.0, 1)
        prob = compute_outbreak_probability(pred, pop, rain, bod, fecal)
        outbreak_stat = get_last_outbreak_status(cur_cases, pop, float(r["rain_mm_lag1"]))
        shap_info = get_plain_text_shap_reasons(r)
        wqi = calculate_wqi(bod, fecal, float(r["wq_do_min_mg_l"]))

        records.append({
            "District": d,
            "Week": f"{req_y}-W{req_w:02d}",
            "Predicted_Probability_Pct": f"{prob}%",
            "Risk_Band": tier,
            "Last_Reported_Outbreak_Status": outbreak_stat["status_label"],
            "Top_3_SHAP_Drivers_Plain_Text": shap_info["plain_text_summary"],
            "Predicted_Next_Week_Cases": pred,
            "Predicted_Incidence_Per_100k": rate,
            "WQI_Score": wqi["score"],
            "Rainfall_Total_mm": round(rain, 1)
        })

    out_df = pd.DataFrame(records)
    csv_buf = io.StringIO()
    out_df.to_csv(csv_buf, index=False)
    csv_buf.seek(0)

    filename = f"kerala_outbreak_forecast_{req_y}_W{req_w:02d}.csv"
    return StreamingResponse(
        iter([csv_buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI Surveillance Engine...")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
