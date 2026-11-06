# AI-Based Spatiotemporal Early-Warning System & Web-GIS Surveillance for Waterborne Disease Outbreaks in Kerala

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-1.0.0-009688.svg)](https://fastapi.tiangolo.com/)
[![Flask](https://img.shields.io/badge/Flask-2.3%2B-black.svg)](https://flask.palletsprojects.com/)
[![Machine Learning](https://img.shields.io/badge/ML-LightGBM%20%7C%20Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Explainability](https://img.shields.io/badge/XAI-SHAP-red.svg)](https://github.com/slundberg/shap)

An integrated, near-real-time spatiotemporal early-warning system and interactive Web-GIS dashboard designed to predict, monitor, and explain waterborne disease outbreak risks across all 14 districts of Kerala, India.

---

## 📌 Project Overview

Waterborne infectious diseases (such as Acute Diarrheal Diseases, Cholera, Typhoid, and Hepatitis) represent recurring public health challenges influenced by monsoon patterns, river water quality, and district-level demographic factors.

This project delivers:
1. **Automated ETL & Database Pipeline:** Automated extraction of historical disease surveillance data (IDSP/Kerala Health Department), river water quality parameters (Kerala State Pollution Control Board), and Open-Meteo climate records into a unified SQLite relational database (`kerala_health.db`).
2. **Predictive Machine Learning Engine:** Multi-model ensemble (LightGBM Regressor + Gradient Boosting Classifier) predicting weekly outbreak probabilities and case count trends ($t+1$ week forecasting).
3. **Explainable AI (XAI):** District-specific and global SHAP (SHapley Additive exPlanations) values to interpret key clinical, environmental, and hydrological drivers behind each prediction.
4. **Interactive Web-GIS Dashboards:** High-performance FastAPI and Flask interfaces rendering interactive Choropleth maps, water-quality monitoring stations, and time-series outbreak analytics.

---

## 🌟 Key Features

- **14 District Spatiotemporal Surveillance:** Comprehensive geographic coverage across Kerala with GeoJSON boundary integration.
- **Weekly Early-Warning Alerts:** High, Moderate, and Low risk stratification to empower proactive municipal and healthcare interventions.
- **Explainable Predictions:** Built-in SHAP explanations showing top positive and negative features influencing each district's forecast (e.g., rainfall anomalies, coliform levels, pH, temperature).
- **Dual Server Implementations:**
  - **FastAPI Backend (`main.py`):** High-speed asynchronous REST API with automatic Swagger/OpenAPI docs and Jinja2 templating.
  - **Flask Backend (`app.py`):** Lightweight microservice implementation for local deployment and dashboard rendering.
- **Interactive Web-GIS Visualizations:** Dynamic Leaflet/Mapbox maps with real-time river station water quality metrics and district risk heatmaps.

---

## 📁 Repository Structure

```text
main project/
│
├── app.py                         # Flask web application & REST API
├── main.py                        # FastAPI enterprise backend & Web-GIS server
├── requirements.txt               # Python package dependencies
├── .gitignore                     # Git ignore rules for virtualenvs, caches & artifacts
│
├── data/                          # Data store
│   ├── raw/                       # Original GeoJSON, boundary shapefiles, PDFs
│   ├── processed/                 # Cleaned tabular datasets
│   └── kerala_health.db           # SQLite master health database
│
├── outputs/                       # Model artifacts and visualizations
│   ├── models/                    # Spatial and temporal trained model bundles
│   ├── models_weekly/             # Weekly outbreak models (LightGBM + SHAP data)
│   └── eda_plots/                 # Exploratory data analysis charts
│
├── scripts/                       # Maintenance and auxiliary scripts
│   ├── generate_2026_forecasts.py # Pre-computation of forward forecasts
│   ├── generate_station_coords.py # Water quality station coordinate mappings
│   └── test_endpoints.py          # API automated endpoint test suite
│
├── static/                        # CSS, JavaScript, and static web assets
│   ├── css/
│   └── js/
│
├── templates/                     # Frontend HTML templates
│   ├── dashboard.html             # FastAPI interactive Web-GIS dashboard
│   └── index.html                 # Flask dashboard template
│
└── [Pipeline Scripts]             # Data engineering & ML training scripts
    ├── fetch_climate_data.py      # Climate API ingest
    ├── extract_water_quality_save.py # River water quality extraction
    ├── extract_disease_data.py    # Health report extraction
    ├── merge_datasets.py          # Unified data joining & preprocessing
    ├── build_database.py          # SQLite database schema builder
    ├── build_weekly_master.py     # Weekly aggregated master dataset builder
    ├── train_models.py            # Baseline spatial and temporal ML models
    ├── train_weekly_models.py     # Production weekly models & benchmarking
    └── compute_shap_explanations.py # SHAP XAI calculation engine
```

---

## 🚀 Getting Started

### 1. Prerequisites
- **Python 3.10+**
- **Git**

### 2. Clone the Repository
```bash
git clone https://github.com/nimisha2005-com/Kerala-waterborne-disease-surveillance.git
cd Kerala-waterborne-disease-surveillance
```

### 3. Create a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 💻 Running the Application

### Option A: Run the FastAPI Web-GIS Dashboard (Recommended)
FastAPI provides high performance and interactive API documentation.
```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```
- **Dashboard:** Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.
- **Interactive API Docs (Swagger):** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Alternative Docs (ReDoc):** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

### Option B: Run the Flask Web Application
```bash
python app.py
```
- Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

---

## 🛠️ Data Pipeline & Model Training

If you wish to re-train the models or rebuild datasets from scratch:

1. **Extract & Merge Datasets:**
   ```bash
   python fetch_climate_data.py
   python extract_water_quality_save.py
   python extract_disease_data.py
   python merge_datasets.py
   python build_database.py
   python build_weekly_master.py
   ```

2. **Train Models & Compute Explainability:**
   ```bash
   python train_weekly_models.py
   python compute_shap_explanations.py
   ```

---

## 📊 Machine Learning & Metrics

- **Models Evaluated:** LightGBM, Gradient Boosting, Random Forest, Extra Trees, Ridge Regression, Support Vector Regressor (SVR).
- **Target Variables:**
  - Weekly Outbreak Risk Classification (Binary / Multi-class risk stratification).
  - Weekly Expected Case Count (Continuous regression).
- **Validation:** Spatiotemporal GroupKFold cross-validation across Kerala's 14 districts to prevent data leakage.
- **Interpretability:** TreeSHAP used to produce global feature rankings and local district-level waterfall contributions.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!
1. Fork the Project.
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`).
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the Branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
