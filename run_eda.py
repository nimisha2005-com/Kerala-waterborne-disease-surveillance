"""
Exploratory Data Analysis (EDA) for Kerala Waterborne Disease Surveillance (2011–2025).

Generates 5 publication-quality charts in outputs/eda/:
  1. 01_monthly_seasonality_timeline.png  - 15-year monthly timeline & seasonal monsoon curve
  2. 02_kerala_choropleth_map.png         - Geographic map of Kerala districts colored by disease risk
  3. 03_district_hotspot_ranking.png      - Bar chart of ADD & Leptospirosis rates by district
  4. 04_rainfall_and_literacy_scatter.png - Key drivers: Monsoon rainfall & Literacy rate vs ADD
  5. 05_correlation_heatmap.png           - Correlation matrix across climate, water quality, demographics & disease

Run:  python run_eda.py
"""
import json
import os

os.environ.setdefault("MPLCONFIGDIR", os.path.abspath("outputs/.mplcache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd
import seaborn as sns

OUT_DIR = "outputs/eda"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs("outputs/.mplcache", exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
})
sns.set_style("whitegrid")

# ── Load datasets ─────────────────────────────────────────────────────────
df_dist = pd.read_csv("data/processed/master_district_yearly.csv")
df_mon = pd.read_csv("data/processed/master_state_monthly.csv")
with open("data/raw/boundaries/kerala_districts.geojson", "r", encoding="utf-8") as f:
    geojson = json.load(f)

df_dist_noncovid = df_dist[~df_dist["covid_year"]].copy()
df_mon_noncovid = df_mon[~df_mon["covid_year"]].copy()

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# ═══════════════════════════════════════════════════════════════════════════
# CHART 1: 15-Year Monthly Timeline & Seasonal Monsoon Curve
# ═══════════════════════════════════════════════════════════════════════════
print("Generating Chart 1: Monthly Seasonality & 15-Year Timeline...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), gridspec_kw={"height_ratios": [1.3, 1]})

df_mon["date"] = pd.to_datetime(
    df_mon["year"].astype(str) + "-" + df_mon["month"].astype(str).str.zfill(2) + "-01"
)
df_mon = df_mon.sort_values("date")

# Top panel: 15-year timeline of ADD cases vs Rainfall
ax1_rain = ax1.twinx()
ax1_rain.fill_between(df_mon["date"], df_mon["rain_mm"], color="#6baed6", alpha=0.35, label="Monthly Rainfall (mm)")
ax1_rain.set_ylabel("State Avg Rainfall (mm)", color="#2171b5", fontweight="bold")
ax1_rain.tick_params(axis="y", labelcolor="#2171b5")
ax1_rain.grid(False)

ax1.plot(df_mon["date"], df_mon["add_cases"] / 1000, color="#cb181d", linewidth=1.8, label="ADD Cases (Thousands)")
ax1.axvspan(pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31"), color="#fdd0a2", alpha=0.45, label="COVID-19 Reporting Dip (2020–21)")
ax1.set_title("Kerala Monthly Acute Diarrhoeal Disease (ADD) Cases vs. Rainfall (2011–2025)")
ax1.set_ylabel("ADD Cases (in Thousands)", color="#cb181d", fontweight="bold")
ax1.tick_params(axis="y", labelcolor="#cb181d")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_rain.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=True)

# Bottom panel: Average seasonal profile by month (excluding COVID)
seasonal = df_mon_noncovid.groupby("month").agg(
    add_mean=("add_cases", "mean"),
    lepto_mean=("leptospirosis_cases", "mean"),
    rain_mean=("rain_mm", "mean"),
).reindex(range(1, 13))

x = np.arange(12)
ax2_rain = ax2.twinx()
ax2_rain.bar(x - 0.18, seasonal["rain_mean"], width=0.35, color="#6baed6", alpha=0.65, label="Avg Rainfall (mm)")
ax2_rain.set_ylabel("Avg Rainfall (mm)", color="#2171b5", fontweight="bold")
ax2_rain.tick_params(axis="y", labelcolor="#2171b5")
ax2_rain.grid(False)

ax2.bar(x + 0.18, seasonal["add_mean"] / 1000, width=0.35, color="#ef3b2c", alpha=0.85, label="Avg ADD Cases (Thousands)")
ax2.set_xticks(x)
ax2.set_xticklabels(MONTH_NAMES)
ax2.set_ylabel("Avg Monthly ADD Cases (Thousands)", color="#cb181d", fontweight="bold")
ax2.tick_params(axis="y", labelcolor="#cb181d")
ax2.set_title("Seasonal Monsoon Peak (June–August): Average Monthly Pattern (Excl. COVID Years)")

l1, lb1 = ax2.get_legend_handles_labels()
l2, lb2 = ax2_rain.get_legend_handles_labels()
ax2.legend(l1 + l2, lb1 + lb2, loc="upper right", frameon=True)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/01_monthly_seasonality_timeline.png")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════
# CHART 2: Geographic Choropleth Map of Kerala Districts
# ═══════════════════════════════════════════════════════════════════════════
print("Generating Chart 2: Kerala District Choropleth Maps...")
dist_summary = df_dist_noncovid.groupby("district").agg(
    add_per_lakh=("add_per_lakh", "mean"),
    lepto_per_lakh=("leptospirosis_per_lakh", "mean"),
    hepa_per_lakh=("hepatitis_a_per_lakh", "mean"),
    rain_annual=("rain_annual_mm", "mean"),
).to_dict(orient="index")


def plot_kerala_map(ax, metric_key, title, cmap_name, unit_label):
    patches = []
    values = []
    centroids = {}

    for feat in geojson["features"]:
        d_name = feat["properties"]["district"]
        val = dist_summary.get(d_name, {}).get(metric_key, 0.0)
        geom = feat["geometry"]
        coords_list = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]

        largest_ring = None
        max_pts = 0
        for poly in coords_list:
            exterior = np.array(poly[0])
            patches.append(Polygon(exterior, closed=True))
            values.append(val)
            if len(exterior) > max_pts:
                max_pts = len(exterior)
                largest_ring = exterior

        if largest_ring is not None:
            centroids[d_name] = (largest_ring[:, 0].mean(), largest_ring[:, 1].mean(), val)

    cmap = plt.get_cmap(cmap_name)
    vmin, vmax = min(values), max(values)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    pc = PatchCollection(patches, cmap=cmap, norm=norm, edgecolor="#222222", linewidth=0.9)
    pc.set_array(np.array(values))
    ax.add_collection(pc)
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, pad=10)

    for d_name, (cx, cy, val) in centroids.items():
        short = d_name[:3].upper() if len(d_name) > 8 else d_name
        ax.text(cx, cy, f"{short}\n({val:.0f})", ha="center", va="center",
                fontsize=7.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white", alpha=0.75, edgecolor="none"))

    cbar = plt.colorbar(pc, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label(unit_label, fontsize=9)


fig, axes = plt.subplots(1, 2, figsize=(13, 7.5))
plot_kerala_map(axes[0], "add_per_lakh",
                "ADD Incidence Rate by District\n(Avg Cases per 1 Lakh Pop, 2011–2025)",
                "YlOrRd", "Cases per 1 Lakh Population")
plot_kerala_map(axes[1], "lepto_per_lakh",
                "Leptospirosis Incidence Rate by District\n(Avg Cases per 1 Lakh Pop, 2011–2025)",
                "PuBuGn", "Cases per 1 Lakh Population")
plt.suptitle("Geographic Distribution of Waterborne Diseases Across Kerala Districts", fontsize=14, fontweight="bold", y=0.98)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/02_kerala_choropleth_map.png")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════
# CHART 3: District Hotspot Ranking (Horizontal Bar Comparison)
# ═══════════════════════════════════════════════════════════════════════════
print("Generating Chart 3: District Hotspot Ranking...")
rank_df = df_dist_noncovid.groupby("district").agg(
    add_per_lakh=("add_per_lakh", "mean"),
    hepa_per_lakh=("hepatitis_a_per_lakh", "mean"),
    lepto_per_lakh=("leptospirosis_per_lakh", "mean"),
).sort_values("add_per_lakh", ascending=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), sharey=True)

colors_add = plt.cm.YlOrRd(np.linspace(0.35, 0.9, len(rank_df)))
ax1.barh(rank_df.index, rank_df["add_per_lakh"], color=colors_add, edgecolor="#333333", height=0.7)
for i, v in enumerate(rank_df["add_per_lakh"]):
    ax1.text(v + 30, i, f"{v:.0f}", va="center", fontsize=8.5, fontweight="bold")
ax1.set_xlim(0, rank_df["add_per_lakh"].max() * 1.15)
ax1.set_xlabel("Avg Annual ADD Cases per 1 Lakh Population")
ax1.set_title("Acute Diarrhoeal Disease (ADD) Burden")

y_pos = np.arange(len(rank_df))
ax2.barh(y_pos - 0.18, rank_df["hepatitis_a_per_lakh"] if "hepatitis_a_per_lakh" in rank_df else rank_df["hepa_per_lakh"],
         height=0.35, color="#fd8d3c", label="Hepatitis A (Total)", edgecolor="#333333")
ax2.barh(y_pos + 0.18, rank_df["lepto_per_lakh"],
         height=0.35, color="#2b8cbe", label="Leptospirosis (Total)", edgecolor="#333333")
ax2.set_xlabel("Avg Annual Cases per 1 Lakh Population")
ax2.set_title("Hepatitis A & Leptospirosis Burden")
ax2.legend(loc="lower right", frameon=True)

plt.suptitle("Kerala District Disease Hotspot Ranking (2011–2025 Average, Excl. COVID)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/03_district_hotspot_ranking.png")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════
# CHART 4: Key Drivers Scatter Plots (Monsoon Rainfall & Literacy Rate)
# ═══════════════════════════════════════════════════════════════════════════
print("Generating Chart 4: Key Drivers Scatter Plots...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

# Left: Monthly Rainfall vs Monthly ADD Cases
sns.regplot(
    data=df_mon_noncovid, x="rain_mm", y="add_cases",
    ax=ax1, color="#2171b5",
    scatter_kws={"alpha": 0.65, "s": 45, "edgecolor": "white"},
    line_kws={"color": "#cb181d", "linewidth": 2}
)
r_mon = df_mon_noncovid["rain_mm"].corr(df_mon_noncovid["add_cases"])
ax1.set_title(f"Monthly Rainfall vs. State ADD Cases (r = {r_mon:+.2f})")
ax1.set_xlabel("State Monthly Rainfall (mm)")
ax1.set_ylabel("Monthly ADD Cases (Whole State)")

# Right: District Literacy Rate vs Avg ADD per Lakh
dist_avg = df_dist_noncovid.groupby("district").agg(
    add_per_lakh=("add_per_lakh", "mean"),
    literacy_rate=("literacy_rate", "first"),
    wq_bod=("wq_bod_mg_l", "first"),
).reset_index()

sns.regplot(
    data=dist_avg, x="literacy_rate", y="add_per_lakh",
    ax=ax2, color="#238b45",
    scatter_kws={"alpha": 0.85, "s": 70, "edgecolor": "black"},
    line_kws={"color": "#cb181d", "linewidth": 2}
)
for _, row in dist_avg.iterrows():
    ax2.annotate(row["district"], (row["literacy_rate"], row["add_per_lakh"]),
                 xytext=(4, 4), textcoords="offset points", fontsize=8)
r_lit = dist_avg["literacy_rate"].corr(dist_avg["add_per_lakh"])
ax2.set_title(f"District Literacy Rate vs. ADD Rate (r = {r_lit:+.2f})")
ax2.set_xlabel("Literacy Rate (% — Census 2011)")
ax2.set_ylabel("Avg Annual ADD Cases per 1 Lakh")

plt.suptitle("Environmental & Demographic Drivers of Waterborne Disease in Kerala", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/04_rainfall_and_literacy_scatter.png")
plt.close()

# ═══════════════════════════════════════════════════════════════════════════
# CHART 5: Multi-Factor Correlation Heatmap
# ═══════════════════════════════════════════════════════════════════════════
print("Generating Chart 5: Correlation Heatmap...")
corr_cols = {
    "add_per_lakh": "ADD / Lakh",
    "hepatitis_a_per_lakh": "Hepatitis A / Lakh",
    "leptospirosis_per_lakh": "Leptospirosis / Lakh",
    "typhoid_per_lakh": "Typhoid / Lakh",
    "rain_annual_mm": "Annual Rain (mm)",
    "rain_sw_monsoon_mm": "SW Monsoon Rain (mm)",
    "heavy_rain_weeks": "Heavy Rain Weeks (>150mm)",
    "temp_mean_c": "Mean Temp (°C)",
    "humidity_mean_pct": "Mean Humidity (%)",
    "wq_fecal_coliform_mpn": "River Fecal Coliform",
    "wq_bod_mg_l": "River BOD (mg/L)",
    "wq_do_min_mg_l": "River Min DO (mg/L)",
    "population_density": "Population Density",
    "literacy_rate": "Literacy Rate (%)",
}
corr_df = df_dist_noncovid[list(corr_cols.keys())].rename(columns=corr_cols).corr()

fig, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(
    corr_df, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-0.8, vmax=0.8,
    center=0, linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8}
)
ax.set_title("Correlation Matrix: Disease Incidence, Climate, Water Quality & Demographics\n(District-Year Dataset, 2011–2025 Excl. COVID)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/05_correlation_heatmap.png")
plt.close()

print("\n" + "=" * 60)
print(f"All 5 EDA charts saved to: {OUT_DIR}/")
for fname in sorted(os.listdir(OUT_DIR)):
    print(f"  [OK] {fname}")
