import geopandas as gpd
import os

# ── Step 1: Open the full India shapefile ─────────────────────────────────
print("Opening the India shapefile...")
india = gpd.read_file(r"data\raw\boundaries\gadm41_IND_2.shp")

# ── Step 2: Filter only Kerala ────────────────────────────────────────────
kerala = india[india["NAME_1"] == "Kerala"].copy()

# ── Step 3: Keep only the columns we need ─────────────────────────────────
kerala = kerala[["NAME_2", "geometry"]].copy()
kerala = kerala.rename(columns={"NAME_2": "district"})

# ── Step 4: Save as GeoJSON ───────────────────────────────────────────────
os.makedirs("data/raw/boundaries", exist_ok=True)
output_path = "data/raw/boundaries/kerala_districts.geojson"
kerala.to_file(output_path, driver="GeoJSON")

print(f"Saved! File: {output_path}")
print(f"Districts saved: {len(kerala)}")
print()
print("Districts in your new file:")
for name in sorted(kerala["district"].tolist()):
    print(" -", name)
