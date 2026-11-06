import geopandas as gpd

# ── Step 1: Open the shapefile ─────────────────────────────────────────────
print("Opening the India shapefile...")
india = gpd.read_file(r"data\raw\boundaries\gadm41_IND_2.shp")

print(f"Total districts in India: {len(india)}")
print()

# ── Step 2: Show the column names ─────────────────────────────────────────
print("Column names in the file:")
print(list(india.columns))
print()

# ── Step 3: Filter only Kerala districts ──────────────────────────────────
print("Filtering Kerala...")
kerala = india[india["NAME_1"] == "Kerala"]

print(f"Number of Kerala districts found: {len(kerala)}")
print()

# ── Step 4: Print the district names ──────────────────────────────────────
print("Kerala district names:")
for name in sorted(kerala["NAME_2"].tolist()):
    print(" -", name)

print()
print("Done! The shapefile is working correctly.")
