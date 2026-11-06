import pdfplumber
import pandas as pd
import os
import re

# ── Kerala districts (to match from location names) ───────────────────────
KERALA_DISTRICTS = [
    "Thiruvananthapuram", "Kollam", "Pathanamthitta", "Alappuzha",
    "Kottayam", "Idukki", "Ernakulam", "Thrissur", "Palakkad",
    "Malappuram", "Kozhikode", "Wayanad", "Kannur", "Kasaragod",
]

def guess_district(location_text):
    """Try to find a Kerala district name inside the location string."""
    text = location_text.upper()
    for district in KERALA_DISTRICTS:
        if district.upper() in text:
            return district
    # Extra common name variations
    if "TRIVANDRUM" in text or "THIRUVANANTHAPURAM" in text:
        return "Thiruvananthapuram"
    if "CALICUT" in text or "KOZHIKODE" in text:
        return "Kozhikode"
    if "TRICHUR" in text or "THRISSUR" in text:
        return "Thrissur"
    if "COCHIN" in text or "ERNAKULAM" in text:
        return "Ernakulam"
    if "PALGHAT" in text or "PALAKKAD" in text:
        return "Palakkad"
    if "ALLEPPEY" in text or "ALAPPUZHA" in text:
        return "Alappuzha"
    if "QUILON" in text:
        return "Kollam"
    return "Unknown"

def clean_number(value):
    """Convert a string to float, return None if not possible."""
    try:
        v = str(value).strip().replace(",", "")
        if v in ("", "None", "-", "N/A", "ND"):
            return None
        return float(v)
    except:
        return None

# ── Column names for Medium & Minor Rivers PDF ────────────────────────────
# Based on the PDF table structure seen in the screenshot
MED_MINOR_COLS = [
    "station_code", "monitoring_location", "river_name", "state",
    "temp_min_c", "temp_max_c",
    "do_min_mg_l", "do_max_mg_l",
    "ph_min", "ph_max",
    "conductivity_min", "conductivity_max",
    "bod_min_mg_l", "bod_max_mg_l",
    "nitrate_min_mg_l", "nitrate_max_mg_l",
    "fecal_coliform_min_mpn", "fecal_coliform_max_mpn",
    "total_coliform_min_mpn", "total_coliform_max_mpn",
    "fecal_streptococci_min_mpn", "fecal_streptococci_max_mpn",
]

def extract_pdf(pdf_path, col_names, state_col_index, label):
    """Extract Kerala rows from a PDF and return as a DataFrame."""
    print(f"\nExtracting: {label}")

    if not os.path.exists(pdf_path):
        print(f"  ✗ File not found: {pdf_path}")
        return pd.DataFrame()

    kerala_rows = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                for row in table:
                    if row is None:
                        continue
                    clean = [str(c).strip() if c else "" for c in row]
                    row_text = " ".join(clean).upper()
                    if "KERALA" in row_text and len(clean) > 5:
                        kerala_rows.append(clean)

    print(f"  Found {len(kerala_rows)} Kerala rows")

    if not kerala_rows:
        return pd.DataFrame()

    # Assign column names (pad or trim each row to match)
    records = []
    for row in kerala_rows:
        # Pad row with empty strings if it has fewer columns than expected
        padded = row + [""] * max(0, len(col_names) - len(row))
        # Trim row if it has more columns than expected
        padded = padded[:len(col_names)]
        records.append(dict(zip(col_names, padded)))

    df = pd.DataFrame(records)
    df["source_pdf"] = label
    df["year"] = 2024   # This is the 2024 dataset

    # Add district column by guessing from location name
    df["district"] = df["monitoring_location"].apply(guess_district)

    return df


# ── Extract from Medium & Minor Rivers PDF (primary - 94 rows) ────────────
df_med = extract_pdf(
    pdf_path  = r"data\raw\water_quality\Water_Quality_data_of_Med_Min_River_2024.pdf",
    col_names = MED_MINOR_COLS,
    state_col_index = 3,
    label     = "Medium & Minor Rivers 2024",
)

# ── Extract from Major Rivers PDF (secondary - 6 rows) ────────────────────
# Major rivers PDF has slightly different structure (extra empty columns)
# We use same col names but it will be approximate
df_rivers = extract_pdf(
    pdf_path  = r"data\raw\water_quality\WQuality_River-Data-2024.pdf",
    col_names = MED_MINOR_COLS,
    state_col_index = 4,
    label     = "Major Rivers 2024",
)

# ── Combine both DataFrames ────────────────────────────────────────────────
df_all = pd.concat([df_med, df_rivers], ignore_index=True)

# ── Convert numeric columns ────────────────────────────────────────────────
numeric_cols = [
    "temp_min_c", "temp_max_c",
    "do_min_mg_l", "do_max_mg_l",
    "ph_min", "ph_max",
    "bod_min_mg_l", "bod_max_mg_l",
    "nitrate_min_mg_l", "nitrate_max_mg_l",
    "fecal_coliform_min_mpn", "fecal_coliform_max_mpn",
    "total_coliform_min_mpn", "total_coliform_max_mpn",
    "fecal_streptococci_min_mpn", "fecal_streptococci_max_mpn",
]
for col in numeric_cols:
    if col in df_all.columns:
        df_all[col] = df_all[col].apply(clean_number)

# ── Save to CSV ───────────────────────────────────────────────────────────
output_path = "data/raw/water_quality/kerala_water_quality_2024.csv"
df_all.to_csv(output_path, index=False)

# ── Print summary ─────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"Saved to: {output_path}")
print(f"Total rows: {len(df_all)}")
print(f"\nDistricts found in data:")
print(df_all["district"].value_counts().to_string())
print(f"\nColumn names saved:")
print(list(df_all.columns))
print(f"\nPreview (first 3 rows — key columns only):")
preview_cols = ["district", "monitoring_location", "river_name",
                "ph_min", "ph_max", "do_min_mg_l", "bod_max_mg_l",
                "fecal_coliform_max_mpn", "total_coliform_max_mpn"]
print(df_all[preview_cols].head(3).to_string(index=False))
