import pandas as pd

# ── Load the CSV ──────────────────────────────────────────────────────────
df = pd.read_csv("data/raw/water_quality/kerala_water_quality_2024.csv")
print(f"Unknown districts before fix: {(df['district'] == 'Unknown').sum()}")

# ── River name → district mapping ─────────────────────────────────────────
# Each Kerala river flows through specific districts.
# When location text contains the river name, we assign that district.
RIVER_TO_DISTRICT = {
    # Thiruvananthapuram
    "KARAMANA":     "Thiruvananthapuram",
    "VAMANAPURAM":  "Thiruvananthapuram",
    "NEYYAR":       "Thiruvananthapuram",
    "MAMOM":        "Thiruvananthapuram",
    "AYIROOR":      "Thiruvananthapuram",

    # Kollam
    "KALLADA":      "Kollam",
    "ITHIKKARA":    "Kollam",
    "PALLICKAL":    "Kollam",

    # Pathanamthitta
    "ACHANKOVIL":   "Pathanamthitta",
    "MANIMALA":     "Pathanamthitta",
    "RAMAPURAM":    "Pathanamthitta",

    # Alappuzha
    "PAMBA":        "Alappuzha",

    # Kottayam
    "MEENACHIL":    "Kottayam",

    # Ernakulam
    "PERIYAR":      "Ernakulam",
    "MUVATTAPUZHA": "Ernakulam",
    "CHITHRAPUZHA": "Ernakulam",
    "KADAMBRAYAR":  "Ernakulam",

    # Thrissur
    "KARUVANNUR":   "Thrissur",
    "PUZHAKKAL":    "Thrissur",
    "PUZHACKAL":    "Thrissur",

    # Palakkad
    "BHARATHAPUZHA":"Palakkad",

    # Malappuram
    "CHALIYAR":     "Malappuram",
    "KADALUNDI":    "Malappuram",

    # Kozhikode
    "KORAPUZHA":    "Kozhikode",
    "KUTTIYADY":    "Kozhikode",
    "KUPPAM":       "Kasaragod",   # Kuppam river is in Kasaragod

    # Kannur
    "VALAPATTANAM": "Kannur",
    "ANCHARAKANDY": "Kannur",
    "PERUVAMBA":    "Kannur",
    "PULLUR":       "Kannur",
    "KORAYAR":      "Idukki",

    # Kasaragod
    "MOGRAL":       "Kasaragod",
    "SHIRIYA":      "Kasaragod",
    "MANJESHWAR":   "Kasaragod",
}

# ── Truncated suffix patterns ──────────────────────────────────────────────
# Some location names end with a truncated district name (e.g. "KANNU" = KANNUR)
SUFFIX_TO_DISTRICT = {
    "KANNU":   "Kannur",
    "ALAPP":   "Alappuzha",
    "THIRUVANANTH": "Thiruvananthapuram",
    "PATHANAM":     "Pathanamthitta",
    "ERNAK":        "Ernakulam",
    "THRISS":       "Thrissur",
    "MALAPP":       "Malappuram",
    "KOZHI":        "Kozhikode",
}

def full_district_guess(location_text):
    """Full multi-strategy district lookup."""
    text = str(location_text).upper()

    # Strategy 1: direct district name in text
    district_names = {
        "THIRUVANANTHAPURAM": "Thiruvananthapuram",
        "TRIVANDRUM":         "Thiruvananthapuram",
        "KOLLAM":   "Kollam",    "QUILON":   "Kollam",
        "PATHANAMTHITTA": "Pathanamthitta",
        "ALAPPUZHA": "Alappuzha", "ALLEPPEY": "Alappuzha",
        "KOTTAYAM": "Kottayam",
        "IDUKKI":   "Idukki",
        "ERNAKULAM": "Ernakulam", "COCHIN": "Ernakulam",
        "THRISSUR": "Thrissur",  "TRICHUR": "Thrissur",
        "PALAKKAD": "Palakkad",  "PALGHAT": "Palakkad",
        "MALAPPURAM": "Malappuram",
        "KOZHIKODE": "Kozhikode", "CALICUT": "Kozhikode",
        "WAYANAD": "Wayanad",
        "KANNUR":  "Kannur",     "CANNANORE": "Kannur",
        "KASARAGOD": "Kasaragod",
    }
    for key, district in district_names.items():
        if key in text:
            return district

    # Strategy 2: truncated district suffix at end of string
    for suffix, district in SUFFIX_TO_DISTRICT.items():
        if text.endswith(suffix) or text.endswith(suffix + "."):
            return district

    # Strategy 3: river name in location text
    for river, district in RIVER_TO_DISTRICT.items():
        if river in text:
            return district

    # Strategy 4: known place names (from previous script)
    place_names = {
        "ALUVA": "Ernakulam", "ELOOR": "Ernakulam", "KALADY": "Ernakulam",
        "IRUMPANAM": "Ernakulam", "BRAHMAPURAM": "Ernakulam",
        "MUPPATHADAM": "Ernakulam", "PATHALAM": "Ernakulam",
        "KALAMASSERY": "Ernakulam",
        "CHALAKUDY": "Thrissur", "KARUVANNUR": "Thrissur",
        "ARUVIKARA": "Thiruvananthapuram", "ARUVIPURAM": "Thiruvananthapuram",
        "AMARAVILA": "Thiruvananthapuram", "THIRUVALLOM": "Thiruvananthapuram",
        "THALIPARAMBA": "Kannur", "TALIPARAMBA": "Kannur",
        "PARASSINIKADAVU": "Kannur", "AZHEEKAL": "Kannur",
        "KUTTIPPURAM": "Malappuram", "HAJIRAPPALLY": "Malappuram",
        "TIRURRANGADI": "Malappuram",
        "THOTTAPPALLY": "Alappuzha", "THAKAZHY": "Alappuzha",
        "KALLOOPARA": "Pathanamthitta", "THUNDRA": "Pathanamthitta",
        "THUMPAMON": "Pathanamthitta", "KALLARAKADAVU": "Pathanamthitta",
        "PUNALO": "Kollam",
        "KOOLIMADU": "Malappuram",    # Chaliyar river area
        "CHUNGAPALLY": "Malappuram",  # Chaliyar river area
        "VETTIKKATTUMUKKU": "Ernakulam",
        "MOONNATTUMUKKU": "Thiruvananthapuram",
        "KANAYANKODE": "Kozhikode",
        "MERUVAMBAYI": "Kannur",
        "RAYAROM": "Kasaragod",
        "CHANDAPURA": "Kannur",
        "ANGADIMOGARU": "Kasaragod",
        "BAJRAKKARA": "Kasaragod",
        "NARAGAMPALLY": "Idukki",
        "MANIYANKODE": "Wayanad",
    }
    for place, district in place_names.items():
        if place in text:
            return district

    return "Unknown"


# ── Apply the improved function only to Unknown rows ──────────────────────
mask = df["district"] == "Unknown"
df.loc[mask, "district"] = df.loc[mask, "monitoring_location"].apply(
    full_district_guess
)

# ── Save updated CSV ──────────────────────────────────────────────────────
df.to_csv("data/raw/water_quality/kerala_water_quality_2024.csv", index=False)

# ── Summary ───────────────────────────────────────────────────────────────
remaining = (df["district"] == "Unknown").sum()
print(f"Unknown districts after fix: {remaining}")
print(f"\nFinal district distribution:")
print(df["district"].value_counts().to_string())

if remaining > 0:
    print(f"\nStill-unknown locations:")
    for loc in df[df["district"] == "Unknown"]["monitoring_location"].tolist():
        print(f"  - {loc}")

print(f"\nDone! kerala_water_quality_2024.csv updated.")
