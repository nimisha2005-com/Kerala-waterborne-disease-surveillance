import pandas as pd

# ── Load the saved CSV ─────────────────────────────────────────────────────
df = pd.read_csv("data/raw/water_quality/kerala_water_quality_2024.csv")
print(f"Rows before fix: {len(df)}")
print(f"Unknown districts before fix: {(df['district'] == 'Unknown').sum()}")

# ── Detailed place-name → district mapping ────────────────────────────────
# Many CPCB station names use town/village names instead of district names.
# This lookup maps those place names to the correct Kerala district.
PLACE_TO_DISTRICT = {
    # Ernakulam
    "ALUVA": "Ernakulam", "ELOOR": "Ernakulam", "KALADY": "Ernakulam",
    "PERUMBAVOOR": "Ernakulam", "MUVATTUPUZHA": "Ernakulam",
    "KOTHAMANGALAM": "Ernakulam", "ANGAMALY": "Ernakulam",
    "NORTH PARAVUR": "Ernakulam", "VYPIN": "Ernakulam",
    "EDAPALLY": "Ernakulam", "KAKKANAD": "Ernakulam",

    # Thrissur
    "CHALAKUDY": "Thrissur", "IRINJALAKUDA": "Thrissur",
    "GURUVAYUR": "Thrissur", "KODUNGALLUR": "Thrissur",
    "TRICHUR": "Thrissur", "KUNNAMKULAM": "Thrissur",

    # Palakkad
    "SHORNUR": "Palakkad", "OTTAPPALAM": "Palakkad",
    "MANNARKKAD": "Palakkad", "PATTAMBI": "Palakkad",
    "CHITTUR": "Palakkad", "NENMARA": "Palakkad",
    "ELACHIVAZHY": "Palakkad",

    # Malappuram
    "TIRUR": "Malappuram", "MANJERI": "Malappuram",
    "PERINTHALMANNA": "Malappuram", "PONNANI": "Malappuram",
    "NILAMBUR": "Malappuram", "TIRURRANGADI": "Malappuram",
    "TANUR": "Malappuram",

    # Kozhikode
    "VADAKARA": "Kozhikode", "KOYILANDY": "Kozhikode",
    "FEROKE": "Kozhikode", "QUILANDY": "Kozhikode",
    "CALICUT": "Kozhikode", "BEYPORE": "Kozhikode",

    # Kannur
    "THALASSERY": "Kannur", "PAYYANUR": "Kannur",
    "TELLICHERRY": "Kannur", "IRITTY": "Kannur",
    "TALIPARAMBA": "Kannur", "ETTIKULAM": "Kannur",

    # Kasaragod
    "HOSDURG": "Kasaragod", "KANJANGAD": "Kasaragod",
    "BEKAL": "Kasaragod", "NILESHWAR": "Kasaragod",

    # Wayanad
    "KALPETTA": "Wayanad", "MANANTHAVADY": "Wayanad",
    "SULTHAN BATHERY": "Wayanad", "BATHERY": "Wayanad",
    "MUTHANGA": "Wayanad", "MUTHANKARA": "Wayanad",
    "VYTHIRI": "Wayanad",

    # Kottayam
    "VAIKOM": "Kottayam", "CHANGANACHERRY": "Kottayam",
    "PALA": "Kottayam", "ETTUMANOOR": "Kottayam",
    "ERATTUPETTA": "Kottayam",

    # Alappuzha
    "CHENGANNUR": "Alappuzha", "AMBALAPPUZHA": "Alappuzha",
    "ALLEPPEY": "Alappuzha", "HARIPAD": "Alappuzha",
    "KAYAMKULAM": "Alappuzha", "CHERTHALA": "Alappuzha",
    "MARARIKULAM": "Alappuzha",

    # Pathanamthitta
    "THIRUVALLA": "Pathanamthitta", "RANNI": "Pathanamthitta",
    "KONNI": "Pathanamthitta", "ADOOR": "Pathanamthitta",
    "PANDALAM": "Pathanamthitta",

    # Kollam
    "PUNALUR": "Kollam", "KUNDARA": "Kollam",
    "KOTTARAKKARA": "Kollam", "KARUNAGAPPALLY": "Kollam",
    "PARAVUR": "Kollam", "CHAVARA": "Kollam",

    # Thiruvananthapuram
    "NEYYATTINKARA": "Thiruvananthapuram", "ATTINGAL": "Thiruvananthapuram",
    "VARKALA": "Thiruvananthapuram", "NEMOM": "Thiruvananthapuram",
    "TRIVANDRUM": "Thiruvananthapuram", "NEDUMANGAD": "Thiruvananthapuram",

    # Idukki
    "MUNNAR": "Idukki", "THODUPUZHA": "Idukki",
    "KATTAPPANA": "Idukki", "KUMILY": "Idukki",
    "ADIMALI": "Idukki",
}

def improved_district_guess(location_text):
    """Match district using expanded place name lookup."""
    text = str(location_text).upper()

    # First try: direct district name match
    district_names = [
        "THIRUVANANTHAPURAM", "KOLLAM", "PATHANAMTHITTA", "ALAPPUZHA",
        "KOTTAYAM", "IDUKKI", "ERNAKULAM", "THRISSUR", "PALAKKAD",
        "MALAPPURAM", "KOZHIKODE", "WAYANAD", "KANNUR", "KASARAGOD",
    ]
    for d in district_names:
        if d in text:
            return d.title()

    # Extra district name aliases
    aliases = {
        "TRIVANDRUM": "Thiruvananthapuram", "CALICUT": "Kozhikode",
        "TRICHUR": "Thrissur", "PALGHAT": "Palakkad",
        "ALLEPPEY": "Alappuzha", "QUILON": "Kollam",
        "COCHIN": "Ernakulam",
    }
    for alias, district in aliases.items():
        if alias in text:
            return district

    # Second try: place name lookup
    for place, district in PLACE_TO_DISTRICT.items():
        if place.upper() in text:
            return district

    return "Unknown"


# ── Apply improved district matching ─────────────────────────────────────
df["district"] = df["monitoring_location"].apply(improved_district_guess)

# ── Save the improved CSV ─────────────────────────────────────────────────
df.to_csv("data/raw/water_quality/kerala_water_quality_2024.csv", index=False)

# ── Summary ───────────────────────────────────────────────────────────────
print(f"\nAfter fix:")
print(f"Unknown districts remaining: {(df['district'] == 'Unknown').sum()}")
print(f"\nDistrict distribution:")
print(df["district"].value_counts().to_string())

# Show still-unknown rows so we can see what's left
unknown = df[df["district"] == "Unknown"]["monitoring_location"].tolist()
if unknown:
    print(f"\nStill-unknown locations ({len(unknown)}):")
    for loc in unknown:
        print(f"  - {loc}")
