"""
Extract REAL Kerala communicable-disease data (DHS Kerala / IDSP annual reports, 2011-2025).

Input : data/raw/disease/2011.pdf ... 2025.pdf
Output: data/processed/disease/
    kerala_disease_district_yearly_long.csv  - every disease, every district, every year
    kerala_disease_district_yearly.csv       - waterborne diseases, one row per district-year
    kerala_disease_state_monthly.csv         - waterborne diseases, one row per month (whole state)
    extraction_report.txt                    - validation checks

Run:  python extract_disease_data.py
"""
import glob
import os
import re

import pandas as pd
import pdfplumber

RAW_DIR = "data/raw/disease"
OUT_DIR = "data/processed/disease"

# ── District name aliases (letters de-duplicated, see collapse()) ─────────
DISTRICT_ALIASES = {
    "Thiruvananthapuram": ["THIRUVANANTHAPURAM", "THIRUVANATHAPURAM", "TRIVANDRUM", "TVM"],
    "Kollam": ["KOLLAM", "KLM", "QUILON"],
    "Pathanamthitta": ["PATHANAMTHITTA", "PTA"],
    "Idukki": ["IDUKKI", "IDK"],
    "Kottayam": ["KOTTAYAM", "KTM"],
    "Alappuzha": ["ALAPPUZHA", "ALP", "ALLEPPEY"],
    "Ernakulam": ["ERNAKULAM", "EARNAKULAM", "ERANAKULAM", "EKM"],
    "Thrissur": ["THRISSUR", "TRISSUR", "TRICHUR", "TSR"],
    "Palakkad": ["PALAKKAD", "PKD"],
    "Malappuram": ["MALAPPURAM", "MLP"],
    "Kozhikode": ["KOZHIKODE", "KOZHIKKODE", "KKD", "CALICUT"],
    "Wayanad": ["WAYANAD", "WYD"],
    "Kannur": ["KANNUR", "KNR"],
    "Kasaragod": ["KASARAGOD", "KASARAGODE", "KASARGODE", "KASARGOD", "KSD"],
}


def collapse(s):
    """Upper-case, keep letters only, remove repeated letters (fixes 'KKOLLAM', 'MMaallaa...')."""
    s = re.sub(r"[^A-Z]", "", s.upper())
    return re.sub(r"(.)\1+", r"\1", s)


ALIAS_LOOKUP = {collapse(a): d for d, aliases in DISTRICT_ALIASES.items() for a in aliases}

MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}

NUM = r"(?:[\d,]+\*?|-)"
# one record: "<sl.no> <disease name> <cases> <deaths>", possibly two per line
RECORD_RE = re.compile(
    rf"(\d{{1,2}})\s+([A-Za-z][^\n]*?)\s+({NUM})\s+({NUM})(?=\s+\d{{1,2}}\s+[A-Za-z]|\s*$)")
VALUES_ONLY_RE = re.compile(rf"^\s*({NUM})\s+({NUM})\s*$")        # 2020-21: "460 4"
NAME_ONLY_RE = re.compile(r"^\s*(\d{1,2})\s+([A-Za-z].*?)\s*$")   # 2020-21: "1 Dengue Fever"


def to_int(tok):
    tok = tok.replace(",", "").replace("*", "").strip()
    return 0 if tok in ("-", "") else int(tok)


def district_header(line):
    """Return list of districts if the line is ONLY district names (profile page header)."""
    if re.search(r"\d", line):
        return None
    tokens = line.split()
    if not tokens:
        return None
    found = [ALIAS_LOOKUP.get(collapse(t)) for t in tokens]
    return found if all(found) else None


def normalize_disease(raw):
    r = raw.lower()
    prob = "prob" in r
    if "suspected cholera" in r or "sus. cholera" in r:
        return "cholera_suspected"
    if re.search(r"\badd\b|diarrh", r):
        return "add"
    if "typhoid" in r:
        return "typhoid"
    if "cholera" in r:
        return "cholera"
    if "shigella" in r:
        return "shigella"
    if re.search(r"hepatitis\s*-?\s*a\b|hep\w*\s*a\b", r):
        return "hepatitis_a_probable" if prob else "hepatitis_a"
    if re.search(r"hepatitis\s*-?\s*e\b|hep\w*\s*e\b", r):
        return "hepatitis_e"
    if re.search(r"hepatitis\s*-?\s*b\b", r):
        return "hepatitis_b"
    if re.search(r"hepatitis\s*-?\s*c\b", r):
        return "hepatitis_c"
    if "lepto" in r:
        return "leptospirosis_probable" if prob else "leptospirosis"
    if "dengue" in r:
        return "dengue_probable" if prob else "dengue"
    if "food" in r and "poison" in r:
        return "food_poisoning"
    if "scrub" in r:
        return "scrub_typhus"
    if "malaria" in r:
        return "malaria"
    if "chick" in r or "c.pox" in r or "c. pox" in r:
        return "chickenpox"
    return re.sub(r"[^a-z0-9]+", "_", r).strip("_")


# ═══════════════════════════════════════════════════════════════════════════
# 1. District / state annual totals (from the per-district "profile" pages)
# ═══════════════════════════════════════════════════════════════════════════
def parse_profiles(lines, year):
    records = []
    current = None          # list of 1 or 2 district names
    pending_vals = None     # for 2020-21 layout (values line before name line)

    for line in lines:
        low = line.lower()
        if "month wise" in low or "district wise" in low:
            current, pending_vals = None, None
            continue
        if ("communicable disease" in low and "data" in low) or "kerala state" in low \
                or line.strip().upper() in ("KERALA",):
            current, pending_vals = ["Kerala"], None
            continue
        hdr = district_header(line)
        if hdr:
            current, pending_vals = hdr, None
            continue
        if current is None or re.search(r"\b(OP|IP)\b", line):
            continue

        m = VALUES_ONLY_RE.match(line)
        if m:
            pending_vals = (to_int(m.group(1)), to_int(m.group(2)))
            continue
        if pending_vals:
            m = NAME_ONLY_RE.match(line)
            if m and not RECORD_RE.search(line):
                records.append((year, current[0], m.group(2).strip(), *pending_vals))
                pending_vals = None
                continue

        found = RECORD_RE.findall(line)
        for i, (_, name, c, d) in enumerate(found[:len(current)]):
            name = name.strip()
            if collapse(name) in ALIAS_LOOKUP or name[:3].upper() in MONTHS:
                continue   # a summary-table row, not a profile row
            records.append((year, current[i], name, to_int(c), to_int(d)))
    return records


# ═══════════════════════════════════════════════════════════════════════════
# 2. State monthly tables
# ═══════════════════════════════════════════════════════════════════════════
def parse_month_blocks(lines):
    """Return list of blocks: {'months': {1: [vals], ...}, 'total': [vals]}"""
    blocks, cur = [], None
    for line in lines:
        toks = line.split()
        if toks and re.fullmatch(r"\d{1,2}", toks[0]) and len(toks) > 1:
            toks = toks[1:]
        # a new table starts at its January row (some tables have no "Month wise" title)
        if toks and toks[0][:3].upper() == "JAN" and len(toks) > 1 \
                and all(re.fullmatch(NUM, v) for v in toks[1:]):
            cur = {"months": {}, "total": None}
            blocks.append(cur)
        if cur is None:
            continue
        toks = line.split()
        if toks and re.fullmatch(r"\d{1,2}", toks[0]) and len(toks) > 1:
            toks = toks[1:]                         # 2011: "1 January 57 6 ..."
        if not toks:
            continue
        head = toks[0][:3].upper()
        vals = toks[1:]
        if head in MONTHS and vals and all(re.fullmatch(NUM, v) for v in vals):
            cur["months"][MONTHS[head]] = [to_int(v) for v in vals]
        elif head == "TOT" and vals and all(re.fullmatch(NUM, v) for v in vals):
            cur["total"] = [to_int(v) for v in vals]
            cur = None
    return [b for b in blocks if b["total"] and len(b["months"]) >= 11]


def match_month_columns(blocks, state_totals, year, log):
    """Find each disease's Case/Death columns by matching the TOTAL row to annual state totals."""
    rows = {}   # (month, disease) -> cases
    used = [set() for _ in blocks]
    for disease, (cases, deaths) in sorted(state_totals.items(), key=lambda x: -x[1][0]):
        if cases == 0:
            for mth in range(1, 13):
                rows[(mth, disease)] = 0
            continue
        hits = [(b, i) for b, blk in enumerate(blocks)
                for i in range(len(blk["total"]) - 1)
                if blk["total"][i] == cases and blk["total"][i + 1] == deaths
                and i not in used[b] and i + 1 not in used[b]]
        if not hits:   # some tables only give a single column; try cases alone
            hits = [(b, i) for b, blk in enumerate(blocks)
                    for i in range(len(blk["total"]))
                    if blk["total"][i] == cases and i not in used[b]]
        if not hits:
            log.append(f"  {year}: monthly table not found for {disease} (annual {cases})")
            continue
        if len(hits) > 1:
            log.append(f"  {year}: {disease} ambiguous ({len(hits)} matches) - took first")
        b, i = hits[0]
        used[b].update({i, i + 1})
        for mth, vals in blocks[b]["months"].items():
            if i < len(vals):
                rows[(mth, disease)] = vals[i]
    return rows


# ═══════════════════════════════════════════════════════════════════════════
WATERBORNE = ["add", "typhoid", "cholera", "hepatitis_a", "hepatitis_a_probable",
              "hepatitis_e", "shigella", "leptospirosis", "leptospirosis_probable"]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_records, monthly_rows, log = [], [], []

    for path in sorted(glob.glob(os.path.join(RAW_DIR, "*.pdf"))):
        year = int(re.search(r"(20\d\d)", os.path.basename(path)).group(1))
        with pdfplumber.open(path) as pdf:
            lines = []
            for page in pdf.pages:
                # dedupe_chars() removes letters printed twice (bold text glitch: "NNoovveemmbbeerr")
                lines += (page.dedupe_chars().extract_text() or "").splitlines()

        recs = parse_profiles(lines, year)
        all_records += recs

        # State totals (normalized). Use Kerala summary page, fallback to sum of 14 districts.
        state = {}
        for _, dist, name, c, d in recs:
            if dist == "Kerala":
                key = normalize_disease(name)
                if key in WATERBORNE and key not in state:
                    state[key] = (c, d)
        for key in WATERBORNE:
            if key not in state:
                c_sum = sum(c for _, dist, name, c, _ in recs if dist != "Kerala" and normalize_disease(name) == key)
                d_sum = sum(d for _, dist, name, _, d in recs if dist != "Kerala" and normalize_disease(name) == key)
                if c_sum > 0:
                    state[key] = (c_sum, d_sum)

        blocks = parse_month_blocks(lines)
        month_vals = match_month_columns(blocks, state, year, log)
        for (mth, dis), v in month_vals.items():
            monthly_rows.append({"year": year, "month": mth, "disease": dis, "cases": v})

        n_dist = len({r[1] for r in recs if r[1] != "Kerala"})
        print(f"{year}: {len(recs):4d} records | districts found: {n_dist:2d} | "
              f"month tables: {len(blocks)}")
        if n_dist != 14:
            log.append(f"  {year}: only {n_dist} districts found!")

    # ── Long table ────────────────────────────────────────────────────────
    long = pd.DataFrame(all_records, columns=["year", "district", "disease_raw", "cases", "deaths"])
    long["disease"] = long["disease_raw"].apply(normalize_disease)
    long = long.drop_duplicates(["year", "district", "disease"], keep="first")
    long["source"] = "DHS Kerala IDSP annual report"

    # ── Fill a district whose page is missing from the PDF (e.g. Alappuzha 2025) ──
    # value = Kerala state total - sum of the other 13 districts  (marked as derived)
    all_d = set(DISTRICT_ALIASES)
    extra = []
    for y, grp in long.groupby("year"):
        missing = all_d - set(grp["district"])
        if len(missing) != 1:
            continue
        miss = missing.pop()
        st = grp[grp["district"] == "Kerala"].set_index("disease")
        others = grp[grp["district"] != "Kerala"].groupby("disease")[["cases", "deaths"]].sum()
        for dis, row in st.iterrows():
            if dis in others.index:
                c = max(int(row["cases"] - others.loc[dis, "cases"]), 0)
                d = max(int(row["deaths"] - others.loc[dis, "deaths"]), 0)
                extra.append({"year": y, "district": miss, "disease_raw": row["disease_raw"],
                              "cases": c, "deaths": d, "disease": dis,
                              "source": "DERIVED: state total minus other districts"})
        log.append(f"  {y}: {miss} page missing in PDF -> derived from state total")
    if extra:
        long = pd.concat([long, pd.DataFrame(extra)], ignore_index=True)
    long.to_csv(os.path.join(OUT_DIR, "kerala_disease_district_yearly_long.csv"), index=False)

    # ── Wide district-year table (waterborne only) ────────────────────────
    dist = long[(long["district"] != "Kerala") & (long["disease"].isin(WATERBORNE))]
    cases = dist.pivot_table(index=["district", "year"], columns="disease",
                             values="cases", aggfunc="sum")
    deaths = dist.pivot_table(index=["district", "year"], columns="disease",
                              values="deaths", aggfunc="sum")
    wide = cases.add_suffix("_cases").join(deaths.add_suffix("_deaths")).reset_index()
    for base in ["hepatitis_a", "leptospirosis"]:
        cols = [c for c in [f"{base}_cases", f"{base}_probable_cases"] if c in wide]
        wide[f"{base}_total_cases"] = wide[cols].fillna(0).sum(axis=1)
    wide = wide.sort_values(["year", "district"])
    derived = set(zip(*long.loc[long["source"].str.startswith("DERIVED"), ["district", "year"]].values.T)) \
        if (long["source"].str.startswith("DERIVED")).any() else set()
    wide["is_derived"] = [(d, y) in derived for d, y in zip(wide["district"], wide["year"])]
    wide.to_csv(os.path.join(OUT_DIR, "kerala_disease_district_yearly.csv"), index=False)

    # ── State monthly table ───────────────────────────────────────────────
    mon = pd.DataFrame(monthly_rows)
    mon = mon.pivot_table(index=["year", "month"], columns="disease",
                          values="cases", aggfunc="first").reset_index()
    mon.columns.name = None
    mon = mon.rename(columns={c: f"{c}_cases" for c in mon.columns if c not in ("year", "month")})
    mon.to_csv(os.path.join(OUT_DIR, "kerala_disease_state_monthly.csv"), index=False)

    # ── Validation: sum of districts vs state total ───────────────────────
    log.append("\nCHECK: sum of 14 districts vs Kerala total (should match)")
    state = long[long["district"] == "Kerala"].set_index(["year", "disease"])["cases"]
    dsum = long[long["district"] != "Kerala"].groupby(["year", "disease"])["cases"].sum()
    for (y, dis) in sorted(state.index):
        if dis in WATERBORNE and (y, dis) in dsum.index:
            s, d = state[(y, dis)], dsum[(y, dis)]
            flag = "OK " if s == d else ("~  " if abs(s - d) <= max(2, 0.01 * s) else "XX ")
            log.append(f"  {flag}{y} {dis:24s} state={s:>8} districts={d:>8}")

    report = "\n".join(log)
    with open(os.path.join(OUT_DIR, "extraction_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    print("\n" + "=" * 60)
    print(f"District-year rows : {len(wide)}  (expected 15 years x 14 = 210)")
    print(f"State-month rows   : {len(mon)}  (expected 15 x 12 = 180)")
    bad = [l for l in log if l.strip().startswith("XX")]
    print(f"Validation mismatches: {len(bad)}  (details in extraction_report.txt)")
    print(f"Saved to: {OUT_DIR}/")


if __name__ == "__main__":
    main()
