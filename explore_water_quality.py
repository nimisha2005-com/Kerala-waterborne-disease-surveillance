import pdfplumber
import pandas as pd
import os

# ── PDF file paths ─────────────────────────────────────────────────────────
PDF_FILES = {
    "rivers":         r"data\raw\water_quality\WQuality_River-Data-2024.pdf",
    "medium_minor":   r"data\raw\water_quality\Water_Quality_data_of_Med_Min_River_2024.pdf",
}

os.makedirs("data/raw/water_quality", exist_ok=True)

def extract_kerala_from_pdf(pdf_path, label):
    """Extract all tables from a PDF and filter Kerala rows."""
    print(f"\n{'='*60}")
    print(f"Processing: {label}")
    print(f"File: {pdf_path}")
    print(f"{'='*60}")

    if not os.path.exists(pdf_path):
        print(f"  ✗ File not found: {pdf_path}")
        print(f"    Make sure the file is in: data\\raw\\water_quality\\")
        return None

    kerala_rows = []

    with pdfplumber.open(pdf_path) as pdf:
        print(f"  Total pages in PDF: {len(pdf.pages)}")

        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables()

            for table in tables:
                if not table:
                    continue

                for row in table:
                    if row is None:
                        continue

                    # Convert all cells to strings and clean up
                    clean_row = [
                        str(cell).strip() if cell is not None else ""
                        for cell in row
                    ]

                    # Join the row as text to search for "Kerala"
                    row_text = " ".join(clean_row).lower()

                    if "kerala" in row_text:
                        kerala_rows.append({
                            "page":     page_num,
                            "raw_row":  clean_row,
                        })

    print(f"  Kerala rows found: {len(kerala_rows)}")

    if kerala_rows:
        print(f"\n  Preview of first 3 Kerala rows:")
        for i, row in enumerate(kerala_rows[:3]):
            print(f"    Page {row['page']}: {row['raw_row']}")

    return kerala_rows


# ── Run extraction on both PDFs ────────────────────────────────────────────
all_results = {}

for label, path in PDF_FILES.items():
    rows = extract_kerala_from_pdf(path, label)
    all_results[label] = rows

# ── Summary ───────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
for label, rows in all_results.items():
    count = len(rows) if rows else 0
    status = "✓" if count > 0 else "✗"
    print(f"  {status} {label}: {count} Kerala rows found")

print()
print("Next step: Run extract_water_quality_save.py to save the data.")
