"""
Look inside ONE downloaded disease PDF so we can see how its tables are laid out.

Usage:
    python explore_disease_pdf.py "data/raw/disease/pdfs/2022/somefile.pdf"
"""
import sys
import pdfplumber

if len(sys.argv) < 2:
    print('Usage: python explore_disease_pdf.py "path/to/file.pdf"')
    sys.exit(1)

path = sys.argv[1]
with pdfplumber.open(path) as pdf:
    print(f"File: {path}")
    print(f"Pages: {len(pdf.pages)}")
    for pno, page in enumerate(pdf.pages[:3], 1):        # first 3 pages only
        print("\n" + "=" * 60 + f"\nPAGE {pno}\n" + "=" * 60)
        text = page.extract_text() or ""
        print("--- first 15 text lines ---")
        print("\n".join(text.splitlines()[:15]) or "(no text - may be a scanned image)")
        tables = page.extract_tables()
        print(f"\n--- {len(tables)} table(s) found ---")
        for tno, table in enumerate(tables, 1):
            print(f"\nTable {tno}: {len(table)} rows")
            for row in table[:8]:                            # first 8 rows
                print("   ", row)
