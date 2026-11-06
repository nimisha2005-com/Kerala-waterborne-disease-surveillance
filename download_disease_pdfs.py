"""
Download Kerala DHS "Data on Communicable Diseases" PDFs automatically.

Step 1 (list only, downloads nothing):
    python download_disease_pdfs.py

Step 2 (actually download):
    python download_disease_pdfs.py --download

Optional: only certain years
    python download_disease_pdfs.py --download --years 2018 2019 2020
"""
import argparse
import os
import re
import time
from urllib.parse import urljoin, urlparse, unquote

import requests
from bs4 import BeautifulSoup

START_URL = "https://dhs.kerala.gov.in/en/data-on-communicable-diseases"
DOMAIN = "dhs.kerala.gov.in"
OUT_DIR = "data/raw/disease/pdfs"
MAX_DEPTH = 2          # start page -> year/sub pages -> PDFs
DELAY_SEC = 1.0        # be polite to the government server
HEADERS = {"User-Agent": "Mozilla/5.0 (student research project)"}

YEAR_RE = re.compile(r"20(1[1-9]|2[0-5])")
FOLLOW_WORDS = ("communicable", "data", "disease", "idsp", "daily", "report")


def get_soup(url):
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")


def should_follow(url, text):
    """Only follow internal pages that look related to the disease data."""
    p = urlparse(url)
    if p.netloc and DOMAIN not in p.netloc:
        return False
    s = (url + " " + text).lower()
    return bool(YEAR_RE.search(s)) or any(w in s for w in FOLLOW_WORDS)


def crawl():
    pdfs = {}            # pdf_url -> (link_text, found_on_page)
    seen = set()
    queue = [(START_URL, 0)]

    while queue:
        url, depth = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)
        print(f"[depth {depth}] scanning {url}")
        try:
            soup = get_soup(url)
        except Exception as e:
            print(f"   ! failed: {e}")
            continue
        time.sleep(DELAY_SEC)

        for a in soup.find_all("a", href=True):
            link = urljoin(url, a["href"]).split("#")[0]
            text = a.get_text(" ", strip=True)
            if link.lower().endswith(".pdf"):
                pdfs.setdefault(link, (text, url))
            elif depth < MAX_DEPTH and link not in seen and should_follow(link, text):
                queue.append((link, depth + 1))

    return pdfs


def guess_year(url, text):
    m = YEAR_RE.search(text) or YEAR_RE.search(unquote(url))
    return m.group(0) if m else "unknown_year"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true", help="actually download files")
    ap.add_argument("--years", nargs="*", help="only these years, e.g. 2018 2019")
    args = ap.parse_args()

    pdfs = crawl()

    rows = []
    for url, (text, page) in pdfs.items():
        year = guess_year(url, text)
        if args.years and year not in args.years:
            continue
        rows.append((year, url, text))
    rows.sort()

    print("\n" + "=" * 60)
    print(f"PDFs found: {len(rows)}")
    by_year = {}
    for y, _, _ in rows:
        by_year[y] = by_year.get(y, 0) + 1
    for y in sorted(by_year):
        print(f"  {y}: {by_year[y]} files")

    # Save the full list so you can inspect it
    os.makedirs(OUT_DIR, exist_ok=True)
    list_path = os.path.join(OUT_DIR, "_pdf_list.csv")
    with open(list_path, "w", encoding="utf-8") as f:
        f.write("year,url,link_text\n")
        for y, u, t in rows:
            f.write(f'{y},"{u}","{t.replace(chr(34), "")}"\n')
    print(f"\nFull list saved to: {list_path}")

    if not args.download:
        print("\nNothing downloaded (list mode). Run with --download to fetch files.")
        return

    for i, (year, url, _) in enumerate(rows, 1):
        folder = os.path.join(OUT_DIR, year)
        os.makedirs(folder, exist_ok=True)
        name = os.path.basename(unquote(urlparse(url).path))
        path = os.path.join(folder, name)
        if os.path.exists(path):
            continue  # already downloaded -> safe to re-run
        try:
            r = requests.get(url, headers=HEADERS, timeout=60)
            r.raise_for_status()
            with open(path, "wb") as f:
                f.write(r.content)
            print(f"[{i}/{len(rows)}] saved {path}")
        except Exception as e:
            print(f"[{i}/{len(rows)}] ! failed {url}: {e}")
        time.sleep(DELAY_SEC)

    print("\nDone.")


if __name__ == "__main__":
    main()
