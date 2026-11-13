"""Download weekly US beef export sales to Mexico from USDA FAS (Open Data ESR API).

Export Sales Reporting (ESR) data, weekly, available from 1999.
Commodity 1701 = fresh, chilled or frozen muscle cuts of beef. Country 2010 = Mexico.

Needs a free API key from https://api.data.gov/signup/ , saved in .env as FAS_API_KEY
(see .env.example).

Usage:
    python src/fetch_usda_fas_exports.py --inspect      # check key, codes and columns
    python src/fetch_usda_fas_exports.py               # download 1999 to this year
    python src/fetch_usda_fas_exports.py --start-year 2020

Output:
    data/raw/usda_fas_beef_exports_mexico.csv
"""

import argparse
import os
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

BASE_URL = "https://api.fas.usda.gov/api/esr"
BEEF_CODE = 1701        # fresh, chilled or frozen muscle cuts of beef
MEXICO_CODE = 2010      # Mexico
FIRST_YEAR = 1999       # earliest year ESR data is published for
ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "raw" / "usda_fas_beef_exports_mexico.csv"
TIMEOUT_SECONDS = 60


def get_json(path: str, key: str):
    """Call the ESR API. The key goes in a header; two header names are tried.

    The FAS specification names the header API_KEY, while api.data.gov documents
    X-Api-Key. Trying both avoids a dead end if one is rejected.
    """
    url = f"{BASE_URL}{path}"
    last = None
    for header in ("API_KEY", "X-Api-Key"):
        try:
            response = requests.get(url, headers={header: key}, timeout=TIMEOUT_SECONDS)
        except requests.exceptions.Timeout:
            sys.exit(f"The FAS server did not answer within {TIMEOUT_SECONDS} seconds. Try again later.")
        if response.ok:
            return response.json()
        last = response
        if response.status_code not in (401, 403):
            break   # not an authentication problem, so another header will not help

    sys.exit(
        f"FAS returned HTTP {last.status_code} for {url}\n"
        f"Response: {last.text[:300]}\n"
        "If this says the key is invalid, check FAS_API_KEY in .env. "
        "A free key comes from https://api.data.gov/signup/"
    )


def inspect(key: str) -> None:
    """Confirm the key works and that the beef and Mexico codes are right."""
    commodities = get_json("/commodities", key)
    beef = [c for c in commodities if "BEEF" in str(c.get("commodityName", "")).upper()]
    print("Beef commodities:")
    for c in beef:
        print("  ", c.get("commodityCode"), "-", c.get("commodityName"))

    countries = get_json("/countries", key)
    mexico = [c for c in countries if "MEXICO" in str(c.get("countryName", "")).upper()]
    print("\nMexico:")
    for c in mexico:
        print("  ", c.get("countryCode"), "-", c.get("countryName"))

    year = date.today().year
    rows = get_json(
        f"/exports/commodityCode/{BEEF_CODE}/countryCode/{MEXICO_CODE}/marketYear/{year}", key
    )
    print(f"\nBeef to Mexico, market year {year}: {len(rows)} weekly records")
    if rows:
        print("Columns:", list(rows[0].keys()))
        print("First record:", rows[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inspect", action="store_true", help="check key, codes and columns, then exit")
    parser.add_argument("--start-year", type=int, default=FIRST_YEAR)
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    key = (os.getenv("FAS_API_KEY") or "").strip().strip('"\'')
    if not key:
        sys.exit(
            "FAS_API_KEY is missing. Add it to .env (see .env.example).\n"
            "A free key comes from https://api.data.gov/signup/"
        )

    if args.inspect:
        inspect(key)
        return

    frames = []
    for year in range(args.start_year, date.today().year + 1):
        rows = get_json(
            f"/exports/commodityCode/{BEEF_CODE}/countryCode/{MEXICO_CODE}/marketYear/{year}", key
        )
        print(f"{year}: {len(rows)} weekly records")
        if rows:
            frames.append(pd.DataFrame(rows))
        time.sleep(0.5)

    if not frames:
        sys.exit("No data returned. Run with --inspect to check the key and the codes.")

    df = pd.concat(frames, ignore_index=True)
    if "weekEndingDate" in df.columns:
        df["weekEndingDate"] = pd.to_datetime(df["weekEndingDate"], errors="coerce")
        df = df.sort_values("weekEndingDate").drop_duplicates(subset=["weekEndingDate"])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"\nSaved {len(df):,} rows to {OUT_PATH.relative_to(ROOT)}")
    print("Columns:", list(df.columns))
    if "weekEndingDate" in df.columns:
        print(f"Date range: {df['weekEndingDate'].min().date()} to {df['weekEndingDate'].max().date()}")


if __name__ == "__main__":
    main()
