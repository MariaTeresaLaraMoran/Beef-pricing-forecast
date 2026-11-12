"""Download the US boxed beef cutout from USDA AMS Market News (MPR Datamart API).

Report 2453 (LM_XB403): National Daily Boxed Beef Cutout & Boxed Beef Cuts,
Negotiated Sales, afternoon. The "Current Cutout Values" section holds the daily
Choice and Select cutout values in US dollars per 100 pounds. No API key is needed.

Usage:
    python src/fetch_usda_boxed_beef.py --inspect      # list sections and columns
    python src/fetch_usda_boxed_beef.py                # download the full history
    python src/fetch_usda_boxed_beef.py --start-year 2015

Output:
    data/raw/usda_boxed_beef_cutout.csv   (all columns the API returns, one row per day)
"""

import argparse
import sys
import time
from datetime import date
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

REPORT_ID = 2453
SECTION = "Current Cutout Values"
BASE_URL = f"https://mpr.datamart.ams.usda.gov/services/v1.1/reports/{REPORT_ID}"
ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "raw" / "usda_boxed_beef_cutout.csv"


def get_json(url: str) -> dict:
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    return response.json()


def inspect() -> None:
    """Print the report's sections and the columns of the section we use."""
    overview = get_json(BASE_URL)
    print("Top-level keys:", list(overview.keys()))
    print("Sections:", overview.get("reportSections"))

    today = date.today()
    sample_url = (
        f"{BASE_URL}/{quote(SECTION)}"
        f"?q=report_date={today.replace(day=1):%m/%d/%Y}:{today:%m/%d/%Y}"
    )
    sample = get_json(sample_url)
    results = sample.get("results", [])
    print(f"\nSection '{SECTION}': {len(results)} rows this month")
    if results:
        print("Columns:", list(results[0].keys()))
        print("Example row:", results[0])


def fetch_year(year: int) -> list[dict]:
    end = min(date(year, 12, 31), date.today())
    url = (
        f"{BASE_URL}/{quote(SECTION)}"
        f"?q=report_date=01/01/{year}:{end:%m/%d/%Y}"
    )
    return get_json(url).get("results", [])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inspect", action="store_true", help="show sections and columns, then exit")
    parser.add_argument("--start-year", type=int, default=2000)
    args = parser.parse_args()

    if args.inspect:
        inspect()
        return

    rows: list[dict] = []
    for year in range(args.start_year, date.today().year + 1):
        try:
            year_rows = fetch_year(year)
        except requests.HTTPError as error:
            print(f"{year}: request failed ({error}); skipping")
            continue
        print(f"{year}: {len(year_rows)} rows")
        rows.extend(year_rows)
        time.sleep(1)  # be polite to the server

    if not rows:
        sys.exit("No data returned. Run with --inspect to check the section name.")

    df = pd.DataFrame(rows)
    if "report_date" in df.columns:
        df["report_date"] = pd.to_datetime(df["report_date"], format="%m/%d/%Y", errors="coerce")
        df = df.sort_values("report_date").drop_duplicates(subset=["report_date"])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"\nSaved {len(df):,} rows to {OUT_PATH.relative_to(ROOT)}")
    print("Columns:", list(df.columns))
    if "report_date" in df.columns:
        print(f"Date range: {df['report_date'].min().date()} to {df['report_date'].max().date()}")


if __name__ == "__main__":
    main()
