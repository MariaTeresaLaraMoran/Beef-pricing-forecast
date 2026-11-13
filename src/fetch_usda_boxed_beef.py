"""Download the US boxed beef cutout from USDA AMS MyMarketNews (MARS API).

Report slug 2453 (LM_XB403): National Daily Boxed Beef Cutout & Boxed Beef Cuts,
Negotiated Sales, afternoon. The "Current Cutout Values" section holds the daily
Choice and Select cutout values in US dollars per 100 pounds.

Needs a free MyMarketNews API key in .env as USDA_MMN_API_KEY (see .env.example).
The key is sent as the HTTP basic-auth username, with no password.

Note: USDA's older Datamart service (mpr.datamart.ams.usda.gov) serves the same
report without a key, but it was unreachable when this project was built.

Usage:
    python src/fetch_usda_boxed_beef.py --inspect       # list the report's sections
    python src/fetch_usda_boxed_beef.py --inspect --section "Current Cutout Values"
    python src/fetch_usda_boxed_beef.py                 # download the full history
    python src/fetch_usda_boxed_beef.py --start-year 2015

Output:
    data/raw/usda_boxed_beef_cutout.csv
"""

import argparse
import os
import sys
import time
from datetime import date
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests
from dotenv import load_dotenv

BASE_URL = "https://marsapi.ams.usda.gov/services/v1.2/reports"
SLUG = 2453                        # National Daily Boxed Beef Cutout (LM_XB403)
SECTION = "Current Cutout Values"
FIRST_YEAR = 2000
ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "raw" / "usda_boxed_beef_cutout.csv"
TIMEOUT_SECONDS = 90


def get_json(url: str, key: str):
    """Call the MARS API. The key is the basic-auth username; the password is empty."""
    try:
        response = requests.get(url, auth=(key, ""), timeout=TIMEOUT_SECONDS)
    except requests.exceptions.Timeout:
        sys.exit(
            f"The USDA server did not answer within {TIMEOUT_SECONDS} seconds.\n"
            "This is a server or network problem, not a problem with the script."
        )
    if not response.ok:
        sys.exit(
            f"USDA returned HTTP {response.status_code} for {url}\n"
            f"Response: {response.text[:300]}\n"
            "If this is 401 or 403, check USDA_MMN_API_KEY in .env."
        )
    return response.json()


def inspect(key: str, section: str | None) -> None:
    """Show the report's sections, or the columns of one section."""
    if not section:
        data = get_json(f"{BASE_URL}/{SLUG}", key)
        record = data[0] if isinstance(data, list) and data else data
        print("Report:", record.get("report_title") or record.get("reportTitle"))
        print("Sections:", record.get("report_section") or record.get("reportSections"))
        print("\nRun again with --section \"<name>\" to see that section's columns.")
        return

    today = date.today()
    url = (
        f"{BASE_URL}/{SLUG}/{quote(section)}"
        f"?q=report_begin_date={today.replace(day=1):%m/%d/%Y}:{today:%m/%d/%Y}"
    )
    print("Requesting:", url)
    data = get_json(url, key)
    rows = data.get("results", data) if isinstance(data, dict) else data
    print(f"Rows this month: {len(rows)}")
    if rows:
        print("Columns:", list(rows[0].keys()))
        print("Example row:", rows[0])


def fetch_year(year: int, key: str) -> list[dict]:
    end = min(date(year, 12, 31), date.today())
    url = (
        f"{BASE_URL}/{SLUG}/{quote(SECTION)}"
        f"?q=report_begin_date=01/01/{year}:{end:%m/%d/%Y}"
    )
    data = get_json(url, key)
    return data.get("results", data) if isinstance(data, dict) else data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inspect", action="store_true", help="show the report's structure, then exit")
    parser.add_argument("--section", default=None, help="section name to inspect")
    parser.add_argument("--start-year", type=int, default=FIRST_YEAR)
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    key = (os.getenv("USDA_MMN_API_KEY") or "").strip().strip('"\'')
    if not key:
        sys.exit(
            "USDA_MMN_API_KEY is missing. Add it to .env (see .env.example).\n"
            "The key comes from your MyMarketNews account: https://mymarketnews.ams.usda.gov"
        )

    if args.inspect:
        inspect(key, args.section)
        return

    rows: list[dict] = []
    for year in range(args.start_year, date.today().year + 1):
        year_rows = fetch_year(year, key)
        print(f"{year}: {len(year_rows)} rows")
        rows.extend(year_rows)
        time.sleep(1)

    if not rows:
        sys.exit("No data returned. Run with --inspect to check the section name.")

    df = pd.DataFrame(rows)
    date_column = next((c for c in ("report_date", "report_begin_date") if c in df.columns), None)
    if date_column:
        df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
        df = df.sort_values(date_column).drop_duplicates(subset=[date_column])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"\nSaved {len(df):,} rows to {OUT_PATH.relative_to(ROOT)}")
    print("Columns:", list(df.columns))
    if date_column:
        print(f"Date range: {df[date_column].min().date()} to {df[date_column].max().date()}")


if __name__ == "__main__":
    main()
