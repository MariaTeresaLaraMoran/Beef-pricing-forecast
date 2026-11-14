"""Download US boxed beef prices from USDA AMS, report LM_XB403 (slug 2453).

Source: the Livestock Mandatory Reporting (LMR) Datamart
        https://mpr.datamart.ams.usda.gov/
No API key and no registration are required; unregistered callers are allowed
99,999 rows per request.

Note on which USDA service to use: boxed beef is an LMR report and lives on the
Datamart above, NOT on the MyMarketNews MARS API (marsapi.ams.usda.gov). Report 2453
is listed on MyMarketNews as a "Non Mars Location", and querying MARS for it returns
401 "User is not found" whatever key is supplied. The Datamart was returning HTTP 500
for part of 2026-11-12 and 13; if it fails, wait and retry rather than switching
service.

Two sections are collected:

  Current Cutout Values -- the daily Choice and Select cutout, USD per 100 lb
  Choice Cuts           -- 42 individual cuts per day, each with its IMPS code,
                           weighted average price, price range, number of trades
                           and pounds traded

The individual cuts are what pair with the Mexican SNIIM cuts; see
data/reference/us_cut_map.csv for the pairing.

The Datamart is slow on wide date ranges and times out on whole-year requests, so
this script asks for one quarter at a time, retries with backoff, and caches each
chunk to data/raw/usda_cache/. Re-running skips chunks already downloaded, so an
interrupted run can simply be restarted.

Usage:
    python src/fetch_usda_boxed_beef.py --inspect        # sections and columns
    python src/fetch_usda_boxed_beef.py                  # both sections, full history
    python src/fetch_usda_boxed_beef.py --chunk month     # if quarters still time out
    python src/fetch_usda_boxed_beef.py --start-year 2020
    python src/fetch_usda_boxed_beef.py --section "Select Cuts"

Output:
    data/raw/usda_boxed_beef_cutout.csv
    data/raw/usda_boxed_beef_choice_cuts.csv
    data/raw/usda_cache/<section>/<start>_<end>.csv    (resume cache)
"""

import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

BASE_URL = "https://mpr.datamart.ams.usda.gov/services/v1.1/reports"
SLUG = 2453
CUTOUT_SECTION = "Current Cutout Values"
CUTS_SECTION = "Choice Cuts"
FIRST_YEAR = 2000
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
TIMEOUT_SECONDS = 180
RETRIES = 4
CACHE = Path(__file__).resolve().parents[1] / "data" / "raw" / "usda_cache"
NUMERIC_COLUMNS = ["choice_600_900_current", "select_600_900_current", "number_trades",
                   "total_pounds", "price_range_low", "price_range_high", "weighted_average"]


def get_json(url: str, required: bool = True):
    """Fetch one URL, retrying on timeouts and server errors with growing backoff."""
    last = ""
    for attempt in range(RETRIES):
        try:
            response = requests.get(url, timeout=TIMEOUT_SECONDS)
            if response.status_code < 500:
                response.raise_for_status()
                return response.json()
            last = f"HTTP {response.status_code}"
        except requests.exceptions.Timeout:
            last = f"timed out after {TIMEOUT_SECONDS}s"
        except requests.RequestException as error:
            last = str(error)[:120]
        if attempt < RETRIES - 1:
            wait = 10 * (attempt + 1)
            print(f"    {last}; retrying in {wait}s ({attempt + 2}/{RETRIES})")
            time.sleep(wait)

    message = (f"USDA failed {RETRIES} times ({last}). The Datamart is intermittently "
               "unavailable; wait and re-run - completed chunks are cached and skipped.")
    if required:
        sys.exit(message)
    print(f"    giving up on this chunk: {last}")
    return None


def service_is_up() -> bool:
    """One small request to see whether the Datamart is answering at all.

    Without this, a pass over 108 windows during an outage takes nearly two hours,
    because each window burns its full retry backoff before being skipped. The probe
    turns that into a few seconds, so a retry loop can check often and catch a short
    window of availability.
    """
    today = date.today()
    url = section_url(CUTOUT_SECTION, today - timedelta(days=3), today)
    try:
        response = requests.get(url, timeout=45)
    except requests.RequestException as error:
        print(f"Datamart probe failed: {str(error)[:90]}")
        return False
    if response.status_code >= 500:
        print(f"Datamart probe failed: HTTP {response.status_code} (service is down)")
        return False
    if not response.ok:
        print(f"Datamart probe returned HTTP {response.status_code}")
        return False
    print("Datamart probe OK - the service is answering.")
    return True


def extract_rows(data) -> list[dict]:
    """Pull the record list out of whatever shape the Datamart returned.

    The service is inconsistent: a normal response is an object with a "results" list,
    but a date range it holds no data for can come back as a bare JSON string (such as
    a message), or as a list, or as null. Anything that is not a list of objects is
    treated as "no rows for this window".
    """
    if data is None:
        return []
    if isinstance(data, dict):
        rows = data.get("results", [])
    elif isinstance(data, list):
        rows = data
    else:
        return []        # a string or number: no data for this window
    return [row for row in rows if isinstance(row, dict)]


def section_url(section: str, start: date, end: date) -> str:
    return (f"{BASE_URL}/{SLUG}/{quote(section)}"
            f"?q=report_date={start:%m/%d/%Y}:{end:%m/%d/%Y}")


def inspect() -> None:
    today = date.today()
    data = get_json(section_url(CUTOUT_SECTION, today.replace(day=1), today))
    rows = extract_rows(data)
    print("Report:", (rows or [{}])[0].get("report_title", "?"))
    if isinstance(data, dict):
        print("Rows allowed per request:", data.get("stats", {}).get("userAllowedRows:"))
        print("\nSections available:")
        for name in data.get("reportSections", []):
            print("  -", name)

    cuts = extract_rows(get_json(section_url(CUTS_SECTION, today, today)))
    print(f"\n'{CUTS_SECTION}' for {today}: {len(cuts)} items")
    if cuts:
        print("Columns:", list(cuts[0].keys()))
        print("\nFirst 8 items:")
        for row in cuts[:8]:
            print(f"  {row.get('item_description','?'):38s} "
                  f"{row.get('weighted_average','-'):>10} USD/cwt  "
                  f"{row.get('total_pounds','-'):>12} lb")


def chunk_ranges(start_year: int, chunk: str) -> list[tuple[date, date]]:
    """Split the history into request windows the Datamart can actually serve."""
    today = date.today()
    step = {"year": 12, "quarter": 3, "month": 1}[chunk]
    ranges, year, month = [], start_year, 1
    while date(year, month, 1) <= today:
        start = date(year, month, 1)
        end_index = month + step - 1
        end_year, end_month = year + (end_index - 1) // 12, (end_index - 1) % 12 + 1
        following = date(end_year + end_month // 12, end_month % 12 + 1, 1)
        ranges.append((start, min(following - timedelta(days=1), today)))
        month += step
        if month > 12:
            month, year = month - 12, year + 1
    return ranges


def fetch_section(section: str, start_year: int, chunk: str) -> pd.DataFrame:
    cache_dir = CACHE / section.lower().replace(" ", "_")
    cache_dir.mkdir(parents=True, exist_ok=True)
    windows = chunk_ranges(start_year, chunk)
    print(f"  {len(windows)} {chunk} window(s) to cover {start_year}-{date.today().year}")

    frames, failed = [], []
    for start, end in windows:
        path = cache_dir / f"{start:%Y%m%d}_{end:%Y%m%d}.csv"
        if path.exists():
            cached = pd.read_csv(path, dtype=str)
            if len(cached):
                frames.append(cached)
            continue

        data = get_json(section_url(section, start, end), required=False)
        if data is None:
            failed.append((start, end))
            continue
        rows = extract_rows(data)
        pd.DataFrame(rows, dtype=str).to_csv(path, index=False)
        if rows:
            frames.append(pd.DataFrame(rows, dtype=str))
        print(f"  {start:%Y-%m} to {end:%Y-%m}: {len(rows):,} rows")
        time.sleep(1)

    if failed:
        print(f"\n  {len(failed)} window(s) failed and were skipped; re-run to retry them:")
        for start, end in failed[:5]:
            print(f"    {start} to {end}")

    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)

    df["report_date"] = pd.to_datetime(df["report_date"], format="%m/%d/%Y", errors="coerce")
    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            # Values arrive as strings with thousands separators; ".00" and blanks mean
            # no trades were reported that day.
            df[column] = pd.to_numeric(
                df[column].astype(str).str.replace(",", "", regex=False).str.strip(),
                errors="coerce")
    sort_keys = ["report_date"] + (["item_description"] if "item_description" in df.columns else [])
    return df.sort_values(sort_keys).drop_duplicates(subset=sort_keys).reset_index(drop=True)


def report(df: pd.DataFrame, path: Path, label: str) -> None:
    if df.empty:
        print(f"  no data returned for {label}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\nSaved {len(df):,} rows to {path.relative_to(ROOT)}")
    print(f"  date range: {df['report_date'].min().date()} to {df['report_date'].max().date()}")
    if "item_description" in df.columns:
        print(f"  distinct items: {df['item_description'].nunique()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inspect", action="store_true", help="show sections and columns, then exit")
    parser.add_argument("--start-year", type=int, default=FIRST_YEAR)
    parser.add_argument("--section", default=None,
                        help="fetch only this section instead of both defaults")
    parser.add_argument("--chunk", choices=["year", "quarter", "month"], default="quarter",
                        help="request window size; use 'month' if quarters time out")
    parser.add_argument("--no-probe", action="store_true",
                        help="skip the health check and attempt every window regardless")
    args = parser.parse_args()

    if not args.inspect and not args.no_probe and not service_is_up():
        print("Nothing attempted. Re-run later; cached windows are kept and skipped.")
        raise SystemExit(2)

    if args.inspect:
        inspect()
        return

    if args.section:
        print(f"Section: {args.section}")
        df = fetch_section(args.section, args.start_year, args.chunk)
        slug = args.section.lower().replace(" ", "_")
        report(df, RAW / f"usda_boxed_beef_{slug}.csv", args.section)
        return

    print(f"Section: {CUTOUT_SECTION}")
    report(fetch_section(CUTOUT_SECTION, args.start_year, args.chunk),
           RAW / "usda_boxed_beef_cutout.csv", CUTOUT_SECTION)

    print(f"\nSection: {CUTS_SECTION}")
    report(fetch_section(CUTS_SECTION, args.start_year, args.chunk),
           RAW / "usda_boxed_beef_choice_cuts.csv", CUTS_SECTION)


if __name__ == "__main__":
    main()
