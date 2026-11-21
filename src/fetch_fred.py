"""Download macroeconomic series from FRED and aggregate them to weeks.

Three series, each chosen because it has a mechanism that reaches US beef prices
within weeks rather than years. Series with a plausible story but a multi-month lag
(corn, cattle on feed) and monthly series generally are deliberately excluded - see
the note at the bottom of this docstring.

  DTWEXBGS   Nominal Broad U.S. Dollar Index, daily, 2006 onwards.
             Channel: a stronger dollar makes US beef dearer to foreign buyers,
             exports soften, more beef stays in the domestic market, and domestic
             prices ease. Mexico is the largest-volume destination, but Japan, Korea
             and China matter more by value, so a trade-weighted basket is the right
             measure rather than any single bilateral rate.

  ICSA       Initial unemployment claims, weekly, 1967 onwards, seasonally adjusted.
             Channel: a weakening labour market makes households trade down the
             carcass - from ribeye to chuck and ground beef. The prediction is
             specific and testable: claims should help the premium cuts more than
             the cheap ones.

  WPU0131    Producer price index for slaughter cattle, monthly. Included only as an
             optional cross-check; it is monthly and therefore excluded from the
             weekly feature set by default (see --include-monthly).

No API key is needed: FRED publishes every series as CSV at
https://fred.stlouisfed.org/graph/fredgraph.csv?id=<SERIES_ID>

Why monthly series are avoided
------------------------------
Forward-filling a monthly series onto a weekly calendar produces a step function that
holds flat for four or five weeks and then jumps. A network can learn "the month just
changed", which is a calendar artifact rather than economics, and it consumes capacity
that a dataset of roughly 800 training windows cannot spare. If a monthly series is
genuinely wanted, carry it as a change recorded on its release week and zero elsewhere,
not as a forward-filled level.

Usage:
    python src/fetch_fred.py
    python src/fetch_fred.py --start 2000-01-01

Output:
    data/raw/fred_weekly.csv
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"

# FRED serves fredgraph.csv primarily to browsers and is slow or unresponsive to
# requests that do not look like one. A plain library user-agent reliably times out.
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept": "text/csv,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://fred.stlouisfed.org/",
}

# column name -> (FRED id, how to aggregate a week, what it is)
SERIES = {
    "usd_index":      ("DTWEXBGS", "mean", "Nominal Broad U.S. Dollar Index"),
    "initial_claims": ("ICSA",     "last", "Initial unemployment claims, weekly"),
}


def week_of(dates: pd.Series) -> pd.Series:
    """Map dates to the Sunday that ends their week.

    Identical to build_dataset.week_of, so every series in the project lands on the
    same weekly grid. Duplicating four lines is better than importing across scripts
    that are otherwise independent.
    """
    return dates.dt.to_period("W-SUN").dt.end_time.dt.normalize()


def fetch_series(series_id: str, timeout: int) -> pd.DataFrame | None:
    """Download one series, or read a copy already saved next to the output.

    Each successful download is cached as data/raw/fred_<ID>.csv, so a run that gets
    one series and then fails on the next does not have to repeat the first. The cache
    also doubles as a manual escape hatch: if FRED refuses the script entirely, open
    https://fred.stlouisfed.org/series/<ID>, use the Download button, and save the file
    to that path - the script will pick it up without touching the network.
    """
    cache = RAW / f"fred_{series_id}.csv"
    if cache.exists():
        print(f"    using cached {cache.name}")
        return pd.read_csv(cache)

    url = f"{BASE}?id={series_id}"
    for attempt in range(1, 4):
        try:
            response = requests.get(url, timeout=timeout, headers=HEADERS)
            response.raise_for_status()
            from io import StringIO
            frame = pd.read_csv(StringIO(response.text))
            RAW.mkdir(parents=True, exist_ok=True)
            frame.to_csv(cache, index=False)
            return frame
        except Exception as error:                      # noqa: BLE001
            print(f"    attempt {attempt}/3 failed: {error}")
            if attempt < 3:
                wait = 15 * attempt
                print(f"    retrying in {wait}s")
                time.sleep(wait)

    print(f"    could not download {series_id}. Either re-run, or download it by hand:")
    print(f"      https://fred.stlouisfed.org/series/{series_id}  ->  Download -> CSV")
    print(f"      save it as {cache.relative_to(ROOT)}")
    return None


def tidy(frame: pd.DataFrame, series_id: str, column: str, how: str) -> pd.Series:
    """One weekly series, on the project's Sunday-ending grid.

    FRED writes missing observations as a single '.', which pandas reads as a string
    and which would otherwise poison the whole column, so values are coerced and the
    gaps dropped before aggregation.
    """
    date_column = frame.columns[0]
    frame = frame.rename(columns={date_column: "date", series_id: "value"})
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")

    missing = frame["value"].isna().sum()
    frame = frame.dropna(subset=["date", "value"])
    if missing:
        print(f"    dropped {missing:,} missing observation(s) (FRED writes these '.')")

    frame["week"] = week_of(frame["date"])
    grouped = frame.groupby("week")["value"]
    weekly = grouped.mean() if how == "mean" else grouped.last()
    weekly.name = column
    return weekly


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", default=None,
                        help="drop weeks before this date, e.g. 2000-01-01")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    collected = []
    for column, (series_id, how, description) in SERIES.items():
        print(f"{series_id}  ({description})")
        frame = fetch_series(series_id, args.timeout)
        if frame is None:
            print(f"  could not download {series_id}; skipped")
            continue
        print(f"  {len(frame):,} raw observations, aggregated per week by {how}")
        weekly = tidy(frame, series_id, column, how)
        print(f"  -> {len(weekly):,} weeks, {weekly.index.min().date()} to "
              f"{weekly.index.max().date()}")
        collected.append(weekly)

    if not collected:
        print("\nNothing downloaded.")
        return

    table = pd.concat(collected, axis=1).sort_index()
    table.index.name = "week"
    if args.start:
        table = table.loc[args.start:]

    RAW.mkdir(parents=True, exist_ok=True)
    out_path = RAW / "fred_weekly.csv"
    table.to_csv(out_path)

    print(f"\nSaved {out_path.relative_to(ROOT)}")
    print(f"  {len(table):,} weeks, {table.index.min().date()} to "
          f"{table.index.max().date()}")
    print("\n  column          weeks   first        last         mean")
    for column in table.columns:
        series = table[column].dropna()
        print(f"  {column:15s} {len(series):5d}   {series.index.min().date()}   "
              f"{series.index.max().date()}   {series.mean():,.2f}")

    overlap = table.dropna()
    if len(overlap):
        print(f"\n  all series present from {overlap.index.min().date()} "
              f"({len(overlap):,} weeks)")
        print("  NOTE: the dollar index starts in 2006, so using it costs about two")
        print("  years of training data relative to the 2004 start of the US series.")


if __name__ == "__main__":
    main()
