"""Download US fed cattle prices from the USDA AMS LMR Datamart.

Report 2477, "5 Area Weekly Weighted Average Direct Slaughter Cattle" (LM_CT150).
This is the headline US fed cattle series: the weighted average price paid by packers
for slaughter cattle across the five major feeding regions (TX/OK/NM, KS, NE, CO,
IA/MN). It is the open USDA original of the commercial "five area" files that vendors
resell.

Why it matters for this project
-------------------------------
The boxed beef cutout already in the dataset is what the packer SELLS. This report is
what the packer BUYS. The difference between them, on the same carcass-weight basis,
is the gross packer margin:

    margin = cutout (USD/cwt carcass) - dressed steer price (USD/cwt carcass)

Margin is a supply-side driver of beef prices: when it is wide, packers raise slaughter
volume and more beef reaches the market; when it is thin, they cut kills and prices
firm. It is also the natural US-side analogue of the Mexico-minus-US spread that
build_dataset.py already builds.

Pick the DRESSED basis, not LIVE, when comparing with the cutout. Dressed prices are
quoted per hundredweight of CARCASS, which is what the cutout is quoted on. Live
prices are per hundredweight of the animal, roughly 63 per cent of which becomes
carcass, so comparing live with cutout would be comparing different things.

No API key is needed and the whole history comes back in one request, so unlike the
boxed beef download there is no chunking, no cache and no retry loop to babysit.

Known break in the series
-------------------------
AMS suppresses a region when too few packers report, to protect confidentiality, and
the 5-Area aggregate has excluded regions at various times: Colorado from 26 Dec 2022,
TX/OK/NM from 29 Jun 2026, Kansas from 17 Aug 2026. From 26 Oct 2026 the report
includes all five regions again. The composition of the average therefore changes
within the modelling period. The script writes the report's own narrative text to a
companion file so this is documented rather than silently absorbed.

Usage:
    python src/fetch_usda_cattle.py
    python src/fetch_usda_cattle.py --timeout 300

Output:
    data/raw/usda_5area_cattle.csv         one row per week per class/basis/grade
    data/raw/usda_5area_cattle_notes.txt   AMS narratives, for the methodology slide
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

BASE = "https://mpr.datamart.ams.usda.gov/services/v1.1/reports"
SLUG = "2477"
SECTION = "Detail"

KEEP = ["report_date", "class_description", "selling_basis_description",
        "grade_description", "head_count", "weight_range_low", "weight_range_high",
        "weight_range_avg", "price_range_low", "price_range_high",
        "weighted_avg_price"]

RENAME = {
    "class_description": "cattle_class",
    "selling_basis_description": "selling_basis",
    "grade_description": "grade",
    "weight_range_avg": "weight_avg_lb",
    "weighted_avg_price": "price_usd_cwt",
    "price_range_low": "price_low_usd_cwt",
    "price_range_high": "price_high_usd_cwt",
    "weight_range_low": "weight_low_lb",
    "weight_range_high": "weight_high_lb",
}


def extract_rows(data) -> list[dict]:
    """Pull the record list out of whatever shape Datamart returns.

    The service answers with an object carrying "results" when it has data, but with a
    bare JSON string, null, or a plain list in other cases. The boxed beef download
    learned this the hard way, so the same guard is used here.
    """
    if data is None:
        return []
    if isinstance(data, dict):
        rows = data.get("results", [])
    elif isinstance(data, list):
        rows = data
    else:
        return []
    return [row for row in rows if isinstance(row, dict)]


def to_number(series: pd.Series) -> pd.Series:
    """Datamart sends numbers as strings with thousands separators."""
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce")


def tidy(rows: list[dict]) -> pd.DataFrame:
    """Normalise the raw records into a tidy weekly table."""
    frame = pd.DataFrame(rows)
    for column in KEEP:
        if column not in frame.columns:
            frame[column] = pd.NA
    frame = frame[KEEP].rename(columns=RENAME)

    frame["report_date"] = pd.to_datetime(frame["report_date"], format="%m/%d/%Y",
                                          errors="coerce")
    for column in ("head_count", "weight_avg_lb", "weight_low_lb", "weight_high_lb",
                   "price_usd_cwt", "price_low_usd_cwt", "price_high_usd_cwt"):
        frame[column] = to_number(frame[column])

    for column in ("cattle_class", "selling_basis", "grade"):
        frame[column] = frame[column].astype("string").str.strip()

    # Rows exist for every class/basis/grade combination each week, most of them empty
    # because nothing in that category traded. Those carry no information.
    before = len(frame)
    frame = frame.dropna(subset=["report_date", "price_usd_cwt"])
    print(f"  dropped {before - len(frame):,} row(s) with no price reported "
          f"({(before - len(frame)) / before * 100:.1f}% - categories that did not trade)")

    return frame.sort_values(["report_date", "cattle_class", "selling_basis", "grade"])


TOTAL = "Total all grades"


def collapse_grades(group: pd.DataFrame) -> float:
    """One price per week for a class and selling basis.

    AMS publishes a "Total all grades" line, but only from late 2003; in 2001 and 2002
    the report carries the four Choice-percentage grades and no total. Where the total
    is published it is used as-is. Where it is absent, it is rebuilt as the head-count
    weighted average of the grade lines, which is how AMS itself computes it (verified
    against the weeks where both are present).
    """
    total = group[group["grade"] == TOTAL]["price_usd_cwt"]
    if len(total) and pd.notna(total.iloc[0]):
        return float(total.iloc[0])

    parts = group[group["grade"] != TOTAL].dropna(subset=["price_usd_cwt"])
    if parts.empty:
        return float("nan")
    weights = parts["head_count"].fillna(0)
    if weights.sum() > 0:
        return float((parts["price_usd_cwt"] * weights).sum() / weights.sum())
    return float(parts["price_usd_cwt"].mean())


def weekly_series(frame: pd.DataFrame) -> pd.DataFrame:
    """Wide weekly table: one column per series build_dataset.py needs."""
    wanted = {
        "us_steer_dressed": ("STEER", "DRESSED DELIVERED"),
        "us_steer_live":    ("STEER", "LIVE FOB"),
        "us_cattle_dressed_all": ("ALL BEEF TYPE", "DRESSED DELIVERED"),
    }
    out = pd.DataFrame()
    for name, (cattle_class, basis) in wanted.items():
        subset = frame[(frame["cattle_class"] == cattle_class)
                       & (frame["selling_basis"] == basis)]
        if subset.empty:
            continue
        out[name] = subset.groupby("report_date").apply(collapse_grades,
                                                        include_groups=False)
        heads = subset[subset["grade"] != TOTAL].groupby("report_date")["head_count"].sum(
            min_count=1)
        out[f"{name}_head"] = heads
    return out.sort_index()


def fetch(timeout: int) -> tuple[list[dict], str]:
    url = f"{BASE}/{SLUG}/{SECTION}"
    print(f"Requesting {url}")
    print("  (one request, whole history, no API key)")

    last_error = None
    for attempt in range(1, 4):
        try:
            started = time.time()
            response = requests.get(url, timeout=timeout,
                                    headers={"Accept": "application/json"})
            response.raise_for_status()
            payload = response.json()
            print(f"  responded in {time.time() - started:.1f}s")
            return extract_rows(payload), response.text[:0]
        except Exception as error:          # noqa: BLE001 - report and retry
            last_error = error
            print(f"  attempt {attempt}/3 failed: {error}")
            if attempt < 3:
                wait = 15 * attempt
                print(f"  retrying in {wait}s")
                time.sleep(wait)

    print(f"\nCould not reach the Datamart: {last_error}")
    print("The service goes down from time to time. Check "
          "https://mpr.datamart.ams.usda.gov in a browser and try again later.")
    return [], ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--timeout", type=int, default=180,
                        help="seconds to wait for the response")
    args = parser.parse_args()

    rows = fetch(args.timeout)[0]
    if not rows:
        sys.exit(1)
    print(f"  received {len(rows):,} records")

    frame = tidy(rows)
    RAW.mkdir(parents=True, exist_ok=True)
    out_path = RAW / "usda_5area_cattle.csv"
    frame.to_csv(out_path, index=False)

    print(f"\nSaved {len(frame):,} rows to {out_path.relative_to(ROOT)}")
    print(f"  date range: {frame['report_date'].min().date()} to "
          f"{frame['report_date'].max().date()}")
    print(f"  weeks: {frame['report_date'].nunique():,}")

    # The weekly series build_dataset.py will join.
    weekly = weekly_series(frame)
    weekly_path = RAW / "usda_5area_cattle_weekly.csv"
    weekly.to_csv(weekly_path)
    print(f"\nSaved weekly series to {weekly_path.relative_to(ROOT)}")
    units = {"us_steer_dressed": "USD/cwt carcass", "us_steer_live": "USD/cwt live",
             "us_cattle_dressed_all": "USD/cwt carcass"}
    for column in weekly.columns:
        if column.endswith("_head"):
            continue
        series = weekly[column].dropna()
        if series.empty:
            continue
        print(f"  {column:24s} {len(series):,} weeks, "
              f"{series.index.min().date()} to {series.index.max().date()}, "
              f"latest {series.iloc[-1]:,.2f} {units.get(column, '')}")

    # Keep the AMS narratives: they document the regional suppressions and corrections.
    narratives = {}
    for row in rows:
        text = (row.get("narrative") or "").strip()
        if text and text not in narratives:
            narratives[text] = row.get("report_date")
    notes_path = RAW / "usda_5area_cattle_notes.txt"
    with notes_path.open("w") as handle:
        handle.write("AMS narratives attached to report 2477 (LM_CT150)\n")
        handle.write("Kept because they record regional suppressions and corrections\n")
        handle.write("that change what the 5-Area average covers.\n")
        handle.write("=" * 70 + "\n\n")
        for text, date in sorted(narratives.items(), key=lambda kv: str(kv[1])):
            handle.write(f"[{date}] {text}\n\n")
    print(f"\n{len(narratives)} distinct narrative(s) written to "
          f"{notes_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
