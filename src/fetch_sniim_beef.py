"""Download daily beef carcass prices in Mexico from SNIIM (Secretaría de Economía).

Source page: Mercados Nacionales > Pecuarios > Bovinos > Carne en canal
http://www.economia-sniim.gob.mx/SNIIM-Pecuarios-Nacionales/SelCan.asp?var=Bov

SNIIM has no API. The site answers one month per query as an HTML table, so this
script requests every month since 2000, reads the table, and saves one CSV per month
in data/raw/sniim_carne_canal/. Months already downloaded are skipped, so the script
can be stopped and restarted. At the end all months are combined into one file.

Usage:
    python src/fetch_sniim_beef.py                    # full history, 2000 to today
    python src/fetch_sniim_beef.py --start-year 2025  # a quick test

Output:
    data/raw/sniim_carne_canal/YYYY-MM.csv            (one file per month)
    data/raw/sniim_beef_carcass_prices.csv            (all months combined)

Columns:
    date, slaughterhouse, type, avg_weight_kg, carcasses, price_min_mxn_kg, price_max_mxn_kg
"""

import argparse
import calendar
import re
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

URL = "http://www.economia-sniim.gob.mx/SNIIM-Pecuarios-Nacionales/e_Can.asp"
ROOT = Path(__file__).resolve().parents[1]
MONTH_DIR = ROOT / "data" / "raw" / "sniim_carne_canal"
OUT_PATH = ROOT / "data" / "raw" / "sniim_beef_carcass_prices.csv"
COLUMNS = [
    "date", "slaughterhouse", "type", "avg_weight_kg",
    "carcasses", "price_min_mxn_kg", "price_max_mxn_kg",
]
PAUSE_SECONDS = 1.0   # wait between requests, to be polite to the server
TIMEOUT_SECONDS = 90


def get_page(year: int, month: int, page: int) -> str:
    last_day = calendar.monthrange(year, month)[1]
    params = {
        "var": "Bov",
        "pag": page,
        "destino": 0,              # 0 = all slaughterhouses
        "del": "01",
        "al": f"{last_day:02d}",
        "mes": f"{month:02d}",
        "anio": year,
        "RegPag": 1000,            # records per page (the site's maximum)
    }
    for attempt in range(3):
        try:
            response = requests.get(URL, params=params, timeout=TIMEOUT_SECONDS)
            response.raise_for_status()
            response.encoding = "iso-8859-1"   # the site does not use UTF-8
            return response.text
        except requests.RequestException as error:
            if attempt == 2:
                raise
            print(f"  retrying {year}-{month:02d} page {page} after error: {error}")
            time.sleep(5)
    return ""


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parse_page(html: str, state: dict) -> tuple[list[dict], int]:
    """Read one results page. Returns the data rows and the total number of pages.

    The table has three kinds of rows:
      - a slaughterhouse name (one cell, class encabDES)
      - a date row ("Fecha:01/10/2026" and "Canales vendidos:290", class encabFEC)
      - data rows with five cells: type, weight, carcasses, minimum price, maximum price
    `state` remembers the current slaughterhouse and date from one page to the next.
    """
    soup = BeautifulSoup(html, "html.parser")

    pages = 1
    pager = re.search(r"P.gina\s+\d+\s+de\s+(\d+)", soup.get_text(" "))
    if pager:
        pages = int(pager.group(1))

    rows: list[dict] = []
    for tr in soup.find_all("tr"):
        if tr.find("table"):          # skip layout rows that wrap other tables
            continue
        cells = tr.find_all("td", recursive=False)
        if not cells:
            continue
        css = " ".join(cells[0].get("class") or []).lower()
        texts = [clean(td.get_text(" ")) for td in cells]

        if css == "encabdes":
            state["slaughterhouse"] = texts[0]
        elif css == "encabfec" and texts[0].startswith("Fecha:"):
            state["date"] = pd.to_datetime(texts[0].split(":", 1)[1], format="%d/%m/%Y")
        elif css == "datos" and len(cells) == 5 and state.get("date") is not None:
            rows.append({
                "date": state["date"],
                "slaughterhouse": state.get("slaughterhouse"),
                "type": texts[0],
                "avg_weight_kg": texts[1],
                "carcasses": texts[2],
                "price_min_mxn_kg": texts[3],
                "price_max_mxn_kg": texts[4],
            })
    return rows, pages


def fetch_month(year: int, month: int) -> pd.DataFrame:
    state: dict = {}
    rows, pages = parse_page(get_page(year, month, 1), state)
    for page in range(2, pages + 1):
        time.sleep(PAUSE_SECONDS)
        more, _ = parse_page(get_page(year, month, page), state)
        rows.extend(more)

    df = pd.DataFrame(rows, columns=COLUMNS)
    for column in ["avg_weight_kg", "carcasses", "price_min_mxn_kg", "price_max_mxn_kg"]:
        # SNIIM writes "-" when a value was not reported.
        df[column] = pd.to_numeric(df[column].str.replace(",", ""), errors="coerce")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start-year", type=int, default=2000)
    args = parser.parse_args()

    MONTH_DIR.mkdir(parents=True, exist_ok=True)
    today = date.today()

    for year in range(args.start_year, today.year + 1):
        for month in range(1, 13):
            if (year, month) > (today.year, today.month):
                break
            month_path = MONTH_DIR / f"{year}-{month:02d}.csv"
            is_current_month = (year, month) == (today.year, today.month)
            if month_path.exists() and not is_current_month:
                continue   # already downloaded

            df = fetch_month(year, month)
            df.to_csv(month_path, index=False)
            print(f"{year}-{month:02d}: {len(df):,} rows")
            time.sleep(PAUSE_SECONDS)

    files = sorted(MONTH_DIR.glob("*.csv"))
    if not files:
        sys.exit("No monthly files were downloaded.")
    combined = pd.concat([pd.read_csv(f, parse_dates=["date"]) for f in files], ignore_index=True)
    combined = combined.sort_values(["date", "slaughterhouse", "type"]).reset_index(drop=True)
    combined.to_csv(OUT_PATH, index=False)

    print(f"\nSaved {len(combined):,} rows to {OUT_PATH.relative_to(ROOT)}")
    print(f"Date range: {combined['date'].min().date()} to {combined['date'].max().date()}")
    print(f"Slaughterhouses: {combined['slaughterhouse'].nunique()}")
    print("Rows by type:")
    print(combined["type"].value_counts().to_string())


if __name__ == "__main__":
    main()
