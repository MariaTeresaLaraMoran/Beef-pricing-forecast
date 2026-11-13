"""Download daily beef CUT prices in Mexico from SNIIM (Secretaría de Economía).

Source page: Mercados Nacionales > Pecuarios > Bovinos > Cortes de empacadoras y
distribuidoras
http://www.economia-sniim.gob.mx/SNIIM-Pecuarios-Nacionales/SelCor.asp?var=Bov

These are prices for individual cuts (Arrachera, Costillar, Chuleton Diezmillo, Lomo,
Pecho, and so on) reported by packers and distributors, in pesos per kilo. They are the
target series for this project; the carcass prices downloaded by fetch_sniim_beef.py are
used as a predictor.

SNIIM has no API. The site answers one month per query as an HTML table, so this script
requests every month since 2000, reads the table, and saves one CSV per month in
data/raw/sniim_cortes/. Months already downloaded are skipped, so the script can be
stopped and restarted. At the end all months are combined into one file.

Usage:
    python src/fetch_sniim_cuts.py                    # full history, 2000 to today
    python src/fetch_sniim_cuts.py --start-year 2025  # a quick test

Output:
    data/raw/sniim_cortes/YYYY-MM.csv                 (one file per month)
    data/raw/sniim_beef_cut_prices.csv                (all months combined)

Columns:
    date, establishment, origin, cut, price_min_mxn_kg, price_max_mxn_kg
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

URL = "http://www.economia-sniim.gob.mx/SNIIM-Pecuarios-Nacionales/Cor.asp"
ROOT = Path(__file__).resolve().parents[1]
MONTH_DIR = ROOT / "data" / "raw" / "sniim_cortes"
OUT_PATH = ROOT / "data" / "raw" / "sniim_beef_cut_prices.csv"
COLUMNS = ["date", "establishment", "origin", "cut", "price_min_mxn_kg", "price_max_mxn_kg"]
DATE_PATTERN = re.compile(r"^\d{2}/\d{2}/\d{4}$")
PAUSE_SECONDS = 1.0
TIMEOUT_SECONDS = 90


def get_page(year: int, month: int, page: int) -> str:
    last_day = calendar.monthrange(year, month)[1]
    params = {
        "Var": "Bov",
        "pag": page,
        "origen": 0,              # 0 = all origins (states, imports, local, frozen)
        "destino": 0,             # 0 = all establishments
        "del": "01",
        "al": f"{last_day:02d}",
        "mes": f"{month:02d}",
        "anio": year,
        "RegPag": 1000,           # records per page (the site's maximum)
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

    The table holds two kinds of row:
      - an establishment name in a single cell of class encabDES
      - a data row of five cells: date, origin, cut, minimum price, maximum price
    `state` carries the current establishment across page boundaries.
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
            state["establishment"] = texts[0]
        elif len(cells) == 5 and DATE_PATTERN.match(texts[0]):
            rows.append({
                "date": pd.to_datetime(texts[0], format="%d/%m/%Y"),
                "establishment": state.get("establishment"),
                "origin": texts[1],
                "cut": texts[2],
                "price_min_mxn_kg": texts[3],
                "price_max_mxn_kg": texts[4],
            })
    return rows, pages


def to_number(series: pd.Series) -> pd.Series:
    """Turn '$1,234.00' into 1234.0 and '-' into NaN."""
    cleaned = (series.astype(str)
               .str.replace("$", "", regex=False)
               .str.replace(",", "", regex=False)
               .str.strip())
    return pd.to_numeric(cleaned, errors="coerce")


def fetch_month(year: int, month: int) -> pd.DataFrame:
    state: dict = {}
    rows, pages = parse_page(get_page(year, month, 1), state)
    for page in range(2, pages + 1):
        time.sleep(PAUSE_SECONDS)
        more, _ = parse_page(get_page(year, month, page), state)
        rows.extend(more)

    df = pd.DataFrame(rows, columns=COLUMNS)
    for column in ["price_min_mxn_kg", "price_max_mxn_kg"]:
        df[column] = to_number(df[column])
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
            print(f"{year}-{month:02d}: {len(df):,} rows, {df['cut'].nunique()} cuts")
            time.sleep(PAUSE_SECONDS)

    files = sorted(MONTH_DIR.glob("*.csv"))
    if not files:
        sys.exit("No monthly files were downloaded.")
    combined = pd.concat([pd.read_csv(f, parse_dates=["date"]) for f in files], ignore_index=True)
    combined = combined.sort_values(["date", "establishment", "cut"]).reset_index(drop=True)
    combined.to_csv(OUT_PATH, index=False)

    print(f"\nSaved {len(combined):,} rows to {OUT_PATH.relative_to(ROOT)}")
    print(f"Date range: {combined['date'].min().date()} to {combined['date'].max().date()}")
    print(f"Establishments: {combined['establishment'].nunique()}  "
          f"Origins: {combined['origin'].nunique()}  Cuts: {combined['cut'].nunique()}")
    print("\nTop 25 cuts by number of records:")
    print(combined["cut"].value_counts().head(25).to_string())


if __name__ == "__main__":
    main()
