"""Download the daily USD/MXN FIX exchange rate from Banco de México (SIE API).

Series SF43718: "Tipo de cambio pesos por dólar E.U.A., FIX, fecha de determinación".
Requires a free token in the .env file (see .env.example).

Usage:
    python src/fetch_banxico.py
    python src/fetch_banxico.py --start 2000-01-01 --end 2026-11-12

Output:
    data/raw/banxico_usdmxn_fix.csv   (columns: date, usd_mxn_fix)
"""

import argparse
import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

SERIES_ID = "SF43718"
BASE_URL = "https://www.banxico.org.mx/SieAPIRest/service/v1/series"
CHUNK_YEARS = 5
ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "raw" / "banxico_usdmxn_fix.csv"


def fetch_chunk(start: str, end: str, token: str) -> list[dict]:
    url = f"{BASE_URL}/{SERIES_ID}/datos/{start}/{end}"
    response = requests.get(url, headers={"Bmx-Token": token}, timeout=60)
    if not response.ok:
        # Show Banxico's own explanation instead of a bare status code.
        sys.exit(
            f"Banxico returned HTTP {response.status_code} for {start} to {end}.\n"
            f"Response: {response.text[:500]}"
        )
    series = response.json()["bmx"]["series"][0]
    return series.get("datos", [])


def fetch(start: str, end: str, token: str) -> pd.DataFrame:
    # Request the history in blocks of a few years to keep each call small.
    edges = pd.date_range(start, end, freq=f"{CHUNK_YEARS}YS").tolist()
    starts = sorted({pd.Timestamp(start), *edges})
    rows: list[dict] = []
    for i, chunk_start in enumerate(starts):
        chunk_end = starts[i + 1] - pd.Timedelta(days=1) if i + 1 < len(starts) else pd.Timestamp(end)
        chunk = fetch_chunk(f"{chunk_start:%Y-%m-%d}", f"{chunk_end:%Y-%m-%d}", token)
        print(f"{chunk_start:%Y-%m-%d} to {chunk_end:%Y-%m-%d}: {len(chunk)} rows")
        rows.extend(chunk)

    if not rows:
        sys.exit(f"Banxico returned no data for {start} to {end}.")

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["fecha"], format="%d/%m/%Y")
    # Banxico marks days without a published value as "N/E".
    df["usd_mxn_fix"] = pd.to_numeric(df["dato"].str.replace(",", ""), errors="coerce")
    df = df.dropna(subset=["usd_mxn_fix"]).drop_duplicates(subset=["date"])
    return df[["date", "usd_mxn_fix"]].sort_values("date").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", default="2000-01-01", help="YYYY-MM-DD")
    parser.add_argument("--end", default=date.today().isoformat(), help="YYYY-MM-DD")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    token = (os.getenv("BANXICO_TOKEN") or "").strip().strip('"\'')
    if not token:
        sys.exit("BANXICO_TOKEN is missing. Copy .env.example to .env and paste your token.")

    df = fetch(args.start, args.end, token)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"Saved {len(df):,} rows to {OUT_PATH.relative_to(ROOT)}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(df.tail(3).to_string(index=False))


if __name__ == "__main__":
    main()
