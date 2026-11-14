"""Build the weekly modelling table from the raw series.

Steps, in order:
  1. Normalise SNIIM cut labels with data/reference/cut_name_map.csv
  2. Aggregate the six target cuts to weekly maximum price
  3. Join the carcass price (predictor), USD/MXN and US beef exports to Mexico
  4. Apply a publication lag to each series, so that at week t the model only sees
     what had actually been published by week t
  5. Build log levels, weekly log returns, lagged features and seasonal terms
  6. Write one modelling table to data/processed/

Target definition
-----------------
For each cut c, the target is the log return over the next HORIZON weeks:

    y_c(t) = log P_c(t + HORIZON) - log P_c(t)

A predicted price is recovered as P_c(t) * exp(y_hat). Log returns are used because
the EDA showed the price levels are non-stationary and their variance grows with the
level: a 1% move was 0.23 MXN/kg in 2000 and is about 1.15 MXN/kg now.

Leakage
-------
Every predictor column is shifted by that series' publication lag before any feature
is built, so no row can contain a value that was not public at that week. The target
is the only column that looks forward.

Usage:
    python src/build_dataset.py
    python src/build_dataset.py --horizon 4 --origin domestic
    python src/build_dataset.py --targets us          # forecast US prices instead
    python src/build_dataset.py --targets both        # both, to compare them (default)

Output:
    data/processed/weekly_model_table.csv
    data/processed/weekly_model_table_columns.txt   (a description of every column)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
REFERENCE = ROOT / "data" / "reference"

# Five targets, one per primal, each with a direct USDA counterpart (see
# data/reference/us_cut_map.csv). Arrachera was dropped: skirts are absent from the
# daily USDA boxed beef report, so it was the only target without a clean US pair.
TARGET_CUTS = ["Sirloin", "Costillar", "Chuleton Diezmillo", "Pulpa Negra", "Pecho"]
CARCASS_TYPE = "corral"
HORIZON = 4          # weeks ahead
FEATURE_LAGS = [1, 2, 3, 4]

# Publication lag in weeks: how long after a week ends before its value is public.
# SNIIM and Banxico publish daily, so their values for week t are available in week t.
# USDA FAS export sales run about two weeks behind (the latest record on 2026-11-13
# was for the week ending 2026-10-29).
PUBLICATION_LAG_WEEKS = {
    "carcass": 0,
    "usd_mxn": 0,
    "exports_tonnes": 2,
    "net_sales_tonnes": 2,
    # US cut prices and cutout slot in here once collected; USDA AMS publishes weekly
    # boxed beef with a few days' delay, so 1 week is the expected value.
    "us_ribeye": 1,
    "us_chuck_roll": 1,
    "us_brisket": 1,
    "us_top_inside_round": 1,
    "us_top_butt": 1,
    "us_cutout_choice": 1,
    "us_cutout_select": 1,
    # LM_CT150 covers a week and is published the Monday after it closes.
    "us_steer_dressed": 1,
    "us_steer_live": 1,
}

# Columns reserved for the US series, created empty so the table's shape is stable
# and they can be filled in without changing anything downstream.
US_COLUMNS = ["us_ribeye", "us_chuck_roll", "us_brisket", "us_top_inside_round",
              "us_top_butt", "us_cutout_choice", "us_cutout_select",
              "us_steer_dressed", "us_steer_live"]

# USDA boxed beef items paired with each Mexican target, by IMPS code. See
# data/reference/us_cut_map.csv for why each pairing was chosen.
US_ITEMS = {
    "us_ribeye":            "Rib, ribeye, lip-on, bn-in (109E  1)",   # <- Costillar
    "us_chuck_roll":        "Chuck, roll, lxl, neck/off (116A  3)",   # <- Chuleton Diezmillo
    "us_brisket":           "Brisket, deckle-off, bnls (120  1)",     # <- Pecho
    "us_top_inside_round":  "Round, top inside round (168  3)",       # <- Pulpa Negra
    "us_top_butt":          "Loin, top butt, boneless (184  3)",      # <- Sirloin
}

# The US series that can be forecast instead of, or alongside, the Mexican cuts.
# Chosen with --targets us or --targets both. The cutout comes first because it is the
# headline series: a volume-weighted index of the whole carcass, and the number the US
# trade actually quotes.
US_TARGET_SERIES = [
    "us_cutout_choice", "us_cutout_select",
    "us_ribeye", "us_chuck_roll", "us_brisket", "us_top_inside_round", "us_top_butt",
]


def week_of(dates: pd.Series) -> pd.Series:
    """Map dates to the Sunday that ends their week, so all series align."""
    return dates.dt.to_period("W-SUN").dt.end_time.dt.normalize()


# ---------------------------------------------------------------------------
# 1 and 2. Cut prices: normalise labels, then aggregate to weekly
# ---------------------------------------------------------------------------

AGGREGATION = "mean"        # set by --agg; how establishments are combined per week
OUTLIER_FACTOR = 5.0        # drop records this many times above or below the yearly median


def drop_outlier_records(cuts: pd.DataFrame, factor: float) -> pd.DataFrame:
    """Remove individual records that cannot be real prices.

    SNIIM is hand-entered and contains occasional errors: Pecho has records at
    10 MXN/kg and at 2,850 against a median near 49. With only 3-5 establishments
    reporting a cut, one such record moves the weekly average by an order of
    magnitude. Each record is compared with the median for its own cut in its own
    year, which allows for inflation across 26 years, and dropped if it lies beyond
    `factor` times that median in either direction.
    """
    work = cuts.copy()
    work["year"] = work["date"].dt.year
    reference = work.groupby(["cut", "year"])["price_max_mxn_kg"].transform("median")
    too_high = work["price_max_mxn_kg"] > reference * factor
    too_low = work["price_max_mxn_kg"] < reference / factor
    bad = (too_high | too_low) & work["price_max_mxn_kg"].notna()

    if bad.any():
        print(f"  dropped {bad.sum():,} record(s) beyond {factor:g}x the yearly median "
              f"({bad.mean() * 100:.3f}% of rows)")
        worst = work[bad].nlargest(3, "price_max_mxn_kg")[["date", "cut", "price_max_mxn_kg"]]
        for _, row in worst.iterrows():
            print(f"    {row['date'].date()}  {row['cut']:20s} {row['price_max_mxn_kg']:>10,.2f}")
    return cuts[~bad]


def load_cuts(origin_filter: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cuts = pd.read_csv(RAW / "sniim_beef_cut_prices.csv", parse_dates=["date"])
    mapping = pd.read_csv(REFERENCE / "cut_name_map.csv")

    before = cuts["cut"].nunique()
    name_map = dict(zip(mapping["sniim_cut_raw"].str.strip(), mapping["sniim_cut_normalized"]))
    cuts["cut"] = cuts["cut"].str.strip()
    unmapped = sorted(set(cuts["cut"]) - set(name_map))
    if unmapped:
        print(f"WARNING: cut labels not in the mapping, left unchanged: {unmapped}")
    cuts["cut"] = cuts["cut"].map(lambda c: name_map.get(c, c))
    dropped = (cuts["cut"] == "DROP").sum()
    cuts = cuts[cuts["cut"] != "DROP"]
    print(f"Cut labels normalised: {before} raw -> {cuts['cut'].nunique()} cuts "
          f"({dropped} row(s) dropped as unassignable)")

    if origin_filter == "domestic":
        # Exclude imported and frozen stock so the target is domestic product only.
        mask = ~cuts["origin"].str.contains("Importaci|Congelado", case=False, na=False)
        print(f"Origin filter 'domestic': kept {mask.sum():,} of {len(cuts):,} rows")
        cuts = cuts[mask]

    if OUTLIER_FACTOR:
        cuts = drop_outlier_records(cuts, OUTLIER_FACTOR)

    cuts["week"] = week_of(cuts["date"])
    target = cuts[cuts["cut"].isin(TARGET_CUTS)]

    # No quantity column exists for cuts, so establishments are combined unweighted.
    # The median is the default: only 3-5 establishments report some cuts, and SNIIM
    # contains occasional data-entry errors (Pecho has records at 10 and at 2,850
    # MXN/kg against a median near 49), which a mean of five values cannot absorb.
    # The carcass file does carry carcass counts and is weighted below.
    wide = target.pivot_table(index="week", columns="cut",
                              values="price_max_mxn_kg", aggfunc=AGGREGATION)
    reporters = target.pivot_table(index="week", columns="cut",
                                   values="establishment", aggfunc="nunique")
    wide = wide.reindex(columns=TARGET_CUTS)
    reporters = reporters.reindex(columns=TARGET_CUTS)
    wide.columns = [f"price_{slug(c)}" for c in wide.columns]
    reporters.columns = [f"n_reporters_{slug(c)}" for c in reporters.columns]
    return wide, reporters


def slug(name: str) -> str:
    return name.lower().replace(" ", "_")


# ---------------------------------------------------------------------------
# 3. The other series
# ---------------------------------------------------------------------------

def load_carcass() -> pd.Series:
    df = pd.read_csv(RAW / "sniim_beef_carcass_prices.csv", parse_dates=["date"])
    df["type"] = df["type"].astype(str).str.strip().str.lower()
    df = df[df["type"] == CARCASS_TYPE].copy()
    df["week"] = week_of(df["date"])

    def weighted(group: pd.DataFrame) -> float:
        values = group["price_max_mxn_kg"]
        weights = group["carcasses"].fillna(0)
        mask = values.notna()
        if not mask.any():
            return np.nan
        v, w = values[mask], weights[mask]
        return float(np.average(v, weights=w)) if w.sum() > 0 else float(v.mean())

    return df.groupby("week").apply(weighted, include_groups=False).rename("carcass")


def load_fx() -> pd.Series:
    df = pd.read_csv(RAW / "banxico_usdmxn_fix.csv", parse_dates=["date"])
    df["week"] = week_of(df["date"])
    return df.groupby("week")["usd_mxn_fix"].mean().rename("usd_mxn")


def to_number(series: pd.Series) -> pd.Series:
    """USDA sends numbers as strings with thousands separators."""
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.strip(), errors="coerce")


def load_us_cuts() -> pd.DataFrame:
    """Weekly US prices for the five cuts paired with the Mexican targets, USD per cwt."""
    path = RAW / "usda_boxed_beef_choice_cuts.csv"
    if not path.exists():
        print("  usda_boxed_beef_choice_cuts.csv not found - US cut columns left empty")
        return pd.DataFrame(columns=list(US_ITEMS))

    df = pd.read_csv(path, parse_dates=["report_date"], low_memory=False)
    df["item_description"] = df["item_description"].astype(str).str.strip()
    df["weighted_average"] = to_number(df["weighted_average"])
    # ".00" means the item was listed but no trades were reported that day.
    df.loc[df["weighted_average"] <= 0, "weighted_average"] = np.nan
    df["week"] = week_of(df["report_date"])

    out = pd.DataFrame()
    for column, item in US_ITEMS.items():
        rows = df[df["item_description"] == item]
        if rows.empty:
            print(f"  WARNING: USDA item not found: {item}")
            continue
        out[column] = rows.groupby("week")["weighted_average"].mean()
    print(f"  joined {len(out.columns)} US cut series "
          f"({out.index.min().date() if len(out) else '-'} onwards)")
    return out


def load_us_cutout() -> pd.DataFrame:
    """Weekly US Choice and Select boxed beef cutout, USD per cwt."""
    path = RAW / "usda_boxed_beef_cutout.csv"
    if not path.exists():
        print("  usda_boxed_beef_cutout.csv not found - cutout columns left empty")
        return pd.DataFrame(columns=["us_cutout_choice", "us_cutout_select"])

    df = pd.read_csv(path, parse_dates=["report_date"], low_memory=False)
    df["week"] = week_of(df["report_date"])
    out = pd.DataFrame()
    for column, source in (("us_cutout_choice", "choice_600_900_current"),
                           ("us_cutout_select", "select_600_900_current")):
        if source in df.columns:
            values = to_number(df[source])
            out[column] = values.groupby(df["week"]).mean()
    print(f"  joined cutout ({out.index.min().date() if len(out) else '-'} onwards)")
    return out


def load_cattle() -> pd.DataFrame:
    """Weekly US fed cattle prices from AMS report LM_CT150, USD per cwt.

    Produced by fetch_usda_cattle.py. The dressed series is quoted per hundredweight
    of CARCASS, the same basis as the boxed beef cutout, which is what makes the
    packer margin below a meaningful subtraction.
    """
    path = RAW / "usda_5area_cattle_weekly.csv"
    if not path.exists():
        print("  usda_5area_cattle_weekly.csv not found - run fetch_usda_cattle.py; "
              "cattle columns left empty")
        return pd.DataFrame(columns=["us_steer_dressed", "us_steer_live"])

    df = pd.read_csv(path, parse_dates=["report_date"])
    df["week"] = week_of(df["report_date"])
    out = pd.DataFrame()
    for column in ("us_steer_dressed", "us_steer_live"):
        if column in df.columns:
            out[column] = df.groupby("week")[column].mean()
    print(f"  joined fed cattle ({out.index.min().date() if len(out) else '-'} onwards)")
    return out


def load_exports() -> pd.DataFrame:
    df = pd.read_csv(RAW / "usda_fas_beef_exports_mexico.csv", parse_dates=["weekEndingDate"])
    df["week"] = week_of(df["weekEndingDate"])
    out = df.groupby("week")[["weeklyExports", "currentMYNetSales"]].sum(min_count=1)
    return out.rename(columns={"weeklyExports": "exports_tonnes",
                               "currentMYNetSales": "net_sales_tonnes"})


# ---------------------------------------------------------------------------
# 4, 5. Lags, returns, features
# ---------------------------------------------------------------------------

def apply_publication_lags(table: pd.DataFrame) -> pd.DataFrame:
    """Shift each predictor forward by its publication delay.

    After this, the value sitting on row t is the most recent value that was
    actually published by week t.
    """
    out = table.copy()
    for column, lag in PUBLICATION_LAG_WEEKS.items():
        if column in out.columns and lag > 0:
            out[column] = out[column].shift(lag)
            print(f"  {column}: shifted {lag} week(s) for publication delay")
    return out


# Which US series each Mexican target is compared with when building the spread.
SPREAD_PAIRS = {
    "price_costillar":           "us_ribeye",
    "price_chuleton_diezmillo":  "us_chuck_roll",
    "price_pecho":               "us_brisket",
    "price_pulpa_negra":         "us_top_inside_round",
    "price_sirloin":             "us_top_butt",
}
POUNDS_PER_KG = 0.45359237


def add_spreads(out: pd.DataFrame) -> pd.DataFrame:
    """Add the Mexico-minus-US price gap for each paired cut.

    This is the error-correction term from the price-transmission literature. US cut
    prices are quoted in USD per hundredweight, so they are converted to MXN per kg at
    the week's exchange rate before comparison:

        spread = log(Mexican MXN/kg) - log(US USD/cwt / 100 * MXN per USD / kg per lb)

    It matters because the raw US series carry almost no information at weekly
    frequency: US cut returns correlate with Mexican cut returns at between -0.04 and
    0.05, and US price levels at 0.01 to 0.04. The spread, by contrast, correlates with
    the next four weeks of Mexican returns at -0.16 to -0.32, and with the next thirteen
    weeks at -0.20 to -0.40 -- negative, as error correction predicts, and stronger at
    longer horizons, which is the signature of the slow adjustment (3-4% per month)
    reported by Jaramillo Villanueva et al. (2022).
    """
    if "usd_mxn" not in out.columns:
        return out
    added = []
    for mexican, american in SPREAD_PAIRS.items():
        if mexican not in out.columns or american not in out.columns:
            continue
        us_mxn_per_kg = out[american] / 100.0 * out["usd_mxn"] / POUNDS_PER_KG
        name = f"spread_{mexican.replace('price_', '')}"
        out[name] = (np.log(out[mexican].where(out[mexican] > 0))
                     - np.log(us_mxn_per_kg.where(us_mxn_per_kg > 0)))
        out[f"{name}_change"] = out[name].diff()
        added.append(name)
    if added:
        print(f"  built {len(added)} Mexico-minus-US spread(s): {', '.join(added)}")
    return out


def add_margins(out: pd.DataFrame) -> pd.DataFrame:
    """Add the gross packer margin: what the packer sells minus what it buys.

    The boxed beef cutout is the packer's revenue per hundredweight of carcass; the
    dressed fed cattle price is its raw material cost on the same basis. The gap drives
    slaughter decisions: a wide margin pulls cattle forward and puts more beef on the
    market a few weeks later, a thin one makes packers cut kills and firms prices. So
    unlike the cutout's own return, which is already in the price, the margin is a
    supply-side state variable with a lag built into it.

    Both forms are kept: the dollar margin because that is what the trade quotes, and
    the log ratio because it is scale-free across a period when the price level
    roughly tripled.
    """
    cost = "us_steer_dressed"
    if cost not in out.columns or out[cost].notna().sum() == 0:
        return out

    added = []
    for cutout in ("us_cutout_choice", "us_cutout_select"):
        if cutout not in out.columns:
            continue
        grade = cutout.replace("us_cutout_", "")
        revenue = out[cutout]
        out[f"margin_{grade}_usd"] = revenue - out[cost]
        out[f"margin_{grade}_log"] = (np.log(revenue.where(revenue > 0))
                                      - np.log(out[cost].where(out[cost] > 0)))
        out[f"margin_{grade}_log_change"] = out[f"margin_{grade}_log"].diff()
        added.append(f"margin_{grade}")
    if added:
        print(f"  built {len(added)} packer margin(s): {', '.join(added)}")
    return out


def add_features(table: pd.DataFrame, price_columns: list[str], horizon: int,
                 add_lags: bool, target_columns: list[str] | None = None) -> pd.DataFrame:
    out = table.copy()

    # Log levels and weekly log returns for every price or volume series.
    series_columns = price_columns + ["carcass", "usd_mxn"] + US_COLUMNS
    for column in series_columns:
        if column not in out.columns:
            continue
        positive = out[column].where(out[column] > 0)
        out[f"log_{column}"] = np.log(positive)
        out[f"ret_{column}"] = out[f"log_{column}"].diff()

    # Export volumes can be zero, so use log1p and a simple change.
    for column in ["exports_tonnes", "net_sales_tonnes"]:
        if column in out.columns:
            out[f"log_{column}"] = np.log1p(out[column].clip(lower=0))
            out[f"ret_{column}"] = out[f"log_{column}"].diff()

    # Seasonality: week of year as a smooth pair rather than 52 dummies.
    week_number = out.index.isocalendar().week.astype(float).to_numpy()
    out["season_sin"] = np.sin(2 * np.pi * week_number / 52.0)
    out["season_cos"] = np.cos(2 * np.pi * week_number / 52.0)

    out = add_spreads(out)
    out = add_margins(out)

    # Lagged returns, for the ARIMA and linear-regression baselines. The recurrent
    # models build their own windows from the return columns instead.
    if add_lags:
        return_columns = [c for c in out.columns
                          if c.startswith("ret_") or c.startswith("spread_")
                          or c.startswith("margin_")]
        # Built in one concat rather than hundreds of inserts: adding ~500 columns one
        # at a time makes pandas re-block the frame each time and emit a
        # PerformanceWarning for every one.
        lagged = {f"{column}_lag{lag}": out[column].shift(lag)
                  for column in return_columns for lag in FEATURE_LAGS}
        out = pd.concat([out, pd.DataFrame(lagged, index=out.index)], axis=1)

    # Targets: the log return over the next `horizon` weeks. The only forward-looking
    # columns in the table.
    #
    # A series with a publication lag needs care when it is the target rather than a
    # predictor. The lag belongs on the features (what was knowable at week t), not on
    # the thing being forecast. So main() stashes an unlagged copy as tgt_<column>
    # before the lags are applied, and the target is built from that when it exists.
    if target_columns is None:
        target_columns = price_columns
    targets = {}
    for column in target_columns:
        stash = f"tgt_{column}"
        if stash in out.columns:
            base = np.log(out[stash].where(out[stash] > 0))
        elif f"log_{column}" in out.columns:
            base = out[f"log_{column}"]
        else:
            print(f"  WARNING: no series available for target {column}")
            continue
        targets[f"target_{column}_h{horizon}"] = base.shift(-horizon) - base
    out = pd.concat([out, pd.DataFrame(targets, index=out.index)], axis=1)

    # The unlagged stash columns would be leakage if a model were handed every column,
    # so they are removed now that the targets are built.
    out = out.drop(columns=[c for c in out.columns if c.startswith("tgt_")])

    return out


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--horizon", type=int, nargs="+", default=[HORIZON],
                        help="forecast horizon(s) in weeks. Several may be given, e.g. "
                             "--horizon 4 12: one table then carries target_<series>_h4 "
                             "and target_<series>_h12 side by side, so the two can be "
                             "compared without rebuilding or overwriting anything")
    parser.add_argument("--origin", choices=["all", "domestic"], default="all",
                        help="'domestic' excludes imported and frozen stock from the target")
    parser.add_argument("--no-lags", action="store_true", help="skip the lagged feature columns")
    parser.add_argument("--agg", choices=["mean", "median"], default="mean",
                        help="how to combine establishments reporting the same cut in a week")
    parser.add_argument("--outlier-factor", type=float, default=5.0,
                        help="drop records beyond this multiple of the cut's yearly median; "
                             "0 disables the filter")
    parser.add_argument("--targets", choices=["mexico", "us", "both"], default="both",
                        help="which series to build target columns for: the five Mexican "
                             "cuts, the US cutout and cuts, or both (the default, so the "
                             "two can be compared from one table)")
    args = parser.parse_args()

    global AGGREGATION, OUTLIER_FACTOR
    AGGREGATION = args.agg
    OUTLIER_FACTOR = args.outlier_factor
    print(f"Combining establishments per cut by: {AGGREGATION}")
    print(f"Record outlier filter: {OUTLIER_FACTOR or 'disabled'}\n")

    print("1-2. Cut prices")
    cuts, reporters = load_cuts(args.origin)
    price_columns = list(cuts.columns)

    print("\n3. Joining the other series")
    table = cuts.join(reporters, how="left")
    for series in (load_carcass(), load_fx()):
        table = table.join(series, how="left")
        print(f"  joined {series.name}")
    table = table.join(load_exports(), how="left")
    print("  joined exports_tonnes, net_sales_tonnes")

    table = table.sort_index()
    full_weeks = pd.date_range(table.index.min(), table.index.max(), freq="W-SUN")
    missing = len(full_weeks) - len(table)
    table = table.reindex(full_weeks)       # insert any absent weeks as blank rows
    table.index.name = "week"
    if missing:
        print(f"  {missing} calendar week(s) had no cut report and were inserted as blank rows")

    print("\n4. US series")
    for frame in (load_us_cuts(), load_us_cutout(), load_cattle()):
        for column in frame.columns:
            table[column] = frame[column].reindex(table.index)
    for column in US_COLUMNS:               # keep the shape stable if a file is missing
        if column not in table.columns:
            table[column] = np.nan

    # Which series get target columns.
    target_columns = []
    if args.targets in ("mexico", "both"):
        target_columns += price_columns
    if args.targets in ("us", "both"):
        target_columns += [c for c in US_TARGET_SERIES if c in table.columns]
    print(f"\n  targets ({args.targets}): {len(target_columns)} series")

    # Stash an unlagged copy of any target that carries a publication lag, so the lag
    # lands on the predictors only. See the note in add_features().
    for column in target_columns:
        if PUBLICATION_LAG_WEEKS.get(column, 0) > 0:
            table[f"tgt_{column}"] = table[column]

    print("\n5. Publication lags")
    table = apply_publication_lags(table)

    print("\n6. Features")
    horizons = sorted(set(args.horizon))
    for position, horizon in enumerate(horizons):
        # Features are identical across horizons, so they are built once on the first
        # pass and only the target columns are added on later passes.
        table = add_features(table, price_columns, horizon,
                             add_lags=(not args.no_lags) and position == 0,
                             target_columns=target_columns)
    primary = horizons[0]

    PROCESSED.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED / "weekly_model_table.csv"
    table.to_csv(out_path)

    core = price_columns + ["carcass", "usd_mxn", "exports_tonnes"]

    print(f"\nSaved {out_path.relative_to(ROOT)}")
    print(f"  rows: {len(table):,}  columns: {len(table.columns)}")
    print(f"  weeks: {table.index.min().date()} to {table.index.max().date()}")

    # Report each target family separately: the Mexican and US series start in different
    # years, so one combined count would hide how much data each model actually has.
    families = []
    if args.targets in ("mexico", "both"):
        families.append(("Mexican cuts", price_columns, core))
    if args.targets in ("us", "both"):
        us_present = [c for c in US_TARGET_SERIES if c in table.columns]
        families.append(("US cutout and cuts", us_present,
                         [c for c in us_present if c in table.columns] + ["usd_mxn"]))
    for label, series, required in families:
        targets = [f"target_{c}_h{h}" for c in series for h in horizons]
        targets = [c for c in targets if c in table.columns]
        needed = [c for c in required + targets if c in table.columns]
        complete = table.dropna(subset=needed)
        print(f"  {label}: {len(complete):,} rows with every target and predictor present")
        if len(complete):
            print(f"    usable modelling range: "
                  f"{complete.index.min().date()} to {complete.index.max().date()}")

    print("\nMissing values per core column:")
    print(table[core].isna().sum().to_string())

    notes = PROCESSED / "weekly_model_table_columns.txt"
    with notes.open("w") as handle:
        handle.write("Columns in weekly_model_table.csv\n")
        handle.write("=" * 40 + "\n\n")
        handle.write("week                 week ending Sunday (index)\n")
        handle.write("price_<cut>          weekly mean of the maximum reported price, MXN/kg\n")
        handle.write("n_reporters_<cut>    establishments reporting that cut that week\n")
        handle.write("carcass              Corral carcass price, weighted by carcass count, MXN/kg\n")
        handle.write("usd_mxn              weekly mean of the Banxico FIX rate\n")
        handle.write("exports_tonnes       US beef exports to Mexico, tonnes\n")
        handle.write("net_sales_tonnes     US beef net sales to Mexico, tonnes\n")
        handle.write("us_*                 reserved for US cut prices, cutout and live cattle\n")
        handle.write("log_<series>         natural log of the series\n")
        handle.write("ret_<series>         weekly log return (first difference of the log)\n")
        handle.write("ret_<series>_lagN    that return, N weeks earlier\n")
        handle.write("spread_<cut>         log(Mexican price) - log(paired US price in MXN/kg):\n")
        handle.write("                     the error-correction term. The single most\n")
        handle.write("                     informative US-derived feature; the raw US series\n")
        handle.write("                     carry almost nothing at weekly frequency.\n")
        handle.write("spread_<cut>_change  weekly change in that spread\n")
        handle.write("us_steer_dressed     5-Area fed steer price, USD/cwt CARCASS\n")
        handle.write("                     (LM_CT150) - the packer's input cost\n")
        handle.write("us_steer_live        the same cattle quoted per cwt LIVE weight\n")
        handle.write("margin_<grade>_usd   cutout minus dressed steer price, USD/cwt:\n")
        handle.write("                     the gross packer margin, a supply-side driver\n")
        handle.write("margin_<grade>_log   the same as a log ratio, scale-free\n")
        handle.write("margin_<grade>_log_change   weekly change in that log margin\n")
        handle.write("season_sin, season_cos   week of year as a smooth annual cycle\n")
        handle.write(f"target_<cut>_h{primary}    log return over the NEXT {primary} weeks "
                     "(the only forward-looking columns)\n\n")
        handle.write("Publication lags applied (weeks):\n")
        for key, value in PUBLICATION_LAG_WEEKS.items():
            handle.write(f"  {key:20s} {value}\n")
    print(f"\nColumn notes written to {notes.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
