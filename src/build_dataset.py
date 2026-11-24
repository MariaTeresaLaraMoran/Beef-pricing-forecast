"""Build the weekly modelling table from the raw series.

Steps, in order:
  1. Load the USDA boxed beef cut and cutout prices and aggregate them to weekly
  2. Join the 5-Area fed cattle price, negotiated head counts and the FRED macro series
  3. Build the gross packer margin: cutout minus dressed steer, on a carcass basis
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
PUBLICATION_LAG_WEEKS = {
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
    "us_steer_dressed_head": 1,
    "us_steer_live_head": 1,
    # The dollar index publishes each day's value on the next business day, so a
    # week's average is not complete until the Monday after the week closes.
    "usd_index": 1,
    # Initial claims for a week ending Saturday are released the following Thursday.
    "initial_claims": 1,
}

# Columns reserved for the US series, created empty so the table's shape is stable
# and they can be filled in without changing anything downstream.
US_COLUMNS = ["us_ribeye", "us_chuck_roll", "us_brisket", "us_top_inside_round",
              "us_top_butt", "us_cutout_choice", "us_cutout_select",
              "us_steer_dressed", "us_steer_live"]

# Weekly negotiated head counts from the same AMS report: how much cattle actually
# traded. The packer margin is the incentive to slaughter; this is closer to the
# decision itself. Counts, not prices, so they are logged with log1p alongside exports.
#
# One caveat for the write-up: this is 5-Area NEGOTIATED volume, not total federally
# inspected slaughter. The negotiated share of US fed cattle trade has fallen over the
# period as formula and contract purchasing grew, so the level carries a downward drift
# from market structure rather than activity. Use the return, never the level.
VOLUME_COLUMNS = ["us_steer_dressed_head", "us_steer_live_head"]

# Macro series from FRED, added as a testable ablation arm rather than as core inputs.
FRED_COLUMNS = ["usd_index", "initial_claims"]

# The USDA boxed beef items forecast here, by IMPS code. See
# data/reference/us_cut_map.csv for why each pairing was chosen.
US_ITEMS = {
    "us_ribeye":            "Rib, ribeye, lip-on, bn-in (109E  1)",   # <- Costillar
    "us_chuck_roll":        "Chuck, roll, lxl, neck/off (116A  3)",   # <- Chuleton Diezmillo
    "us_brisket":           "Brisket, deckle-off, bnls (120  1)",     # <- Pecho
    "us_top_inside_round":  "Round, top inside round (168  3)",       # <- Pulpa Negra
    "us_top_butt":          "Loin, top butt, boneless (184  3)",      # <- Sirloin
}

# The US series that carry target columns.
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


def to_number(series: pd.Series) -> pd.Series:
    """USDA sends numbers as strings with thousands separators."""
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.strip(), errors="coerce")


def load_us_cuts() -> pd.DataFrame:
    """Weekly US prices for the five individual cuts, USD per cwt."""
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
        return pd.DataFrame({c: pd.Series(dtype="float64")
                             for c in ["us_steer_dressed", "us_steer_live"]})

    df = pd.read_csv(path, parse_dates=["report_date"])
    df["week"] = week_of(df["report_date"])
    out = pd.DataFrame()
    for column in ("us_steer_dressed", "us_steer_live"):
        if column in df.columns:
            out[column] = df.groupby("week")[column].mean()
    for column in VOLUME_COLUMNS:
        if column in df.columns:
            out[column] = df.groupby("week")[column].sum(min_count=1)
    print(f"  joined fed cattle ({out.index.min().date() if len(out) else '-'} onwards)")
    return out


def load_fred() -> pd.DataFrame:
    """Weekly macro series produced by fetch_fred.py.

    These are not core inputs. They exist so the `plus-macro` arm of the ablation has
    something to test: whether broad economic conditions carry information about US
    beef prices that the beef market's own data does not already contain.
    """
    path = RAW / "fred_weekly.csv"
    if not path.exists():
        print("  fred_weekly.csv not found - run fetch_fred.py; macro columns left empty")
        return pd.DataFrame({c: pd.Series(dtype="float64") for c in FRED_COLUMNS})

    df = pd.read_csv(path, index_col="week", parse_dates=True)
    keep = [c for c in FRED_COLUMNS if c in df.columns]
    out = df[keep]
    if len(out):
        starts = ", ".join(f"{c} from {out[c].dropna().index.min().date()}"
                           for c in keep if out[c].notna().any())
        print(f"  joined FRED macro ({starts})")
    return out


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


def add_features(table: pd.DataFrame, horizon: int, add_lags: bool,
                 target_columns: list[str]) -> pd.DataFrame:
    out = table.copy()

    # Log levels and weekly log returns for every price or volume series.
    series_columns = US_COLUMNS + ["usd_index"]
    for column in series_columns:
        if column not in out.columns:
            continue
        # Coerce before taking logs. A series whose source file was absent arrives as
        # an all-NaN column of dtype object, and np.log raises on object dtype rather
        # than propagating NaN, which would stop the whole build over a missing
        # optional input.
        values = pd.to_numeric(out[column], errors="coerce")
        positive = values.where(values > 0)
        out[f"log_{column}"] = np.log(positive)
        out[f"ret_{column}"] = out[f"log_{column}"].diff()

    # Counts and claims rather than prices, and they can touch zero, so log1p.
    for column in ["initial_claims"] + VOLUME_COLUMNS:
        if column in out.columns:
            values = pd.to_numeric(out[column], errors="coerce")
            out[f"log_{column}"] = np.log1p(values.clip(lower=0))
            out[f"ret_{column}"] = out[f"log_{column}"].diff()

    # Seasonality: week of year as a smooth pair rather than 52 dummies.
    week_number = out.index.isocalendar().week.astype(float).to_numpy()
    out["season_sin"] = np.sin(2 * np.pi * week_number / 52.0)
    out["season_cos"] = np.cos(2 * np.pi * week_number / 52.0)

    out = add_margins(out)

    # Lagged returns, for the ARIMA and linear-regression baselines. The recurrent
    # models build their own windows from the return columns instead.
    if add_lags:
        return_columns = [c for c in out.columns
                          if c.startswith("ret_") or c.startswith("margin_")]
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
    parser.add_argument("--no-lags", action="store_true", help="skip the lagged feature columns")
    args = parser.parse_args()

    print("1. US cut and cutout prices, fed cattle, macro")
    frames = [load_us_cuts(), load_us_cutout(), load_cattle(), load_fred()]
    # Built column by column off the union of the indexes rather than by concat: the
    # sources do not share a calendar, and a duplicate week in any one of them would
    # make a concat raise rather than align.
    index = frames[0].index
    for frame in frames[1:]:
        index = index.union(frame.index)
    table = pd.DataFrame(index=index.sort_values())
    for frame in frames:
        for column in frame.columns:
            table[column] = frame[column].reindex(table.index)

    full_weeks = pd.date_range(table.index.min(), table.index.max(), freq="W-SUN")
    missing = len(full_weeks) - len(table)
    table = table.reindex(full_weeks)       # insert any absent weeks as blank rows
    table.index.name = "week"
    if missing:
        print(f"  {missing} calendar week(s) had no report and were inserted as blank rows")
    for column in US_COLUMNS + VOLUME_COLUMNS + FRED_COLUMNS:  # stable shape if absent
        if column not in table.columns:
            table[column] = np.nan

    target_columns = [c for c in US_TARGET_SERIES if c in table.columns]
    print(f"\n  targets: {len(target_columns)} series")

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
        table = add_features(table, horizon,
                             add_lags=(not args.no_lags) and position == 0,
                             target_columns=target_columns)
    primary = horizons[0]

    PROCESSED.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED / "weekly_model_table.csv"
    table.to_csv(out_path)

    core = ["us_cutout_choice", "us_steer_dressed", "margin_choice_usd"]

    print(f"\nSaved {out_path.relative_to(ROOT)}")
    print(f"  rows: {len(table):,}  columns: {len(table.columns)}")
    print(f"  weeks: {table.index.min().date()} to {table.index.max().date()}")

    # How many rows each horizon can actually be scored on. The count falls as the
    # horizon grows, because the last `horizon` weeks have no target yet.
    us_present = [c for c in US_TARGET_SERIES if c in table.columns]
    for horizon in horizons:
        targets = [f"target_{c}_h{horizon}" for c in us_present]
        needed = [c for c in us_present + targets if c in table.columns]
        complete = table.dropna(subset=needed)
        print(f"  h={horizon}: {len(complete):,} rows with every target and predictor present")
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
        handle.write("us_<cut>             weekly USDA boxed beef price, USD/cwt\n")
        handle.write("log_<series>         natural log of the series\n")
        handle.write("ret_<series>         weekly log return (first difference of the log)\n")
        handle.write("ret_<series>_lagN    that return, N weeks earlier\n")

        handle.write("us_steer_dressed     5-Area fed steer price, USD/cwt CARCASS\n")
        handle.write("                     (LM_CT150) - the packer's input cost\n")
        handle.write("us_steer_live        the same cattle quoted per cwt LIVE weight\n")
        handle.write("margin_<grade>_usd   cutout minus dressed steer price, USD/cwt:\n")
        handle.write("                     the gross packer margin, a supply-side driver\n")
        handle.write("margin_<grade>_log   the same as a log ratio, scale-free\n")
        handle.write("margin_<grade>_log_change   weekly change in that log margin\n")
        handle.write("us_steer_*_head      weekly NEGOTIATED head count, 5-Area:\n")
        handle.write("                     throughput, not total federal slaughter\n")
        handle.write("usd_index            FRED DTWEXBGS, broad trade-weighted dollar\n")
        handle.write("initial_claims       FRED ICSA, weekly initial jobless claims\n")
        handle.write("season_sin, season_cos   week of year as a smooth annual cycle\n")
        for horizon in horizons:
            handle.write(f"target_<cut>_h{horizon}{'':8s} log return over the NEXT {horizon} "
                         "weeks (the only forward-looking columns)\n")
        handle.write("\n")
        handle.write("Publication lags applied (weeks):\n")
        for key, value in PUBLICATION_LAG_WEEKS.items():
            handle.write(f"  {key:20s} {value}\n")
    print(f"\nColumn notes written to {notes.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
