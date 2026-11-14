"""Exploratory analysis of the beef cut series, Mexican and US.

Reads the weekly modelling table produced by build_dataset.py and reports, for each
target series in the chosen family:

  - summary statistics on the price level and on the weekly log return
    (mean, median, percentiles, dispersion, skew, kurtosis)
  - stationarity: ADF and KPSS on the log level and on the log return
  - how often the price does not move at all (administered prices are sticky)
  - how the series move together, within a family and across the border

Two families:

  mexico   the five SNIIM target cuts, MXN/kg, weekly mean of the maximum reported
           price across establishments
  us       USDA boxed beef: the Choice and Select cutout plus the five cuts paired
           with the Mexican targets by IMPS code, USD/cwt

Figures written to reports/figures/:

  Mexican family
    fig10_cuts_levels_facet.png      one panel per cut, price level over time
    fig11_cuts_returns_facet.png     one panel per cut, weekly log return
    fig12_cuts_correlation.png       correlation between the cuts' returns
    fig13_cuts_seasonality.png       monthly distribution of weekly returns per cut
    fig14_acf_pacf_<cut>.png         ACF and PACF per cut, level and log return

  US family
    fig20..fig24                     the same five, for the US series

  Both families together
    fig30_pairs_mx_vs_us.png         each Mexican cut against its US pair, both in
                                     MXN/kg, so the gap between them is visible
    fig31_spreads.png                the five log spreads, the error-correction term
    fig32_cross_border_correlation.png  all returns against each other

One caveat on the US panels. The table's us_* columns carry their one-week publication
lag, because that is what a forecast may use. A uniform one-week shift leaves the
distribution, the autocorrelation and the stationarity verdicts untouched and moves
only the dates on the x axis, so the plots and tables below are unaffected by it.

Usage:
    python src/eda_cuts.py
    python src/eda_cuts.py --family mexico
    python src/eda_cuts.py --max-lag 104      # two years of lags instead of one
"""

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "data" / "processed" / "weekly_model_table.csv"
CUT_MAP = ROOT / "data" / "reference" / "cut_name_map.csv"
FIGURES = ROOT / "reports" / "figures"
PERCENTILES = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
POUNDS_PER_KG = 0.45359237

# The US series, in reporting order, with the Mexican cut each one is paired with.
# The pairings are by IMPS code and are documented in data/reference/us_cut_map.csv.
US_SERIES = ["us_cutout_choice", "us_cutout_select", "us_ribeye", "us_chuck_roll",
             "us_brisket", "us_top_inside_round", "us_top_butt"]

US_LABELS = {
    "us_cutout_choice":    "US Choice cutout (600-900 lb)",
    "us_cutout_select":    "US Select cutout (600-900 lb)",
    "us_ribeye":           "US Rib, ribeye lip-on bone-in (IMPS 109E)",
    "us_chuck_roll":       "US Chuck roll (IMPS 116A)",
    "us_brisket":          "US Brisket, deckle-off boneless (IMPS 120)",
    "us_top_inside_round": "US Top inside round (IMPS 168)",
    "us_top_butt":         "US Top sirloin butt boneless (IMPS 184)",
}

# Mexican cut <-> US series, the same pairs build_dataset.py uses for the spreads.
PAIRS = {
    "price_costillar":          "us_ribeye",
    "price_chuleton_diezmillo": "us_chuck_roll",
    "price_pecho":              "us_brisket",
    "price_pulpa_negra":        "us_top_inside_round",
    "price_sirloin":            "us_top_butt",
}

UNITS = {"mexico": "MXN/kg", "us": "USD/cwt"}

pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 40)


def heading(text: str) -> None:
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


def english_names() -> dict[str, str]:
    """Spanish SNIIM cut name -> English name, from the reference mapping."""
    if not CUT_MAP.exists():
        return {}
    mapping = pd.read_csv(CUT_MAP)
    pairs = mapping[["sniim_cut_normalized", "english_name"]].drop_duplicates()
    return {row.sniim_cut_normalized.lower().replace(" ", "_"): row.english_name
            for row in pairs.itertuples()
            if isinstance(row.english_name, str) and row.sniim_cut_normalized != "DROP"}


ENGLISH = english_names()


def pretty(column: str, short: bool = False) -> str:
    """Label a series for a table or a plot title.

    price_chuleton_diezmillo -> 'Chuleton Diezmillo (Chuck chop / chuck eye steak)'
    us_chuck_roll            -> 'US Chuck roll (IMPS 116A) ~ Chuleton Diezmillo'

    The English names come from data/reference/cut_name_map.csv so the labels and the
    reference table stay in step. With short=True the name is trimmed to fit a title.
    """
    if column.startswith("us_"):
        label = US_LABELS.get(column, column)
        if short:
            return label.split(" (")[0]
        partner = next((m for m, u in PAIRS.items() if u == column), None)
        if partner:
            spanish = partner[len("price_"):].replace("_", " ").title()
            return f"{label} ~ {spanish}"
        return label

    key = column[len("price_"):] if column.startswith("price_") else column
    spanish = key.replace("_", " ").title()
    name = ENGLISH.get(key)
    if not name:
        return spanish
    if short:
        name = name.split("/")[0].strip()
    return f"{spanish} ({name})"


def load() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    if not TABLE.exists():
        raise SystemExit(f"{TABLE.relative_to(ROOT)} not found. Run build_dataset.py first.")
    table = pd.read_csv(TABLE, index_col="week", parse_dates=True)

    mexican = [c for c in table.columns
               if c.startswith("price_") and not c.startswith("price_min")]
    american = [c for c in US_SERIES if c in table.columns and table[c].notna().any()]
    families = {name: columns for name, columns
                in (("mexico", mexican), ("us", american)) if columns}
    if not families:
        raise SystemExit("No target series found in the modelling table.")
    return table, families


def returns_of(table: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    levels = table[columns]
    out = np.log(levels.where(levels > 0)).diff()
    out.columns = columns
    return out


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def describe_block(frame: pd.DataFrame, title: str, decimals: int) -> None:
    heading(title)
    summary = frame.describe(percentiles=PERCENTILES).T
    summary["median"] = frame.median()
    summary["cv"] = summary["std"] / summary["mean"]
    summary["skew"] = frame.skew()
    summary["kurtosis"] = frame.kurtosis()
    order = ["count", "mean", "median", "std", "cv", "min", "1%", "5%", "25%", "50%",
             "75%", "95%", "99%", "max", "skew", "kurtosis"]
    summary = summary[[c for c in order if c in summary.columns]]
    summary.index = [pretty(i) for i in summary.index]
    print(summary.round(decimals).to_string())


def report_statistics(table: pd.DataFrame, columns: list[str], family: str,
                      returns: pd.DataFrame) -> None:
    describe_block(table[columns], f"1. Price level ({UNITS[family]})", 2)
    describe_block(returns * 100, "2. Weekly log return (%)", 3)

    heading("3. Price stickiness and coverage")
    rows = []
    for column in columns:
        series_returns = returns[column].dropna()
        reporters = table.get(f"n_reporters_{column.replace('price_', '')}")
        observed = table[column].dropna()
        rows.append({
            "series": pretty(column, short=True),
            "weeks": int(len(observed)),
            "first": observed.index.min().date(),
            "last": observed.index.max().date(),
            "zero-change weeks %": round((series_returns == 0).mean() * 100, 1),
            "sd of weekly return %": round(float(series_returns.std() * 100), 2),
            "median reporters": (round(float(reporters.median()), 1)
                                 if reporters is not None else np.nan),
        })
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nA high zero-change share means the naive forecast is a hard floor: the")
    print("series genuinely does not move in those weeks, so predicting no change is")
    print("right by default.")

    heading("4. How the series move together (correlation of weekly log returns)")
    correlation = returns.corr()
    correlation.index = [pretty(i, short=True) for i in correlation.index]
    correlation.columns = [pretty(c, short=True) for c in correlation.columns]
    print(correlation.round(3).to_string())
    off_diagonal = correlation.where(~np.eye(len(correlation), dtype=bool)).stack()
    print(f"\nOff-diagonal correlations run {off_diagonal.min():.2f} to "
          f"{off_diagonal.max():.2f}, mean {off_diagonal.mean():.2f}.")
    print("The higher these are, the more a single joint network gains from sharing")
    print("one series' history when predicting another. Low values mean separate")
    print("models would lose little, and the multi-output design has to be justified")
    print("on other grounds.")


# ---------------------------------------------------------------------------
# Stationarity
# ---------------------------------------------------------------------------

def verdict(adf_p: float, kpss_p: float) -> str:
    if adf_p < 0.05 and kpss_p >= 0.05:
        return "stationary (both agree)"
    if adf_p >= 0.05 and kpss_p < 0.05:
        return "non-stationary (both agree)"
    return "tests disagree"


def report_stationarity(table: pd.DataFrame, columns: list[str],
                        returns: pd.DataFrame) -> None:
    heading("5. Stationarity: ADF and KPSS")
    print("ADF   H0 = unit root.  p < 0.05 -> stationary")
    print("KPSS  H0 = stationary. p < 0.05 -> non-stationary\n")

    rows = []
    for column in columns:
        level = np.log(table[column].where(table[column] > 0)).dropna()
        for label, data in (("log level", level), ("log return", returns[column].dropna())):
            if len(data) < 30:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                adf_p = adfuller(data, autolag="AIC")[1]
                kpss_p = kpss(data, regression="c", nlags="auto")[1]
            rows.append({"series": pretty(column, short=True), "form": label,
                         "n": len(data), "ADF p": round(adf_p, 4),
                         "KPSS p": round(kpss_p, 4),
                         "verdict": verdict(adf_p, kpss_p)})
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nLevels non-stationary and returns stationary means d = 1 on the log price")
    print("is the right differencing order for ARIMA, and confirms the modelling table")
    print("was right to define the target as a log return rather than a price.")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

PALETTE = ["#8b0000", "#1f4e79", "#2e7d32", "#b36b00", "#5b2c6f", "#00695c", "#a0522d"]


def save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.relative_to(ROOT)}")


def facet(index, columns: list[str], values, title: str, ylabel: str, name: str,
          zero_line: bool = False) -> None:
    """One panel per series, stacked, sharing the x axis."""
    fig, axes = plt.subplots(len(columns), 1, figsize=(12, 2.1 * len(columns)),
                             sharex=True)
    axes = np.atleast_1d(axes)
    for ax, column, colour in zip(axes, columns, PALETTE * 3):
        ax.plot(index, values[column], lw=0.7, color=colour)
        if zero_line:
            ax.axhline(0, color="black", lw=0.6)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(pretty(column, short=True), fontsize=10, loc="left")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("week")
    fig.suptitle(title, y=1.0)
    fig.tight_layout()
    save(fig, name)


def correlation_figure(returns: pd.DataFrame, title: str, name: str) -> None:
    correlation = returns.corr()
    labels = [pretty(c, short=True) for c in correlation.columns]
    size = max(6.0, 0.62 * len(labels) + 3.5)
    fig, ax = plt.subplots(figsize=(size, size * 0.88))
    image = ax.imshow(correlation.values, vmin=-0.2, vmax=1, cmap="RdYlBu_r")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = correlation.values[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if value > 0.65 or value < -0.05 else "black")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    save(fig, name)


def seasonality_figure(index, returns: pd.DataFrame, columns: list[str],
                       name: str) -> None:
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig, axes = plt.subplots(len(columns), 1, figsize=(11, 2.0 * len(columns)),
                             sharex=True)
    axes = np.atleast_1d(axes)
    for ax, column in zip(axes, columns):
        series = returns[column] * 100
        data = [series[index.month == m].dropna().values for m in range(1, 13)]
        try:
            ax.boxplot(data, tick_labels=months, widths=0.6, showfliers=False)
        except TypeError:
            ax.boxplot(data, labels=months, widths=0.6, showfliers=False)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_ylabel("% / week", fontsize=8)
        ax.set_title(pretty(column, short=True), fontsize=10, loc="left")
        ax.grid(alpha=0.3)
    fig.suptitle("Weekly return by month - is there seasonality?", y=1.0)
    fig.tight_layout()
    save(fig, name)


def acf_pacf_figures(table: pd.DataFrame, returns: pd.DataFrame, columns: list[str],
                     max_lag: int, prefix: str) -> None:
    for column in columns:
        level = np.log(table[column].where(table[column] > 0)).dropna()
        change = returns[column].dropna()
        lags = max(4, min(max_lag, len(change) // 3))

        fig, axes = plt.subplots(2, 2, figsize=(13, 7))
        plot_acf(level, lags=lags, ax=axes[0, 0], title="ACF - log price level")
        plot_pacf(level, lags=lags, ax=axes[0, 1], title="PACF - log price level",
                  method="ywm")
        plot_acf(change, lags=lags, ax=axes[1, 0], title="ACF - log return")
        plot_pacf(change, lags=lags, ax=axes[1, 1], title="PACF - log return",
                  method="ywm")
        for ax in axes.ravel():
            ax.grid(alpha=0.3)
        fig.suptitle(f"{pretty(column)} - autocorrelation ({lags} weekly lags)")
        fig.tight_layout()
        stem = column[len("price_"):] if column.startswith("price_") else column
        save(fig, f"{prefix}_acf_pacf_{stem}.png")


def pair_figures(table: pd.DataFrame) -> None:
    """Each Mexican cut against its US pair, both converted to MXN/kg.

    This is the picture behind the spread feature: two series that drift apart and
    then close the gap. If the gap were constant the spread would carry no
    information; if the series were unrelated it would never close.
    """
    pairs = [(m, u) for m, u in PAIRS.items()
             if m in table.columns and u in table.columns]
    if not pairs or "usd_mxn" not in table.columns:
        print("  pair figures skipped: need both families and usd_mxn")
        return

    fig, axes = plt.subplots(len(pairs), 1, figsize=(12, 2.3 * len(pairs)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (mexican, american) in zip(axes, pairs):
        us_mxn = table[american] / 100.0 * table["usd_mxn"] / POUNDS_PER_KG
        ax.plot(table.index, table[mexican], lw=0.8, color="#8b0000",
                label="Mexico (SNIIM)")
        ax.plot(table.index, us_mxn, lw=0.8, color="#1f4e79",
                label="US (USDA, converted)")
        ax.set_ylabel("MXN/kg", fontsize=8)
        ax.set_title(f"{pretty(mexican, short=True)}  vs  {pretty(american, short=True)}",
                     fontsize=10, loc="left")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")
    axes[-1].set_xlabel("week")
    fig.suptitle("Paired cuts, both in MXN/kg - the gap is the spread", y=1.0)
    fig.tight_layout()
    save(fig, "fig30_pairs_mx_vs_us.png")

    # Name the five spread columns explicitly. Matching on the "spread_" prefix would
    # also pick up spread_<cut>_lag1 and spread_<cut>_change_lag1, which belong to the
    # feature set, not to this figure.
    spreads = [(mexican, f"spread_{mexican[len('price_'):]}") for mexican in PAIRS]
    spreads = [(m, s) for m, s in spreads if s in table.columns]
    if not spreads:
        return

    # One panel each: the Costillar spread sits near -1.5 while the others hover around
    # zero, so a shared axis would flatten four of the five into a single line.
    fig, axes = plt.subplots(len(spreads), 1, figsize=(12, 1.9 * len(spreads)),
                             sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (mexican, column), colour in zip(axes, spreads, PALETTE * 3):
        series = table[column]
        ax.plot(table.index, series, lw=0.8, color=colour)
        ax.axhline(0, color="black", lw=0.6)
        mean = series.mean()
        ax.axhline(mean, color="black", lw=0.8, ls="--",
                   label=f"mean {mean:+.2f}")
        ax.set_ylabel("log gap", fontsize=8)
        ax.set_title(f"{pretty(mexican, short=True)} vs "
                     f"{pretty(PAIRS[mexican], short=True)}", fontsize=10, loc="left")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")
    axes[-1].set_xlabel("week")
    fig.suptitle("The error-correction term: log(Mexican price) - log(US price in "
                 "MXN/kg).\nA spread that returns to its mean is forecastable; one "
                 "that drifts away is not.", y=1.0)
    fig.tight_layout()
    save(fig, "fig31_spreads.png")


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--family", choices=["mexico", "us", "both"], default="both")
    parser.add_argument("--max-lag", type=int, default=52,
                        help="lags shown in the ACF and PACF plots (52 = one year)")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    table, families = load()
    print(f"Modelling table: {len(table):,} weeks, "
          f"{table.index.min().date()} to {table.index.max().date()}")

    wanted = list(families) if args.family == "both" else [args.family]
    prefixes = {"mexico": ("fig10", "fig11", "fig12", "fig13", "fig14"),
                "us": ("fig20", "fig21", "fig22", "fig23", "fig24")}
    titles = {"mexico": "Mexican beef cut prices (SNIIM, weekly)",
              "us": "US boxed beef prices (USDA, weekly)"}

    all_returns = {}
    for family in wanted:
        if family not in families:
            print(f"\n{family}: no series in the table, skipped")
            continue
        columns = families[family]
        returns = returns_of(table, columns)
        all_returns[family] = returns

        print("\n\n" + "#" * 78)
        print(f"# {family.upper()} FAMILY - {len(columns)} series")
        print("#" * 78)
        print(f"Series: {', '.join(pretty(c, short=True) for c in columns)}")

        report_statistics(table, columns, family, returns)
        report_stationarity(table, columns, returns)

        if not args.no_figures:
            heading(f"6. Figures ({family})")
            a, b, c, d, e = prefixes[family]
            facet(table.index, columns, table, titles[family], UNITS[family],
                  f"{a}_{family}_levels_facet.png")
            facet(table.index, columns, returns * 100,
                  f"Weekly log returns - {family}", "% / week",
                  f"{b}_{family}_returns_facet.png", zero_line=True)
            correlation_figure(returns,
                               f"Correlation of weekly log returns - {family}",
                               f"{c}_{family}_correlation.png")
            seasonality_figure(table.index, returns, columns,
                               f"{d}_{family}_seasonality.png")
            acf_pacf_figures(table, returns, columns, args.max_lag, e)

    if not args.no_figures and len(all_returns) == 2:
        heading("7. Cross-border figures")
        pair_figures(table)
        combined = pd.concat([all_returns["mexico"], all_returns["us"]], axis=1)
        correlation_figure(combined,
                           "Weekly log returns: Mexican and US series together",
                           "fig32_cross_border_correlation.png")

        heading("8. Cross-border correlation of weekly returns")
        block = combined.corr().loc[families["mexico"], families["us"]]
        block.index = [pretty(i, short=True) for i in block.index]
        block.columns = [pretty(c, short=True) for c in block.columns]
        print(block.round(3).to_string())
        print("\nEach row is a Mexican cut, each column a US series. Values near zero")
        print("mean US weekly moves carry no contemporaneous information about Mexican")
        print("ones, which is why the level spread, not the US return, is the feature")
        print("that matters. Read the paired diagonal entries first.")


if __name__ == "__main__":
    main()
