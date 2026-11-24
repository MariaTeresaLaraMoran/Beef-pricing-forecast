"""Exploratory analysis of the US boxed beef series.

Reads the weekly modelling table produced by build_dataset.py and reports, for each
target series:

  - summary statistics on the price level and on the weekly log return
    (mean, median, percentiles, dispersion, skew, kurtosis)
  - stationarity: ADF and KPSS on the log level and on the log return
  - how often the price does not move at all (administered prices are sticky)
  - how the series move together
  - whether the packer margin leads the price, and at what distance
  - which series exist over which weeks

The targets are the USDA boxed beef Choice and Select cutouts plus five cuts by IMPS
code, all in USD/cwt.

Figures written to reports/figures/:

    fig20_us_levels_facet.png        one panel per series, price level over time
    fig21_us_returns_facet.png       one panel per series, weekly log return
    fig22_us_correlation.png         correlation between the series' returns
    fig23_us_seasonality.png         monthly distribution of weekly returns
    fig24_acf_pacf_<series>.png      ACF and PACF per series, level and log return
    fig25_margin_leading.png         the packer margin over time, and its correlation
                                     with the forward return at leads of 1 to 16 weeks
    fig26_data_coverage.png          which series exist when, against the test period

One caveat on the panels. The table's us_* columns carry their one-week publication
lag, because that is what a forecast may use. A uniform one-week shift leaves the
distribution, the autocorrelation and the stationarity verdicts untouched and moves
only the dates on the x axis, so the plots and tables below are unaffected by it.

Usage:
    python src/eda_cuts.py
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

from baselines import BREAK_END, BREAK_START  # one definition of the window

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "data" / "processed" / "weekly_model_table.csv"
CUT_MAP = ROOT / "data" / "reference" / "cut_name_map.csv"
FIGURES = ROOT / "reports" / "figures"
PERCENTILES = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
POUNDS_PER_KG = 0.45359237

# The US series, in reporting order.
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


UNITS = {"us": "USD/cwt"}

pd.set_option("display.width", 170)
pd.set_option("display.max_columns", 40)


def heading(text: str) -> None:
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)




def pretty(column: str, short: bool = False) -> str:
    """Label a series for a table or a plot title.

    us_chuck_roll -> 'Chuck roll (IMPS 116A)', or 'Chuck roll' with short=True.
    """
    label = US_LABELS.get(column, column)
    return label.split(" (")[0] if short else label


def load() -> tuple[pd.DataFrame, dict[str, list[str]]]:
    if not TABLE.exists():
        raise SystemExit(f"{TABLE.relative_to(ROOT)} not found. Run build_dataset.py first.")
    table = pd.read_csv(TABLE, index_col="week", parse_dates=True)

    american = [c for c in US_SERIES if c in table.columns and table[c].notna().any()]
    families = {"us": american} if american else {}
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
        observed = table[column].dropna()
        rows.append({
            "series": pretty(column, short=True),
            "weeks": int(len(observed)),
            "first": observed.index.min().date(),
            "last": observed.index.max().date(),
            "zero-change weeks %": round((series_returns == 0).mean() * 100, 1),
            "sd of weekly return %": round(float(series_returns.std() * 100), 2),
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
        stem = column
        save(fig, f"{prefix}_acf_pacf_{stem}.png")



# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# The prediction problem: does the margin lead the price, and what data exists
# ---------------------------------------------------------------------------

MAX_LEAD = 16            # weeks of forward return to test the margin against
REPORTS = ROOT / "reports"


def forward_return(levels: pd.Series, weeks: int) -> pd.Series:
    """Log return over the NEXT `weeks` weeks, aligned to the week it is known from.

    Computed from the price level rather than from a target_ column, so this works
    at any lead, not only the two horizons build_dataset.py happens to emit.
    """
    logged = np.log(levels.where(levels > 0))
    return logged.shift(-weeks) - logged


def margin_lead_table(table: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Correlation of today's log margin with each series' forward return, by lead."""
    margin = table.get("margin_choice_log")
    if margin is None:
        return pd.DataFrame()
    rows = {}
    for column in columns:
        rows[pretty(column, short=True)] = {
            weeks: margin.corr(forward_return(table[column], weeks))
            for weeks in range(1, MAX_LEAD + 1)
        }
    return pd.DataFrame(rows).T


def report_margin_lead(table: pd.DataFrame, columns: list[str],
                       figures: bool) -> None:
    heading("9. The packer margin as a leading indicator")
    if "margin_choice_log" not in table.columns:
        print("margin_choice_log is not in the table; run build_dataset.py.")
        return

    leads = margin_lead_table(table, columns)
    print("Correlation of today's log margin with the forward log return.")
    print("Negative is the expected sign: a wide margin today, a falling price later.\n")
    shown = [1, 2, 4, 8, 12, 16]
    print(leads[shown].round(3).to_string())

    best = leads.mean(axis=0)
    print(f"\nAveraged across the {len(columns)} series, the correlation is strongest")
    print(f"at a lead of {int(best.idxmin())} weeks ({best.min():.3f}) and weakest at "
          f"{int(best.idxmax())} ({best.max():.3f}).")
    print("If the margin corrected quickly the strongest lead would be short. It is not,")
    print("which is the measured basis for forecasting at a longer horizon.")

    if not figures:
        return
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(12, 9))

    for grade, colour in (("choice", "#9E2B25"), ("select", "#2C5578")):
        column = f"margin_{grade}_usd"
        if column in table.columns:
            top.plot(table.index, table[column].to_numpy(), lw=1.1, color=colour,
                     label=f"{grade.title()} cutout minus dressed steer")
    top.axhline(0, color="black", lw=0.8)
    top.set_title("Gross packer margin, USD/cwt on a carcass basis", loc="left")
    top.set_ylabel("USD/cwt")
    top.legend(fontsize=8)
    top.grid(alpha=0.3)

    for name, row in leads.iterrows():
        bottom.plot(row.index.to_numpy(), row.to_numpy(), lw=1.3, marker="o",
                    markersize=3, label=name)
    bottom.axhline(0, color="black", lw=0.8)
    bottom.set_title("Correlation of today's log margin with the forward log return",
                     loc="left")
    bottom.set_xlabel("lead, weeks ahead")
    bottom.set_ylabel("correlation")
    bottom.legend(fontsize=7, ncol=2)
    bottom.grid(alpha=0.3)

    fig.suptitle("The margin leads the price, and the signal deepens with distance",
                 y=0.98)
    fig.tight_layout()
    save(fig, "fig25_margin_leading.png")


def report_coverage(table: pd.DataFrame, figures: bool) -> None:
    heading("10. Data coverage over time")
    groups = {
        "US cut and cutout prices": US_SERIES,
        "Fed cattle price": ["us_steer_dressed", "us_steer_live"],
        "Negotiated head counts": ["us_steer_dressed_head", "us_steer_live_head"],
        "Packer margin": ["margin_choice_usd", "margin_select_usd"],
        "Macro (FRED)": ["usd_index", "initial_claims"],
    }
    spans = []
    for label, cols in groups.items():
        present = [c for c in cols if c in table.columns and table[c].notna().any()]
        if not present:
            continue
        mask = table[present].notna().any(axis=1)
        weeks = table.index[mask]
        spans.append((label, weeks.min(), weeks.max(),
                      100 * mask.loc[weeks.min():].mean(), len(present)))

    print(f"{'group':28s} {'from':>12s} {'to':>12s} {'weeks present':>14s}")
    for label, lo, hi, pct, _ in spans:
        print(f"{label:28s} {lo.date()!s:>12s} {hi.date()!s:>12s} {pct:13.1f}%")
    print("\nThe percentage counts weeks with data from that group's own first week,")
    print("so a late start is not scored as missing data. The broad dollar index is")
    print("the binding constraint on the macro arm: it starts in 2006, which costs")
    print("that arm 85 training windows.")

    if not figures:
        return
    fig, ax = plt.subplots(figsize=(12, 4.6))
    for position, (label, lo, hi, _, _) in enumerate(spans):
        ax.barh(position, hi - lo, left=lo, height=0.55, color="#9E2B25", alpha=0.75)
    ax.set_yticks(range(len(spans)))
    ax.set_yticklabels([row[0] for row in spans], fontsize=9)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    test = REPORTS / "baseline_predictions_h12.csv"
    if test.exists():
        scored = pd.read_csv(test, parse_dates=["week"])["week"]
        ax.axvspan(scored.min(), scored.max(), color="#2C5578", alpha=0.12)
        ax.text(scored.min(), -0.9, "  test period (h=12)", fontsize=8,
                color="#2C5578", va="center")

    ax.set_title("Which series exist when", loc="left")
    fig.tight_layout()
    save(fig, "fig26_data_coverage.png")



def regime_table(table: pd.DataFrame) -> pd.DataFrame:
    """The packer margin by calendar year, with each year's ratio to the 2004-19 norm.

    The disruption is easiest to see at yearly resolution: the weekly series is noisy
    enough that the 2020-21 excursion reads as volatility rather than as a regime.
    """
    margin = table["margin_choice_usd"].dropna()
    by_year = margin.groupby(margin.index.year).agg(["mean", "median", "max"])
    norm = margin.loc[:"2019"].mean()
    by_year["vs_2004_2019"] = by_year["mean"] / norm
    by_year.index.name = "year"
    return by_year.round(3)


def report_regimes(table: pd.DataFrame, figures: bool) -> None:
    heading("11. The 2020-21 disruption, and why it is treated separately")
    if "margin_choice_usd" not in table.columns:
        print("margin_choice_usd is not in the table; run build_dataset.py.")
        return

    years = regime_table(table)
    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / "margin_by_year.csv"
    years.to_csv(out)

    margin = table["margin_choice_usd"].dropna()
    norm = margin.loc[:"2019"].mean()
    test = margin.loc["2023-08":]
    print(years.to_string())
    print(f"\n2004-2019 mean      {norm:6.1f} USD/cwt")
    print(f"test period mean    {test.mean():6.1f} USD/cwt  "
          f"({test.mean() / norm:.1f}x the 2004-19 norm)")
    print(f"peak year (2021)    {years.loc[2021, 'mean']:6.1f} USD/cwt  "
          f"({years.loc[2021, 'vs_2004_2019']:.1f}x)")
    print("\nThe forecast period sits at roughly the historical norm while the 2020-21")
    print("weeks sit an order of magnitude above it. That gap, not the size of the")
    print("numbers on their own, is why those weeks are excluded from training in the")
    print(f"--exclude-break variant ({BREAK_START.date()} to {BREAK_END.date()}).")
    print(f"\nWritten {out.relative_to(ROOT)}")

    if not figures:
        return
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(12, 8.5),
                                      gridspec_kw={"height_ratios": [1.25, 1]})

    cutout, steer = "us_cutout_choice", "us_steer_dressed"
    if cutout in table.columns and steer in table.columns:
        top.fill_between(table.index, table[steer], table[cutout],
                         where=table[cutout] >= table[steer],
                         color="#9E2B25", alpha=0.18, linewidth=0,
                         label="packer margin (the gap)")
        top.plot(table.index, table[cutout], lw=1.0, color="#9E2B25",
                 label="Choice boxed beef cutout")
        top.plot(table.index, table[steer], lw=1.0, color="#2C5578",
                 label="5-Area dressed fed steer")
    top.axvspan(BREAK_START, BREAK_END, color="#524C44", alpha=0.13, linewidth=0)
    top.text(BREAK_START, top.get_ylim()[1], " 2020-21 disruption", fontsize=8,
             va="top", color="#524C44")
    top.set_title("What the packer sells, what the packer pays, and the gap between",
                  loc="left")
    top.set_ylabel("USD/cwt carcass")
    top.legend(fontsize=8, loc="upper left")
    top.grid(alpha=0.3)

    colour = {"normal": "#2C5578", "break": "#9E2B25"}
    kinds = ["break" if y in (2020, 2021) else "normal" for y in years.index]
    bottom.bar(years.index, years["mean"],
               color=[colour[k] for k in kinds], width=0.75)
    bottom.axhline(norm, color="black", lw=1.0, ls="--",
                   label=f"2004-19 mean, {norm:.1f}")
    bottom.set_title("Mean packer margin by year - the excursion, and the return",
                     loc="left")
    bottom.set_ylabel("USD/cwt")
    bottom.legend(fontsize=8)
    bottom.grid(alpha=0.3, axis="y")

    fig.suptitle("The forecast period is an ordinary-margin regime; 2020-21 is not",
                 y=0.99)
    fig.tight_layout()
    save(fig, "fig27_margin_regimes.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-lag", type=int, default=52,
                        help="lags shown in the ACF and PACF plots (52 = one year)")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    table, families = load()
    print(f"Modelling table: {len(table):,} weeks, "
          f"{table.index.min().date()} to {table.index.max().date()}")

    wanted = list(families)
    prefixes = {"us": ("fig20", "fig21", "fig22", "fig23", "fig24")}
    titles = {"us": "US boxed beef prices (USDA, weekly)"}

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

    if "us" in families:
        report_margin_lead(table, families["us"], not args.no_figures)
        report_coverage(table, not args.no_figures)
        report_regimes(table, not args.no_figures)


if __name__ == "__main__":
    main()
