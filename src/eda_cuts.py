"""Exploratory analysis of the five Mexican beef cut series.

Reads the weekly modelling table produced by build_dataset.py and reports, for each
of the five target cuts:

  - summary statistics on the price level and on the weekly log return
    (mean, median, percentiles, dispersion, skew, kurtosis)
  - stationarity: ADF and KPSS on the level and on the log return
  - how often the price does not move at all (SNIIM prices are sticky)

Figures written to reports/figures/:

  fig10_cuts_levels_facet.png     one panel per cut, price level over time
  fig11_cuts_returns_facet.png    one panel per cut, weekly log return
  fig12_cuts_correlation.png      correlation between the cuts' returns
  fig13_cuts_seasonality.png      monthly distribution of weekly returns per cut
  fig14_acf_pacf_<cut>.png        ACF and PACF per cut, level and log return

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

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "data" / "processed" / "weekly_model_table.csv"
CUT_MAP = ROOT / "data" / "reference" / "cut_name_map.csv"
FIGURES = ROOT / "reports" / "figures"
PERCENTILES = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]

pd.set_option("display.width", 150)
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
    """price_chuleton_diezmillo -> 'Chuleton Diezmillo (Chuck chop / chuck eye steak)'

    The English name comes from data/reference/cut_name_map.csv so the two stay in step.
    With short=True the English name is trimmed at the first '/' to fit plot titles.
    """
    key = column.replace("price_", "")
    spanish = key.replace("_", " ").title()
    name = ENGLISH.get(key)
    if not name:
        return spanish
    if short:
        name = name.split("/")[0].strip()
    return f"{spanish} ({name})"


def load() -> tuple[pd.DataFrame, list[str]]:
    if not TABLE.exists():
        raise SystemExit(f"{TABLE.relative_to(ROOT)} not found. Run build_dataset.py first.")
    table = pd.read_csv(TABLE, index_col="week", parse_dates=True)
    price_columns = [c for c in table.columns
                     if c.startswith("price_") and not c.startswith("price_min")]
    if not price_columns:
        raise SystemExit("No price_ columns found in the modelling table.")
    return table, price_columns


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


def report_statistics(table: pd.DataFrame, price_columns: list[str]) -> pd.DataFrame:
    levels = table[price_columns]
    returns = np.log(levels.where(levels > 0)).diff()
    returns.columns = price_columns

    describe_block(levels, "1. Price level by cut (MXN/kg)", 2)
    describe_block(returns * 100, "2. Weekly log return by cut (%)", 3)

    heading("3. Price stickiness and coverage")
    rows = []
    for column in price_columns:
        series_returns = returns[column].dropna()
        reporters = table.get(f"n_reporters_{column.replace('price_', '')}")
        rows.append({
            "cut": pretty(column),
            "weeks": int(table[column].notna().sum()),
            "first": table[column].dropna().index.min().date(),
            "last": table[column].dropna().index.max().date(),
            "zero-change weeks %": round((series_returns == 0).mean() * 100, 1),
            "median reporters": (round(float(reporters.median()), 1)
                                 if reporters is not None else np.nan),
        })
    print(pd.DataFrame(rows).to_string(index=False))

    heading("4. How the cuts move together (correlation of weekly log returns)")
    correlation = returns.corr()
    correlation.index = [pretty(i) for i in correlation.index]
    correlation.columns = [pretty(c) for c in correlation.columns]
    print(correlation.round(3).to_string())
    print("\nHigher correlations mean a joint model has more to gain: one network can use")
    print("one cut's history to help predict another, which separate models cannot.")
    return returns


# ---------------------------------------------------------------------------
# Stationarity
# ---------------------------------------------------------------------------

def verdict(adf_p: float, kpss_p: float) -> str:
    if adf_p < 0.05 and kpss_p >= 0.05:
        return "stationary (both agree)"
    if adf_p >= 0.05 and kpss_p < 0.05:
        return "non-stationary (both agree)"
    return "tests disagree"


def report_stationarity(table: pd.DataFrame, price_columns: list[str],
                        returns: pd.DataFrame) -> None:
    heading("5. Stationarity: ADF and KPSS")
    print("ADF   H0 = unit root.  p < 0.05 -> stationary")
    print("KPSS  H0 = stationary. p < 0.05 -> non-stationary\n")

    rows = []
    for column in price_columns:
        for label, data in (("log level", np.log(table[column].where(table[column] > 0)).dropna()),
                            ("log return", returns[column].dropna())):
            if len(data) < 30:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                adf_p = adfuller(data, autolag="AIC")[1]
                kpss_p = kpss(data, regression="c", nlags="auto")[1]
            rows.append({"cut": pretty(column), "form": label, "n": len(data),
                         "ADF p": round(adf_p, 4), "KPSS p": round(kpss_p, 4),
                         "verdict": verdict(adf_p, kpss_p)})
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nIf the levels are non-stationary and the returns are stationary, d = 1 on the")
    print("log price is the right differencing order for ARIMA.")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.relative_to(ROOT)}")


def facet(table: pd.DataFrame, columns: list[str], values, title: str,
          ylabel: str, name: str, zero_line: bool = False) -> None:
    """One panel per cut, stacked, sharing the x axis."""
    fig, axes = plt.subplots(len(columns), 1, figsize=(12, 2.1 * len(columns)), sharex=True)
    axes = np.atleast_1d(axes)
    palette = ["#8b0000", "#1f4e79", "#2e7d32", "#b36b00", "#5b2c6f", "#00695c"]
    for ax, column, colour in zip(axes, columns, palette):
        ax.plot(table.index, values[column], lw=0.7, color=colour)
        if zero_line:
            ax.axhline(0, color="black", lw=0.6)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(pretty(column, short=True), fontsize=10, loc="left")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("week")
    fig.suptitle(title, y=1.0)
    fig.tight_layout()
    save(fig, name)


def correlation_figure(returns: pd.DataFrame) -> None:
    correlation = returns.corr()
    labels = [pretty(c, short=True) for c in correlation.columns]
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(correlation.values, vmin=0, vmax=1, cmap="YlGnBu")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(labels)), labels, fontsize=9)
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = correlation.values[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if value > 0.6 else "black")
    ax.set_title("Correlation of weekly log returns between cuts")
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    save(fig, "fig12_cuts_correlation.png")


def seasonality_figure(table: pd.DataFrame, returns: pd.DataFrame,
                       columns: list[str]) -> None:
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig, axes = plt.subplots(len(columns), 1, figsize=(11, 2.0 * len(columns)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, column in zip(axes, columns):
        series = returns[column] * 100
        data = [series[table.index.month == m].dropna().values for m in range(1, 13)]
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
    save(fig, "fig13_cuts_seasonality.png")


def acf_pacf_figures(table: pd.DataFrame, returns: pd.DataFrame,
                     columns: list[str], max_lag: int) -> None:
    for column in columns:
        level = np.log(table[column].where(table[column] > 0)).dropna()
        change = returns[column].dropna()
        lags = min(max_lag, len(change) // 3)

        fig, axes = plt.subplots(2, 2, figsize=(13, 7))
        plot_acf(level, lags=lags, ax=axes[0, 0], title="ACF - log price level")
        plot_pacf(level, lags=lags, ax=axes[0, 1], title="PACF - log price level",
                  method="ywm")
        plot_acf(change, lags=lags, ax=axes[1, 0], title="ACF - log return")
        plot_pacf(change, lags=lags, ax=axes[1, 1], title="PACF - log return", method="ywm")
        for ax in axes.ravel():
            ax.grid(alpha=0.3)
        fig.suptitle(f"{pretty(column)} - autocorrelation ({lags} weekly lags)")
        fig.tight_layout()
        save(fig, f"fig14_acf_pacf_{column.replace('price_', '')}.png")


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-lag", type=int, default=52,
                        help="lags shown in the ACF and PACF plots (52 = one year)")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    table, price_columns = load()
    print(f"Modelling table: {len(table):,} weeks, "
          f"{table.index.min().date()} to {table.index.max().date()}")
    print(f"Target cuts: {', '.join(pretty(c) for c in price_columns)}")

    returns = report_statistics(table, price_columns)
    report_stationarity(table, price_columns, returns)

    if not args.no_figures:
        heading("6. Figures")
        facet(table, price_columns, table, "Mexican beef cut prices (weekly)",
              "MXN/kg", "fig10_cuts_levels_facet.png")
        facet(table, price_columns, returns * 100, "Weekly log returns by cut",
              "% / week", "fig11_cuts_returns_facet.png", zero_line=True)
        correlation_figure(returns)
        seasonality_figure(table, returns, price_columns)
        acf_pacf_figures(table, returns, price_columns, args.max_lag)


if __name__ == "__main__":
    main()
