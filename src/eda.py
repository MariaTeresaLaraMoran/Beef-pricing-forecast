"""Exploratory data analysis of the three collected series.

Prints a text report (previews, coverage, summary statistics, stationarity tests)
and saves figures to reports/figures/.

Series examined:
  1. SNIIM beef carcass prices, Corral type   -> the target
  2. USD/MXN FIX exchange rate
  3. US beef exports to Mexico

The target is built two ways, by maximum reported price and by the midpoint of the
minimum and maximum, so the choice between them can be made from evidence.

Usage:
    python src/eda.py
    python src/eda.py --type Vaca      # look at a different cattle type
"""

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # save figures to file instead of opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
FIGURES = ROOT / "reports" / "figures"
PROCESSED = ROOT / "data" / "processed"

SNIIM_PATH = RAW / "sniim_beef_carcass_prices.csv"
FX_PATH = RAW / "banxico_usdmxn_fix.csv"
EXPORTS_PATH = RAW / "usda_fas_beef_exports_mexico.csv"

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 30)


def heading(text: str) -> None:
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


# ----------------------------------------------------------------------------
# 1. Load and preview
# ----------------------------------------------------------------------------

def preview(df: pd.DataFrame, name: str, date_column: str) -> None:
    heading(f"{name}")
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print("\nColumns and types:")
    print(df.dtypes.to_string())
    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))
    print("\nMissing values per column:")
    missing = df.isna().sum()
    print(missing[missing > 0].to_string() if missing.any() else "  none")
    dates = df[date_column]
    print(f"\nDate range: {dates.min().date()} to {dates.max().date()}")
    spacing = dates.drop_duplicates().sort_values().diff().dt.days.value_counts().head(5)
    print("Spacing between observation dates (days -> count):")
    print(spacing.to_string())


def load_all(cattle_type: str):
    sniim = pd.read_csv(SNIIM_PATH, parse_dates=["date"])
    # SNIIM writes a few type labels in the wrong case, e.g. "vaca", "PRADERA".
    sniim["type"] = sniim["type"].astype(str).str.strip().str.lower()
    fx = pd.read_csv(FX_PATH, parse_dates=["date"])
    exports = pd.read_csv(EXPORTS_PATH, parse_dates=["weekEndingDate"])

    preview(sniim, "1. SNIIM beef carcass prices (raw, all types)", "date")
    print("\nRows by cattle type:")
    print(sniim["type"].value_counts().to_string())

    preview(fx, "2. USD/MXN FIX exchange rate (Banco de Mexico)", "date")
    preview(exports, "3. US beef exports to Mexico (USDA FAS)", "weekEndingDate")

    return sniim[sniim["type"] == cattle_type.lower()].copy(), fx, exports


# ----------------------------------------------------------------------------
# 2. Build weekly series
# ----------------------------------------------------------------------------

def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    """Average weighted by number of carcasses, falling back to a plain mean."""
    mask = values.notna()
    v, w = values[mask], weights[mask].fillna(0)
    if len(v) == 0:
        return np.nan
    return float(np.average(v, weights=w)) if w.sum() > 0 else float(v.mean())


def build_weekly(target_rows: pd.DataFrame, fx: pd.DataFrame, exports: pd.DataFrame,
                 cattle_type: str) -> pd.DataFrame:
    t = target_rows.copy()
    t["price_mid_mxn_kg"] = t[["price_min_mxn_kg", "price_max_mxn_kg"]].mean(axis=1)
    # Weeks end on Sunday for every series, so the three line up.
    t["week"] = t["date"].dt.to_period("W-SUN").dt.end_time.dt.normalize()

    grouped = t.groupby("week")
    weekly = pd.DataFrame({
        "price_max": grouped.apply(
            lambda g: weighted_mean(g["price_max_mxn_kg"], g["carcasses"]), include_groups=False),
        "price_mid": grouped.apply(
            lambda g: weighted_mean(g["price_mid_mxn_kg"], g["carcasses"]), include_groups=False),
        "plants": grouped["slaughterhouse"].nunique(),
        "carcasses": grouped["carcasses"].sum(min_count=1),
        "obs": grouped.size(),
        # Share of underlying rows where SNIIM reported no minimum price. Where this
        # is 1, the midpoint simply equals the maximum and carries no extra information.
        "min_missing_share": grouped["price_min_mxn_kg"].apply(lambda s: s.isna().mean()),
    })

    fx = fx.copy()
    fx["week"] = fx["date"].dt.to_period("W-SUN").dt.end_time.dt.normalize()
    weekly["usd_mxn"] = fx.groupby("week")["usd_mxn_fix"].mean()

    ex = exports.copy()
    ex["week"] = ex["weekEndingDate"].dt.to_period("W-SUN").dt.end_time.dt.normalize()
    weekly["exports_tonnes"] = ex.groupby("week")["weeklyExports"].sum(min_count=1)
    weekly["net_sales_tonnes"] = ex.groupby("week")["currentMYNetSales"].sum(min_count=1)

    weekly.index.name = "week"
    return weekly.sort_index()


def report_coverage(weekly: pd.DataFrame) -> None:
    heading("4. Weekly series: coverage and frequency")
    full = pd.date_range(weekly.index.min(), weekly.index.max(), freq="W-SUN")
    print(f"Weeks present: {len(weekly):,}")
    print(f"Weeks in the calendar range: {len(full):,}")
    missing = full.difference(weekly.index)
    print(f"Calendar weeks with no target observation: {len(missing)}")
    if len(missing):
        print("  " + ", ".join(str(d.date()) for d in missing[:12]) + (" ..." if len(missing) > 12 else ""))

    print("\nFirst and last available week per column:")
    for column in weekly.columns:
        s = weekly[column].dropna()
        if len(s):
            print(f"  {column:18s} {s.index.min().date()} to {s.index.max().date()}  "
                  f"({s.notna().sum():,} weeks, {weekly[column].isna().sum():,} missing)")

    overlap = weekly.dropna(subset=["price_max", "usd_mxn", "exports_tonnes"])
    print(f"\nWeeks where all three series are present: {len(overlap):,}")
    if len(overlap):
        print(f"Usable modelling range: {overlap.index.min().date()} to {overlap.index.max().date()}")


# ----------------------------------------------------------------------------
# 3. Summary statistics
# ----------------------------------------------------------------------------

def report_statistics(weekly: pd.DataFrame, cattle_type: str) -> None:
    heading("5. Summary statistics (weekly)")
    columns = ["price_max", "price_mid", "usd_mxn", "exports_tonnes", "net_sales_tonnes"]
    summary = weekly[columns].describe().T
    summary["cv"] = summary["std"] / summary["mean"]      # coefficient of variation
    summary["skew"] = weekly[columns].skew()
    summary["kurtosis"] = weekly[columns].kurtosis()
    print(summary.round(3).to_string())

    heading("6. Maximum price vs midpoint: which to use as the target")
    both = weekly[["price_max", "price_mid"]].dropna()
    gap = (both["price_max"] - both["price_mid"])
    print(f"Weeks with both: {len(both):,}")
    print(f"Correlation between the two: {both['price_max'].corr(both['price_mid']):.4f}")
    print(f"Correlation of their weekly changes: "
          f"{both['price_max'].pct_change().corr(both['price_mid'].pct_change()):.4f}")
    print(f"\nGap (max - mid): mean {gap.mean():.2f}, std {gap.std():.2f} MXN/kg")
    print(f"Gap as share of max: mean {(gap / both['price_max']).mean() * 100:.1f}%")
    print("\nShare of underlying records with no minimum price reported, by year.")
    print("Where this is high the midpoint is mostly just the maximum, so it adds little:")
    share = weekly.groupby(weekly.index.year)["min_missing_share"].mean()
    print((share * 100).round(1).to_string())

    print("\nVolatility of weekly changes:")
    for column in ["price_max", "price_mid"]:
        changes = weekly[column].pct_change().dropna() * 100
        print(f"  {column}: std {changes.std():.3f}%, min {changes.min():.2f}%, max {changes.max():.2f}%")

    print("\nYear-by-year mean of the target (level, MXN/kg):")
    yearly = weekly.groupby(weekly.index.year)[["price_max", "price_mid", "plants", "carcasses"]].mean()
    print(yearly.round(2).to_string())


# ----------------------------------------------------------------------------
# 4. Stationarity
# ----------------------------------------------------------------------------

def stationarity_table(weekly: pd.DataFrame) -> None:
    heading("7. Stationarity: ADF and KPSS, on levels and first differences")
    print("ADF   H0 = unit root (non-stationary). p < 0.05 -> stationary.")
    print("KPSS  H0 = stationary.              p < 0.05 -> non-stationary.")
    print("Both are reported because they disagree when a series has a trend or a break.\n")

    rows = []
    for column in ["price_max", "price_mid", "usd_mxn", "exports_tonnes", "net_sales_tonnes"]:
        series = weekly[column].dropna()
        for label, data in (("level", series), ("diff", series.diff().dropna())):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                adf_p = adfuller(data, autolag="AIC")[1]
                kpss_p = kpss(data, regression="c", nlags="auto")[1]
            rows.append({
                "series": column,
                "form": label,
                "n": len(data),
                "ADF p": round(adf_p, 4),
                "KPSS p": round(kpss_p, 4),
                "verdict": verdict(adf_p, kpss_p),
            })
    print(pd.DataFrame(rows).to_string(index=False))


def verdict(adf_p: float, kpss_p: float) -> str:
    adf_stationary = adf_p < 0.05
    kpss_stationary = kpss_p >= 0.05
    if adf_stationary and kpss_stationary:
        return "stationary (both agree)"
    if not adf_stationary and not kpss_stationary:
        return "non-stationary (both agree)"
    return "tests disagree - trend or break likely"


# ----------------------------------------------------------------------------
# 5. Figures
# ----------------------------------------------------------------------------

def boxplot(ax, data, tick_labels=None, **kwargs):
    """Box plot that works across matplotlib versions.

    The `labels` argument was renamed to `tick_labels` in matplotlib 3.9 and
    removed in 3.11, so try the new name first and fall back to the old one.
    """
    if tick_labels is None:
        return ax.boxplot(data, **kwargs)
    try:
        return ax.boxplot(data, tick_labels=tick_labels, **kwargs)
    except TypeError:
        return ax.boxplot(data, labels=tick_labels, **kwargs)


def save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.relative_to(ROOT)}")


def make_figures(weekly: pd.DataFrame, cattle_type: str) -> None:
    heading("8. Figures")

    # --- Figure 1: the three series over time -------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(weekly.index, weekly["price_max"], lw=0.8, color="#8b0000")
    axes[0].set_ylabel("MXN / kg")
    axes[0].set_title(f"Mexican beef carcass price, {cattle_type} (weekly, maximum reported)")
    axes[1].plot(weekly.index, weekly["usd_mxn"], lw=0.8, color="#1f4e79")
    axes[1].set_ylabel("MXN per USD")
    axes[1].set_title("USD/MXN exchange rate (weekly mean of FIX)")
    axes[2].plot(weekly.index, weekly["exports_tonnes"], lw=0.8, color="#2e7d32")
    axes[2].set_ylabel("tonnes")
    axes[2].set_title("US beef exports to Mexico (weekly)")
    axes[2].set_xlabel("week")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, "fig01_series_overview.png")

    # --- Figure 2: boxplots ------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, column, title in zip(
        axes,
        ["price_max", "usd_mxn", "exports_tonnes"],
        ["Beef price (MXN/kg)", "USD/MXN", "Exports (tonnes)"],
    ):
        boxplot(ax, weekly[column].dropna(), widths=0.5)
        ax.set_title(title)
        ax.grid(alpha=0.3)
    fig.suptitle("Distribution of each weekly series (whole period)")
    fig.tight_layout()
    save(fig, "fig02_boxplots.png")

    # --- Figure 3: price by year, to show trend and spread -----------------
    by_year = [g["price_max"].dropna().values for _, g in weekly.groupby(weekly.index.year)]
    years = [int(y) for y, _ in weekly.groupby(weekly.index.year)]
    fig, ax = plt.subplots(figsize=(13, 5))
    boxplot(ax, by_year, tick_labels=years, widths=0.6)
    ax.set_title(f"Beef carcass price by year ({cattle_type}) - level and spread")
    ax.set_ylabel("MXN / kg")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, "fig03_price_by_year.png")

    # --- Figure 4: seasonality by month ------------------------------------
    changes = weekly["price_max"].pct_change() * 100
    by_month = [changes[weekly.index.month == m].dropna().values for m in range(1, 13)]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    boxplot(ax, by_month, tick_labels=["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], widths=0.6)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Weekly price change by month - is there seasonality?")
    ax.set_ylabel("% change week on week")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, "fig04_seasonality_by_month.png")

    # --- Figure 5: ACF and PACF, levels and differences --------------------
    series = weekly["price_max"].dropna()
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    plot_acf(series, lags=52, ax=axes[0, 0], title="ACF - price level")
    plot_pacf(series, lags=52, ax=axes[0, 1], title="PACF - price level", method="ywm")
    plot_acf(series.diff().dropna(), lags=52, ax=axes[1, 0], title="ACF - first difference")
    plot_pacf(series.diff().dropna(), lags=52, ax=axes[1, 1], title="PACF - first difference",
              method="ywm")
    for ax in axes.ravel():
        ax.grid(alpha=0.3)
    fig.suptitle("Autocorrelation of the target (52 weekly lags = one year)")
    fig.tight_layout()
    save(fig, "fig05_acf_pacf.png")

    # --- Figure 6: maximum vs midpoint ------------------------------------
    both = weekly[["price_max", "price_mid"]].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(both.index, both["price_max"], lw=0.8, label="maximum")
    axes[0].plot(both.index, both["price_mid"], lw=0.8, label="midpoint")
    axes[0].set_title("Target candidates over time")
    axes[0].set_ylabel("MXN / kg")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(both.index, both["price_max"] - both["price_mid"], lw=0.8, color="#b36b00")
    axes[1].set_title("Gap: maximum minus midpoint")
    axes[1].set_ylabel("MXN / kg")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    save(fig, "fig06_max_vs_midpoint.png")

    # --- Figure 7: how many plants report each week -----------------------
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(weekly.index, weekly["plants"], lw=0.8, color="#555555")
    ax.set_title("Number of slaughterhouses reporting this cattle type each week")
    ax.set_ylabel("plants")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, "fig07_reporting_plants.png")


# ----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--type", default="Corral", help="cattle type to use as target")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    target_rows, fx, exports = load_all(args.type)
    if target_rows.empty:
        raise SystemExit(f"No SNIIM rows for cattle type {args.type!r}.")

    weekly = build_weekly(target_rows, fx, exports, args.type)
    report_coverage(weekly)
    report_statistics(weekly, args.type)
    stationarity_table(weekly)
    if not args.no_figures:
        make_figures(weekly, args.type)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    out = PROCESSED / f"weekly_eda_{args.type.lower()}.csv"
    weekly.to_csv(out)
    print(f"\nWeekly table written to {out.relative_to(ROOT)} "
          f"({len(weekly):,} rows) for reuse in the notebook.")


if __name__ == "__main__":
    main()
