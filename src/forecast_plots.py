"""Forecast charts on the price scale, with prediction bands.

Everything so far has been reported as a log return, which is the right scale for
modelling and the wrong one for showing anybody. These figures put the forecast back
into dollars per hundredweight and draw it against what actually happened.

Reading the chart
-----------------
At week t the model predicts the log return over the next `horizon` weeks. The implied
price is

    P_hat(t + h) = P(t) * exp(y_hat(t))

and it is plotted at t + h, the week being forecast, so the forecast line can be read
straight against the actual price for the same week.

Where the bands come from
-------------------------
Two kinds, both honest about using only information available at the time:

  seed band    For a deep model trained over several random seeds, the spread across
               seeds at each week. This is the model's own instability, and it is why
               models.py now keeps the per-seed predictions.

  residual band For any model, an expanding empirical interval: at week t the band is
               the 10th to 90th percentile of the model's own past forecast errors,
               counting only errors that had already been observed by t - horizon.
               Nothing from the present or future enters the band, so it widens or
               narrows as the model's recent record changes. The first MIN_HISTORY
               weeks have no band because there is no record yet.

A residual band is not a confidence interval in the textbook sense - it makes no
distributional assumption and inherits whatever bias the model has. It answers the
practical question instead: given how wrong this model has been lately, how wrong
might it be now.

Usage:
    python src/forecast_plots.py
    python src/forecast_plots.py --series us_chuck_roll us_ribeye
    python src/forecast_plots.py --models ridge arima gru-joint-core
    python src/forecast_plots.py --horizon 12

Output:
    reports/figures/fig50_forecast_<series>.png
    reports/figures/fig51_forecast_overview.png
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

MIN_HISTORY = 20          # weeks of observed errors before a residual band is drawn
LOW, HIGH = 10, 90        # percentiles of the band

SHORT = {
    "us_cutout_choice": "Choice cutout", "us_cutout_select": "Select cutout",
    "us_ribeye": "Ribeye, lip-on bone-in (IMPS 109E)",
    "us_chuck_roll": "Chuck roll (IMPS 116A)",
    "us_brisket": "Brisket, deckle-off (IMPS 120)",
    "us_top_inside_round": "Top inside round (IMPS 168)",
    "us_top_butt": "Top sirloin butt (IMPS 184)",
}
# compare.py writes final_comparison.csv with its own shorter row labels. Matching on
# the plot titles above would silently drop five of the seven series, so the mapping is
# written out rather than inferred.
COMPARE_LABEL = {
    "us_cutout_choice": "Choice cutout", "us_cutout_select": "Select cutout",
    "us_ribeye": "Ribeye (109E)", "us_chuck_roll": "Chuck roll (116A)",
    "us_brisket": "Brisket (120)", "us_top_inside_round": "Top inside round (168)",
    "us_top_butt": "Top sirloin butt (184)",
}

COLOURS = {"ridge": "#1f4e79", "arima": "#2e7d32", "naive": "#9e9e9e",
           "drift": "#bdbdbd"}
DEEP_COLOURS = ["#8b0000", "#b36b00", "#5b2c6f", "#00695c"]


def tagged(name: str, horizon: int, exclude_break: bool = False) -> str:
    """h=4 full-sample keeps the plain filename; other horizons get an _h<N> suffix and
    the excluded-disruption variant a _nobreak one (see baselines.py).

    Without this a --horizon 12 run would read the four-week prediction files and
    stamp them twelve weeks forward, silently producing a plausible-looking but
    meaningless chart.
    """
    stem, _, suffix = name.rpartition(".")
    if horizon != 4:
        stem = f"{stem}_h{horizon}"
    if exclude_break:
        stem = f"{stem}_nobreak"
    return f"{stem}.{suffix}"


def load_prices() -> pd.DataFrame:
    path = PROCESSED / "weekly_model_table.csv"
    if not path.exists():
        raise SystemExit(f"{path.relative_to(ROOT)} not found. Run build_dataset.py.")
    return pd.read_csv(path, index_col=0, parse_dates=True)


def load_predictions(horizon: int, exclude_break: bool = False
                     ) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_path = REPORTS / tagged("baseline_predictions.csv", horizon, exclude_break)
    if not baseline_path.exists():
        raise SystemExit(f"{baseline_path.relative_to(ROOT)} not found. "
                         f"Run baselines.py --horizon {horizon}.")
    baselines = pd.read_csv(baseline_path, index_col=0, parse_dates=True)

    dl_path = REPORTS / tagged("dl_predictions.csv", horizon, exclude_break)
    deep = (pd.read_csv(dl_path, index_col=0, parse_dates=True)
            if dl_path.exists() else pd.DataFrame())
    return baselines, deep


def to_price_path(prices: pd.Series, returns: pd.Series, horizon: int) -> pd.Series:
    """Turn predicted log returns into a predicted price, stamped at the target week."""
    implied = prices.reindex(returns.index) * np.exp(returns)
    implied.index = returns.index + pd.Timedelta(weeks=horizon)
    return implied.dropna()


def residual_band(predicted: pd.Series, actual: pd.Series, horizon: int
                  ) -> tuple[pd.Series, pd.Series]:
    """Expanding out-of-sample interval from the model's own past errors.

    At each week only errors whose target week had already passed are used, so the band
    never sees its own outcome. Returns empty series if there is never enough history.
    """
    frame = pd.concat([predicted.rename("p"), actual.rename("a")], axis=1).dropna()
    errors = frame["p"] - frame["a"]

    low, high, stamps = [], [], []
    for position, week in enumerate(frame.index):
        # Errors are only knowable `horizon` weeks after the forecast was made.
        observed = errors.iloc[:max(0, position - horizon)]
        if len(observed) < MIN_HISTORY:
            continue
        low.append(frame["p"].iloc[position] - np.percentile(observed, HIGH))
        high.append(frame["p"].iloc[position] - np.percentile(observed, LOW))
        stamps.append(week)
    if not stamps:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    return pd.Series(low, index=stamps), pd.Series(high, index=stamps)


def seed_band(frame: pd.DataFrame, prices: pd.Series, horizon: int
              ) -> tuple[pd.Series, pd.Series] | None:
    """Spread across random seeds, on the price scale."""
    seeds = [c for c in frame.columns if c.startswith("seed_")]
    if len(seeds) < 3:
        return None
    paths = pd.concat([to_price_path(prices, frame[c], horizon) for c in seeds], axis=1)
    return paths.min(axis=1), paths.max(axis=1)


def panel(ax, series: str, prices: pd.Series, actual_return: pd.Series,
          chosen: dict[str, pd.DataFrame | pd.Series], horizon: int,
          show_bands: bool, band_for: str | None = None) -> None:
    """Draw one series. Only one model gets a band - several overlapping translucent
    bands average out to an indistinct smear and hide the lines underneath."""
    realised = to_price_path(prices, actual_return, horizon)
    ax.plot(realised.index, realised.to_numpy(), color="black", lw=1.8,
            label="actual", zorder=5)

    # Band the model the caller asked for, else the last one listed, which by default
    # is the deep configuration - the one whose uncertainty is worth showing.
    banded = band_for if band_for in chosen else (list(chosen)[-1] if chosen else None)

    deep_index = 0
    for name, payload in chosen.items():
        returns = payload["prediction"] if isinstance(payload, pd.DataFrame) else payload
        path = to_price_path(prices, returns, horizon)
        if path.empty:
            continue
        colour = COLOURS.get(name) or DEEP_COLOURS[deep_index % len(DEEP_COLOURS)]
        if name not in COLOURS:
            deep_index += 1
        ax.plot(path.index, path.to_numpy(), lw=1.2, color=colour,
                label=name.replace("-core", ""))

        if not show_bands or name != banded:
            continue
        # Same two-band treatment as fig52: the wide band is where the actual is
        # likely to fall, the narrow one is only seed-to-seed wobble. Drawing the
        # seed spread alone would look like a tight forecast interval.
        outer = residual_band(path, realised, horizon)
        if len(outer[0]):
            ax.fill_between(outer[0].index, outer[0].to_numpy(),
                            outer[1].to_numpy(), color=colour, alpha=0.11,
                            linewidth=0, zorder=1,
                            label=f"{HIGH - LOW}% prediction band")
        inner = (seed_band(payload, prices, horizon)
                 if isinstance(payload, pd.DataFrame) else None)
        if inner is not None and len(inner[0]):
            ax.fill_between(inner[0].index, inner[0].to_numpy(),
                            inner[1].to_numpy(), color=colour, alpha=0.30,
                            linewidth=0, zorder=2, label="seed spread")

    ax.set_title(SHORT.get(series, series), fontsize=10, loc="left")
    ax.set_ylabel("USD/cwt", fontsize=8)
    ax.grid(alpha=0.3)


def best_per_series(table: pd.DataFrame, prices_table: pd.DataFrame,
                    baselines: pd.DataFrame, deep: pd.DataFrame,
                    horizon: int, show_bands: bool,
                    exclude_break: bool = False) -> None:
    """One panel per series showing only the method that actually won it.

    Every other figure in this project compares methods. This one answers the
    different question a reader asks last: given the best available model, how good
    is the forecast and how uncertain should I be about it. So each panel carries one
    forecast, its band, and nothing else to argue with.
    """
    panels = []
    missing = []
    for series, label in SHORT.items():
        key = COMPARE_LABEL.get(series, label)
        if key not in table.index or series not in prices_table.columns:
            missing.append(series)
            continue
        row = table.loc[key]
        methods = {c: row[c] for c in table.columns
                   if not c.endswith("__sd") and c != "naive"
                   and np.isfinite(row.get(c, np.nan))}
        if not methods:
            continue
        winner = min(methods, key=methods.get)

        block = baselines[baselines["series"] == series]
        if winner in block.columns:
            payload = block[winner].dropna()
        else:
            sub = deep[(deep["series"] == series) & (deep["config"] == winner)]
            if sub.empty:
                continue
            payload = sub
        panels.append((series, label, winner, methods[winner], payload, block))

    if missing:
        print(f"  fig52: no comparison row or price column for {missing}")
    if not panels:
        return

    fig, axes = plt.subplots(len(panels), 1, figsize=(12, 2.5 * len(panels)),
                             sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (series, label, winner, rmse, payload, block) in zip(axes, panels):
        prices = prices_table[series]
        realised = to_price_path(prices, block["actual"].dropna(), horizon)
        ax.plot(realised.index, realised.to_numpy(), color="black", lw=1.6,
                label="actual", zorder=5)

        returns = payload["prediction"] if isinstance(payload, pd.DataFrame) else payload
        path = to_price_path(prices, returns, horizon)
        colour = COLOURS.get(winner, "#8b0000")
        ax.plot(path.index, path.to_numpy(), lw=1.3, color=colour,
                label=winner.replace("-single", "").replace("-core", ""))

        if show_bands:
            # Two different quantities, drawn separately because they answer
            # different questions and the seed band is far the narrower of the two.
            # Outer: where the actual is likely to fall, from the model's own past
            # errors. Inner: how much the answer moves when only the random seed
            # changes. Labelling seed spread as forecast uncertainty would overstate
            # how well the forecast is pinned down.
            outer = residual_band(path, realised, horizon)
            if len(outer[0]):
                ax.fill_between(outer[0].index, outer[0].to_numpy(),
                                outer[1].to_numpy(), color=colour, alpha=0.13,
                                linewidth=0, zorder=1)
                ax.plot([], [], color=colour, alpha=0.25, lw=6,
                        label=f"{HIGH - LOW}% prediction band")
                cover = pd.concat([realised.rename("a"), outer[0].rename("lo"),
                                   outer[1].rename("hi")], axis=1).dropna()
                if len(cover):
                    hit = 100 * ((cover.a >= cover.lo) & (cover.a <= cover.hi)).mean()
                    print(f"    {label}: {HIGH - LOW}% band covers "
                          f"{hit:.0f}% of actual weeks ({len(cover)} scored)")
            inner = (seed_band(payload, prices, horizon)
                     if isinstance(payload, pd.DataFrame) else None)
            if inner is not None and len(inner[0]):
                ax.fill_between(inner[0].index, inner[0].to_numpy(),
                                inner[1].to_numpy(), color=colour, alpha=0.34,
                                linewidth=0, zorder=2)
                ax.plot([], [], color=colour, alpha=0.6, lw=6,
                        label="seed spread")

        ax.set_title(f"{label}   -   best: "
                     f"{winner.replace('-single', '').replace('-core', '')}, "
                     f"RMSE {rmse:.5f}", fontsize=10, loc="left")
        ax.set_ylabel("USD/cwt", fontsize=8)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=4, loc="upper left")

    axes[-1].set_xlabel("week being forecast")
    fig.suptitle(f"Best method per series: {horizon}-week-ahead forecast against what "
                 f"happened\nlight band = {HIGH - LOW}% interval from the model's own "
                 f"past errors; dark band = spread across random seeds", y=1.0)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    path_out = FIGURES / tagged("fig52_best_per_series.png", horizon, exclude_break)
    fig.savefig(path_out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path_out.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--series", nargs="*", default=None,
                        help="series to plot; default is every US series available")
    parser.add_argument("--models", nargs="*", default=None,
                        help="which models to draw; default is the best benchmark and "
                             "the best deep configuration for each series")
    parser.add_argument("--exclude-break", action="store_true",
                        help="read the run that excluded the 2020-21 disruption "
                             "from training (the _nobreak files)")
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--no-bands", action="store_true")
    parser.add_argument("--no-best", action="store_true",
                        help="skip fig52, the best-method-per-series summary")
    parser.add_argument("--include-joint", action="store_true",
                        help="also consider the joint multi-output runs")
    parser.add_argument("--all-configs", action="store_true",
                        help="draw every deep configuration rather than the best one")
    parser.add_argument("--band-for", default=None,
                        help="which model gets the shaded band; default is the last "
                             "one drawn, normally the deep configuration")
    args = parser.parse_args()

    prices_table = load_prices()
    baselines, deep = load_predictions(args.horizon, args.exclude_break)

    available = [s for s in SHORT if s in baselines["series"].unique()]
    series_list = args.series or available
    if not series_list:
        raise SystemExit("No US series found in baseline_predictions.csv. "
                         "Run baselines.py with --family us or both.")

    FIGURES.mkdir(parents=True, exist_ok=True)
    comparison = REPORTS / tagged("final_comparison.csv", args.horizon, args.exclude_break)
    ranking = pd.read_csv(comparison, index_col=0) if comparison.exists() else None

    panels = []
    for series in series_list:
        block = baselines[baselines["series"] == series]
        if block.empty:
            print(f"  {series}: no baseline predictions, skipped")
            continue
        prices = prices_table[series] if series in prices_table.columns else None
        if prices is None:
            print(f"  {series}: no price column, skipped")
            continue

        chosen: dict[str, pd.DataFrame | pd.Series] = {}
        wanted = args.models
        if wanted:
            for name in wanted:
                if name in block.columns:
                    chosen[name] = block[name].dropna()
                elif not deep.empty:
                    sub = deep[(deep["series"] == series) & (deep["config"] == name)]
                    if len(sub):
                        chosen[name] = sub
        else:
            # Default: naive for reference, the strongest benchmark, and every deep
            # configuration that was run for this series.
            chosen["naive"] = block["naive"].dropna()
            errors = {m: float(np.sqrt(np.mean((block[m] - block["actual"]) ** 2)))
                      for m in ("ridge", "arima") if m in block.columns}
            if errors:
                chosen[min(errors, key=errors.get)] = block[min(errors, key=errors.get)]
            if not deep.empty:
                # Joint multi-output runs lost the comparison and are out of the
                # reported story, so they are not drawn unless asked for. Among the
                # rest, draw only the best one per series: a panel with five model
                # lines is a tangle, and the point of this figure is to show one
                # forecast against what happened.
                candidates = [c for c in sorted(deep["config"].unique())
                              if args.include_joint or "-joint-" not in c]
                scored = {}
                for config in candidates:
                    sub = deep[(deep["series"] == series) & (deep["config"] == config)]
                    if sub.empty:
                        continue
                    path = to_price_path(prices, sub["prediction"], args.horizon)
                    realised = to_price_path(prices, block["actual"].dropna(),
                                             args.horizon)
                    both = pd.concat([path.rename("p"), realised.rename("a")],
                                     axis=1).dropna()
                    if len(both):
                        scored[config] = (float(np.sqrt(np.mean(
                            (both["p"] - both["a"]) ** 2))), sub)
                if scored:
                    if args.all_configs:
                        for config, (_, sub) in scored.items():
                            chosen[config] = sub
                    else:
                        best = min(scored, key=lambda c: scored[c][0])
                        chosen[best] = scored[best][1]

        if not chosen:
            print(f"  {series}: nothing to draw, skipped")
            continue

        fig, ax = plt.subplots(figsize=(12, 4.6))
        panel(ax, series, prices, block["actual"].dropna(), chosen, args.horizon,
              not args.no_bands, args.band_for)
        ax.set_xlabel("week being forecast")
        note = ("shaded: spread across random seeds, or the model's own past "
                f"{LOW}-{HIGH}th percentile errors")
        ax.legend(fontsize=8, ncol=3, loc="upper left")
        fig.suptitle(f"{args.horizon}-week-ahead forecast against what happened\n{note}",
                     y=1.04, fontsize=11)
        fig.tight_layout()
        path = FIGURES / tagged(f"fig50_forecast_{series}.png", args.horizon, args.exclude_break)
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {path.relative_to(ROOT)}")
        panels.append((series, prices, block, chosen))

    comparison_path = REPORTS / tagged("final_comparison.csv", args.horizon, args.exclude_break)
    if not args.no_best and comparison_path.exists():
        best_per_series(pd.read_csv(comparison_path, index_col=0), prices_table,
                        baselines, deep, args.horizon, not args.no_bands,
                        args.exclude_break)

    if len(panels) > 1:
        fig, axes = plt.subplots(len(panels), 1, figsize=(12, 2.6 * len(panels)),
                                 sharex=True)
        axes = np.atleast_1d(axes)
        for ax, (series, prices, block, chosen) in zip(axes, panels):
            panel(ax, series, prices, block["actual"].dropna(), chosen, args.horizon,
                  not args.no_bands, args.band_for)
        axes[0].legend(fontsize=7, ncol=4, loc="upper left")
        axes[-1].set_xlabel("week being forecast")
        fig.suptitle(f"{args.horizon}-week-ahead forecasts, US boxed beef", y=1.0)
        fig.tight_layout()
        path = FIGURES / tagged("fig51_forecast_overview.png", args.horizon, args.exclude_break)
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
