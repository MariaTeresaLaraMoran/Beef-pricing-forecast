"""Forward price path for each US boxed beef series.

The models in models.py predict the cumulative log return over a single horizon, so they
answer "where is the price in twelve weeks" and say nothing about the weeks in between.
This script produces the weekly path instead: one model per series with twelve outputs,
the log return at each horizon from one to twelve weeks, run forward from the most recent
complete window.

Two bands are drawn, and they measure different things:

  - the wider band is an empirical prediction interval, taken from the model's own
    out-of-sample errors at that horizon. It answers "where is the price likely to fall".
  - the narrower band is the spread across random seeds. It is model instability, not
    forecast uncertainty, and is far narrower. Showing only this one would overstate how
    well the forecast is pinned down.

One model with twelve outputs, rather than twelve models, because the horizons of a
single series are strongly related and share a representation usefully. Sharing across
SERIES is a different proposition and was tested and rejected - see the joint ablation in
models.py. The headline RMSE tables still come from the dedicated single-output models;
this script is for the picture, and prints how far the two disagree at h=12 so the
difference is visible rather than assumed.

Requires target columns for every horizon:

    python src/build_dataset.py --horizon 1 2 3 4 5 6 7 8 9 10 11 12

Usage:
    python src/forecast_path.py
    python src/forecast_path.py --seeds 3 --no-exclude-break
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from baselines import BREAK_END, BREAK_START
from models import (CORE_FEATURES, WINDOW, build_model, build_windows, get_keras,
                    load_table)

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

US_TARGETS = ["us_cutout_choice", "us_cutout_select", "us_ribeye", "us_chuck_roll",
              "us_brisket", "us_top_inside_round", "us_top_butt"]

LABELS = {
    "us_cutout_choice": "Choice cutout",
    "us_cutout_select": "Select cutout",
    "us_ribeye": "Ribeye, lip-on bone-in (IMPS 109E)",
    "us_chuck_roll": "Chuck roll (IMPS 116A)",
    "us_brisket": "Brisket, deckle-off (IMPS 120)",
    "us_top_inside_round": "Top inside round (IMPS 168)",
    "us_top_butt": "Top sirloin butt (IMPS 184)",
}

MAX_HORIZON = 12
LOW, HIGH = 5, 95           # percentile bounds of the prediction band
HISTORY_WEEKS = 52          # how much past price to draw behind the forecast


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def target_columns(series: str, horizons: list[int]) -> list[str]:
    return [f"target_{series}_h{h}" for h in horizons]


def standardise(x: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and sd per feature, over the fitting rows only."""
    flat = x[rows].reshape(-1, x.shape[2])
    mean = np.nanmean(flat, axis=0)
    sd = np.nanstd(flat, axis=0)
    sd[sd == 0] = 1.0
    return mean, sd


def fit_seed(x: np.ndarray, y: np.ndarray, train: np.ndarray, val: np.ndarray,
             seed: int, args, epochs: int | None = None) -> tuple[object, int]:
    """Train one network. With `epochs` given, train that many and skip early stopping.

    The refit on the full sample has no held-out block to stop on, so it reuses the
    epoch count the split run settled at rather than guessing one.
    """
    keras = get_keras()
    keras.utils.set_random_seed(seed)
    model = build_model(args.arch, WINDOW, x.shape[2], y.shape[1], args.units,
                        args.dropout, args.learning_rate)
    if epochs is None:
        stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience,
                                             restore_best_weights=True)
        history = model.fit(x[train], y[train], validation_data=(x[val], y[val]),
                            epochs=args.epochs, batch_size=args.batch_size,
                            callbacks=[stop], verbose=0)
        return model, len(history.history["loss"])
    model.fit(x[train], y[train], epochs=epochs, batch_size=args.batch_size, verbose=0)
    return model, epochs


def final_window(table: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, pd.Timestamp]:
    """The most recent complete feature window, and the week it ends on.

    Walks back from the end of the table: the last rows often have a feature missing
    because a source publishes late, and a window with a gap in it cannot be used.
    """
    values = table[features].to_numpy(dtype="float32")
    for end in range(len(table) - 1, WINDOW - 2, -1):
        block = values[end - WINDOW + 1:end + 1]
        if not np.isnan(block).any():
            return block, table.index[end]
    raise SystemExit("no complete feature window at the end of the table")


def run_series(table: pd.DataFrame, series: str, features: list[str],
               horizons: list[int], args) -> dict | None:
    """Split run for the error bands, then a full-sample refit for the forecast."""
    columns = target_columns(series, horizons)
    missing = [c for c in columns if c not in table.columns]
    if missing:
        raise SystemExit(
            f"target column(s) not in the table: {missing[:3]}...\n"
            f"run: python src/build_dataset.py --horizon "
            + " ".join(str(h) for h in horizons))

    x, y, stamps = build_windows(table, features, columns, WINDOW, max(horizons))
    if len(x) < 200:
        print(f"  {series}: only {len(x)} windows, skipped")
        return None

    n = len(stamps)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    embargo = max(horizons)
    train = np.arange(0, train_end)
    val = np.arange(train_end + embargo, val_end)
    test = np.arange(val_end + embargo, n)

    if args.exclude_break:
        ordinary = (stamps < BREAK_START) | (stamps > BREAK_END)
        train, val = train[ordinary[train]], val[ordinary[val]]

    mean, sd = standardise(x, train)
    xs = (x - mean) / sd

    # --- pass A: honest out-of-sample errors, one row per seed -------------
    errors, epoch_counts = [], []
    for seed in range(args.seeds):
        model, used = fit_seed(xs, y, train, val, seed, args)
        predicted = model.predict(xs[test], verbose=0)
        errors.append(predicted - y[test])
        epoch_counts.append(used)
    errors = np.concatenate(errors, axis=0)        # (seeds * test, horizons)

    # --- pass B: refit on everything, then run the last window forward -----
    whole = np.arange(n)
    if args.exclude_break:
        whole = whole[(stamps < BREAK_START) | (stamps > BREAK_END)]
    mean_all, sd_all = standardise(x, whole)
    xs_all = (x - mean_all) / sd_all
    block, as_of = final_window(table, features)
    forward = ((block - mean_all) / sd_all)[None, :, :]

    paths = []
    for seed in range(args.seeds):
        model, _ = fit_seed(xs_all, y, whole, whole, seed, args,
                            epochs=int(np.median(epoch_counts)))
        paths.append(model.predict(forward, verbose=0)[0])
    paths = np.stack(paths)                        # (seeds, horizons)

    return {"series": series, "as_of": as_of, "horizons": horizons,
            "returns": paths.mean(axis=0), "seed_sd": paths.std(axis=0, ddof=1),
            "error_low": np.percentile(errors, LOW, axis=0),
            "error_high": np.percentile(errors, HIGH, axis=0),
            "test_weeks": len(test), "epochs": int(np.median(epoch_counts))}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def to_prices(last_price: float, result: dict) -> dict:
    """Log returns to a price path, with both bands on the price scale."""
    r = result["returns"]
    return {
        "mid": last_price * np.exp(r),
        "seed_low": last_price * np.exp(r - result["seed_sd"]),
        "seed_high": last_price * np.exp(r + result["seed_sd"]),
        # the error band is centred on the forecast, so subtract the error quantiles
        "band_low": last_price * np.exp(r - result["error_high"]),
        "band_high": last_price * np.exp(r - result["error_low"]),
    }


def figure(table: pd.DataFrame, results: list[dict], args) -> None:
    rows = len(results)
    fig, axes = plt.subplots(rows, 1, figsize=(11, 3.1 * rows), squeeze=False)
    for ax, result in zip(axes[:, 0], results):
        series = result["series"]
        history = table[series].dropna().iloc[-HISTORY_WEEKS:]
        last_price = float(history.iloc[-1])
        weeks = pd.date_range(result["as_of"], periods=max(result["horizons"]) + 1,
                              freq="W-SUN")
        prices = to_prices(last_price, result)

        # history, as a plain line
        ax.plot(history.index, history.to_numpy(), lw=1.4, color="#1C1A17",
                label=f"actual, last {HISTORY_WEEKS} weeks")

        # the forecast starts at the last observed price, so the two meet
        mid = np.concatenate([[last_price], prices["mid"]])
        band_lo = np.concatenate([[last_price], prices["band_low"]])
        band_hi = np.concatenate([[last_price], prices["band_high"]])
        seed_lo = np.concatenate([[last_price], prices["seed_low"]])
        seed_hi = np.concatenate([[last_price], prices["seed_high"]])

        ax.fill_between(weeks, band_lo, band_hi, color="#2C5578", alpha=0.16,
                        linewidth=0, label=f"{HIGH - LOW}% prediction band")
        ax.fill_between(weeks, seed_lo, seed_hi, color="#9E2B25", alpha=0.38,
                        linewidth=0, label="seed spread")
        ax.plot(weeks, mid, lw=1.8, color="#9E2B25", label="forecast")
        ax.axvline(result["as_of"], color="#5A534A", lw=0.9, ls="--")

        ax.set_title(f"{LABELS.get(series, series)}  -  "
                     f"{max(result['horizons'])} weeks from {result['as_of'].date()}",
                     loc="left", fontsize=10)
        ax.set_ylabel("USD/cwt")
        ax.legend(fontsize=7, ncol=4, loc="upper left")
        ax.grid(alpha=0.3)

    fig.suptitle("Forward price path per cut: the weeks ahead, not a backtest\n"
                 "blue = where the price is likely to fall; red = spread across "
                 "random seeds", y=1.0)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    name = "fig60_forward_path"
    if args.exclude_break:
        name += "_nobreak"
    path = FIGURES / f"{name}.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  saved {path.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizons", type=int, nargs="+",
                        default=list(range(1, MAX_HORIZON + 1)))
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--arch", choices=["lstm", "gru"], default="lstm")
    parser.add_argument("--units", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-exclude-break", dest="exclude_break",
                        action="store_false",
                        help="keep the 2020-21 disruption in the fitting data")
    parser.set_defaults(exclude_break=True)
    args = parser.parse_args()

    table = load_table()
    features = [c for c in CORE_FEATURES if c in table.columns]
    if len(features) < len(CORE_FEATURES):
        missing = [c for c in CORE_FEATURES if c not in table.columns]
        raise SystemExit(f"feature column(s) not in the table: {missing}")

    print(f"Forward path, horizons {min(args.horizons)}-{max(args.horizons)}, "
          f"{args.seeds} seeds, {len(features)} features"
          f"{', disruption excluded' if args.exclude_break else ''}")

    results, rows = [], []
    for series in US_TARGETS:
        if series not in table.columns:
            continue
        print(f"  {series}")
        result = run_series(table, series, features, args.horizons, args)
        if result is None:
            continue
        results.append(result)
        last_price = float(table[series].dropna().iloc[-1])
        prices = to_prices(last_price, result)
        for position, horizon in enumerate(result["horizons"]):
            rows.append({
                "series": series, "as_of": result["as_of"].date(), "horizon": horizon,
                "last_price": round(last_price, 2),
                "forecast": round(float(prices["mid"][position]), 2),
                "band_low": round(float(prices["band_low"][position]), 2),
                "band_high": round(float(prices["band_high"][position]), 2),
                "seed_sd_pct": round(float(result["seed_sd"][position]) * 100, 2),
            })

    if not results:
        raise SystemExit("no series produced a forecast")

    frame = pd.DataFrame(rows)
    out = REPORTS / ("forward_path_nobreak.csv" if args.exclude_break
                     else "forward_path.csv")
    frame.to_csv(out, index=False)

    # The two horizons the rest of the project reports, side by side.
    headline = [h for h in (4, 12) if h in args.horizons]
    print(f"\n{'series':24s} {'as of':>12s} {'last':>9s}", end="")
    for h in headline:
        print(f" {('+' + str(h) + 'w'):>9s} {'change':>8s} {str(HIGH - LOW) + '% band':>21s}", end="")
    print()
    print("-" * (47 + 40 * len(headline)))
    for result in results:
        series = result["series"]
        last = float(table[series].dropna().iloc[-1])
        prices = to_prices(last, result)
        print(f"{series:24s} {result['as_of'].date()!s:>12s} {last:9.2f}", end="")
        for h in headline:
            i = result["horizons"].index(h)
            mid = float(prices["mid"][i])
            print(f" {mid:9.2f} {100 * (mid / last - 1):7.1f}%"
                  f" {float(prices['band_low'][i]):9.2f} to {float(prices['band_high'][i]):7.2f}",
                  end="")
        print()

    if headline:
        summary = frame[frame["horizon"].isin(headline)].pivot(
            index="series", columns="horizon",
            values=["forecast", "band_low", "band_high"])
        summary_path = REPORTS / ("forward_summary_nobreak.csv" if args.exclude_break
                                  else "forward_summary.csv")
        summary.to_csv(summary_path)
        print(f"\nWritten {summary_path.relative_to(ROOT)}")
    print(f"\nWritten {out.relative_to(ROOT)}")
    print(f"\nThe band is a {HIGH - LOW}% interval from the model's own "
          "out-of-sample errors.")
    print("The seed spread is narrower and is model instability, not forecast")
    print("uncertainty - do not read it as a confidence interval.")

    figure(table, results, args)


if __name__ == "__main__":
    main()
