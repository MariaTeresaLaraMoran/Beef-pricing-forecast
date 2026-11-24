"""LSTM and GRU forecasts of US boxed beef prices four weeks ahead.

This is the deep learning half of the project. The benchmark half lives in
baselines.py, and this script is written to be compared with it exactly: it scores
itself on the same test weeks, with the same metrics, by importing them rather than
reimplementing them.

What the network sees
---------------------
A window of WINDOW consecutive weeks. Each timestep carries contemporaneous features
only - weekly log returns of the seven US series, the fed cattle price, the two packer
margins, export volumes, and a smooth annual seasonal pair. It is given no lagged
columns, because reading a sequence and working out which lags matter is the thing a
recurrent layer is for. Ridge needed ret_x_lag1..lag4 spelled out; this does not.

What it predicts
----------------
The log return over the next four weeks, directly - not one week ahead iterated four
times. This matters. ARIMA is fitted to one-step dynamics and iterated, which compounds
its errors, and that is the likeliest reason it loses to the naive forecast on both
cutout series despite those series having the strongest weekly momentum in the dataset.
Predicting the horizon directly avoids that failure mode, and matches what ridge does.

Three comparisons 
-------------------------------------
  1. Does nonlinearity help?      this script against baselines.py
  2. Does sharing across cuts help?  --mode joint (one network, seven outputs) against
                                     --mode single (one network per series)
  3. Does extra information help?    --features core against --features plus-volume,
                                     a straight ablation on the same architecture

Why several seeds
-----------------
With 169 test weeks, the spread between random initialisations of a small recurrent
network can be larger than the gap between the models being compared. A single run is
not evidence. Every configuration is therefore trained --seeds times and reported as
mean and standard deviation across seeds, with the seed-level results written out so
the variation is visible rather than hidden.

Usage:
    python src/models.py                                   # LSTM, joint, 5 seeds
    python src/models.py --arch gru
    python src/models.py --mode single
    python src/models.py --features plus-macro             # the ablation
    python src/models.py --arch gru --seeds 10 --window 12

Output:
    reports/dl_results.csv       one row per series per configuration, averaged
    reports/dl_results_seeds.csv the same before averaging, one row per seed
    reports/dl_predictions.csv   test-set predictions, for the final comparison
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Keras is noisy on import and the oneDNN notice confuses more than it helps.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from baselines import (BREAK_END, BREAK_START, diebold_mariano, score,
                       tagged)  # noqa: E402  - shared with the benchmarks

HORIZON = 4
WINDOW = 26
US_TARGETS = ["us_cutout_choice", "us_cutout_select", "us_ribeye", "us_chuck_roll",
              "us_brisket", "us_top_inside_round", "us_top_butt"]

# One timestep's features. Contemporaneous only - the window supplies the history.
CORE_FEATURES = (
    [f"ret_{s}" for s in US_TARGETS]
    + ["ret_us_steer_dressed", "margin_choice_log", "margin_select_log",
       "margin_choice_log_change", "season_sin", "season_cos"]
)

# Beef-market volume: how much cattle actually changed hands. The packer margin is the
# incentive to slaughter; this is nearer the decision itself. Separate from the macro arm
# because it is beef data, not economy-wide data - confounding them would make a result
# uninterpretable.
VOLUME_FEATURES = ["ret_us_steer_dressed_head", "ret_us_steer_live_head"]

# Economy-wide conditions. Two channels, each with a specific prediction:
#   usd_index      a stronger dollar softens export demand, so more beef stays domestic
#   initial_claims a weakening labour market makes households trade down the carcass,
#                  which should show up more in the premium cuts than the cheap ones
# Note that the dollar index begins in 2006 while the US price series begin in 2004, so
# this arm costs roughly two years of training windows. If it underperforms, check
# plus-volume first to see whether the loss is the features or the shorter history.
MACRO_FEATURES = ["ret_usd_index", "ret_initial_claims"]



# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_table() -> pd.DataFrame:
    path = PROCESSED / "weekly_model_table.csv"
    if not path.exists():
        raise SystemExit(f"{path.relative_to(ROOT)} not found. Run build_dataset.py first.")
    return pd.read_csv(path, index_col=0, parse_dates=True)


def load_baseline_test_index(horizon: int = HORIZON,
                             exclude_break: bool = False) -> pd.DatetimeIndex:
    """The exact weeks baselines.py scored, so the comparison is like-for-like.

    Deriving the split again here would give almost the same weeks, and "almost" is not
    good enough when the whole point is to compare two models on identical rows.

    The two disruption variants share a test period by construction, so either file
    gives the same answer; the matching one is preferred and the plain one is the
    fallback, which lets the variants be run in either order.
    """
    path = REPORTS / tagged("baseline_predictions.csv", horizon, exclude_break)
    if not path.exists():
        path = REPORTS / tagged("baseline_predictions.csv", horizon)
    if not path.exists():
        raise SystemExit(f"{path.relative_to(ROOT)} not found. Run baselines.py first - "
                         "this script scores itself on the weeks it used.")
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    us = frame[frame["series"].isin(US_TARGETS)] if "series" in frame.columns else frame
    return pd.DatetimeIndex(sorted(us.index.unique()))


def build_windows(table: pd.DataFrame, features: list[str], targets: list[str],
                  window: int, horizon: int
                  ) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Stack consecutive weeks into (samples, window, features).

    build_dataset.py reindexes the table onto a complete weekly calendar, so adjacent
    rows are always adjacent weeks and a positional slice cannot silently straddle a
    gap. A window is kept only if every feature in it and every target at its final
    week is present.
    """
    feature_values = table[features].to_numpy(dtype="float32")
    target_values = table[targets].to_numpy(dtype="float32")

    samples, labels, stamps = [], [], []
    for end in range(window - 1, len(table)):
        block = feature_values[end - window + 1:end + 1]
        label = target_values[end]
        if np.isnan(block).any() or np.isnan(label).any():
            continue
        samples.append(block)
        labels.append(label)
        stamps.append(table.index[end])

    if not samples:
        return (np.empty((0, window, len(features)), dtype="float32"),
                np.empty((0, len(targets)), dtype="float32"),
                pd.DatetimeIndex([]))
    return np.stack(samples), np.stack(labels), pd.DatetimeIndex(stamps)


def split_by_test_start(stamps: pd.DatetimeIndex, test_start: pd.Timestamp,
                        horizon: int, val_fraction: float
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train and validate strictly before the benchmark's test period.

    Two embargoes of `horizon` weeks are applied: one between training and validation,
    and one between validation and the test period. Without them the last training
    windows carry targets that reach forward into the next block, and the model would
    be scored partly on weeks it had already learned from.
    """
    pre = np.flatnonzero(stamps <= test_start - pd.Timedelta(weeks=horizon + 1))
    test = np.flatnonzero(stamps >= test_start)

    cut = int(len(pre) * (1 - val_fraction))
    train = pre[:cut]
    val = pre[cut + horizon:]
    return train, val, test


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def get_keras():
    """Return the Keras module, whichever way it is installed.

    Keras 3 ships as its own package and runs on TensorFlow, PyTorch or JAX. Importing
    it directly first means this script still works if TensorFlow has no wheel for the
    installed Python version - set KERAS_BACKEND=torch and install torch instead. The
    model code below is identical either way.
    """
    try:
        import keras
        return keras
    except ImportError:
        pass
    try:
        from tensorflow import keras
        return keras
    except ImportError as error:
        raise SystemExit(
            "Keras is not available. Either:\n"
            "  pip install tensorflow\n"
            "or, if that has no wheel for your Python version:\n"
            "  pip install keras torch\n"
            "  export KERAS_BACKEND=torch\n"
            f"({error})")


def build_model(arch: str, window: int, n_features: int, n_targets: int,
                units: int, dropout: float, learning_rate: float):
    """One recurrent layer, dropout, a linear output head.

    Deliberately small. There are about 800 training windows; a deeper or wider network
    has more parameters than the dataset has observations, and the literature review's
    feasibility section commits to a single layer, dropout and early stopping for
    exactly this reason.
    """
    keras = get_keras()
    layers = keras.layers

    recurrent = layers.LSTM if arch == "lstm" else layers.GRU
    model = keras.Sequential([
        layers.Input(shape=(window, n_features)),
        recurrent(units),
        layers.Dropout(dropout),
        layers.Dense(n_targets),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss="mse", metrics=["mae"])
    return model


def train_once(arch: str, data: dict, seed: int, args) -> tuple[np.ndarray, int, float]:
    """Train one network on one seed and return its test predictions."""
    keras = get_keras()
    keras.utils.set_random_seed(seed)

    model = build_model(arch, args.window, data["x_train"].shape[2],
                        data["y_train"].shape[1], args.units, args.dropout,
                        args.learning_rate)
    stopper = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=args.patience, restore_best_weights=True)

    history = model.fit(
        data["x_train"], data["y_train"],
        validation_data=(data["x_val"], data["y_val"]),
        epochs=args.epochs, batch_size=args.batch_size,
        callbacks=[stopper], verbose=0, shuffle=True)

    epochs_run = len(history.history["loss"])
    best_val = float(np.min(history.history["val_loss"]))
    # Call the model directly rather than through predict(). predict() compiles a new
    # tf.function for every model, and training many seeds in one process makes Keras
    # warn about repeated retracing; a direct call avoids that and is faster here.
    scaled = np.asarray(model(data["x_test"], training=False), dtype="float64")

    # Undo the target scaling so predictions are log returns again.
    predictions = scaled * data["y_scale"] + data["y_mean"]
    keras.backend.clear_session()
    return predictions, epochs_run, best_val


def prepare(table: pd.DataFrame, features: list[str], targets: list[str],
            test_start: pd.Timestamp, args) -> dict | None:
    """Window, split and scale. Scaling statistics come from the training block only."""
    # The targets are the forward log-return columns, never the price columns. Passing
    # the series names here instead would train the network to predict the current
    # price, which also sits in its own feature window - a leak, not a forecast.
    target_columns = [f"target_{series}_h{args.horizon}" for series in targets]
    missing = [c for c in target_columns if c not in table.columns]
    if missing:
        raise SystemExit(f"target column(s) not in the table: {missing}")

    x, y, stamps = build_windows(table, features, target_columns, args.window,
                                 args.horizon)
    if len(x) == 0:
        return None

    train, val, test = split_by_test_start(stamps, test_start, args.horizon,
                                           args.val_fraction)
    if args.exclude_break:
        # Windows whose target week falls inside the disruption, dropped from FITTING
        # only. Test is untouched, so the two variants are scored on identical weeks.
        # This also keeps those weeks out of the standardisation computed below from
        # the training rows, which is where an |z| of 8.9 does the most damage.
        ordinary = (stamps < BREAK_START) | (stamps > BREAK_END)
        was = (len(train), len(val))
        train, val = train[ordinary[train]], val[ordinary[val]]
        print(f"    disruption excluded: train {was[0]} -> {len(train)}, "
              f"validation {was[1]} -> {len(val)}")
    if len(train) < 100 or len(val) < 20 or len(test) == 0:
        return None

    # Standardise per feature across training timesteps only, so nothing from the
    # validation or test period influences the scaling.
    flat = x[train].reshape(-1, x.shape[2])
    x_mean, x_scale = flat.mean(axis=0), flat.std(axis=0)
    x_scale[x_scale < 1e-8] = 1.0

    # Targets are standardised too. The seven series have very different volatility -
    # the ribeye's four-week return has three times the standard deviation of the
    # cutout's - so an unscaled multi-output MSE would mostly optimise the noisiest
    # series and neglect the rest.
    y_mean, y_scale = y[train].mean(axis=0), y[train].std(axis=0)
    y_scale[y_scale < 1e-8] = 1.0

    def shape(values):
        return ((values - x_mean) / x_scale).astype("float32")

    return {
        "x_train": shape(x[train]), "y_train": ((y[train] - y_mean) / y_scale),
        "x_val": shape(x[val]), "y_val": ((y[val] - y_mean) / y_scale),
        "x_test": shape(x[test]),
        "y_test": y[test],
        "test_stamps": stamps[test],
        "train_stamps": stamps[train], "val_stamps": stamps[val],
        "y_mean": y_mean, "y_scale": y_scale,
    }


# ---------------------------------------------------------------------------

def evaluate(table: pd.DataFrame, series: str, stamps: pd.DatetimeIndex,
             predicted: np.ndarray, horizon: int) -> dict:
    target = f"target_{series}_h{horizon}"
    actual = table.loc[stamps, target]
    price = table[series].reindex(stamps) if series in table.columns else pd.Series(
        np.nan, index=stamps)
    guess = pd.Series(predicted, index=stamps)

    metrics = score(actual, guess, price, horizon)
    naive = pd.Series(0.0, index=stamps)
    naive_metrics = score(actual, naive, price, horizon)
    metrics["skill_vs_naive"] = (metrics["rmse"] / naive_metrics["rmse"]
                                 if naive_metrics.get("rmse") else np.nan)

    frame = pd.concat([actual.rename("y"), guess.rename("p")], axis=1).dropna()
    statistic, p_value = diebold_mariano(
        (0.0 - frame["y"]).to_numpy(), (frame["p"] - frame["y"]).to_numpy(), horizon)
    metrics["dm_vs_naive"] = statistic
    metrics["dm_p_value"] = p_value
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arch", choices=["lstm", "gru"], default="lstm")
    parser.add_argument("--mode", choices=["joint", "single"], default="single",
                        help="single (default): one network per series. joint: one "
                             "network with seven outputs - kept so the earlier "
                             "sharing comparison stays reproducible, but single won "
                             "it (0.893 against 0.981) and is the model to use")
    parser.add_argument("--features",
                        choices=["core", "plus-volume", "plus-macro", "plus-all"],
                        default="core",
                        help="which ablation arm to run. core: US prices and margins "
                             "only. plus-volume: adds negotiated head counts. "
                             "plus-macro: adds the dollar index and jobless claims. "
                             "plus-all: everything")
    parser.add_argument("--exclude-break", action="store_true",
                        help="drop the 2020-03 to 2021-06 market disruption from "
                             "training and validation. The test period is unchanged, "
                             "so results stay comparable with the full-sample run; "
                             "output files gain a _nobreak suffix")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--window", type=int, default=WINDOW)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--units", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--val-fraction", type=float, default=0.18)
    args = parser.parse_args()

    get_keras()        # fail early and clearly if Keras is not installed

    table = load_table()
    test_start = load_baseline_test_index(args.horizon, args.exclude_break).min()
    extra = {
        "core": [],
        "plus-volume": VOLUME_FEATURES,
        "plus-macro": MACRO_FEATURES,
        "plus-all": VOLUME_FEATURES + MACRO_FEATURES,
    }[args.features]
    features = CORE_FEATURES + extra
    # An arm whose own features are absent must not quietly become `core` under the
    # arm's name: that writes a mislabelled row into reports/ that looks like evidence
    # the features did not help, when in fact they were never used. Refuse instead.
    missing_extra = [c for c in extra if c not in table.columns]
    if missing_extra:
        raise SystemExit(
            f"\nArm '{args.features}' needs columns the model table does not have:\n"
            f"  {missing_extra}\n"
            "The table is stale or was built without the source these come from.\n"
            "Rebuild it first:  python src/build_dataset.py\n"
            "(For plus-macro you also need data/raw/fred_weekly.csv - "
            "run src/fetch_fred.py.)\n"
            "Refusing to run, because dropping them would record this arm's result "
            "as if the features had been tested.")

    missing = [c for c in features if c not in table.columns]
    if missing:
        print(f"WARNING: core features not in the table, dropped: {missing}")
        features = [c for c in features if c in table.columns]

    # Present but entirely empty is the same problem wearing a disguise.
    empty_extra = [c for c in extra if table[c].notna().sum() == 0]
    if empty_extra:
        raise SystemExit(
            f"\nArm '{args.features}' has these columns but they are completely "
            f"empty:\n  {empty_extra}\n"
            "Their source file is missing, so the arm would silently reduce to core.\n"
            "For plus-macro: run src/fetch_fred.py (or save the CSVs by hand as "
            "data/raw/fred_ICSA.csv and data/raw/fred_DTWEXBGS.csv), then rebuild.")

    targets = [t for t in US_TARGETS if f"target_{t}_h{args.horizon}" in table.columns]
    groups = [targets] if args.mode == "joint" else [[t] for t in targets]

    label = f"{args.arch}-{args.mode}-{args.features}"
    print(f"{label}: {len(features)} features/timestep, window {args.window}, "
          f"{len(targets)} target(s), {args.seeds} seed(s)")
    print(f"test period starts {test_start.date()} (taken from baseline_predictions.csv)")

    seed_rows, mean_rows, prediction_frames = [], [], []
    for group in groups:
        data = prepare(table, features, group, test_start, args)
        if data is None:
            print(f"  {', '.join(group)}: not enough complete windows, skipped")
            continue
        print(f"\n  {'all seven jointly' if len(group) > 1 else group[0]}: "
              f"train {len(data['x_train'])} ({data['train_stamps'].min().date()}.."
              f"{data['train_stamps'].max().date()}) | val {len(data['x_val'])} | "
              f"test {len(data['x_test'])}")

        collected = {series: [] for series in group}
        for seed in range(args.seeds):
            predictions, epochs_run, best_val = train_once(args.arch, data, seed, args)
            for position, series in enumerate(group):
                metrics = evaluate(table, series, data["test_stamps"],
                                   predictions[:, position], args.horizon)
                collected[series].append(predictions[:, position])
                seed_rows.append({"config": label, "series": series, "seed": seed,
                                  "epochs": epochs_run, "val_loss": best_val, **metrics})
            print(f"    seed {seed}: {epochs_run:3d} epochs, val_loss {best_val:.4f}, "
                  + "  ".join(
                      f"{s.replace('us_', '')} {seed_rows[-len(group) + i]['skill_vs_naive']:.3f}"
                      for i, s in enumerate(group)))

        # The reported prediction is the mean across seeds: an ensemble, and the honest
        # summary of a stochastic procedure.
        for series in group:
            stacked = np.mean(np.stack(collected[series]), axis=0)
            metrics = evaluate(table, series, data["test_stamps"], stacked, args.horizon)
            own = [r for r in seed_rows
                   if r["series"] == series and r["config"] == label]
            spread = [r["skill_vs_naive"] for r in own]
            errors = [r["rmse"] for r in own]
            mean_rows.append({"config": label, "series": series,
                              "rmse_seed_mean": float(np.mean(errors)),
                              "rmse_seed_sd": float(np.std(errors)),
                              "rmse_ensemble": metrics["rmse"],
                              "skill_seed_mean": float(np.mean(spread)),
                              "skill_seed_sd": float(np.std(spread)),
                              "skill_ensemble": metrics["skill_vs_naive"],
                              **{k: v for k, v in metrics.items()
                                 if k not in ("skill_vs_naive",)}})
            # Keep the per-seed predictions as well as the ensemble. The spread across
            # seeds is the model's own uncertainty estimate and is what forecast_plots.py
            # draws as a band; thrown away here it cannot be recovered without retraining.
            frame = pd.DataFrame({"prediction": stacked, "series": series,
                                  "config": label}, index=data["test_stamps"])
            for seed, values in enumerate(collected[series]):
                frame[f"seed_{seed}"] = values
            prediction_frames.append(frame)

    if not mean_rows:
        print("\nNothing was trained.")
        return

    REPORTS.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(mean_rows)
    seeds_frame = pd.DataFrame(seed_rows)
    for frame, name in ((results, tagged("dl_results.csv", args.horizon, args.exclude_break)),
                        (seeds_frame, tagged("dl_results_seeds.csv", args.horizon, args.exclude_break))):
        path = REPORTS / name
        if path.exists():                     # keep earlier configurations for comparison
            frame = pd.concat([pd.read_csv(path), frame], ignore_index=True)
            frame = frame.drop_duplicates(subset=["config", "series"]
                                          + (["seed"] if "seed" in frame.columns else []),
                                          keep="last")
        frame.to_csv(path, index=False)
    # Accumulate across configurations rather than overwriting, so every run stays
    # available for the final comparison and the forecast plots.
    predictions = pd.concat(prediction_frames)
    predictions.index.name = "week"
    path = REPORTS / tagged("dl_predictions.csv", args.horizon, args.exclude_break)
    if path.exists():
        earlier = pd.read_csv(path, index_col=0, parse_dates=True)
        earlier.index.name = "week"
        predictions = pd.concat([earlier, predictions])
        predictions = predictions[~predictions.reset_index().duplicated(
            subset=["week", "series", "config"], keep="last").to_numpy()]
    predictions.to_csv(path)

    print("\n" + "=" * 78)
    print(f"{label}: RMSE on the four-week log return, benchmark's own test weeks")
    print("(seed mean +/- sd across runs, then the across-seed ensemble;")
    print(" * marks a Diebold-Mariano difference from naive at p < 0.05)")
    print("=" * 78)
    print(f"  {'series':22s} {'across seeds':>20s} {'ensemble':>10s} "
          f"{'dir':>7s} {'mape%':>7s}")
    for _, row in results.iterrows():
        star = "*" if row.get("dm_p_value", 1) < 0.05 else " "
        print(f"  {row['series']:22s} "
              f"{row['rmse_seed_mean']:.5f}+/-{row['rmse_seed_sd']:.5f} "
              f"{row['rmse_ensemble']:9.5f}{star} "
              f"{row.get('direction', np.nan):6.1%} {row.get('mape_pct', np.nan):7.2f}")

    print("\nRun compare.py for these against the benchmarks on the same weeks.")
    print("If the seed standard deviation is as large as the gap to the best")
    print("benchmark, the difference is not evidence of anything.")


if __name__ == "__main__":
    main()
