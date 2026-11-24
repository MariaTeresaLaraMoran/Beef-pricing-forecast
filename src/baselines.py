"""Benchmark models for the four-week-ahead beef price forecast.

This is the "benchmark model" section of the project: traditional methods that
establish the performance floor the deep learning model has to clear. Nothing here
is a neural network.

Four models, in increasing order of ambition:

  naive    Predict a log return of zero, i.e. the price in four weeks is the price
           today. For a series close to a random walk this is very hard to beat, and
           it is the honest floor. Mexican cut prices are sticky -- 21 to 33 per cent
           of weeks show no change at all -- so this bar is higher than it looks.

  drift    Predict the mean four-week return seen in the training period. Tests
           whether a constant trend adds anything over the naive forecast.

  arima    ARIMA(p, 1, q) on the log price, order chosen by AIC on the training
           period, forecast four steps ahead and converted back to a four-week log
           return. Fitted on an expanding window and re-estimated every
           --refit-every weeks; between re-estimations the existing parameters are
           applied to the longer history, which is what a forecaster would do in
           practice and keeps the run time reasonable.

  ridge    Ridge regression on lagged features: own lagged returns, the carcass
           price, the exchange rate, US export volumes, the Mexico-minus-US spread
           (for Mexican targets), and a smooth annual seasonal pair. This is the
           multivariate benchmark -- the model that gets to see everything the
           network sees, but can only combine it linearly.

Both target families are evaluated, so the choice between forecasting Mexican cut
prices and forecasting US boxed beef is settled by measurement rather than argument.
Because the two families have very different volatility (US cut returns have roughly
twice the standard deviation of Mexican ones at this horizon), the headline number is
*skill* -- RMSE divided by the naive model's RMSE on the same rows. Below 1.00 means
the model beat the floor; 1.00 or above means it did not.

Splitting
---------
Time-ordered, never random: train, then validation, then test, in calendar order.
A gap of HORIZON weeks is left between consecutive splits. Without it the last
training rows would have targets that reach into the validation period, and the
model would be scored partly on weeks it had already seen.

Usage:
    python src/baselines.py
    python src/baselines.py --family mexico
    python src/baselines.py --models naive,drift,ridge      # skip the slow one
    python src/baselines.py --refit-every 4 --max-order 3

Output:
    reports/baseline_results.csv     one row per target per model
    reports/baseline_predictions.csv the test-set predictions, for the DL comparison
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

HORIZON = 4
LAGS = [1, 2, 3, 4]


# The documented 2020-21 beef market disruption: plant closures, and the period USDA
# AMS formally investigated. Declared as a fixed calendar window rather than picked
# from the data, so the exclusion cannot be tuned to flatter a result.
BREAK_START = pd.Timestamp("2020-03-01")
BREAK_END = pd.Timestamp("2021-06-30")


def drop_break(index: pd.DatetimeIndex, label: str) -> pd.DatetimeIndex:
    """Remove the disruption window from a FITTING index.

    Applied to training and validation only, never to test. The split is computed
    before this runs, so both variants are scored on exactly the same weeks: only what
    the model learned from differs, which is what makes the two directly comparable.
    """
    keep = index[(index < BREAK_START) | (index > BREAK_END)]
    dropped = len(index) - len(keep)
    if dropped:
        print(f"    {label}: dropped {dropped} week(s) in the 2020-21 disruption")
    return keep


def tagged(name: str, horizon: int, exclude_break: bool = False) -> str:
    """Output filename for a horizon and disruption setting.

    The default four-week, full-sample results keep their plain names so earlier runs
    stay valid; any other horizon gets an _h<N> suffix and the excluded-disruption
    variant a _nobreak one, so all four combinations sit beside each other instead of
    overwriting.
    """
    stem, _, suffix = name.rpartition(".")
    if horizon != HORIZON:
        stem = f"{stem}_h{horizon}"
    if exclude_break:
        stem = f"{stem}_nobreak"
    return f"{stem}.{suffix}"
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]


# ---------------------------------------------------------------------------
# Targets and features
# ---------------------------------------------------------------------------

def discover_targets(table: pd.DataFrame, horizon: int) -> dict[str, list[str]]:
    """Group the table's target columns into the two families."""
    suffix = f"_h{horizon}"
    families: dict[str, list[str]] = {"mexico": [], "us": []}
    for column in table.columns:
        if not (column.startswith("target_") and column.endswith(suffix)):
            continue
        series = column[len("target_"):-len(suffix)]
        if series.startswith("price_"):
            families["mexico"].append(series)
        elif series.startswith("us_"):
            families["us"].append(series)
    return {name: series for name, series in families.items() if series}


def feature_columns(table: pd.DataFrame, series: str) -> list[str]:
    """The predictors for one target, all of them knowable at the forecast origin.

    Note on the spread: spread_<cut> at week t is log(Mexican price at t) minus
    log(US price in MXN/kg), where the US leg already carries its publication lag.
    Both legs were public at week t, so the contemporaneous spread is a legitimate
    feature, not a leak. It is also the single most informative US-derived column.
    """
    # Lag 0 is included deliberately. ret_X on row t is the return into week t, which
    # for a series with a publication lag is already a historical value, and for a
    # same-week series is the move that had finished by the forecast origin. Either
    # way it was knowable at t, and it is the strongest single momentum term; leaving
    # it out would throw away a week of information.
    columns = [f"ret_{series}"] + [f"ret_{series}_lag{lag}" for lag in LAGS]

    if series.startswith("price_"):
        cut = series[len("price_"):]
        for base in ("ret_carcass", "ret_usd_mxn", "ret_exports_tonnes"):
            columns.append(base)
            columns += [f"{base}_lag{lag}" for lag in LAGS]
        for base in (f"spread_{cut}", f"spread_{cut}_change"):
            columns.append(base)
            columns += [f"{base}_lag{lag}" for lag in LAGS]
    else:
        # For a US target the Mexican side is irrelevant, but the cutout index and
        # export demand are both US-side information.
        for base in ("ret_us_cutout_choice", "ret_us_steer_dressed",
                     "ret_exports_tonnes"):
            columns.append(base)
            columns += [f"{base}_lag{lag}" for lag in LAGS]
        # The gross packer margin, the US-side error-correction term. Both legs carry
        # their publication lag already, so the contemporaneous value was knowable at
        # the forecast origin.
        for base in ("margin_choice_log", "margin_choice_log_change"):
            columns.append(base)
            columns += [f"{base}_lag{lag}" for lag in LAGS]

    columns += ["season_sin", "season_cos"]
    return [c for c in dict.fromkeys(columns) if c in table.columns]


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def split_index(index: pd.DatetimeIndex, horizon: int,
                train_fraction: float, val_fraction: float
                ) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex]:
    """Time-ordered three-way split with a `horizon`-week embargo between parts."""
    n = len(index)
    train_end = int(n * train_fraction)
    val_end = int(n * (train_fraction + val_fraction))

    train = index[:train_end]
    val = index[train_end + horizon:val_end]
    test = index[val_end + horizon:]
    return train, val, test


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def fit_ridge(table: pd.DataFrame, target: str, features: list[str],
              fit_index: pd.DatetimeIndex, predict_index: pd.DatetimeIndex,
              alpha: float) -> pd.Series:
    """Standardise on the fitting rows only, then ridge-regress."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train_rows = table.loc[fit_index, features + [target]].dropna()
    if len(train_rows) < 50:
        return pd.Series(np.nan, index=predict_index)

    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(train_rows[features], train_rows[target])

    predict_rows = table.loc[predict_index, features]
    usable = predict_rows.dropna().index
    out = pd.Series(np.nan, index=predict_index)
    if len(usable):
        out.loc[usable] = model.predict(predict_rows.loc[usable])
    return out


def choose_arima_order(log_price: pd.Series, train_end: pd.Timestamp,
                       max_order: int) -> tuple[tuple[int, int, int], str, float]:
    """Grid-search (p, 1, q) and the trend term by AIC on the training period."""
    from statsmodels.tsa.arima.model import ARIMA

    history = log_price.loc[:train_end]
    best = (None, None, np.inf)
    for p in range(max_order + 1):
        for q in range(max_order + 1):
            if p == 0 and q == 0:
                continue
            for trend in ("n", "c"):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        result = ARIMA(history, order=(p, 1, q), trend=trend).fit()
                    if np.isfinite(result.aic) and result.aic < best[2]:
                        best = ((p, 1, q), trend, result.aic)
                except Exception:
                    continue
    if best[0] is None:
        return (1, 1, 0), "n", np.nan
    return best[0], best[1], best[2]


def arima_forecasts(log_price: pd.Series, origins: pd.DatetimeIndex,
                    order: tuple[int, int, int], trend: str,
                    horizon: int, refit_every: int) -> pd.Series:
    """Walk forward over `origins`, forecasting the horizon-week log return.

    Parameters are re-estimated every `refit_every` origins. In between, the stored
    parameters are applied to the lengthened history, so each forecast still uses
    every observation up to its own origin but costs almost nothing.
    """
    from statsmodels.tsa.arima.model import ARIMA

    predictions: dict[pd.Timestamp, float] = {}
    result = None
    for position, origin in enumerate(origins):
        history = log_price.loc[:origin]
        refit = result is None or position % refit_every == 0
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if refit:
                    result = ARIMA(history, order=order, trend=trend).fit()
                else:
                    try:
                        result = result.apply(history, refit=False)
                    except Exception:
                        result = ARIMA(history, order=order, trend=trend).fit()
                forecast = result.forecast(steps=horizon)
        except Exception:
            predictions[origin] = np.nan
            result = None
            continue

        current = log_price.loc[origin]
        predictions[origin] = (float(np.asarray(forecast)[-1]) - float(current)
                               if np.isfinite(current) else np.nan)
    return pd.Series(predictions, dtype=float).reindex(origins)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def diebold_mariano(errors_a: np.ndarray, errors_b: np.ndarray,
                    horizon: int) -> tuple[float, float]:
    """Test whether two forecasts differ in squared-error accuracy.

    Positive statistic means model A has the larger loss, i.e. B is better.
    Overlapping h-step forecasts are serially correlated, so the variance uses a
    Newey-West estimator with h-1 lags, with the Harvey-Leybourne-Newbold small
    sample correction applied.
    """
    difference = errors_a ** 2 - errors_b ** 2
    difference = difference[np.isfinite(difference)]
    n = len(difference)
    if n < 20:
        return np.nan, np.nan

    mean = difference.mean()
    centred = difference - mean
    variance = float(np.mean(centred ** 2))
    for lag in range(1, horizon):
        if lag >= n:
            break
        variance += 2.0 * float(np.mean(centred[:n - lag] * centred[lag:]))
    if variance <= 0:
        return np.nan, np.nan

    statistic = mean / np.sqrt(variance / n)
    correction = np.sqrt(max((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n, 1e-12))
    statistic *= correction
    p_value = 2.0 * (1.0 - stats.t.cdf(abs(statistic), df=n - 1))
    return float(statistic), float(p_value)


def score(actual: pd.Series, predicted: pd.Series, price: pd.Series,
          horizon: int) -> dict[str, float]:
    """Metrics on the log-return scale, plus a price-level error for the slides."""
    frame = pd.concat([actual.rename("y"), predicted.rename("yhat"),
                       price.rename("p")], axis=1).dropna(subset=["y", "yhat"])
    if frame.empty:
        return {"n": 0}

    error = frame["yhat"] - frame["y"]
    out = {
        "n": int(len(frame)),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
    }

    # Direction, ignoring weeks where the price did not move at all: calling "no
    # change" correct or incorrect is arbitrary and these series are sticky. A model
    # that never predicts a direction -- the naive forecast, which is always zero --
    # gets no score here rather than a meaningless zero.
    moved = frame[(frame["y"] != 0) & (frame["yhat"] != 0)]
    out["direction"] = (float(np.mean(np.sign(moved["yhat"]) == np.sign(moved["y"])))
                        if len(moved) else np.nan)

    # Absolute percentage error on the recovered price, which is what a stakeholder
    # reads: P(t) * exp(yhat) against the price actually observed at t + horizon.
    with_price = frame.dropna(subset=["p"])
    if len(with_price):
        predicted_price = with_price["p"] * np.exp(with_price["yhat"])
        actual_price = with_price["p"] * np.exp(with_price["y"])
        out["mape_pct"] = float(np.mean(np.abs(predicted_price - actual_price)
                                        / actual_price) * 100)
    return out


# ---------------------------------------------------------------------------

def run_target(table: pd.DataFrame, series: str, horizon: int, models: list[str],
               refit_every: int, max_order: int, exclude_break: bool = False
               ) -> tuple[list[dict], pd.DataFrame]:
    target = f"target_{series}_h{horizon}"
    features = feature_columns(table, series)
    log_price = table[f"log_{series}"]

    # Rows that every model can be scored on: target present and features complete.
    usable = table.loc[table[target].notna() & table[features].notna().all(axis=1)].index
    if len(usable) < 200:
        print(f"  {series}: only {len(usable)} usable rows, skipped")
        return [], pd.DataFrame()

    train, val, test = split_index(usable, horizon, TRAIN_FRACTION, VAL_FRACTION)
    if exclude_break:
        train = drop_break(train, "train")
        val = drop_break(val, "validation")
    print(f"  {series}: {len(features)} features | "
          f"train {len(train)} ({train.min().date()}..{train.max().date()}) | "
          f"val {len(val)} | test {len(test)} ({test.min().date()}..{test.max().date()})")

    actual = table.loc[test, target]
    price = table[series].reindex(test) if series in table.columns else pd.Series(
        np.nan, index=test)
    predictions = pd.DataFrame(index=test)
    chosen: dict[str, str] = {}

    if "naive" in models:
        predictions["naive"] = 0.0

    if "drift" in models:
        predictions["drift"] = float(table.loc[train, target].mean())

    if "ridge" in models:
        # Alpha on the validation block, then refit on train + val for the test block.
        best_alpha, best_rmse = RIDGE_ALPHAS[0], np.inf
        for alpha in RIDGE_ALPHAS:
            guess = fit_ridge(table, target, features, train, val, alpha)
            result = score(table.loc[val, target], guess, pd.Series(np.nan, index=val),
                           horizon)
            if result.get("n", 0) and result["rmse"] < best_rmse:
                best_alpha, best_rmse = alpha, result["rmse"]
        chosen["ridge"] = f"alpha={best_alpha:g}"
        predictions["ridge"] = fit_ridge(table, target, features,
                                         train.union(val), test, best_alpha)

    if "arima" in models:
        order, trend, aic = choose_arima_order(log_price, val.max(), max_order)
        chosen["arima"] = (f"ARIMA{order} trend={trend}"
                           + (f" AIC={aic:.1f}" if np.isfinite(aic) else ""))
        print(f"    arima order by AIC on train+val: {chosen['arima']}")
        predictions["arima"] = arima_forecasts(log_price, test, order, trend,
                                               horizon, refit_every)

    # Score everything on the same rows, and test each model against naive.
    rows = []
    naive_error = None
    if "naive" in predictions:
        aligned = pd.concat([actual, predictions["naive"]], axis=1).dropna()
        naive_error = (aligned.iloc[:, 1] - aligned.iloc[:, 0])

    naive_rmse = np.nan
    for name in predictions.columns:
        metrics = score(actual, predictions[name], price, horizon)
        if name == "naive":
            naive_rmse = metrics.get("rmse", np.nan)

    for name in predictions.columns:
        metrics = score(actual, predictions[name], price, horizon)
        if not metrics.get("n"):
            continue
        row = {"family": "mexico" if series.startswith("price_") else "us",
               "series": series, "model": name, **metrics}
        row["skill_vs_naive"] = (metrics["rmse"] / naive_rmse
                                 if np.isfinite(naive_rmse) and naive_rmse else np.nan)
        if name in chosen:
            row["settings"] = chosen[name]

        if naive_error is not None and name != "naive":
            both = pd.concat([actual.rename("y"), predictions["naive"].rename("a"),
                              predictions[name].rename("b")], axis=1).dropna()
            statistic, p_value = diebold_mariano((both["a"] - both["y"]).to_numpy(),
                                                 (both["b"] - both["y"]).to_numpy(),
                                                 horizon)
            row["dm_vs_naive"] = statistic
            row["dm_p_value"] = p_value
        rows.append(row)

    predictions["actual"] = actual
    predictions["series"] = series
    return rows, predictions



def merge_results(fresh: pd.DataFrame, path: Path, models: list[str]) -> pd.DataFrame:
    """Keep scored rows for models this run did not compute.

    Re-running a subset is the normal case when a feature change cannot affect every
    model: ARIMA and naive never see the feature set, so when only ridge changes there
    is no reason to pay for ARIMA again. Without this, writing the file would silently
    drop the models left out. Rows are replaced per model, not per file.
    """
    if not path.exists():
        return fresh
    previous = pd.read_csv(path)
    if "model" not in previous.columns:
        return fresh
    kept = previous[~previous["model"].isin(models)]
    if kept.empty:
        return fresh
    print(f"  carried forward {len(kept)} row(s) for model(s) not run this time: "
          f"{', '.join(sorted(kept['model'].unique()))}")
    return pd.concat([kept, fresh], ignore_index=True)


def merge_predictions(fresh: pd.DataFrame, path: Path, models: list[str]) -> pd.DataFrame:
    """The same carry-forward for the wide predictions file, where models are columns."""
    if not path.exists():
        return fresh
    index_name = fresh.index.name or "week"
    previous = pd.read_csv(path, index_col=0, parse_dates=True)
    previous.index.name = index_name
    if "series" not in previous.columns or "series" not in fresh.columns:
        return fresh
    carry = [c for c in previous.columns
             if c != "series" and c not in fresh.columns and c not in models]
    if not carry:
        return fresh
    key = [index_name, "series"]
    merged = (fresh.reset_index()
                   .merge(previous.reset_index()[key + carry], on=key, how="left")
                   .set_index(index_name))
    print(f"  carried forward prediction column(s): {', '.join(carry)}")
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--exclude-break", action="store_true",
                        help="drop the 2020-03 to 2021-06 market disruption from "
                             "training and validation. The test period is unchanged, "
                             "so results stay comparable with the full-sample run; "
                             "output files gain a _nobreak suffix")
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--family", choices=["mexico", "us", "both"], default="both")
    parser.add_argument("--models", default="naive,ridge,arima",
                        help="comma-separated subset of naive,drift,ridge,arima. The "
                             "default drops drift, which scored 0.996-1.000 against "
                             "naive on every series and carries no information. naive "
                             "is kept because an RMSE means nothing without it: it is "
                             "the error you get from assuming the price does not move")
    parser.add_argument("--refit-every", type=int, default=8,
                        help="re-estimate the ARIMA parameters every N test weeks")
    parser.add_argument("--max-order", type=int, default=2,
                        help="largest p and q considered in the ARIMA grid")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if "arima" in models:
        try:
            import statsmodels  # noqa: F401
        except ImportError:
            print("statsmodels is not installed; run `pip install statsmodels` or "
                  "pass --models naive,drift,ridge")
            return

    table = pd.read_csv(PROCESSED / "weekly_model_table.csv",
                        index_col=0, parse_dates=True)
    families = discover_targets(table, args.horizon)
    wanted = list(families) if args.family == "both" else [args.family]

    all_rows: list[dict] = []
    all_predictions: list[pd.DataFrame] = []
    for family in wanted:
        if family not in families:
            print(f"\n{family}: no targets in the table, skipped")
            continue
        print(f"\n{family.upper()} targets ({len(families[family])})")
        for series in families[family]:
            rows, predictions = run_target(table, series, args.horizon, models,
                                           args.refit_every, args.max_order,
                                           args.exclude_break)
            all_rows += rows
            if not predictions.empty:
                all_predictions.append(predictions)

    if not all_rows:
        print("\nNothing was scored.")
        return

    results = pd.DataFrame(all_rows)
    REPORTS.mkdir(parents=True, exist_ok=True)
    results_path = REPORTS / tagged("baseline_results.csv", args.horizon, args.exclude_break)
    results = merge_results(results, results_path, models)
    results.to_csv(results_path, index=False)
    if all_predictions:
        predictions_path = REPORTS / tagged("baseline_predictions.csv", args.horizon, args.exclude_break)
        frame = merge_predictions(pd.concat(all_predictions), predictions_path, models)
        frame.to_csv(predictions_path)

    # Printed summary: skill against naive is the column that matters.
    print("\n" + "=" * 78)
    print("TEST SET: RMSE on the four-week log return")
    print("(lower is better; compare models down a column, within one series;")
    print(" * marks a Diebold-Mariano difference from naive at p < 0.05)")
    print("=" * 78)
    for family, block in results.groupby("family", sort=False):
        print(f"\n{family.upper()}")
        print(f"  {'series':24s} {'model':7s} {'n':>5s} {'rmse':>9s} "
              f"{'dir':>6s} {'mape%':>7s}")
        for series, rows in block.groupby("series", sort=False):
            for _, row in rows.iterrows():
                star = "*" if row.get("dm_p_value", 1) < 0.05 else " "
                print(f"  {series:24s} {row['model']:7s} {row['n']:5.0f} "
                      f"{row['rmse']:9.5f}{star} "
                      f"{row.get('direction', np.nan):6.1%} "
                      f"{row.get('mape_pct', np.nan):7.2f}")

    best = (results[results["model"] != "naive"]
            .sort_values("rmse").groupby("series", sort=False).head(1))
    print("\nBest benchmark per series:")
    for _, row in best.iterrows():
        naive_row = results[(results["series"] == row["series"])
                            & (results["model"] == "naive")]
        naive_rmse = float(naive_row["rmse"].iloc[0]) if len(naive_row) else np.nan
        print(f"  {row['series']:24s} {row['model']:7s} rmse {row['rmse']:.5f}  "
              f"(naive {naive_rmse:.5f}, p={row.get('dm_p_value', np.nan):.3f})")

    print(f"\nWritten: {(REPORTS / tagged('baseline_results.csv', args.horizon, args.exclude_break)).relative_to(ROOT)}")
    print(f"         {(REPORTS / tagged('baseline_predictions.csv', args.horizon, args.exclude_break)).relative_to(ROOT)}")
    print("\nThe deep learning model has to beat the best skill figure above on the "
          "same test rows. Those predictions are saved so the comparison is exact.")


if __name__ == "__main__":
    main()
