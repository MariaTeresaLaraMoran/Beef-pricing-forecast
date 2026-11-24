"""Final comparison of every forecasting method, scored by RMSE.

Reads what baselines.py and models.py wrote and puts them side by side on the same
169 test weeks.

Everything is RMSE on the four-week log return - the quantity actually being forecast
- so the numbers compare directly between methods within one series. They do NOT
compare between series: the ribeye's four-week return is three times as volatile as
the cutout's, so its RMSE is larger whatever the model does. Read each row on its own.

The naive forecast is kept as a column. It is not a benchmark in the usual sense; it
is the error you get from assuming the price in four weeks is the price today. Without
it an RMSE of 0.0383 means nothing.

Methods compared:
    naive                 the price does not move
    ridge                 regularised linear regression on 32 lagged features
    arima                 ARIMA(p,1,q), order chosen by AIC search
    lstm-single-<arm>     one LSTM per series
    gru-single-<arm>      one GRU per series

Joint multi-output runs are excluded by default. They were tested and lost, so they
stay in the result files for the record but out of the comparison. Pass
--include-joint to see them.

Usage:
    python src/compare.py
    python src/compare.py --include-joint
    python src/compare.py --horizon 12

Output:
    reports/final_comparison.csv
    reports/figures/fig40_rmse_by_model.png
    reports/figures/fig41_seed_stability.png
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

SHORT = {
    "us_cutout_choice": "Choice cutout", "us_cutout_select": "Select cutout",
    "us_ribeye": "Ribeye (109E)", "us_chuck_roll": "Chuck roll (116A)",
    "us_brisket": "Brisket (120)", "us_top_inside_round": "Top inside round (168)",
    "us_top_butt": "Top sirloin butt (184)",
}
ORDER = list(SHORT)
BENCHMARKS = ["naive", "ridge", "arima"]


def tagged(name: str, horizon: int, exclude_break: bool = False) -> str:
    """h=4 full-sample keeps the plain filename; other horizons get an _h<N> suffix and
    the excluded-disruption variant a _nobreak one (see baselines.py)."""
    stem, _, suffix = name.rpartition(".")
    if horizon != 4:
        stem = f"{stem}_h{horizon}"
    if exclude_break:
        stem = f"{stem}_nobreak"
    return f"{stem}.{suffix}"


def test_weeks(baselines: pd.DataFrame) -> int:
    """How many weeks were actually scored. The embargo grows with the horizon, so
    this is not the same number at h=4 and h=12 and must not be hard-coded."""
    n = baselines["n"].dropna()
    return int(n.max()) if len(n) else 0


def load(horizon: int, exclude_break: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_path = REPORTS / tagged("baseline_results.csv", horizon, exclude_break)
    dl_path = REPORTS / tagged("dl_results.csv", horizon, exclude_break)
    for path in (baseline_path, dl_path):
        if not path.exists():
            raise SystemExit(f"{path.relative_to(ROOT)} not found. "
                             "Run baselines.py and models.py first.")
    baselines = pd.read_csv(baseline_path)
    return baselines[baselines["family"] == "us"], pd.read_csv(dl_path)


def pick_configs(dl: pd.DataFrame, include_joint: bool) -> list[str]:
    configs = sorted(dl["config"].unique())
    if include_joint:
        return configs
    return [c for c in configs if "-joint-" not in c]


def short_label(config: str) -> str:
    return config.replace("-single", "").replace("-core", "")


def build_table(baselines: pd.DataFrame, dl: pd.DataFrame,
                configs: list[str]) -> pd.DataFrame:
    """One row per series, one column per method, values are RMSE."""
    rows = []
    for series in ORDER:
        row = {"series": SHORT[series]}
        for model in BENCHMARKS:
            match = baselines[(baselines["series"] == series)
                              & (baselines["model"] == model)]
            row[model] = float(match["rmse"].iloc[0]) if len(match) else np.nan
        for config in configs:
            match = dl[(dl["series"] == series) & (dl["config"] == config)]
            if len(match):
                # Deep models are stochastic, so the headline number is the
                # across-seed ensemble and the spread is carried beside it.
                row[config] = float(match["rmse_ensemble"].iloc[0])
                row[f"{config}__sd"] = float(match["rmse_seed_sd"].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows).set_index("series")


def winners(table: pd.DataFrame, configs: list[str]) -> pd.DataFrame:
    """Best method per series, and whether the margin survives seed noise."""
    out = []
    for label, row in table.iterrows():
        benchmarks = [c for c in BENCHMARKS if np.isfinite(row.get(c, np.nan))]
        deep = [c for c in configs if np.isfinite(row.get(c, np.nan))]
        if not benchmarks or not deep:
            continue

        best_benchmark = min(benchmarks, key=lambda c: row[c])
        best_deep = min(deep, key=lambda c: row[c])
        sd = row.get(f"{best_deep}__sd", np.nan)
        gain = row[best_benchmark] - row[best_deep]
        overall = min(benchmarks + deep, key=lambda c: row[c])

        out.append({
            "series": label,
            "naive": round(float(row.get("naive", np.nan)), 5),
            "best benchmark": f"{best_benchmark} {row[best_benchmark]:.5f}",
            "best deep": f"{short_label(best_deep)} {row[best_deep]:.5f}",
            "gain": round(gain, 5),
            "seed sd": round(sd, 5) if np.isfinite(sd) else np.nan,
            "gain/sd": round(gain / sd, 1) if np.isfinite(sd) and sd > 0 else np.nan,
            "winner": "deep" if overall not in BENCHMARKS else "benchmark",
        })
    return pd.DataFrame(out).set_index("series")


def split_config(config: str) -> tuple[str, str, str]:
    """lstm-single-plus-volume -> ("lstm", "single", "plus-volume")."""
    parts = config.split("-")
    if len(parts) < 3:
        return config, "", "core"
    return parts[0], parts[1], "-".join(parts[2:])


def ablation_report(dl: pd.DataFrame, configs: list[str]) -> None:
    """Each feature arm against the core arm of the same architecture and mode."""
    rows = []
    for config in configs:
        block = dl[dl["config"] == config]
        if block.empty:
            continue
        arch, mode, arm = split_config(config)
        rows.append({"arch": arch, "mode": mode, "arm": arm,
                     "rmse": block["rmse_ensemble"].mean(),
                     "seed sd": block["rmse_seed_sd"].mean()})
    table = pd.DataFrame(rows)
    if len(table) < 2:
        return

    print("\n" + "=" * 92)
    print("FEATURE ABLATION - mean RMSE across the seven series")
    print("delta is against the core arm of the same architecture;")
    print("negative delta means the extra features helped")
    print("=" * 92)
    for (arch, mode), group in table.groupby(["arch", "mode"], sort=False):
        core = group[group["arm"] == "core"]
        base = float(core["rmse"].iloc[0]) if len(core) else np.nan
        base_sd = float(core["seed sd"].iloc[0]) if len(core) else np.nan
        print(f"\n{arch}-{mode}" + ("" if len(core) else "   (no core run)"))
        print(f"  {'arm':14s} {'rmse':>9s} {'delta':>10s} {'seed sd':>10s} "
              f"{'sd ratio':>9s}")
        for _, row in group.sort_values("rmse").iterrows():
            delta = row["rmse"] - base
            ratio = (row["seed sd"] / base_sd
                     if base_sd and np.isfinite(base_sd) else np.nan)
            delta_text = f"{delta:+10.5f}" if np.isfinite(delta) else " " * 10
            ratio_text = f"{ratio:8.2f}x" if np.isfinite(ratio) else " " * 9
            print(f"  {row['arm']:14s} {row['rmse']:9.5f} {delta_text} "
                  f"{row['seed sd']:10.5f} {ratio_text}")

    print("\nAn arm that adds features without lowering RMSE, while raising the seed")
    print("standard deviation, has given the network more ways to fit noise and")
    print("nothing new to learn. That is a result, not a failure.")


def figure_rmse(table: pd.DataFrame, horizon: int,
                exclude_break: bool = False) -> None:
    """One panel per series, each on its own scale.

    RMSE cannot share an axis across series - the ribeye's is three times the
    cutout's before any model is fitted - so a single grouped bar chart would say
    more about volatility than about the models. Small multiples keep the comparison
    where it belongs: between methods, within a series.
    """
    columns = [c for c in table.columns if not c.endswith("__sd")]
    labels = [short_label(c) for c in columns]
    palette = ["#9e9e9e", "#1f4e79", "#2e7d32", "#8b0000", "#b36b00", "#5b2c6f",
               "#00695c"]

    panels = len(table)
    per_row = 2
    rows_needed = int(np.ceil(panels / per_row))
    fig, axes = plt.subplots(rows_needed, per_row, figsize=(13, 2.7 * rows_needed))
    axes = np.atleast_1d(axes).ravel()

    for ax, (series, row) in zip(axes, table.iterrows()):
        values = [row[c] for c in columns]
        bars = ax.bar(range(len(columns)), values,
                      color=[palette[i % len(palette)] for i in range(len(columns))])
        naive = row.get("naive", np.nan)
        if np.isfinite(naive):
            ax.axhline(naive, color="black", lw=1.0, ls="--")
        finite = [v for v in values if np.isfinite(v)]
        if finite:
            best = int(np.nanargmin(values))
            bars[best].set_edgecolor("black")
            bars[best].set_linewidth(1.6)
            ax.set_ylim(min(finite) * 0.88, max(finite) * 1.04)
        ax.set_xticks(range(len(columns)), labels, rotation=35, ha="right", fontsize=7)
        ax.set_title(series, fontsize=10, loc="left")
        ax.set_ylabel("RMSE", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    for ax in axes[panels:]:
        ax.axis("off")

    fig.suptitle(f"RMSE on the {horizon}-week log return\ndashed line = naive "
                 "forecast, outlined bar = best; each panel has its own scale", y=1.0)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / tagged("fig40_rmse_by_model.png", horizon, exclude_break)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.relative_to(ROOT)}")


def figure_stability(dl: pd.DataFrame, configs: list[str], horizon: int,
                     exclude_break: bool = False) -> None:
    block = dl[dl["config"].isin(configs)]
    if block.empty:
        return
    pivot = (block.pivot_table(index="config", values="rmse_seed_sd", aggfunc="mean")
             .sort_values("rmse_seed_sd"))

    fig, ax = plt.subplots(figsize=(9, 0.7 * len(pivot) + 2.4))
    ax.barh([short_label(c) for c in pivot.index], pivot["rmse_seed_sd"],
            color="#1f4e79")
    for position, value in enumerate(pivot["rmse_seed_sd"]):
        ax.text(value * 1.02, position, f"{value:.5f}", va="center", fontsize=9)
    ax.set_xlabel("mean standard deviation of RMSE across random seeds")
    ax.set_title("Run-to-run stability: how much the answer moves with the seed")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    path = FIGURES / tagged("fig41_seed_stability.png", horizon, exclude_break)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--exclude-break", action="store_true",
                        help="read the run that excluded the 2020-21 disruption "
                             "from training (the _nobreak files)")
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--include-joint", action="store_true",
                        help="also show the joint multi-output runs, which lost")
    args = parser.parse_args()

    baselines, dl = load(args.horizon, args.exclude_break)
    if "rmse_ensemble" not in dl.columns:
        raise SystemExit(
            "dl_results.csv has no rmse_ensemble column - it was written by an "
            "older models.py. Re-run the configurations you want to compare.")

    configs = pick_configs(dl, args.include_joint)
    if not configs:
        raise SystemExit("No single-output deep learning runs found. "
                         "Run models.py --features core first.")

    table = build_table(baselines, dl, configs)
    display = table[[c for c in table.columns if not c.endswith("__sd")]]
    display.columns = [short_label(c) for c in display.columns]

    print("=" * 92)
    print(f"RMSE ON THE {args.horizon}-WEEK LOG RETURN - "
          f"{test_weeks(baselines)} common test weeks")
    print("lower is better. Compare methods ACROSS a row; each series has its own")
    print("volatility, so RMSE is not comparable between rows.")
    print("=" * 92)
    print(display.round(5).to_string())

    print("\n" + "=" * 92)
    print("BEST METHOD PER SERIES")
    print("gain = best benchmark RMSE minus best deep RMSE (positive means deep won)")
    print("gain/sd compares that gain with the seed spread; under about 2 is noise")
    print("=" * 92)
    summary = winners(table, configs)
    print(summary.to_string())
    print(f"\ndeep learning best on {int((summary['winner'] == 'deep').sum())} "
          f"of {len(summary)} series")

    ablation_report(dl, configs)

    out_name = tagged("final_comparison.csv", args.horizon, args.exclude_break)
    table.to_csv(REPORTS / out_name)
    print(f"\nWritten {(REPORTS / out_name).relative_to(ROOT)}")
    figure_rmse(table, args.horizon, args.exclude_break)
    figure_stability(dl, configs, args.horizon, args.exclude_break)


if __name__ == "__main__":
    main()
