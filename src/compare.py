"""Final comparison: every benchmark against every deep learning configuration.

Reads what baselines.py and models.py wrote and answers the three questions the
project set out to answer, plus the one the rubric asks last - whether the added
complexity is justified.

Everything is reported as skill against the naive forecast on the same 169 test weeks:

    skill = RMSE(model) / RMSE(naive)

Below 1.00 beats "the price in four weeks is the price today". The deep learning
figures are the across-seed ensemble, with the seed standard deviation carried
alongside, because a difference smaller than the seed spread is not a difference.

Usage:
    python src/compare.py

Output:
    reports/final_comparison.csv       one row per series, every model as a column
    reports/figures/fig40_skill_by_model.png
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
BENCHMARKS = ["naive", "drift", "ridge", "arima"]


def tagged(name: str, horizon: int) -> str:
    """h=4 keeps the plain filename; other horizons get a suffix (see baselines.py)."""
    if horizon == 4:
        return name
    stem, _, suffix = name.rpartition(".")
    return f"{stem}_h{horizon}.{suffix}"


def load(horizon: int = 4) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_path = REPORTS / tagged("baseline_results.csv", horizon)
    dl_path = REPORTS / tagged("dl_results.csv", horizon)
    for path in (baseline_path, dl_path):
        if not path.exists():
            raise SystemExit(f"{path.relative_to(ROOT)} not found. "
                             "Run baselines.py and models.py first.")
    baselines = pd.read_csv(baseline_path)
    return baselines[baselines["family"] == "us"], pd.read_csv(dl_path)


def build_table(baselines: pd.DataFrame, dl: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for series in ORDER:
        row = {"series": SHORT.get(series, series)}
        for model in BENCHMARKS:
            match = baselines[(baselines["series"] == series)
                              & (baselines["model"] == model)]
            row[model] = float(match["skill_vs_naive"].iloc[0]) if len(match) else np.nan
        for config in sorted(dl["config"].unique()):
            match = dl[(dl["series"] == series) & (dl["config"] == config)]
            if len(match):
                row[config] = float(match["skill_ensemble"].iloc[0])
                row[f"{config}__sd"] = float(match["skill_seed_sd"].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows).set_index("series")


def winners(table: pd.DataFrame) -> pd.DataFrame:
    """Best model per series, and by how much it beats the best benchmark."""
    model_columns = [c for c in table.columns if not c.endswith("__sd")]
    deep_columns = [c for c in model_columns if c not in BENCHMARKS]

    out = []
    for series, row in table.iterrows():
        best_benchmark = row[BENCHMARKS].idxmin()
        best_deep = row[deep_columns].idxmin()
        overall = row[model_columns].idxmin()
        sd = row.get(f"{best_deep}__sd", np.nan)
        gap = row[best_benchmark] - row[best_deep]
        out.append({
            "series": series,
            "best benchmark": f"{best_benchmark} {row[best_benchmark]:.3f}",
            "best deep": f"{best_deep.replace('-core', '')} {row[best_deep]:.3f}",
            "gap": round(gap, 3),
            "seed sd": round(sd, 3) if np.isfinite(sd) else np.nan,
            "gap/sd": round(gap / sd, 1) if np.isfinite(sd) and sd > 0 else np.nan,
            "winner": "deep" if overall not in BENCHMARKS else "benchmark",
        })
    return pd.DataFrame(out).set_index("series")


def figure_skill(table: pd.DataFrame) -> None:
    model_columns = [c for c in table.columns if not c.endswith("__sd")]
    labels = [c.replace("-core", "") for c in model_columns]
    x = np.arange(len(table))
    width = 0.8 / len(model_columns)

    fig, ax = plt.subplots(figsize=(14, 6))
    palette = ["#9e9e9e", "#bdbdbd", "#1f4e79", "#2e7d32",
               "#8b0000", "#b36b00", "#5b2c6f", "#00695c"]
    for position, (column, label) in enumerate(zip(model_columns, labels)):
        ax.bar(x + position * width - 0.4 + width / 2, table[column], width,
               label=label, color=palette[position % len(palette)])
    ax.axhline(1.0, color="black", lw=1.2, ls="--")
    ax.text(len(table) - 0.45, 1.004, "naive forecast", fontsize=8, va="bottom",
            ha="right")
    ax.set_xticks(x, table.index, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("skill against naive  (RMSE / naive RMSE)")
    ax.set_ylim(0.6, 1.2)
    ax.set_title("Four-week-ahead forecast skill, US boxed beef - 169 test weeks, "
                 "2023-07 to 2026-10\nbars below the dashed line beat "
                 "'the price stays where it is'")
    ax.legend(fontsize=8, ncol=4, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "fig40_skill_by_model.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {(FIGURES / 'fig40_skill_by_model.png').relative_to(ROOT)}")


def figure_stability(dl: pd.DataFrame) -> None:
    """Seed spread per configuration - the cost of a bigger input space."""
    pivot = dl.pivot_table(index="config", values="skill_seed_sd", aggfunc="mean")
    pivot = pivot.sort_values("skill_seed_sd")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    colours = ["#8b0000" if "plus-mexico" in c else "#1f4e79" for c in pivot.index]
    ax.barh([c.replace("-core", "") for c in pivot.index],
            pivot["skill_seed_sd"], color=colours)
    for position, value in enumerate(pivot["skill_seed_sd"]):
        ax.text(value + 0.002, position, f"{value:.3f}", va="center", fontsize=9)
    ax.set_xlabel("mean standard deviation of skill across random seeds")
    ax.set_title("Run-to-run stability.\nAdding Mexican features roughly doubles the "
                 "spread without improving accuracy - more ways to fit noise.")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig41_seed_stability.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {(FIGURES / 'fig41_seed_stability.png').relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--horizon", type=int, default=4,
                        help="which horizon's results to compare")
    args = parser.parse_args()

    baselines, dl = load(args.horizon)
    table = build_table(baselines, dl)

    display = table[[c for c in table.columns if not c.endswith("__sd")]]
    display.columns = [c.replace("-core", "") for c in display.columns]

    print("=" * 100)
    print(f"SKILL AGAINST NAIVE - horizon {args.horizon} weeks")
    print("lower is better; 1.000 means no better than assuming no change")
    print("=" * 100)
    print(display.round(3).to_string())

    print("\n" + "=" * 100)
    print("BEST MODEL PER SERIES")
    print("gap = best benchmark minus best deep model; gap/sd compares it with the")
    print("seed spread, so a gap of less than about 2 sd is not evidence")
    print("=" * 100)
    summary = winners(table)
    print(summary.to_string())
    deep_wins = int((summary["winner"] == "deep").sum())
    print(f"\ndeep learning best on {deep_wins} of {len(summary)} series")

    print("\n" + "=" * 100)
    print("THE THREE COMPARISONS")
    print("=" * 100)

    def mean_sd(config: str) -> tuple[float, float]:
        block = dl[dl["config"] == config]
        return block["skill_ensemble"].mean(), block["skill_seed_sd"].mean()

    # Each comparison names which side represents "the answer is yes", because the
    # better model is not always the one that says yes: in comparison 2 the winner is
    # the model that does NOT share across cuts.
    pairs = [
        ("1. Does nonlinearity help?",
         "best benchmark", "best deep",
         float(table[BENCHMARKS].min(axis=1).mean()),
         float(table[[c for c in table.columns
                      if not c.endswith('__sd') and c not in BENCHMARKS]].min(axis=1).mean()),
         "right",
         "yes - the best deep model beats the best benchmark on average",
         "no - the benchmarks hold on average"),
        ("2. Does sharing across cuts help?",
         "lstm-joint (shares)", "lstm-single (does not)",
         mean_sd("lstm-joint-core")[0], mean_sd("lstm-single-core")[0],
         "left",
         "yes - one joint network beats seven separate ones",
         "no - sharing HURTS; separate networks are better"),
        ("3. Does Mexican information help?",
         "lstm-joint-core", "lstm-joint-plus-mexico",
         mean_sd("lstm-joint-core")[0], mean_sd("lstm-joint-plus-mexico")[0],
         "right",
         "yes - Mexican features improve the US forecast",
         "no - Mexican features do not improve the US forecast"),
    ]
    for question, left_name, right_name, left, right, yes_side, yes, no in pairs:
        better = "right" if right < left else "left"
        print(f"\n{question}")
        print(f"   {left_name:24s} mean skill {left:.3f}")
        print(f"   {right_name:24s} mean skill {right:.3f}   ({right - left:+.3f})")
        print(f"   -> {yes if better == yes_side else no}")

    joint_sd = mean_sd("lstm-joint-core")[1]
    mexico_sd = mean_sd("lstm-joint-plus-mexico")[1]
    print(f"\n   and on stability: seed sd {joint_sd:.3f} -> {mexico_sd:.3f} "
          f"({mexico_sd / joint_sd:.1f}x)")

    REPORTS.mkdir(parents=True, exist_ok=True)
    out_name = tagged("final_comparison.csv", args.horizon)
    table.to_csv(REPORTS / out_name)
    print(f"\nWritten {(REPORTS / out_name).relative_to(ROOT)}")
    figure_skill(table)
    figure_stability(dl)


if __name__ == "__main__":
    main()
