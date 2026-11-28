# Beef Price Forecast

Forecasting weekly **US boxed beef prices** four and twelve weeks ahead with deep learning
(LSTM and GRU), benchmarked against naive, ridge regression and ARIMA. The feature set
adds the **gross packer margin** — what packers sell beef for minus what they pay for
cattle — and controlled ablations test whether extra information helps at all.

Every result is reported in two variants: on the full sample, and with the documented
2020–21 market disruption excluded from training. The difference between them turned out
to be the most informative thing in the project.

## Problem

Beef buyers decide every week what to purchase and when, against a wholesale price that
moves on supply conditions they cannot observe directly. The usual model input is price
history. This project adds the economics on the other side of the packer: the gross
margin between the boxed beef cutout (what the packer sells) and the dressed fed cattle
price (what it buys), on a common carcass-weight basis. A wide margin pulls cattle
forward into slaughter and puts more beef on the market weeks later, so the margin leads
the price in a way the price's own history does not.

**Why two horizons.** The margin works through a slow physical channel: cattle placed on
feed respond to a wide margin, but they reach slaughter weight months later. A forecast
four weeks out mostly captures momentum that a random walk already has. The correlation
between today's margin and the forward return deepens with distance — −0.301 at four
weeks, −0.353 at thirteen — and that measurement was taken before any model was fitted.
Both horizons are run so the prediction can be tested rather than assumed.

The project began as a forecast of **Mexican** cut prices using US prices as predictors.
Both targets were built and benchmarked under the same procedure; the US series proved
substantially more forecastable, and the target moved. The Mexican sources were later
removed entirely — the ablation had shown they added nothing — though the Mexican
price-transmission literature remains in `reports/literature_review.md`, since it is what
establishes the asymmetry that justified dropping them.

### Setup

- **Type:** supervised learning, time-series regression, one model per series
- **Target:** seven US series — the Choice and Select boxed beef cutouts, plus five cuts
  by IMPS code: ribeye lip-on bone-in (109E), chuck roll (116A), brisket deckle-off
  (120), top inside round (168), top sirloin butt (184)
- **Modelling range:** April 2001 to November 2026; 1,185 usable weeks at h=4 and 1,178
  at h=12, from 2004-01-18 onward
- **Transformation:** log prices, first-differenced (weekly log returns)
- **Target definition:** log return over the next h weeks, predicted **directly** from one
  model, not by iterating a one-step model h times
- **Split:** time-ordered train / validation / test with an embargo of one horizon length
  at each boundary. Every row's target looks h weeks forward, so without the gap the last
  h training rows would carry targets covering the first h test weeks — the model would
  have been fitted on the outcome of weeks it is then scored on. The embargo drops those
  rows. A longer horizon needs a longer embargo, which is why the test set shrinks:
  **172 test weeks at h=4, 163 at h=12.** Row counts are printed by every script and
  never hard-coded
- **Metric:** RMSE on the log return over the horizon, plus MAPE on the recovered price
  and Diebold–Mariano tests against naive
- **Benchmarks:** naive (random walk), ridge regression on 32 lagged features,
  ARIMA(p,1,q) with AIC order selection
- **Deep learning:** Keras LSTM and GRU, one recurrent layer, 32 units, dropout 0.2,
  26-week input window, early stopping, **5 random seeds per configuration**. Tables
  report the 5-seed ensemble; the spread across seeds is the unit against which every
  claimed gain is measured

## Results

RMSE on the twelve-week log return, 163 test weeks, disruption excluded from training.
Lower is better. Compare methods **across a row**; each series has its own volatility, so
RMSE is not comparable between rows.

| Series | naive | ridge | arima | gru | lstm | lstm +macro |
|---|---|---|---|---|---|---|
| Choice cutout | 0.06027 | 0.05512 | 0.06291 | 0.06336 | **0.05228** | 0.05415 |
| Select cutout | 0.07155 | 0.06139 | 0.07398 | 0.06143 | 0.05939 | **0.05855** |
| Ribeye (109E) | 0.21243 | 0.18255 | 0.16997 | 0.15590 | **0.15002** | 0.15398 |
| Chuck roll (116A) | 0.18770 | 0.13799 | 0.15667 | 0.12198 | **0.11961** | 0.12223 |
| Brisket (120) | 0.10764 | 0.10374 | **0.09765** | 0.11005 | 0.10875 | 0.10282 |
| Top inside round (168) | 0.15402 | 0.14265 | **0.12675** | 0.13267 | 0.14280 | 0.14532 |
| Top sirloin butt (184) | 0.17449 | 0.13304 | 0.16975 | 0.13407 | 0.12024 | **0.11936** |

A deep model is best on **five of seven series**, ARIMA on two. But the count matters less
than the margin, measured against the spread across the five random seeds:

| Series | best benchmark | best deep | gain | gain/sd |
|---|---|---|---|---|
| Top sirloin butt (184) | ridge 0.13304 | lstm+macro 0.11936 | +0.01369 | **4.2** |
| Chuck roll (116A) | ridge 0.13799 | lstm 0.11961 | +0.01838 | 2.0 |
| Ribeye (109E) | arima 0.16997 | lstm 0.15002 | +0.01995 | 1.8 |
| Select cutout | ridge 0.06139 | lstm+macro 0.05855 | +0.00284 | 1.2 |
| Choice cutout | ridge 0.05512 | lstm 0.05228 | +0.00284 | 0.5 |
| Top inside round (168) | arima 0.12675 | gru 0.13267 | −0.00592 | −0.7 |
| Brisket (120) | arima 0.09765 | lstm+macro 0.10282 | −0.00517 | −1.2 |

A gain smaller than its own seed spread is not evidence, however favourable it looks.
**Brisket and top inside round go to ARIMA in all four variants tested** — a robust
negative result, not a bad draw.

### Is the added complexity justified?

Only under stated conditions, and the conditions matter more than the answer.

| | h=4 full | h=4 excl. | h=12 full | h=12 excl. |
|---|---|---|---|---|
| Deep learning best on | 5/7 | 5/7 | **3/7** | 5/7 |
| Mean improvement over naive | 13.6% | 16.8% | 16.7% | **21.0%** |
| Series clearing 2 seed sd | 2 | **3** | **0** | 1 |
| Largest gain/sd | **5.8** | 4.2 | 2.0 | 4.2 |

Two findings, pulling in opposite directions:

- **The size of the advantage rises with the horizon**, as the margin correlations
  predicted before any model was fitted: 13.6% → 16.7% on the full sample, 16.8% → 21.0%
  with the disruption excluded.
- **Its reliability falls.** At four weeks two series clear two standard deviations of
  seed spread; at twelve weeks, on the full sample, none do. The networks become noisier
  faster than they become better.

A longer horizon offers more to gain and less certainty about having gained it. Reporting
only the first half would overstate the case considerably.

### The 2020–21 disruption

This is the project's main finding.

USDA FAS export data had gaps over the COVID aftermath. Because an exports feature was
required by the model, **17 extreme weeks were silently dropped from training** — margins
of 43.6 to 61.7 USD/cwt against a 2004–2019 median of 6.34, with `margin_choice_log_change`
reaching |z| = 8.9. When that feature was removed for being untested, those weeks entered
training, and:

| | change in RMSE |
|---|---|
| naive, ridge, ARIMA | under 1% |
| LSTM and GRU | **15% to 37% worse** |

The benchmarks were scored on the same weeks and barely moved, so this is a training
effect, not a scoring one — and it is a statement about **how the methods differ in their
sensitivity to extreme observations**, not a general claim about the pandemic.

The exclusion window, 2020-03-01 to 2021-06-30, is declared as a fixed calendar constant
in `baselines.py` rather than chosen from the data, and is applied to the benchmarks too.
The argument for it is that the **forecast period is an ordinary-margin regime** — test
period mean 8.7 USD/cwt against a 2004–2019 norm of 8.4 — while the excluded weeks run an
order of magnitude above that, produced by a processing-capacity constraint that no longer
exists. Training on them calibrates the margin–price relationship to conditions that do
not occur in the period being forecast. `reports/margin_by_year.csv` and
`fig27_margin_regimes.png` show the full arc: ordinary through 2015, rising to 2019,
10.2× the norm in 2021, and back below it by 2026.

Note what this does *not* claim. The margin–return relationship does not break during the
disruption; measured inside the window it is *stronger* (−0.753 against −0.120 in
2004–2019), because one enormous excursion dominates the correlation. The case for
exclusion rests on influence, not on the mechanism failing.

### The comparisons

**1. Does nonlinearity help?** On five of seven series at twelve weeks with the disruption
excluded, and on two of those by more than two standard deviations. Ridge sees the same
features as the networks, so the difference is attributable to nonlinearity and memory
rather than to information.

**2. Does sharing across series help?** **No — it hurts.** One joint seven-output network
scored meaningfully worse than seven independent single-series networks. Cuts correlate
around 0.8 in levels but only 0.08–0.30 in returns, which is the likely reason. Joint runs
were dropped from the pipeline; `--include-joint` still reads them from archived results.

**3. Does extra information help?** Mean RMSE across the seven series, by arm:

| Arm | h=4 full | h=4 excl. | h=12 full | h=12 excl. |
|---|---|---|---|---|
| `core` | 0.07316 | 0.07444 | 0.11364 | 0.10758 |
| `plus-macro` | 0.07546 | **0.06984** | 0.11423 | 0.10806 |
| `plus-volume` | 0.07374 | 0.07356 | — | — |

- **Macro is not a reliable improvement.** The dollar index and jobless claims help in one
  variant of four and are neutral-to-harmful in the other three. An earlier version of this
  project reported macro helping at both horizons; that result did not survive removing the
  exports feature. Per series it still wins — it is the best arm on the Select cutout and
  the top sirloin butt — but the aggregate claim does not hold.
- **Trading volume does nothing.** Negotiated head count is a noisy series (log sd 0.42)
  and, measured directly, nearly orthogonal to the forward return (correlation +0.005). It
  describes *how* cattle were purchased — negotiated versus formula — not how many exist.
- **Exports to Mexico were removed.** `ret_exports_tonnes` sat in the core feature set from
  the project's Mexican framing and was never tested the way other candidates were. Its mean
  correlation with the forward return is 0.016 at h=12 — weaker than the head count that was
  explicitly rejected, against 0.228 for the margin.

**Architecture.** No clear winner. LSTM is ahead on mean RMSE in three of four variants,
GRU in one, and the two trade places across series. The gating difference is not decisive
at this sample size.

Full tables in `reports/final_comparison*.csv`; figures in `reports/figures/`.

## Data

Every series is collected from its original public source by a script in `src/`.

| Series | Source | Script |
|---|---|---|
| US boxed beef cutout and cut prices | USDA AMS LMR Datamart, report 2453 (no key) | `fetch_usda_boxed_beef.py` |
| US fed cattle, 5-Area weighted average (LM_CT150) | USDA AMS LMR Datamart, report 2477 (no key) | `fetch_usda_cattle.py` |
| Negotiated head counts, 5-Area | USDA AMS, same report | `fetch_usda_cattle.py` |
| Broad dollar index, initial jobless claims | FRED (no key) | `fetch_fred.py` |

Raw files are saved to `data/raw/` exactly as downloaded.
`data/processed/weekly_model_table.csv` is **not** committed — it is derived and
regenerates in about thirty seconds with `python src/build_dataset.py`. Its schema is
documented in `data/processed/weekly_model_table_columns.txt`, which is committed.

## Repository structure

```
├── data/
│   ├── raw/          # files exactly as downloaded, plus AMS methodology notes
│   ├── reference/    # US cut map with IMPS codes and evidence for each pairing
│   └── processed/    # merged weekly table (gitignored) and its column documentation
├── src/
│   ├── fetch_*.py          # one collection script per source
│   ├── build_dataset.py    # merge, publication lags, margins, features, targets
│   ├── eda_cuts.py         # statistics, stationarity, margin lead, coverage, regimes
│   ├── baselines.py        # naive, ridge, ARIMA + Diebold-Mariano
│   ├── models.py           # Keras LSTM/GRU, multi-seed, feature ablations
│   ├── compare.py          # consolidated RMSE comparison and figures
│   └── forecast_plots.py   # price-scale forecasts with prediction bands
└── reports/
    ├── figures/            # fig20-27 EDA, fig40-41 comparison, fig50-52 forecasts
    ├── literature_review.md
    ├── paper_summaries.md
    ├── margin_by_year.csv
    └── *.csv               # results and predictions
```

Output files carry an `_h12` suffix for the twelve-week horizon and a `_nobreak` suffix
for the disruption-excluded variant, so all four combinations sit beside each other
instead of overwriting.

## How to reproduce

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1. collect
python src/fetch_usda_boxed_beef.py      # see the note below if this times out
python src/fetch_usda_cattle.py
python src/fetch_fred.py

# 2. build both horizons into one table, and explore
python src/build_dataset.py --horizon 4 12
python src/eda_cuts.py

# 3. benchmarks, then the networks - full sample
python src/baselines.py
python src/baselines.py --horizon 12
python src/models.py --features core
python src/models.py --features core --arch gru
python src/models.py --features plus-macro
python src/models.py --features plus-volume
python src/models.py --horizon 12 --features core
python src/models.py --horizon 12 --features core --arch gru
python src/models.py --horizon 12 --features plus-macro

# 4. the same again, with the 2020-21 disruption out of training
python src/baselines.py --exclude-break
python src/baselines.py --horizon 12 --exclude-break
python src/models.py --exclude-break --features core
python src/models.py --exclude-break --features core --arch gru
python src/models.py --exclude-break --features plus-macro
python src/models.py --exclude-break --features plus-volume
python src/models.py --horizon 12 --exclude-break --features core
python src/models.py --horizon 12 --exclude-break --features core --arch gru
python src/models.py --horizon 12 --exclude-break --features plus-macro

# 5. compare and plot, one call per variant
python src/compare.py
python src/compare.py --horizon 12
python src/compare.py --exclude-break
python src/compare.py --horizon 12 --exclude-break
python src/forecast_plots.py --horizon 12
python src/forecast_plots.py --horizon 12 --exclude-break
```

Allow a q to 2  hours for steps 3 and 4: five seeds times seven series per arm, plus
ARIMA order selection in step 3.

`baselines.py` merges rather than replaces: re-running a subset of models keeps the rows
for the models left out, so a changed feature set does not force an ARIMA re-fit unless
the split actually moves.

## Known issues and caveats

**Absolute accuracy falls with the horizon.** Mean RMSE rises from roughly 0.073 at four
weeks to 0.108 at twelve. A twelve-week return is a larger quantity to miss. What improves
at the longer horizon is standing *relative to the benchmarks*, not absolute error.

**Seed spread is not forecast uncertainty.** It measures how much the answer moves when
the random initialisation changes. The prediction bands in `fig50` and `fig52` come from
the model's own past errors and are considerably wider; both are drawn, labelled
separately.

**ARIMA residuals are not formally checked.** Orders are selected by AIC and stationarity
is tested with ADF and KPSS, but no Ljung-Box test is run on the residuals.

**The USDA boxed beef download times out intermittently.** The LMR Datamart is sometimes
unavailable. `fetch_usda_boxed_beef.py` caches each date window and skips cached ones on
re-run, so repeated passes accumulate progress:

```bash
bash retry_usda.sh          # retries every 10 minutes until the cache is complete
```

Progress: `find data/raw/usda_cache -name '*.csv' | wc -l`; 216 files means complete.

**FRED sometimes refuses scripted requests.** If `fetch_fred.py` times out, download the
two series by hand and save them as `data/raw/fred_DTWEXBGS.csv` and
`data/raw/fred_ICSA.csv`; the script reads those before trying the network. Use the
direct CSV URL rather than the Download button, which defaults to the last five years:
`https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTWEXBGS`

**The 5-Area fed cattle series changes composition.** AMS suppresses a region when too
few packers report: Colorado excluded since December 2022, Texas/Oklahoma/New Mexico
since June 2026, Kansas since August 2026, all five restored on 26 October 2026. Part of
that falls inside the test period. The AMS narratives are kept in
`data/raw/usda_5area_cattle_notes.txt`.

**Head counts are negotiated volume, not total slaughter.** The negotiated share of US fed
cattle trade has fallen over the period as formula and contract purchasing grew, so the
level carries a structural downward drift. The code uses the return, never the level.

## A note on data sources

Every series is collected from its original public source, so the dataset can be rebuilt
by anyone with the repository. Licensed third-party re-exports of the same data
(commercial price-intelligence platforms) are deliberately not used, because they cannot
be redistributed and a reader could not reproduce them.

