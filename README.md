# Beef Price Forecast

Forecasting weekly **US boxed beef prices twelve weeks ahead** with deep learning
(LSTM and GRU), benchmarked against naive, ridge regression and ARIMA. The feature set
adds the **gross packer margin** — what packers sell beef for minus what they pay for
cattle — and controlled ablations test whether extra information helps at all.

Every stage also runs at a **four-week horizon**, and the two are reported side by side.
The horizon is not a detail of configuration here; it turns out to be the main finding.

## Problem

Beef buyers decide every week what to purchase and when, against a wholesale price that
moves on supply conditions they cannot observe directly. The usual model input is price
history. This project adds the economics on the other side of the packer: the gross
margin between the boxed beef cutout (what the packer sells) and the dressed fed cattle
price (what it buys), on a common carcass-weight basis. A wide margin pulls cattle
forward into slaughter and puts more beef on the market weeks later, so the margin leads
the price in a way the price's own history does not.

**Why twelve weeks.** The margin works through a slow physical channel: cattle placed on
feed respond to a wide margin, but they reach slaughter weight months later. A forecast
four weeks out mostly captures momentum that a random walk already has. Twelve weeks is
far enough for the adjustment to show up and still inside the window a buyer can act on
with forward contracts. The project was built at four weeks first and extended to twelve;
the comparison below shows the method's advantage roughly doubles at the longer horizon,
which is what a slow-adjustment mechanism predicts and what the correlations in the raw
data already indicated (margin against the forward return: −0.301 at 4 weeks, −0.353 at
13 weeks).

The project began as a forecast of **Mexican** cut prices using US prices as predictors.
Both targets were built and benchmarked under the same procedure; the US series proved
substantially more forecastable, and the target moved. The Mexican data became an
ablation. See `reports/literature_review.md` section 7.

### Setup

- **Type:** supervised learning, time-series regression, one model per series
- **Target:** seven US series, **twelve weeks ahead** (four weeks also reported) — the
  Choice and Select boxed beef cutouts, plus five cuts by IMPS code: ribeye lip-on
  bone-in (109E), chuck roll (116A), brisket deckle-off (120), top inside round (168),
  top sirloin butt (184)
- **Modelling range:** January 2004 to October 2026, 1,170 complete weeks
- **Transformation:** log prices, first-differenced (weekly log returns)
- **Target definition:** log return over the next 12 weeks, predicted **directly** from
  one model, not by iterating a one-step model twelve times
- **Split:** time-ordered train / validation / test with an embargo of one horizon length
  at each boundary, so no window straddles a split. The embargo grows with the horizon:
  **160 test weeks at h=12** (2023-08-06 to 2026-08-23), 169 at h=4 (2023-07-30 to
  2026-10-18). Row counts are printed by every script and never hard-coded
- **Metric:** RMSE on the log return over the horizon, plus MAPE on the recovered price
  and Diebold–Mariano tests against naive
- **Benchmarks:** naive (random walk), ridge regression on 32 lagged features,
  ARIMA(p,1,q) with AIC order selection
- **Deep learning:** Keras LSTM and GRU, one recurrent layer, 32 units, dropout 0.2,
  26-week input window, early stopping, **5 random seeds per configuration**. Tables
  report the 5-seed ensemble; the spread across seeds is reported alongside and is the
  unit against which every claimed gain is measured

## Results

RMSE on the **twelve-week** log return, 160 common test weeks. Lower is better. Compare
methods **across a row**; each series has its own volatility, so RMSE is not comparable
between rows.

| Series | naive | ridge | arima | gru | lstm | lstm +macro |
|---|---|---|---|---|---|---|
| Choice cutout | 0.06053 | 0.05536 | 0.06412 | 0.05973 | 0.06292 | **0.04593** |
| Select cutout | 0.07213 | 0.05977 | 0.07439 | 0.05791 | 0.06453 | **0.05568** |
| Ribeye (109E) | 0.21187 | 0.19260 | 0.16962 | 0.15928 | 0.15876 | **0.15822** |
| Chuck roll (116A) | 0.18857 | 0.14132 | 0.15613 | 0.11128 | 0.12313 | **0.10818** |
| Brisket (120) | 0.10785 | 0.10296 | 0.09844 | 0.10398 | **0.09271** | 0.10332 |
| Top inside round (168) | 0.15427 | 0.14023 | **0.12509** | 0.13187 | 0.13305 | 0.13261 |
| Top sirloin butt (184) | 0.17379 | 0.12900 | 0.16951 | 0.12912 | **0.12530** | 0.12582 |

A deep model is best on **six of seven series**, ARIMA on one. But the count matters less
than the margin, measured against the spread across the five random seeds:

| Series | best benchmark | best deep | gain | seed sd | gain/sd |
|---|---|---|---|---|---|
| Chuck roll (116A) | ridge 0.14132 | lstm+macro 0.10818 | +0.03314 | 0.00506 | **6.5** |
| Choice cutout | ridge 0.05536 | lstm+macro 0.04593 | +0.00943 | 0.00501 | 1.9 |
| Select cutout | ridge 0.05977 | lstm+macro 0.05568 | +0.00409 | 0.00317 | 1.3 |
| Brisket (120) | arima 0.09844 | lstm 0.09271 | +0.00574 | 0.00582 | 1.0 |
| Ribeye (109E) | arima 0.16962 | lstm+macro 0.15822 | +0.01140 | 0.02143 | 0.5 |
| Top sirloin butt (184) | ridge 0.12900 | lstm 0.12530 | +0.00369 | 0.00832 | 0.4 |
| Top inside round (168) | arima 0.12509 | gru 0.13187 | −0.00678 | 0.00827 | −0.8 |

`gain/sd` compares each gain with the standard deviation across the five seeds. A gain
smaller than its own seed spread is not evidence, however favourable it looks.

**The chuck roll is the result.** A 23% reduction against the best benchmark at **6.5
standard deviations** — the model lands in the same place from every random
initialisation. The two cutouts clear a weaker but real bar. The ribeye's gain is larger
in absolute terms than the Select cutout's and yet less trustworthy, because that network
is unstable across seeds (sd 0.02143); reporting the gain without the spread would invert
the ranking. The top inside round is the one honest loss: ARIMA wins and should be used
there.

### Does the horizon matter? Yes — it is the main finding

Identical pipeline, identical features, identical seeds; only the forecast distance
changes.

| | h=4 | h=12 |
|---|---|---|
| Deep learning best on | 5 of 7 series | **6 of 7** |
| Mean improvement over naive | 14.6% | **24.5%** |
| Mean improvement over best benchmark per series | 4.1% | **8.2%** |
| Series clearing 2 seed sd | 1 (ribeye, 2.3) | 1 (chuck roll, **6.5**) |
| Best chuck roll gain | 16.1% at 1.3 sd | **23.5% at 6.5 sd** |
| Mean directional accuracy, best arm | ~63% | **71.3%** |

Improvement over the naive forecast, best deep model per series:

| Series | h=4 | h=12 |
|---|---|---|
| Chuck roll (116A) | 26.3% | **42.6%** |
| Top sirloin butt (184) | 19.7% | 27.9% |
| Ribeye (109E) | 19.8% | 25.3% |
| Choice cutout | 14.7% | 24.1% |
| Select cutout | 9.8% | 22.8% |
| Brisket (120) | 9.2% | 14.0% |
| Top inside round (168) | 2.5% | 14.5% |

**Every series improves at the longer horizon**, and the ratio roughly doubles on both
measures. This is the opposite of the usual pattern — forecasts normally degrade with
distance, and in absolute terms these do too (mean RMSE rises from 0.073 to 0.104,
because a twelve-week return is simply a bigger number to miss). What improves is the
advantage *over the benchmarks*. The naive forecast degrades faster than the network
does.

That is the signature of a slow-adjusting mechanism. Four weeks out, most of what is
knowable is momentum, and a random walk already encodes it. Twelve weeks out, the
margin's effect on placements and slaughter has had time to reach the market, and the
network can read it while a random walk cannot. The mechanism was visible in the raw
correlations before any model was fitted (margin against the forward return: −0.301 at 4
weeks, −0.353 at 13 weeks), so the horizon result confirms a prediction rather than
discovering one after the fact.

**Both horizons are kept.** The four-week results are not a discarded draft; they are the
control that makes the twelve-week claim interpretable. A single horizon would have shown
"deep learning wins by a little" without revealing why.

### The comparisons

**1. Does nonlinearity help?** At twelve weeks, yes, and substantially: a mean 8.2%
reduction against the best benchmark chosen per series, concentrated in the chuck and rib
primals — the cuts where supply is most inelastic and all adjustment runs through price.
At four weeks the same comparison gives 4.1%, which is within noise on most series.

**2. Does sharing across series help?** **No — it hurts.** One joint seven-output network
scored meaningfully worse than seven independent single-series networks. Joint runs were
dropped from the pipeline; `--include-joint` still reads them from the archived result
files.

**3. Does extra information help?** Same architecture, one thing changed at a time.

At **h=12**:

| Arm | mean RMSE | change | seed sd ratio |
|---|---|---|---|
| `plus-macro` (dollar index, jobless claims) | **0.10425** | −0.00438 | **0.71×** |
| `core` | 0.10863 | — | 1.00× |

At **h=4**:

| Arm | mean RMSE | change | seed sd ratio |
|---|---|---|---|
| `plus-macro` | **0.07282** | −0.00133 | 1.48× |
| `core` | 0.07414 | — | 1.00× |
| `plus-volume` (negotiated head counts) | 0.07433 | +0.00019 | 0.95× |
| `plus-mexico` (Mexican prices and spreads) | no change | 0.00000 | 2.12× |

- **Macro helps, and helps more at twelve weeks.** A 4.0% reduction at h=12 against 1.8%
  at h=4, and the sign of the stability cost flips: at four weeks the gain came with 48%
  *more* run-to-run variance, at twelve weeks with 29% **less**. Macro conditions act on
  the same slow channel as the margin, so a longer horizon gives them room to matter.
  The gain is achieved *despite* training on 648 windows instead of 733, since the broad
  dollar index only starts in 2006.
- **Trading volume does nothing.** Negotiated head count is a noisy series (log sd 0.42)
  and, measured directly, nearly orthogonal to the forward return (correlation +0.005).
  It describes *how* cattle were purchased — negotiated versus formula — not how many
  exist. It was dropped after h=4; the `plus-volume` arm remains runnable.
- **Mexican data does not help US forecasts**, and doubles seed instability. Price
  transmission runs US → Mexico and not back, which is what the monthly VECM literature
  reports; this is that asymmetry measured at weekly frequency in a neural network.

**Architecture.** LSTM beats GRU on five of seven series at h=12 and is the better arm
once macro features are added, but GRU is the more stable learner in isolation (mean seed
sd 0.00570 against 0.01197 for core LSTM) and wins the one series deep learning loses
overall. The gating difference is not decisive at this sample size; LSTM is reported as
primary because the best configuration on six of seven series is an LSTM.

Full tables in `reports/final_comparison.csv` and `reports/final_comparison_h12.csv`;
figures in `reports/figures/`. `fig52_best_per_series_h12.png` shows only the winning
method per series, on the price scale, with prediction bands.

## Data

Every series is collected from its original public source by a script in `src/`.

| Series | Source | Script |
|---|---|---|
| US boxed beef cutout and cut prices | USDA AMS LMR Datamart, report 2453 (no key) | `fetch_usda_boxed_beef.py` |
| US fed cattle, 5-Area weighted average (LM_CT150) | USDA AMS LMR Datamart, report 2477 (no key) | `fetch_usda_cattle.py` |
| US beef exports to Mexico | USDA FAS Open Data, ESR API (free key) | `fetch_usda_fas_exports.py` |
| Broad dollar index, initial jobless claims | FRED (no key) | `fetch_fred.py` |
| USD/MXN FIX exchange rate | Banco de México SIE API (free token) | `fetch_banxico.py` |
| Mexican beef cut prices | SNIIM, Secretaría de Economía (scraped) | `fetch_sniim_cuts.py` |
| Mexican beef carcass prices | SNIIM, Secretaría de Economía (scraped) | `fetch_sniim_beef.py` |

Raw files are saved to `data/raw/` exactly as downloaded.
`data/processed/weekly_model_table.csv` is **not** committed — it is derived, 5.5 MB, and
regenerates in about 30 seconds with `python src/build_dataset.py`. Its schema is
documented in `data/processed/weekly_model_table_columns.txt`, which is committed.

## Repository structure

```
├── data/
│   ├── raw/          # files exactly as downloaded, plus AMS methodology notes
│   ├── reference/    # SNIIM-to-IMPS cut mappings, with evidence for each pairing
│   └── processed/    # merged weekly table (gitignored) and its column documentation
├── src/
│   ├── fetch_*.py          # one collection script per source
│   ├── build_dataset.py    # merge, publication lags, features, targets
│   ├── eda_cuts.py         # statistics, stationarity tests, EDA figures
│   ├── baselines.py        # naive, ridge, ARIMA + Diebold-Mariano
│   ├── models.py           # Keras LSTM/GRU, multi-seed, feature ablations
│   ├── compare.py          # consolidated RMSE comparison and figures
│   └── forecast_plots.py   # price-scale forecasts with prediction bands
└── reports/
    ├── figures/            # fig10-32 EDA, fig40-41 comparison,
    │                       # fig50-52 forecasts; _h12 suffix for the 12-week run
    ├── literature_review.md
    ├── paper_summaries.md
    └── *.csv               # results and predictions
```

## How to reproduce

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # then paste your Banxico token and FAS key into .env

# 1. collect
python src/fetch_banxico.py
python src/fetch_usda_boxed_beef.py      # see the note below if this times out
python src/fetch_usda_cattle.py
python src/fetch_usda_fas_exports.py
python src/fetch_fred.py
python src/fetch_sniim_beef.py
python src/fetch_sniim_cuts.py

# 2. build both horizons into one table, and explore
python src/build_dataset.py --horizon 4 12
python src/eda_cuts.py

# 3. benchmark, then model - primary horizon, 12 weeks
python src/baselines.py --horizon 12
python src/models.py  --horizon 12 --features core
python src/models.py  --horizon 12 --features core --arch gru
python src/models.py  --horizon 12 --features plus-macro

# 4. compare and plot the 12-week results
python src/compare.py        --horizon 12
python src/forecast_plots.py --horizon 12
```

Allow a few hours for step 3: five seeds times seven series per arm, plus ARIMA order
selection in step 2.

Both API keys are free, and FRED and the USDA LMR Datamart need none:

- Banxico token: https://www.banxico.org.mx/SieAPIRest/service/v1/token
- USDA FAS (api.data.gov) key: https://api.data.gov/signup/

### The four-week comparison horizon

Every stage takes `--horizon`, and the four-week run is what the twelve-week result is
measured against. It is also where the `plus-volume` and `plus-mexico` ablations were
settled:

```bash
python src/baselines.py
python src/models.py --features core
python src/models.py --features core --arch gru
python src/models.py --features plus-macro
python src/models.py --features plus-volume
python src/compare.py
python src/forecast_plots.py
```

Horizon 4 is the default and writes unsuffixed filenames; every other horizon gets an
`_h<N>` suffix, so the two runs sit beside each other instead of overwriting. The row
counts differ by design — the embargo at each split boundary is one horizon long, which
leaves 169 test weeks at h=4 and 160 at h=12 — so no script hard-codes a week count.

## Known issues and caveats

**The USDA boxed beef download times out intermittently.** The LMR Datamart is sometimes
unavailable. `fetch_usda_boxed_beef.py` caches each date window and skips cached ones on
re-run, so repeated passes accumulate progress:

```bash
bash retry_usda.sh          # retries every 10 minutes until the cache is complete
```

Progress: `find data/raw/usda_cache -name '*.csv' | wc -l`; 216 files means complete.
`fetch_usda_cattle.py` needs none of this — one request returns the whole history.

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

**Spring 2020 is a genuine structural break.** Plant closures drove the packer margin to
+281 USD/cwt against a median of 8.91 — the widest since Mandatory Price Reporting began,
and the subject of a formal USDA AMS investigation. It sits inside the training period.

**SNIIM is hand-entered** and contains occasional impossible values. Records beyond five
times the median for their own cut in their own year are dropped: 284 of 364,549 (0.078%).

## A note on data sources

Every series is collected from its original public source, so the dataset can be rebuilt
by anyone with the repository. Licensed third-party re-exports of the same data
(commercial price-intelligence platforms) are deliberately not used, because they cannot
be redistributed and a reader could not reproduce them.

Cut names are reconciled against the USMEF US–Mexico nomenclature standard. Each pairing
in `data/reference/us_cut_map.csv` records the IMPS code, the match quality and the
evidence — including two cases where an initially plausible mapping was tested against
the data and rejected.
