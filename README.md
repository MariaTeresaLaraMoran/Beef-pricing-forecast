# Beef Price Forecast

Forecasting weekly **US boxed beef prices** four weeks ahead with deep learning
(LSTM and GRU), benchmarked against naive, ridge regression and ARIMA. The feature set
adds the **gross packer margin** — what packers sell beef for minus what they pay for
cattle — and three ablations test whether extra information helps at all.

## Problem

Beef buyers decide every week what to purchase and when, against a wholesale price that
moves on supply conditions they cannot observe directly. The usual model input is price
history. This project adds the economics on the other side of the packer: the gross
margin between the boxed beef cutout (what the packer sells) and the dressed fed cattle
price (what it buys), on a common carcass-weight basis. A wide margin pulls cattle
forward into slaughter and puts more beef on the market weeks later, so the margin leads
the price in a way the price's own history does not.

The project began as a forecast of **Mexican** cut prices using US prices as predictors.
Both targets were built and benchmarked under the same procedure; the US series proved
substantially more forecastable, and the target moved. The Mexican data became an
ablation. See `reports/literature_review.md` section 7.

### Setup

- **Type:** supervised learning, time-series regression, one model per series
- **Target:** seven US series, four weeks ahead — the Choice and Select boxed beef
  cutouts, plus five cuts by IMPS code: ribeye lip-on bone-in (109E), chuck roll (116A),
  brisket deckle-off (120), top inside round (168), top sirloin butt (184)
- **Modelling range:** January 2004 to October 2026, 1,170 complete weeks
- **Transformation:** log prices, first-differenced (weekly log returns)
- **Target definition:** log return over the next 4 weeks, predicted **directly**, not by
  iterating a one-step model four times
- **Split:** time-ordered train / validation / test with a 4-week embargo at each
  boundary. Train 733, validation 158, test 169 weeks (2023-07-30 to 2026-10-18)
- **Metric:** RMSE on the four-week log return, plus MAPE on the recovered price and
  Diebold–Mariano tests against naive
- **Benchmarks:** naive (random walk), ridge regression on 32 lagged features,
  ARIMA(p,1,q) with AIC order selection
- **Deep learning:** Keras LSTM and GRU, one recurrent layer, 32 units, dropout 0.2,
  26-week input window, early stopping, **5 random seeds per configuration**

## Results

RMSE on the four-week log return, 169 common test weeks. Lower is better. Compare
methods **across a row**; each series has its own volatility, so RMSE is not comparable
between rows.

| Series | naive | ridge | arima | gru | lstm | lstm +macro | lstm +volume |
|---|---|---|---|---|---|---|---|
| Choice cutout | 0.04369 | 0.03833 | 0.04560 | 0.04229 | 0.04166 | **0.03724** | 0.04231 |
| Select cutout | 0.04609 | **0.04014** | 0.04773 | 0.04157 | 0.04456 | 0.04413 | 0.04437 |
| Ribeye (109E) | 0.13450 | 0.12824 | 0.12533 | 0.10790 | **0.10788** | 0.10908 | 0.10838 |
| Chuck roll (116A) | 0.13830 | 0.12377 | 0.12147 | 0.11071 | 0.10505 | **0.10193** | 0.10531 |
| Brisket (120) | 0.06904 | 0.07258 | 0.06688 | 0.07411 | 0.06823 | **0.06272** | 0.06742 |
| Top inside round (168) | 0.09309 | 0.09004 | **0.08489** | 0.09112 | 0.09101 | 0.09074 | 0.09155 |
| Top sirloin butt (184) | 0.07545 | 0.06071 | 0.07299 | 0.06359 | **0.06062** | 0.06390 | 0.06099 |

**No single method wins everywhere.** A deep model is best on five of seven series,
ARIMA on one, ridge on one. But the margin of victory matters more than the count:

| Series | best benchmark | best deep | gain | seed sd | gain/sd |
|---|---|---|---|---|---|
| Ribeye (109E) | arima 0.12533 | lstm 0.10788 | +0.01745 | 0.00772 | **2.3** |
| Chuck roll (116A) | arima 0.12147 | lstm+macro 0.10193 | +0.01953 | 0.01507 | 1.3 |
| Brisket (120) | arima 0.06688 | lstm+macro 0.06272 | +0.00416 | 0.00453 | 0.9 |
| Choice cutout | ridge 0.03833 | lstm+macro 0.03724 | +0.00109 | 0.00358 | 0.3 |
| Top sirloin butt (184) | ridge 0.06071 | lstm 0.06062 | +0.00009 | 0.00171 | 0.1 |
| Select cutout | ridge 0.04014 | gru 0.04157 | −0.00143 | 0.00596 | −0.2 |
| Top inside round (168) | arima 0.08489 | lstm+macro 0.09074 | −0.00585 | 0.01202 | −0.5 |

`gain/sd` compares each gain with the standard deviation across the five random seeds.
Only the ribeye clears two standard deviations. Top sirloin butt is a tie decided in the
fifth decimal place. So the defensible claim is **one substantial, reliable improvement
and several suggestive ones**, not "deep learning forecasts beef prices better."

### The comparisons

**1. Does nonlinearity help?** Marginally, and unevenly. Mean RMSE across the seven
series is 0.07282 for the best deep configuration against roughly 0.079 for the best
benchmark per series — but concentrated in the rib and chuck primals, the scarce
high-value cuts where supply is most inelastic and all adjustment runs through price.

**2. Does sharing across series help?** **No — it hurts.** One joint seven-output network
scored meaningfully worse than seven independent single-series networks. Joint runs are
retained in the result files but excluded from the comparison; `--include-joint` shows
them.

**3. Does extra information help?** Three ablations, same architecture, one thing changed:

| Arm | mean RMSE | change | seed sd ratio |
|---|---|---|---|
| `plus-macro` (dollar index, jobless claims) | **0.07282** | −0.00133 | 1.48× |
| `core` | 0.07414 | — | 1.00× |
| `plus-volume` (negotiated head counts) | 0.07433 | +0.00019 | 0.95× |
| `plus-mexico` (Mexican prices and spreads) | no change | 0.00000 | 2.12× |

- **Macro helps a little** — a 1.8% average reduction, achieved *despite* training on 648
  windows instead of 733, since the dollar index only starts in 2006. The gain sits in
  the smoother series (Choice cutout, brisket) and costs 48% more run-to-run variance.
- **Trading volume does nothing.** Negotiated head count is a noisy measure (log sd 0.42)
  of a response the packer margin already captures as an incentive.
- **Mexican data does not help US forecasts**, and doubles seed instability. Price
  transmission runs US → Mexico and not back, which is what the monthly VECM literature
  reports; this is that asymmetry measured at weekly frequency in a neural network.

**Architecture:** LSTM beat GRU on five of seven series and was also more stable across
seeds (0.00610 against 0.00738).

Full tables in `reports/final_comparison.csv`; figures in `reports/figures/`.

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
    ├── figures/            # fig10-32 EDA, fig40-41 comparison, fig50-51 forecasts
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

# 2. build and explore
python src/build_dataset.py
python src/eda_cuts.py

# 3. benchmark, then model
python src/baselines.py
python src/models.py --features core
python src/models.py --arch gru --features core
python src/models.py --features plus-volume
python src/models.py --features plus-macro

# 4. compare
python src/compare.py
python src/forecast_plots.py
```

Both API keys are free, and FRED and the USDA LMR Datamart need none:

- Banxico token: https://www.banxico.org.mx/SieAPIRest/service/v1/token
- USDA FAS (api.data.gov) key: https://api.data.gov/signup/

### A second horizon

All stages accept `--horizon`. Building both at once keeps one table:

```bash
python src/build_dataset.py --horizon 4 12
python src/baselines.py --horizon 12
python src/models.py --horizon 12
python src/compare.py --horizon 12
```

Results for any horizon other than 4 are written with an `_h<N>` suffix, so they sit
beside the four-week results rather than overwriting them.

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
