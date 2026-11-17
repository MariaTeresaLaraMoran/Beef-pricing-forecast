# Beef Price Forecast

Forecasting weekly **US boxed beef prices**  four weeks ahead with deep learning
(LSTM/GRU), benchmarked against naive, drift, ridge regression and ARIMA. The feature set
adds the **gross packer margin** — what packers sell beef for minus what they pay for
cattle — and the project tests, as an ablation, whether Mexican market data improves a US
forecast.


## Problem

Beef buyers decide every week what to purchase and when, against a wholesale price that
moves on supply conditions they cannot observe directly. The usual inputs are price
history. This project adds the economics on the other side of the packer: the gross margin
between the boxed beef cutout (what the packer sells) and the dressed fed cattle price
(what it buys), on a common carcass-weight basis. A wide margin pulls cattle forward into
slaughter and puts more beef on the market weeks later, so the margin leads the price in a
way the price's own history does not.

The project began as a forecast of **Mexican** cut prices using US prices as predictors.
Both targets were built and benchmarked under the same procedure; the US series proved
substantially more forecastable, and the target moved. The Mexican data became the
ablation. See `reports/literature_review.md` section 7.

### Setup

- **Type:** supervised learning, multi-output time-series regression
- **Target:** seven US series, four weeks ahead — the Choice and Select boxed beef
  cutouts, plus five cuts by IMPS code: ribeye lip-on bone-in (109E), chuck roll (116A),
  brisket deckle-off (120), top inside round (168), top sirloin butt (184)
- **Modelling range:** January 2004 to October 2026, 1,170 complete weeks
- **Transformation:** log prices, first-differenced (weekly log returns)
- **Target definition:** log return over the next 4 weeks, predicted **directly**, not by
  iterating a one-step model
- **Split:** time-ordered train / validation / test with a 4-week embargo at each boundary
- **Benchmarks:** naive (random walk), drift (random walk with drift), ridge regression on
  32 lagged features, ARIMA(p,1,q) with AIC order selection
- **Deep learning:** Keras LSTM and GRU, one recurrent layer, 32 units, dropout 0.2,
  26-week input window, early stopping, **5 random seeds per configuration**

### Headline results

Skill against the naive forecast on 169 common test weeks (2023-07-30 to 2026-10-18).
Below 1.00 beats "the price in four weeks is the price today".

| Series | best benchmark | best deep model | winner |
|---|---|---|---|
| Ribeye (109E) | arima 0.932 | **lstm 0.802** | deep, 3.7 seed sd |
| Chuck roll (116A) | arima 0.878 | **lstm-single 0.760** | deep, 1.4 seed sd |
| Top sirloin butt (184) | ridge 0.805 | gru-joint 0.796 | tie |
| Choice cutout | **ridge 0.877** | gru-joint 0.903 | benchmark |
| Select cutout | **ridge 0.871** | gru-joint 0.881 | benchmark |
| Brisket (120) | **arima 0.969** | lstm-single 0.988 | benchmark |
| Top inside round (168) | **arima 0.912** | lstm-joint 0.973 | benchmark |

**No single model class wins across all seven series.** Deep learning gives a
substantial, significant improvement on the rib primal, a likely one on the chuck, and
does not beat well-specified traditional models on the remaining five.

The three comparisons:

1. **Does nonlinearity help?** Marginally — best deep 0.872 against best benchmark 0.892
   on average, concentrated in two series.
2. **Does sharing across series help?** **No, it hurts.** One joint seven-output network
   scores 0.981; seven separate networks score 0.893.
3. **Does Mexican information help?** **No.** Identical mean skill, and the seed standard
   deviation roughly doubles (0.043 → 0.091) — more parameters, no extra signal.

Full tables in `reports/final_comparison.csv`; figures in `reports/figures/`.

## Data

All data is collected directly from original public sources by the scripts in `src/`.

| Series | Source | Script |
|---|---|---|
| US boxed beef cutout and cut prices | USDA AMS LMR Datamart, report 2453 (no key) | `src/fetch_usda_boxed_beef.py` |
| US fed cattle, 5-Area weighted average (LM_CT150) | USDA AMS LMR Datamart, report 2477 (no key) | `src/fetch_usda_cattle.py` |
| US beef exports to Mexico | USDA FAS Open Data, ESR API (free key) | `src/fetch_usda_fas_exports.py` |
| USD/MXN FIX exchange rate | Banco de México SIE API (free token) | `src/fetch_banxico.py` |
| Mexican beef cut prices | SNIIM, Secretaría de Economía (scraped) | `src/fetch_sniim_cuts.py` |
| Mexican beef carcass prices | SNIIM, Secretaría de Economía (scraped) | `src/fetch_sniim_beef.py` |

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
│   ├── baselines.py        # naive, drift, ridge, ARIMA + Diebold-Mariano
│   ├── models.py           # Keras LSTM/GRU, multi-seed
│   ├── compare.py          # consolidated comparison table and figures
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
python src/fetch_sniim_beef.py
python src/fetch_sniim_cuts.py

# 2. build and explore
python src/build_dataset.py
python src/eda_cuts.py

# 3. benchmark, then model
python src/baselines.py
python src/models.py                     # LSTM, joint
python src/models.py --mode single
python src/models.py --arch gru
python src/models.py --features plus-mexico

# 4. compare
python src/compare.py
python src/forecast_plots.py
```

Both keys are free:

- Banxico token: https://www.banxico.org.mx/SieAPIRest/service/v1/token
- USDA FAS (api.data.gov) key: https://api.data.gov/signup/

The USDA LMR Datamart needs no key at all.

### A second horizon

All three stages accept `--horizon`. Building both at once keeps one table:

```bash
python src/build_dataset.py --horizon 4 12
python src/baselines.py --horizon 12
python src/models.py --horizon 12 --mode single
python src/compare.py --horizon 12
```

Results for any horizon other than 4 are written with an `_h<N>` suffix, so they sit
beside the four-week results rather than overwriting them.

## If the USDA boxed beef download times out

The LMR Datamart is intermittently unavailable. `fetch_usda_boxed_beef.py` caches each
date window it retrieves and skips cached windows on re-run, so repeated passes accumulate
progress:

```bash
bash retry_usda.sh          # retries every 10 minutes until the cache is complete
```

Progress: `find data/raw/usda_cache -name '*.csv' | wc -l`; 216 files means both sections
are complete. `fetch_usda_cattle.py` needs none of this — it returns the whole history in
one request.

## A note on data sources

Every series is collected from its original public source by a script in `src/`, so the
dataset can be rebuilt by anyone with the repository. Licensed third-party re-exports of
the same data (commercial price-intelligence platforms) are deliberately not used, because
they cannot be redistributed and a reader could not reproduce them.

Cut names are reconciled against the USMEF US–Mexico nomenclature standard. Each pairing
in `data/reference/us_cut_map.csv` records the IMPS code, the match quality and the
evidence — including two cases where an initially plausible mapping was tested against the
data and rejected.
