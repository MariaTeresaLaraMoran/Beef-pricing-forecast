# Beef Price Forecast

Forecasting weekly wholesale prices of individual beef cuts in Mexico with deep
learning (LSTM/GRU), using the carcass price, the USD/MXN exchange rate, US beef
exports to Mexico, and US cut prices. The deep learning models are compared against
naive, ARIMA and linear-regression baselines.

Deep learning course project. Presentation: December 1, 2026.

## Problem

Beef buyers and distributors in Mexico decide every week which cuts to purchase and
when. Cut prices depend on the domestic carcass market upstream of them, and on the US
market, because Mexico imports US beef and pays for it in dollars. Published work on
Mexican beef finds that international prices pass through to domestic prices
*asymmetrically* — falling prices transmit faster than rising ones — but models that
asymmetry with linear, monthly methods. This project forecasts several Mexican beef cut
prices four weeks ahead, at weekly frequency, with models that can represent that
nonlinearity directly.

### Scope

Two designs were considered. The one built here is **single-stage**: one multivariate
model that takes observed, publication-lagged US series as inputs. A **two-stage cascade**
-- forecast US cut prices first, then feed those forecasts into a Mexican model -- was
deliberately deferred. Stage 2 would inherit Stage 1's forecast error, and because Mexican
prices adjust toward international prices only gradually (3-4% per month in the published
estimates), today's *observed* US price still carries information about Mexican prices four
weeks out. A cascade is only necessary for horizons longer than that transmission lag, and
it is recorded here as future work.

The question the cascade was meant to answer -- does US market information improve Mexican
price forecasts? -- is answered more directly by an ablation: train the same model with and
without the US inputs, and compare.

### Setup

- **Type:** supervised learning, multi-output time-series regression
- **Target:** weekly prices of six Mexican beef cuts, four weeks ahead: Sirloin,
  Costillar (ribeye), Chuleton Diezmillo (chuck), Arrachera (outside skirt),
  Pulpa Negra (top round) and Pecho (brisket) -- one cut per primal
- **Modelling range:** January 2001 to November 2026, about 1,300 weeks
- **Transformation:** log prices, first-differenced (weekly log returns)
- **Baselines:** naive (last value), ARIMA, linear regression on lagged features
- **Deep learning models:** LSTM and GRU, one network with six outputs

Three comparisons are planned, each answerable from the same pipeline:

1. **Does nonlinearity help?** LSTM/GRU against ARIMA on the same single series.
2. **Does sharing across cuts help?** One joint six-output network against six
   independent single-cut models.
3. **Does US market information help?** The same model trained with and without the
   US inputs.

## Data

All data is collected directly from the original public sources with the scripts in `src/`.

| Series | Source | Frequency | Script | Status |
|---|---|---|---|---|
| USD/MXN FIX exchange rate | Banco de México, SIE API (series SF43718) | Daily | `src/fetch_banxico.py` | Script written |
| US boxed beef cutout (Choice, Select) | USDA AMS MyMarketNews, MARS API, report 2453 / LM_XB403 (needs a free MyMarketNews key) | Daily | `src/fetch_usda_boxed_beef.py` | Script written |
| US beef exports to Mexico | USDA FAS Open Data, ESR API (needs a free api.data.gov key) | Weekly | `src/fetch_usda_fas_exports.py` | Script written |
| Mexican beef **cut** prices (target) | SNIIM, Secretaría de Economía (web pages, no API) | Daily | `src/fetch_sniim_cuts.py` | Script written |
| Mexican beef carcass prices (predictor) | SNIIM, Secretaría de Economía (web pages, no API) | Daily | `src/fetch_sniim_beef.py` | Downloaded, 2000-2026 |

Raw files are saved to `data/raw/` exactly as downloaded. The merged weekly table is
saved to `data/processed/`.

## Repository structure

```
├── data/
│   ├── raw/          # files exactly as downloaded
│   └── processed/    # merged weekly table
├── src/              # one collection script per source
├── notebooks/        # EDA, baseline, deep learning model
└── reports/          # figures and slides
```

## How to reproduce

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # then paste your Banxico token into .env

python src/fetch_banxico.py
python src/fetch_usda_boxed_beef.py      # see the note below if this times out
python src/fetch_sniim_beef.py
python src/fetch_sniim_cuts.py
python src/fetch_usda_fas_exports.py
```

Both keys are free:

- Banxico token: https://www.banxico.org.mx/SieAPIRest/service/v1/token
- USDA FAS (api.data.gov) key: https://api.data.gov/signup/
- USDA MyMarketNews key: register at https://mymarketnews.ams.usda.gov , then "Show API key" in your account


## If the USDA download times out

The USDA LMR Datamart is intermittently unavailable. `fetch_usda_boxed_beef.py` caches
each date window it retrieves and skips cached windows on re-run, so repeated passes
accumulate progress:

```bash
bash retry_usda.sh          # retries every 10 minutes until the cache is complete
```

Progress can be checked with `find data/raw/usda_cache -name '*.csv' | wc -l`;
216 files means both sections are complete.

## A note on data sources

Every series in this project is collected from its original public source by a script in
`src/`, so the whole dataset can be rebuilt by anyone with the repository. Licensed
third-party re-exports of the same data (commercial price-intelligence platforms) are
deliberately not used, because they cannot be redistributed and a reader could not
reproduce them.
