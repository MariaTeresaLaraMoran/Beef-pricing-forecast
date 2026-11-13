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

- **Type:** supervised learning, multi-output time-series regression
- **Target:** weekly prices of 4-6 Mexican beef cuts, four weeks ahead
- **Transformation:** log prices, first-differenced (weekly log returns)
- **Baselines:** naive (last value), ARIMA, linear regression on lagged features
- **Deep learning models:** LSTM and GRU on the same inputs

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
python src/fetch_usda_boxed_beef.py
python src/fetch_sniim_beef.py
python src/fetch_sniim_cuts.py
python src/fetch_usda_fas_exports.py
```

Both keys are free:

- Banxico token: https://www.banxico.org.mx/SieAPIRest/service/v1/token
- USDA FAS (api.data.gov) key: https://api.data.gov/signup/
- USDA MyMarketNews key: register at https://mymarketnews.ams.usda.gov , then "Show API key" in your account


## A note on data sources

Every series in this project is collected from its original public source by a script in
`src/`, so the whole dataset can be rebuilt by anyone with the repository. Licensed
third-party re-exports of the same data (commercial price-intelligence platforms) are
deliberately not used, because they cannot be redistributed and a reader could not
reproduce them.
