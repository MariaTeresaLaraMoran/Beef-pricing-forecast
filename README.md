# Beef Price Forecast

Forecasting weekly wholesale beef prices in Mexico with deep learning (LSTM/GRU),
using US beef prices, the USD/MXN exchange rate, and US beef exports to Mexico.
The deep learning model is compared against a linear regression baseline.

Deep learning course project. Presentation: December 1, 2026.

## Problem

Beef buyers and distributors in Mexico decide every week how much to purchase and
when. Mexican beef prices depend on the domestic market and also on the US market,
because Mexico imports US beef and pays for it in dollars. This project predicts the
Mexican wholesale beef price four weeks ahead.

- **Type:** supervised learning, time-series regression
- **Target:** weekly wholesale beef price in Mexico, four weeks ahead
- **Baseline:** linear regression on lagged values
- **Deep learning model:** LSTM / GRU on the same inputs

## Data

All data is collected directly from the original public sources with the scripts in `src/`.

| Series | Source | Frequency | Script | Status |
|---|---|---|---|---|
| USD/MXN FIX exchange rate | Banco de México, SIE API (series SF43718) | Daily | `src/fetch_banxico.py` | Script written |
| US boxed beef cutout (Choice, Select) | USDA AMS Market News, report LM_XB403 | Daily | `src/fetch_usda_boxed_beef.py` | Script written |
| US beef exports to Mexico | USDA FAS Export Sales | Weekly | pending | Pending |
| Mexican wholesale beef prices (target) | SNIIM, Secretaría de Economía | Daily | pending | Pending |

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
```

The Banxico token is free: https://www.banxico.org.mx/SieAPIRest/service/v1/token
