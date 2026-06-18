# Cinnamon Export Sales Forecasting

Forecast **weekly Sales Qty for the next 12 weeks**, (a) per product and
(b) per product × country, from ~3.5 years of transaction-level export data
(Feb 2022 – Sep 2025, ~60.7k transactions).

A **product** is the first 9 characters of `Product Code`. Demand is bucketed
into **Monday-anchored weeks** using `Order Date`.

## Project layout

```
cinnamon_sales_forecasting/
├── data/
│   ├── raw/         Cinnamon_export_sales.xlsx        (read-only input)
│   ├── interim/     cleaned_transactions.parquet      (step 2 output)
│   └── processed/   weekly_product_sales.csv, weekly_product_country_sales.csv,
│                    product_tiers.csv                 (step 2/3 output)
├── notebooks/       01_data_understanding … 05_forecasting
├── src/             config, data_loader, profiling, preprocessing, features,
│                    train, forecast, evaluate
├── outputs/         figures/  forecasts/  models/
├── report/          final report + presentation
├── requirements.txt
└── README.md
```

## Setup

```bash
# 1. (recommended) create a virtual environment
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 2. install dependencies
pip install -r requirements.txt

# 3. confirm the raw data is in place
#    data/raw/Cinnamon_export_sales.xlsx
```

## Step 1 — Data understanding  ✅ (this delivery)

Understand the data before changing it. Run:

```bash
cd notebooks
jupyter notebook 01_data_understanding.ipynb     # or: jupyter lab
```

Or run it headless:

```bash
jupyter nbconvert --to notebook --execute --inplace \
    notebooks/01_data_understanding.ipynb
```

It produces, in `outputs/figures/`:
* `01_product_pareto.png` – cumulative product-volume curve (justifies tiering)
* `01_txns_per_product.png` – long-tail histogram of transactions per product
* `01_weekly_total_qty.png` – company-wide weekly demand (trend & seasonality)

### What step 1 found
* Target `Sales Qty` is highly skewed; contains returns (negatives), zeros, and
  an implausible giant outlier to investigate in cleaning.
* **Extreme product concentration:** ~170 products = 80% of volume; ~6,000
  products have a single lifetime transaction → modelling must be **tier-based**.
* Minor missingness across columns; 142 missing `Order Date`s; some rows have
  `Invoice Date` before `Order Date`.

The modules used by the notebook live in `src/`:
* `config.py` – all paths, column names, thresholds, the 12-week horizon.
* `data_loader.py` – load + light type coercion (no cleaning).
* `profiling.py` – missing/cardinality/numeric reports, Pareto & sparsity, plots.

## Roadmap (later steps)
2. Preprocessing & EDA – clean, aggregate to weekly, reindex zero-weeks, build tiers.
3. Feature engineering – calendar, lags, rolling stats, intermittency features.
4. Model training – baselines + global LightGBM (+ SARIMA/Prophet on Tier A).
5. Forecasting – recursive 12-week forecasts, write output CSVs.
6. Report & presentation.

## Full pipeline (run in order)

```bash
python run_step1.py   # data understanding
python run_step2.py   # cleaning + weekly panels + tiers
python run_step3.py   # feature engineering
python run_step4.py   # model training + per-tier evaluation
python run_step5.py   # final 12-week forecasts (deliverables)
```

Models by tier (chosen in step 4): Tier A & B → global LightGBM (Tweedie,
recursive, volume-bias-corrected); Tier C → moving-average baseline.

Final deliverables: `outputs/forecasts/product_12_week_forecast.csv` and
`outputs/forecasts/product_country_12_week_forecast.csv`.
