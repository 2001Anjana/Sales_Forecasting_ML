# CLAUDE.md

Context for AI coding assistants (Claude Code, Cursor, etc.) working in this repo.

## What this project is
Forecast weekly `Sales Qty` 12 weeks ahead, per product and per product×country,
from transaction-level cinnamon export data in `data/raw/Cinnamon_export_sales.xlsx`.

## Hard rules
- **Never modify files in `data/raw/`.** Treat the raw xlsx as read-only.
- A **product** = first 9 chars of `Product Code` (`config.PRODUCT_KEY_LEN`).
- Bucket demand into **Monday-anchored weeks** using **Order Date** (not Invoice Date).
- Put every path / threshold / constant in `src/config.py` — do not hard-code them
  in notebooks or other modules.
- Keep **loading** (`data_loader.py`) separate from **cleaning** (`preprocessing.py`).
  Step-1 profiling must reflect the data *as delivered*.

## Key data facts (from step 1)
- ~60.7k rows, 14 columns, Feb 2022 → Sep 2025 (~185 weeks).
- 13,725 products; **median 2 transactions/product**; ~6,000 with a single txn;
  ~170 products = 80% of volume. → modelling is **tier-based** (A/B/C).
- Data quality: 142 missing Order Dates; 140 negative `Sales Qty` (returns);
  38 zeros; an implausible max `Sales Qty` (~14M) to investigate; some rows have
  Invoice Date before Order Date.

## Conventions
- Reusable logic in `src/`; notebooks just orchestrate and display.
- Plots save to `outputs/figures/` via the `profiling.plot_*` / future helpers.
- Use parquet for `data/interim` and `data/processed`.
- Random seed: `config.RANDOM_SEED` (42).

## Status
- [x] Step 1 — data understanding (`notebooks/01_data_understanding.ipynb`, `run_step1.py`)
- [x] Step 2 — preprocessing & cleaning (`notebooks/02_preprocessing_eda.ipynb`, `run_step2.py`)
      → produces weekly panels + `product_tiers.csv` (A=428, B=2519, C=10777)
- [x] Step 3 — feature engineering (`notebooks/03_feature_engineering.ipynb`, `run_step3.py`)
      → `feature_table.parquet` for Tier A+B (461,759 rows, 30 features, leakage-checked)
- [ ] Step 4 — model training (one model per tier; see plan)
- [ ] Step 5 — forecasting
