"""
config.py
=========
Central configuration for the Cinnamon Export Sales Forecasting project.

Every path, threshold, and magic number lives here so the notebooks and other
``src`` modules never hard-code values. Import it as::

    from src import config

or, when running from inside the ``notebooks`` folder::

    import sys; sys.path.append("..")
    from src import config
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# PROJECT_ROOT resolves to the repository root regardless of where code runs
# from (this file lives in <root>/src/config.py, so two parents up is root).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
INTERIM_DIR: Path = DATA_DIR / "interim"
PROCESSED_DIR: Path = DATA_DIR / "processed"

OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
FIGURES_DIR: Path = OUTPUTS_DIR / "figures"
FORECASTS_DIR: Path = OUTPUTS_DIR / "forecasts"
MODELS_DIR: Path = OUTPUTS_DIR / "models"

REPORT_DIR: Path = PROJECT_ROOT / "report"

# The single raw input file.
RAW_FILE: Path = RAW_DIR / "Cinnamon_export_sales.xlsx"

# --------------------------------------------------------------------------- #
# Column names (exactly as they appear in the raw spreadsheet)
# --------------------------------------------------------------------------- #
COL_REGION = "Region"
COL_COUNTRY = "Country"
COL_CUSTOMER_CODE = "Customer Code"
COL_CUSTOMER_ID = "Customer ID"
COL_BRAND = "Brand Category"
COL_RANGE = "Product Range"
COL_CHANNEL = "Sales Channel"
COL_PRODUCT_CODE = "Product Code"
COL_ORDER_DATE = "Order Date"
COL_INVOICE_DATE = "Invoice Date"
COL_INVOICE_NO = "Invoice No"
COL_SALES_USD = "Sales USD"
COL_SALES_QTY = "Sales Qty"   # <-- target variable
COL_SALES_KG = "Sales KG"

DATE_COLS = [COL_ORDER_DATE, COL_INVOICE_DATE]
NUMERIC_COLS = [COL_SALES_USD, COL_SALES_QTY, COL_SALES_KG]

# Derived columns we will create.
COL_PRODUCT9 = "Product9"        # first 9 chars of Product Code  -> product key
COL_WEEK = "week_start"          # Monday-anchored week start date

# The date used to bucket demand into weeks. Order Date reflects when the
# customer placed demand, which is what we forecast (Invoice Date lags it).
DEMAND_DATE_COL = COL_ORDER_DATE

# Number of characters that define a "product".
PRODUCT_KEY_LEN = 9

# --------------------------------------------------------------------------- #
# Forecasting / modeling constants (used in later steps, kept here for one
# source of truth)
# --------------------------------------------------------------------------- #
FORECAST_HORIZON_WEEKS = 12      # forecast 12 weeks (3 months) ahead
WEEK_ANCHOR = "W-MON"            # weeks start on Monday

# --------------------------------------------------------------------------- #
# Cleaning thresholds (step 2a)
# --------------------------------------------------------------------------- #
# Physically-impossible quantities are treated as data-entry errors and removed.
# The legitimate maximum in this data is ~13.5k units / ~20k kg (bulk orders);
# the single error row was ~14.1M units / ~17.6M kg. 100k sits safely between.
MAX_PLAUSIBLE_QTY = 100_000
MAX_PLAUSIBLE_KG = 100_000

# Returns / credit notes appear as negative Sales Qty. Policy:
#   "keep"  -> keep them so weekly NET demand can dip (default; realistic)
#   "drop"  -> remove them entirely
RETURNS_POLICY = "keep"

# Label used to fill missing categorical fields.
UNKNOWN_LABEL = "UNKNOWN"

# --------------------------------------------------------------------------- #
# Product-tier thresholds (step 2c) -- calibrated on the real distribution.
#   Tier A: regular enough to model as a time series   (active in >= N weeks)
#   Tier C: sparse / one-off                           (<= M lifetime txns)
#   Tier B: everything in between (lumpy / intermittent)
# Result on this data: A=430, B=2517, C=10777.
# --------------------------------------------------------------------------- #
TIER_A_MIN_ACTIVE_WEEKS = 20
TIER_C_MAX_TXNS = 5

# Syntetos-Boylan demand-classification thresholds (used for the intermittency
# label that informs which model each product gets).
ADI_THRESHOLD = 1.32     # average demand interval
CV2_THRESHOLD = 0.49     # squared coeff. of variation of non-zero demand sizes

# Reproducibility.
RANDOM_SEED = 42


def ensure_dirs() -> None:
    """Create every output/data directory if it does not already exist."""
    for d in [
        INTERIM_DIR,
        PROCESSED_DIR,
        FIGURES_DIR,
        FORECASTS_DIR,
        MODELS_DIR,
        REPORT_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)
