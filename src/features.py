"""
features.py
===========
STEP 3 — feature engineering. Turns the weekly series (from step 2) into a
supervised-learning table for the products we will model with ML / a global
model (Tiers A and B). Tier C is left to simple baselines and needs no features.

Feature families
----------------
* calendar     : week-of-year, month, quarter, year, week-of-month + sin/cos
* seasonal     : quarter-end flag, peak-month flag (starting point; tune later)
* lag          : Sales Qty at t-1, t-2, t-4, t-8, t-12, t-52
* rolling      : mean / std / median over the trailing 4, 8, 13 weeks
* intermittency: weeks since last sale, trailing zero-fraction, trailing mean
                 non-zero demand (the signals that matter for sparse series)
* static       : product attributes (Brand Category, Product Range, Sales
                 Channel, dominant Region) shared across a product's rows

LEAKAGE RULE (critical)
-----------------------
Every autoregressive feature at week *t* uses only weeks <= t-1. Lags use a
plain grouped ``shift``; rolling features ``shift(1)`` *before* rolling so the
current week is never included in its own predictors. Calendar/seasonal
features are known in advance and carry no leakage.

The same functions accept a ``group_keys`` argument, so they can later build the
product x country table by passing ``[COL_PRODUCT9, COL_COUNTRY]``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

LAGS = [1, 2, 4, 8, 12, 52]
ROLL_WINDOWS = [4, 8, 13]
PEAK_MONTHS = {11, 12}  # provisional festive-season demand months; tune in EDA


# --------------------------------------------------------------------------- #
# Loading the step-2 outputs
# --------------------------------------------------------------------------- #
def load_weekly_product() -> pd.DataFrame:
    df = pd.read_csv(config.PROCESSED_DIR / "weekly_product_sales.csv",
                     parse_dates=[config.COL_WEEK])
    return df


def load_tiers() -> pd.DataFrame:
    return pd.read_csv(config.PROCESSED_DIR / "product_tiers.csv")


def load_cleaned() -> pd.DataFrame:
    """Load cleaned transactions, trying parquet then csv (whichever exists)."""
    pq = config.INTERIM_DIR / "cleaned_transactions.parquet"
    csv = config.INTERIM_DIR / "cleaned_transactions.csv"
    if pq.exists():
        try:
            return pd.read_parquet(pq)
        except (ImportError, ValueError):
            pass
    return pd.read_csv(csv, parse_dates=config.DATE_COLS + [config.COL_WEEK])


# --------------------------------------------------------------------------- #
# Static product attributes
# --------------------------------------------------------------------------- #
def product_static_attributes(clean: pd.DataFrame) -> pd.DataFrame:
    """Most-common Brand/Range/Channel/Region per product (mode)."""
    cols = [config.COL_BRAND, config.COL_RANGE, config.COL_CHANNEL, config.COL_REGION]

    def _mode(s: pd.Series):
        m = s.mode()
        return m.iloc[0] if len(m) else config.UNKNOWN_LABEL

    attrs = (
        clean.groupby(config.COL_PRODUCT9)[cols].agg(_mode).reset_index()
    )
    return attrs


# --------------------------------------------------------------------------- #
# Feature builders (all leakage-safe)
# --------------------------------------------------------------------------- #
def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    wk = df[config.COL_WEEK]
    iso = wk.dt.isocalendar()
    df["woy"] = iso.week.astype(int)
    df["month"] = wk.dt.month
    df["quarter"] = wk.dt.quarter
    df["year"] = wk.dt.year
    df["week_of_month"] = (wk.dt.day - 1) // 7 + 1
    # cyclical encodings (52-week year, 12-month year)
    df["woy_sin"] = np.sin(2 * np.pi * df["woy"] / 52)
    df["woy_cos"] = np.cos(2 * np.pi * df["woy"] / 52)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    # simple seasonal flags
    df["is_quarter_end"] = wk.dt.is_quarter_end.astype(int)
    df["is_peak_month"] = df["month"].isin(PEAK_MONTHS).astype(int)
    return df


def add_lag_features(df: pd.DataFrame, group_keys, target: str,
                     lags=LAGS) -> pd.DataFrame:
    df = df.copy()
    g = df.groupby(group_keys)[target]
    for L in lags:
        df[f"lag_{L}"] = g.shift(L)
    return df


def add_rolling_features(df: pd.DataFrame, group_keys, target: str,
                         windows=ROLL_WINDOWS) -> pd.DataFrame:
    df = df.copy()
    # shift(1) first so the current week is excluded from its own window
    shifted = df.groupby(group_keys)[target].shift(1)
    for w in windows:
        roll = shifted.groupby([df[k] for k in group_keys]).rolling(w, min_periods=1)
        df[f"roll_mean_{w}"] = roll.mean().reset_index(level=list(range(len(group_keys))), drop=True)
        df[f"roll_std_{w}"] = roll.std().reset_index(level=list(range(len(group_keys))), drop=True)
        df[f"roll_med_{w}"] = roll.median().reset_index(level=list(range(len(group_keys))), drop=True)
    return df


def add_intermittency_features(df: pd.DataFrame, group_keys, target: str,
                               window: int = 13) -> pd.DataFrame:
    """Weeks-since-last-sale, trailing zero-fraction, trailing mean non-zero."""
    df = df.copy()
    df = df.sort_values(group_keys + [config.COL_WEEK]).reset_index(drop=True)

    grp = df.groupby(group_keys)
    # row number within each product's series
    rn = grp.cumcount()
    has_sale = (df[target] > 0)

    # row number of sales only; shift by 1 so we look strictly into the past,
    # then forward-fill within group to get "row of most recent past sale".
    sale_rn = rn.where(has_sale)
    past_sale_rn = sale_rn.groupby([df[k] for k in group_keys]).shift(1)
    past_sale_rn = past_sale_rn.groupby([df[k] for k in group_keys]).ffill()
    weeks_since = rn - past_sale_rn
    # if no prior sale yet, count weeks since the series started
    df["weeks_since_last_sale"] = weeks_since.fillna(rn + 1)

    # trailing zero-fraction over `window` (past only)
    is_zero = (df[target] == 0).astype(float)
    shifted_zero = is_zero.groupby([df[k] for k in group_keys]).shift(1)
    df[f"zero_frac_{window}"] = (
        shifted_zero.groupby([df[k] for k in group_keys])
        .rolling(window, min_periods=1).mean()
        .reset_index(level=list(range(len(group_keys))), drop=True)
    )

    # trailing mean of non-zero demand (past only): sum / count of non-zeros
    val = df[target].where(df[target] > 0, np.nan)
    shifted_val = val.groupby([df[k] for k in group_keys]).shift(1)
    df[f"mean_nonzero_{window}"] = (
        shifted_val.groupby([df[k] for k in group_keys])
        .rolling(window, min_periods=1).mean()
        .reset_index(level=list(range(len(group_keys))), drop=True)
    ).fillna(0.0)

    df["weeks_since_start"] = rn
    return df


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def build_feature_table(tiers_to_include=("A", "B")) -> pd.DataFrame:
    """Build the full supervised feature table for the chosen tiers.

    Returns one row per (product, week) with the target ``Sales Qty`` and all
    feature columns, restricted to products in ``tiers_to_include``.
    """
    target = config.COL_SALES_QTY
    keys = [config.COL_PRODUCT9]

    weekly = load_weekly_product()
    tiers = load_tiers()
    clean = load_cleaned()

    keep = tiers.loc[tiers["tier"].isin(tiers_to_include), config.COL_PRODUCT9]
    df = weekly[weekly[config.COL_PRODUCT9].isin(keep)].copy()
    df = df.sort_values(keys + [config.COL_WEEK]).reset_index(drop=True)

    # features
    df = add_calendar_features(df)
    df = add_lag_features(df, keys, target)
    df = add_rolling_features(df, keys, target)
    df = add_intermittency_features(df, keys, target)

    # attach tier / demand class and static product attributes
    df = df.merge(
        tiers[[config.COL_PRODUCT9, "tier", "demand_class"]],
        on=config.COL_PRODUCT9, how="left",
    )
    df = df.merge(product_static_attributes(clean), on=config.COL_PRODUCT9, how="left")

    return df


def save_feature_table(df: pd.DataFrame) -> "object":
    """Save the feature table to parquet (csv fallback). Returns the path."""
    config.ensure_dirs()
    p = config.PROCESSED_DIR / "feature_table.parquet"
    try:
        df.to_parquet(p, index=False)
    except (ImportError, ValueError):
        p = config.PROCESSED_DIR / "feature_table.csv"
        df.to_csv(p, index=False)
    return p


def feature_columns(df: pd.DataFrame) -> list[str]:
    """The model-input columns (everything except keys, target, raw date)."""
    exclude = {config.COL_PRODUCT9, config.COL_WEEK, config.COL_SALES_QTY,
               config.COL_SALES_USD, config.COL_SALES_KG}
    return [c for c in df.columns if c not in exclude]
