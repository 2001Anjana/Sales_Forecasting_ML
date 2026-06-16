"""
profiling.py
============
Pure data-understanding functions for STEP 1. Each function *measures* the data
and returns a tidy DataFrame (or numbers); plotting lives in separate ``plot_*``
helpers so the same stats can be reused in the report without re-plotting.

Nothing here mutates or cleans the input.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config


# --------------------------------------------------------------------------- #
# Column-level profiles
# --------------------------------------------------------------------------- #
def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column missing-value count and percentage, sorted worst-first."""
    n = len(df)
    miss = df.isna().sum()
    out = pd.DataFrame(
        {
            "missing": miss,
            "missing_pct": (miss / n * 100).round(3),
            "dtype": df.dtypes.astype(str),
        }
    )
    return out.sort_values("missing", ascending=False)


def cardinality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Number of unique (non-null) values per column, sorted high-first."""
    out = pd.DataFrame(
        {
            "n_unique": df.nunique(dropna=True),
            "dtype": df.dtypes.astype(str),
        }
    )
    return out.sort_values("n_unique", ascending=False)


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Describe numeric columns plus counts of negative and zero values.

    The negative / zero counts matter here: negatives are returns/credit notes
    and zeros are no-op lines, both of which the cleaning step must handle.
    """
    cols = [c for c in config.NUMERIC_COLS if c in df.columns]
    desc = df[cols].describe().T
    desc["n_negative"] = [(df[c] < 0).sum() for c in cols]
    desc["n_zero"] = [(df[c] == 0).sum() for c in cols]
    desc["n_missing"] = [df[c].isna().sum() for c in cols]
    return desc


def category_value_counts(df: pd.DataFrame, col: str, top: int = 20) -> pd.DataFrame:
    """Top value counts for a categorical column, with share of rows."""
    vc = df[col].value_counts(dropna=False).head(top)
    out = vc.rename("count").to_frame()
    out["pct"] = (out["count"] / len(df) * 100).round(2)
    return out


# --------------------------------------------------------------------------- #
# Product-volume (Pareto) analysis  --  the chart that justifies tiering
# --------------------------------------------------------------------------- #
def product_volume_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-product (first-9) lifetime stats used to motivate segmentation.

    Returns one row per product with transaction count, total quantity, and
    the cumulative share of total volume when products are ranked high->low.
    """
    g = df.groupby(config.COL_PRODUCT9)
    tbl = pd.DataFrame(
        {
            "n_txns": g.size(),
            "total_qty": g[config.COL_SALES_QTY].sum(min_count=1),
        }
    )
    tbl = tbl.sort_values("total_qty", ascending=False)
    total = tbl["total_qty"].sum()
    tbl["cum_qty_share"] = tbl["total_qty"].cumsum() / total
    tbl["volume_rank"] = np.arange(1, len(tbl) + 1)
    return tbl


def sparsity_headline(df: pd.DataFrame) -> dict:
    """Key sparsity numbers as a dict (handy for printing / the report)."""
    tbl = product_volume_table(df)
    n_products = len(tbl)
    txns = tbl["n_txns"]
    cum = tbl["cum_qty_share"]
    return {
        "n_transactions": int(len(df)),
        "n_products": int(n_products),
        "median_txns_per_product": float(txns.median()),
        "products_with_1_txn": int((txns == 1).sum()),
        "products_le_3_txns": int((txns <= 3).sum()),
        "products_le_10_txns": int((txns <= 10).sum()),
        "products_ge_50_txns": int((txns >= 50).sum()),
        "products_for_80pct_volume": int((cum <= 0.80).sum()) + 1,
        "products_for_95pct_volume": int((cum <= 0.95).sum()) + 1,
    }


# --------------------------------------------------------------------------- #
# Date consistency
# --------------------------------------------------------------------------- #
def date_consistency_report(df: pd.DataFrame) -> dict:
    """Compare Order Date and Invoice Date and flag inconsistencies."""
    o = df[config.COL_ORDER_DATE]
    i = df[config.COL_INVOICE_DATE]
    lag_days = (i - o).dt.days
    both = o.notna() & i.notna()
    return {
        "order_date_min": o.min(),
        "order_date_max": o.max(),
        "invoice_date_min": i.min(),
        "invoice_date_max": i.max(),
        "order_date_missing": int(o.isna().sum()),
        "invoice_date_missing": int(i.isna().sum()),
        "invoice_before_order": int((lag_days[both] < 0).sum()),
        "invoice_lag_days_median": float(lag_days[both].median()),
        "invoice_lag_days_mean": float(lag_days[both].mean().round(2)),
    }


# --------------------------------------------------------------------------- #
# Plots  (each saves to outputs/figures and returns the Axes)
# --------------------------------------------------------------------------- #
def _save(fig, name: str):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def plot_pareto(df: pd.DataFrame, name: str = "01_product_pareto.png"):
    """Cumulative-volume (Pareto) curve over products ranked high->low."""
    tbl = product_volume_table(df)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(tbl["volume_rank"], tbl["cum_qty_share"] * 100, color="#7a4a2b")
    ax.axhline(80, ls="--", color="grey", lw=1)
    ax.axhline(95, ls=":", color="grey", lw=1)
    n80 = (tbl["cum_qty_share"] <= 0.80).sum() + 1
    n95 = (tbl["cum_qty_share"] <= 0.95).sum() + 1
    ax.axvline(n80, ls="--", color="#c44", lw=1)
    ax.set_xlabel("Products ranked by volume (high \u2192 low)")
    ax.set_ylabel("Cumulative share of total Sales Qty (%)")
    ax.set_title(
        f"Product volume is extremely concentrated\n"
        f"{n80} products = 80% of volume, {n95} = 95% "
        f"(of {len(tbl):,} total)"
    )
    ax.set_xlim(0, len(tbl))
    ax.set_ylim(0, 100)
    fig.tight_layout()
    path = _save(fig, name)
    return fig, ax, path


def plot_txn_distribution(df: pd.DataFrame, name: str = "01_txns_per_product.png"):
    """Histogram (log y) of transactions per product -- the long tail."""
    tbl = product_volume_table(df)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(tbl["n_txns"], bins=range(1, int(tbl["n_txns"].max()) + 2),
            color="#7a4a2b", edgecolor="white", linewidth=0.3)
    ax.set_yscale("log")
    ax.set_xlabel("Lifetime transactions per product")
    ax.set_ylabel("Number of products (log scale)")
    ax.set_title("Most products have only a handful of transactions")
    fig.tight_layout()
    path = _save(fig, name)
    return fig, ax, path


def plot_weekly_total(df: pd.DataFrame, name: str = "01_weekly_total_qty.png"):
    """Total Sales Qty per week across the whole company -- trend & seasonality."""
    s = (
        df.dropna(subset=[config.DEMAND_DATE_COL, config.COL_SALES_QTY])
        .set_index(config.DEMAND_DATE_COL)[config.COL_SALES_QTY]
        .resample(config.WEEK_ANCHOR)
        .sum()
    )
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(s.index, s.values, color="#7a4a2b", lw=0.9)
    ax.set_xlabel("Week")
    ax.set_ylabel("Total Sales Qty")
    ax.set_title("Company-wide weekly Sales Qty (look for trend & seasonality)")
    fig.tight_layout()
    path = _save(fig, name)
    return fig, ax, path
