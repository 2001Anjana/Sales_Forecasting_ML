"""
preprocessing.py
================
STEP 2 — turn raw transactions into model-ready weekly series and a product-tier
table. Three parts, matching the implementation plan:

  2a. clean_transactions()        -- fix/flag missing values, returns, outliers
  2b. build_weekly_product()      -- weekly Sales Qty per product
      build_weekly_product_country()
  2c. build_product_tiers()       -- segment products into A / B / C + demand class

Design notes
------------
* Loading (light coercion) stays in ``data_loader``; this module does the actual
  cleaning, so step-1 profiling reflects the data as delivered.
* Weeks are **Monday-anchored**: a transaction's week is the Monday of the week
  it falls in. We use explicit Monday-subtraction (not pandas periods) so the
  anchor is unambiguous.
* Aggregated series are **reindexed to a complete weekly calendar** from each
  product's first sale to the global last week, with missing weeks filled with
  0. Those explicit zeros are the intermittent-demand signal models need.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def to_week_start(dates: pd.Series) -> pd.Series:
    """Map each date to the Monday of its week (normalised to midnight)."""
    weekday = dates.dt.weekday  # Monday = 0
    return (dates - pd.to_timedelta(weekday, unit="D")).dt.normalize()


# --------------------------------------------------------------------------- #
# 2a. Cleaning
# --------------------------------------------------------------------------- #
def clean_transactions(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean the (already type-coerced) transaction table.

    Returns the cleaned DataFrame and a ``log`` dict recording how many rows
    each decision affected (handy for the report — nothing happens silently).

    Decisions
    ---------
    1. Drop rows with a missing target (``Sales Qty``) — unusable.
    2. Remove physically-impossible quantities (data-entry errors) using the
       plausibility caps in ``config``.
    3. Drop rows with a missing ``Order Date`` — cannot be placed in a week.
    4. Returns (negative ``Sales Qty``): keep (net demand) or drop, per
       ``config.RETURNS_POLICY``; either way add an ``is_return`` flag.
    5. Fill missing categorical fields with ``UNKNOWN``.
    6. Add the Monday-anchored ``week_start``.
    """
    log: dict[str, int] = {"rows_in": len(df)}
    d = df.copy()

    # 1. missing target
    before = len(d)
    d = d.dropna(subset=[config.COL_SALES_QTY])
    log["dropped_missing_qty"] = before - len(d)

    # 2. impossible quantities (the ~14M unit / ~17M kg error row, etc.)
    qty = d[config.COL_SALES_QTY].abs()
    kg = d[config.COL_SALES_KG].abs()
    impossible = (qty > config.MAX_PLAUSIBLE_QTY) | (kg > config.MAX_PLAUSIBLE_KG)
    log["dropped_impossible_outliers"] = int(impossible.sum())
    d = d[~impossible]

    # 3. missing order date (can't bucket into a week)
    before = len(d)
    d = d.dropna(subset=[config.COL_ORDER_DATE])
    log["dropped_missing_order_date"] = before - len(d)

    # 4. returns / credit notes
    d = d.copy()
    d["is_return"] = d[config.COL_SALES_QTY] < 0
    log["returns_flagged"] = int(d["is_return"].sum())
    if config.RETURNS_POLICY == "drop":
        d = d[~d["is_return"]]
        log["returns_dropped"] = log["returns_flagged"]
    else:
        log["returns_dropped"] = 0

    # 5. fill missing categoricals
    cat_cols = [config.COL_REGION, config.COL_COUNTRY, config.COL_CHANNEL,
                config.COL_BRAND, config.COL_RANGE]
    filled = 0
    for c in cat_cols:
        if c in d.columns:
            filled += int(d[c].isna().sum())
            d[c] = d[c].fillna(config.UNKNOWN_LABEL)
            # normalise stray double-spaces seen in Brand Category etc.
            d[c] = d[c].astype("string").str.replace(r"\s+", " ", regex=True).str.strip()
    log["categorical_values_filled"] = filled

    # 6. week anchor
    d[config.COL_WEEK] = to_week_start(d[config.COL_ORDER_DATE])

    log["rows_out"] = len(d)
    return d.reset_index(drop=True), log


# --------------------------------------------------------------------------- #
# 2b. Weekly aggregation (+ zero-week reindex)
# --------------------------------------------------------------------------- #
def _global_week_index(clean: pd.DataFrame) -> pd.DatetimeIndex:
    """Complete Monday-anchored weekly calendar spanning all data."""
    return pd.date_range(
        clean[config.COL_WEEK].min(),
        clean[config.COL_WEEK].max(),
        freq="W-MON",
    )


def _reindex_panel(clean: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Build a complete weekly panel for the given grouping keys, 0-filled.

    Vectorised approach (fast):
      1. aggregate transactions to weekly sums per key,
      2. build the full key x week grid, keeping only weeks on/after each
         key's first active week (so we don't invent history before a product
         existed, but we DO carry zeros forward to the global last week),
      3. left-merge the actual sums onto the grid and fill gaps with 0.
    """
    value_cols = [config.COL_SALES_QTY, config.COL_SALES_USD, config.COL_SALES_KG]
    agg = clean.groupby([*keys, config.COL_WEEK])[value_cols].sum().reset_index()

    weeks = _global_week_index(clean)
    weeks_df = pd.DataFrame({config.COL_WEEK: weeks})

    # first active week per key
    starts = agg.groupby(keys)[config.COL_WEEK].min().reset_index()
    starts = starts.rename(columns={config.COL_WEEK: "_start"})

    # full key x week grid via cross join, then drop weeks before each start
    grid = starts.merge(weeks_df, how="cross")
    grid = grid[grid[config.COL_WEEK] >= grid["_start"]].drop(columns="_start")

    # attach the actual weekly sums; missing combinations become 0
    out = grid.merge(agg, on=[*keys, config.COL_WEEK], how="left")
    out[value_cols] = out[value_cols].fillna(0.0)

    sort_keys = [*keys, config.COL_WEEK]
    return out.sort_values(sort_keys).reset_index(drop=True)[
        [*keys, config.COL_WEEK, *value_cols]
    ]


def build_weekly_product(clean: pd.DataFrame) -> pd.DataFrame:
    """Weekly Sales Qty (and USD, KG) per product, zero-filled to global end."""
    return _reindex_panel(clean, [config.COL_PRODUCT9])


def build_weekly_product_country(clean: pd.DataFrame) -> pd.DataFrame:
    """Weekly Sales Qty per (product, country), zero-filled to global end."""
    return _reindex_panel(clean, [config.COL_PRODUCT9, config.COL_COUNTRY])


# --------------------------------------------------------------------------- #
# 2c. Product tiers + demand classification
# --------------------------------------------------------------------------- #
def _demand_class(adi: float, cv2: float) -> str:
    """Syntetos-Boylan demand category from ADI and CV^2."""
    if np.isnan(adi) or np.isnan(cv2):
        return "no_demand"
    intermittent = adi >= config.ADI_THRESHOLD
    erratic = cv2 >= config.CV2_THRESHOLD
    if not intermittent and not erratic:
        return "smooth"
    if intermittent and not erratic:
        return "intermittent"
    if not intermittent and erratic:
        return "erratic"
    return "lumpy"


def build_product_tiers(weekly_product: pd.DataFrame,
                        clean: pd.DataFrame) -> pd.DataFrame:
    """One row per product: activity stats, demand class, and routing tier.

    Tier drives which model the product gets in step 4:
      * A -> regular series: SARIMA / Prophet per series, or global LightGBM
      * B -> lumpy/intermittent: global LightGBM or Croston/TSB
      * C -> sparse/one-off: simple baseline (recent mean / category rate)
    """
    qty = config.COL_SALES_QTY
    pid = config.COL_PRODUCT9

    # transaction counts come from the cleaned (un-reindexed) data
    n_txns = clean.groupby(pid).size().rename("n_txns")

    # everything else from the reindexed weekly panel
    grp = weekly_product.groupby(pid)
    rows = []
    for p, g in grp:
        q = g[qty].to_numpy()
        n_weeks = len(q)                       # weeks from first sale to global end
        nonzero = q[q != 0]
        n_active = int((q != 0).sum())
        total_qty = float(q.sum())
        pct_zero = float((q == 0).mean() * 100)
        # ADI on the product's own span; CV^2 on non-zero demand sizes
        adi = n_weeks / n_active if n_active > 0 else np.nan
        if len(nonzero) >= 2 and nonzero.mean() != 0:
            cv2 = float((nonzero.std(ddof=0) / abs(nonzero.mean())) ** 2)
        else:
            cv2 = np.nan
        rows.append((p, n_weeks, n_active, total_qty, pct_zero, adi, cv2))

    tiers = pd.DataFrame(
        rows,
        columns=[pid, "n_weeks_span", "n_active_weeks", "total_qty",
                 "pct_zero_weeks", "adi", "cv2"],
    ).set_index(pid)
    tiers = tiers.join(n_txns)

    tiers["demand_class"] = [
        _demand_class(a, c) for a, c in zip(tiers["adi"], tiers["cv2"])
    ]

    # routing tier
    is_a = tiers["n_active_weeks"] >= config.TIER_A_MIN_ACTIVE_WEEKS
    is_c = (~is_a) & (tiers["n_txns"] <= config.TIER_C_MAX_TXNS)
    tiers["tier"] = np.where(is_a, "A", np.where(is_c, "C", "B"))

    # volume rank for convenience
    tiers = tiers.sort_values("total_qty", ascending=False)
    tiers["volume_rank"] = np.arange(1, len(tiers) + 1)

    return tiers.reset_index()


# --------------------------------------------------------------------------- #
# Save helpers
# --------------------------------------------------------------------------- #
def save_outputs(clean: pd.DataFrame,
                 weekly_product: pd.DataFrame,
                 weekly_product_country: pd.DataFrame,
                 tiers: pd.DataFrame) -> dict:
    """Write all step-2 artefacts to disk; return a dict of written paths."""
    config.ensure_dirs()
    paths = {}

    p = config.INTERIM_DIR / "cleaned_transactions.parquet"
    try:
        clean.to_parquet(p, index=False)
    except (ImportError, ValueError):
        # parquet engine (pyarrow/fastparquet) not installed -> fall back to CSV
        p = config.INTERIM_DIR / "cleaned_transactions.csv"
        clean.to_csv(p, index=False)
    paths["cleaned_transactions"] = p

    p = config.PROCESSED_DIR / "weekly_product_sales.csv"
    weekly_product.to_csv(p, index=False)
    paths["weekly_product_sales"] = p

    p = config.PROCESSED_DIR / "weekly_product_country_sales.csv"
    weekly_product_country.to_csv(p, index=False)
    paths["weekly_product_country_sales"] = p

    p = config.PROCESSED_DIR / "product_tiers.csv"
    tiers.to_csv(p, index=False)
    paths["product_tiers"] = p

    return paths
