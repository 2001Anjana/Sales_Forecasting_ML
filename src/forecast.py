"""
forecast.py
===========
STEP 5 — generate the deliverable 12-week-ahead forecasts.

Differences from step 4 (which was for *evaluation*):
  * models are retrained on ALL history (no holdout),
  * we forecast the 12 weeks AFTER the last observed week,
  * per-tier routing to the chosen "best" model:
        Tier A, B -> global LightGBM, volume-bias-corrected
        Tier C    -> moving-average baseline
  * a per-product forecast is disaggregated to product x country by recent
    country share (top-down hierarchical reconciliation), so country forecasts
    always sum back to the product forecast.

Why bias-correct LightGBM
-------------------------
The recursive LightGBM minimises per-week error but under-forecasts total
volume on intermittent demand (see step 4's `vol_capture`). We estimate a
per-tier correction factor = total_actual / total_forecast on a 12-week
backtest, clip it to a sane range, and scale the future forecast so the total
volume is realistic while keeping the model's weekly pattern.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, features as fe, train as tr, evaluate as ev

KEY, WK, QTY = config.COL_PRODUCT9, config.COL_WEEK, config.COL_SALES_QTY
MA_WINDOW = 8
BIAS_CLIP = (1.0, 6.0)   # keep correction factors sensible


def future_weeks(weekly: pd.DataFrame, horizon: int = None) -> list:
    horizon = horizon or config.FORECAST_HORIZON_WEEKS
    last = pd.Timestamp(np.sort(weekly[WK].unique())[-1])
    return [last + pd.Timedelta(weeks=i) for i in range(1, horizon + 1)]


# --------------------------------------------------------------------------- #
# Bias factors from a 12-week backtest
# --------------------------------------------------------------------------- #
def estimate_bias_factors(weekly, tiers, feat, static) -> dict:
    """Per-tier LightGBM volume-correction factor from a holdout backtest."""
    cutoff = ev.train_test_cutoff(weekly)
    tw = ev.test_weeks(weekly)
    ab = tiers.loc[tiers["tier"].isin(["A", "B"]), KEY]

    model, feats = tr.train_global_lgbm(feat[feat[WK] <= cutoff])
    hist = weekly[(weekly[WK] <= cutoff) & (weekly[KEY].isin(ab))]
    fc = tr.recursive_forecast_lgbm(model, feats, hist, static, tiers, tw)

    actual = weekly[weekly[WK].isin(tw)][[KEY, WK, QTY]]
    df = actual.merge(fc, on=[KEY, WK], how="left").merge(
        tiers[[KEY, "tier"]], on=KEY, how="left")
    df["forecast"] = df["forecast"].fillna(0)

    factors = {}
    for t, g in df.groupby("tier"):
        tot_f = g["forecast"].sum()
        tot_a = g[QTY].sum()
        f = (tot_a / tot_f) if tot_f > 0 else 1.0
        factors[t] = float(np.clip(f, *BIAS_CLIP))
    return factors


# --------------------------------------------------------------------------- #
# Tier-C moving average
# --------------------------------------------------------------------------- #
def moving_average_future(weekly_c: pd.DataFrame, fweeks) -> pd.DataFrame:
    ma = (weekly_c.sort_values(WK).groupby(KEY)[QTY]
          .apply(lambda s: s.tail(MA_WINDOW).mean()))
    rows = [(p, w, max(0.0, float(v))) for p, v in ma.items() for w in fweeks]
    return pd.DataFrame(rows, columns=[KEY, WK, "forecast_qty"])


# --------------------------------------------------------------------------- #
# Main product-level forecast
# --------------------------------------------------------------------------- #
def forecast_products(weekly, tiers, feat, clean) -> tuple[pd.DataFrame, dict]:
    static = fe.product_static_attributes(clean)
    fweeks = future_weeks(weekly)

    # 1. estimate the LightGBM bias correction on a backtest
    factors = estimate_bias_factors(weekly, tiers, feat, static)

    # 2. retrain LightGBM on ALL data and forecast the future for A+B
    ab = tiers.loc[tiers["tier"].isin(["A", "B"]), KEY]
    model, feats = tr.train_global_lgbm(feat)
    ab_hist = weekly[weekly[KEY].isin(ab)]
    fc_ab = tr.recursive_forecast_lgbm(model, feats, ab_hist, static, tiers, fweeks)
    fc_ab = fc_ab.merge(tiers[[KEY, "tier"]], on=KEY, how="left")
    fc_ab["forecast_qty"] = fc_ab.apply(
        lambda r: r["forecast"] * factors.get(r["tier"], 1.0), axis=1)
    fc_ab["model_used"] = "LightGBM(bias-corrected)"
    fc_ab = fc_ab[[KEY, WK, "forecast_qty", "tier", "model_used"]]

    # 3. Tier C -> moving average
    c_ids = tiers.loc[tiers["tier"] == "C", KEY]
    weekly_c = weekly[weekly[KEY].isin(c_ids)]
    fc_c = moving_average_future(weekly_c, fweeks)
    fc_c["tier"] = "C"
    fc_c["model_used"] = f"MovingAvg{MA_WINDOW}"

    product_fc = pd.concat([fc_ab, fc_c], ignore_index=True)
    product_fc["forecast_qty"] = product_fc["forecast_qty"].clip(lower=0).round(2)
    product_fc = product_fc.rename(columns={WK: "week_start"})
    return product_fc, factors


# --------------------------------------------------------------------------- #
# Disaggregate product forecast to product x country
# --------------------------------------------------------------------------- #
def disaggregate_to_country(product_fc, clean, recent_weeks: int = 52) -> pd.DataFrame:
    """Split each product's weekly forecast across countries by recent share."""
    cutoff = clean[WK].max() - pd.Timedelta(weeks=recent_weeks)
    recent = clean[clean[WK] >= cutoff]
    share = (recent.groupby([KEY, config.COL_COUNTRY])[QTY].sum()
             .clip(lower=0).reset_index())
    # fall back to all-history share for products with no recent positive volume
    tot = share.groupby(KEY)["Sales Qty"].transform("sum")
    share = share[tot > 0].copy()
    share["w"] = share[QTY] / share.groupby(KEY)[QTY].transform("sum")

    # products with no usable share -> single UNKNOWN country at full weight
    missing = set(product_fc[KEY]) - set(share[KEY])
    if missing:
        extra = pd.DataFrame({KEY: list(missing)})
        extra[config.COL_COUNTRY] = config.UNKNOWN_LABEL
        extra["w"] = 1.0
        share = pd.concat([share[[KEY, config.COL_COUNTRY, "w"]], extra],
                          ignore_index=True)

    out = product_fc.merge(share[[KEY, config.COL_COUNTRY, "w"]], on=KEY, how="left")
    out["forecast_qty"] = (out["forecast_qty"] * out["w"]).clip(lower=0).round(2)
    out = out[[KEY, config.COL_COUNTRY, "week_start", "forecast_qty",
               "tier", "model_used"]]
    return out


def save_forecasts(product_fc, country_fc) -> dict:
    config.ensure_dirs()
    p1 = config.FORECASTS_DIR / "product_12_week_forecast.csv"
    p2 = config.FORECASTS_DIR / "product_country_12_week_forecast.csv"
    product_fc.to_csv(p1, index=False)
    country_fc.to_csv(p2, index=False)
    return {"product": p1, "product_country": p2}
