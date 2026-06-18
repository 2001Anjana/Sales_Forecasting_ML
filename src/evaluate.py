"""
evaluate.py
===========
STEP 4 helpers — time-based train/test split and forecast-accuracy metrics
suited to intermittent demand.

Why these metrics
-----------------
* MAPE is unusable here: it divides by actuals, and most weeks are zero.
* **WAPE** (weighted absolute percentage error) = sum|err| / sum|actual| is
  robust to zeros and is the headline metric.
* **MASE** scales error against the in-sample naïve forecast, so a value < 1
  means "better than naïve". We report the mean per-series MASE.
* RMSE and MAE are included for completeness.

All metrics are computed on the held-out final weeks, broken out **per tier** so
a strong Tier-A result is never hidden by the un-forecastable tail.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


# --------------------------------------------------------------------------- #
# Time split
# --------------------------------------------------------------------------- #
def train_test_cutoff(weekly: pd.DataFrame, horizon: int = None) -> pd.Timestamp:
    """Return the last training week: everything after it is the test horizon."""
    horizon = horizon or config.FORECAST_HORIZON_WEEKS
    weeks = np.sort(weekly[config.COL_WEEK].unique())
    # the cutoff is the week `horizon` steps before the last observed week
    return pd.Timestamp(weeks[-horizon - 1])


def test_weeks(weekly: pd.DataFrame, horizon: int = None) -> np.ndarray:
    horizon = horizon or config.FORECAST_HORIZON_WEEKS
    weeks = np.sort(weekly[config.COL_WEEK].unique())
    return weeks[-horizon:]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _align(y_true: pd.Series, y_pred: pd.Series):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return y_true, y_pred


def wape(y_true, y_pred) -> float:
    y_true, y_pred = _align(y_true, y_pred)
    denom = np.abs(y_true).sum()
    if denom == 0:
        return np.nan
    return float(np.abs(y_true - y_pred).sum() / denom)


def mae(y_true, y_pred) -> float:
    y_true, y_pred = _align(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = _align(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def series_mase(y_true, y_pred, train_series) -> float:
    """MASE for one series. Scale = in-sample one-step naïve MAE on the train."""
    y_true, y_pred = _align(y_true, y_pred)
    tr = np.asarray(train_series, dtype=float)
    if len(tr) < 2:
        return np.nan
    scale = np.mean(np.abs(np.diff(tr)))
    if scale == 0:
        return np.nan
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


# --------------------------------------------------------------------------- #
# Aggregation over a forecast table
# --------------------------------------------------------------------------- #
def evaluate_per_tier(forecasts: pd.DataFrame,
                      actuals: pd.DataFrame,
                      tiers: pd.DataFrame,
                      train_hist: pd.DataFrame,
                      model_name: str) -> pd.DataFrame:
    """Compute pooled WAPE/MAE/RMSE and mean per-series MASE, per tier.

    Parameters
    ----------
    forecasts : columns [Product9, week_start, forecast]
    actuals   : columns [Product9, week_start, Sales Qty]  (test weeks only)
    tiers     : product_tiers table (for the `tier` column)
    train_hist: weekly history BEFORE the test (for the MASE scale)
    """
    key = config.COL_PRODUCT9
    qty = config.COL_SALES_QTY

    df = actuals.merge(forecasts, on=[key, config.COL_WEEK], how="left")
    df["forecast"] = df["forecast"].fillna(0.0).clip(lower=0)
    df = df.merge(tiers[[key, "tier"]], on=key, how="left")

    # pre-compute per-series train arrays for MASE
    train_by_prod = {p: g[qty].to_numpy()
                     for p, g in train_hist.groupby(key)}

    rows = []
    for tier, g in df.groupby("tier"):
        w = wape(g[qty], g["forecast"])
        m = mae(g[qty], g["forecast"])
        r = rmse(g[qty], g["forecast"])
        # volume capture: total forecast / total actual over the horizon.
        # 1.0 = perfect aggregate volume; ~0 = predicting (near) zero, which
        # minimises point error but is useless for restocking.
        tot_actual = g[qty].sum()
        vol_capture = (g["forecast"].sum() / tot_actual) if tot_actual else np.nan
        # mean per-series MASE
        mases = []
        for p, gp in g.groupby(key):
            tr = train_by_prod.get(p)
            if tr is not None:
                val = series_mase(gp[qty], gp["forecast"], tr)
                if not np.isnan(val):
                    mases.append(val)
        rows.append({
            "tier": tier, "model": model_name,
            "WAPE": round(w, 4),
            "MASE": round(float(np.mean(mases)), 4) if mases else np.nan,
            "MAE": round(m, 3), "RMSE": round(r, 3),
            "vol_capture": round(float(vol_capture), 3) if not np.isnan(vol_capture) else np.nan,
            "n_products": g[key].nunique(),
        })
    return pd.DataFrame(rows)
