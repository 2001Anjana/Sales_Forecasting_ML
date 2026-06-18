"""
train.py
========
STEP 4 — models. One strategy per tier, plus baselines everywhere:

  * baselines (all tiers) : naïve, seasonal-naïve, moving average
  * Croston / TSB (Tier B): purpose-built for intermittent demand
  * global LightGBM (A+B) : gradient boosting on the step-3 feature table,
                            forecast recursively over the 12-week horizon
  * Tier C                : moving-average baseline (too sparse to model)

The LightGBM uses a **Tweedie** objective — designed for non-negative,
zero-inflated targets like intermittent demand — and predictions are clipped at
zero. Forecasting is **recursive**: each predicted week is fed back in as the
lag for the next, mirroring how the model is used in production (step 5).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, features as fe

KEY = config.COL_PRODUCT9
WK = config.COL_WEEK
QTY = config.COL_SALES_QTY


# --------------------------------------------------------------------------- #
# Baseline forecasters  (each returns [Product9, week_start, forecast])
# --------------------------------------------------------------------------- #
def _frame(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=[KEY, WK, "forecast"])


def naive_forecast(train: pd.DataFrame, future_weeks) -> pd.DataFrame:
    """Carry the last observed value forward across the horizon."""
    last = train.sort_values(WK).groupby(KEY)[QTY].last()
    rows = []
    for p, v in last.items():
        for w in future_weeks:
            rows.append((p, w, float(v)))
    return _frame(rows)


def moving_average_forecast(train: pd.DataFrame, future_weeks,
                            window: int = 8) -> pd.DataFrame:
    """Mean of the last `window` weeks, repeated across the horizon."""
    def _ma(g):
        return g.sort_values(WK)[QTY].tail(window).mean()
    ma = train.groupby(KEY).apply(_ma, include_groups=False)
    rows = [(p, w, float(v)) for p, v in ma.items() for w in future_weeks]
    return _frame(rows)


def seasonal_naive_forecast(full_hist: pd.DataFrame, future_weeks,
                            season: int = 52) -> pd.DataFrame:
    """Forecast each future week with the value 52 weeks earlier (0 if absent)."""
    idx = full_hist.set_index([KEY, WK])[QTY]
    rows = []
    for p in full_hist[KEY].unique():
        for w in future_weeks:
            past = pd.Timestamp(w) - pd.Timedelta(weeks=season)
            v = idx.get((p, past), 0.0)
            rows.append((p, w, float(v)))
    return _frame(rows)


def predict_zero_forecast(train: pd.DataFrame, future_weeks) -> pd.DataFrame:
    """Forecast zero everywhere — the honest competitor for the sparse tail."""
    rows = [(p, w, 0.0) for p in train[KEY].unique() for w in future_weeks]
    return _frame(rows)


# --------------------------------------------------------------------------- #
# Croston / TSB  (intermittent demand)
# --------------------------------------------------------------------------- #
def croston_tsb(series: np.ndarray, alpha: float = 0.1,
                beta: float = 0.1, variant: str = "tsb") -> float:
    """Return the per-period demand-rate forecast for an intermittent series.

    TSB (Teunter-Syntetos-Babai) updates the demand *probability* every period,
    so it handles obsolescence better than classic Croston. Returns a single
    rate that we repeat across the horizon.
    """
    y = np.asarray(series, dtype=float)
    n = len(y)
    if n == 0 or y.sum() == 0:
        return 0.0
    # initialise
    first_nz = np.argmax(y > 0)
    z = y[y > 0][0]           # demand size estimate
    p = 1.0 / max(1, (np.diff(np.flatnonzero(y > 0)).mean()
                      if (y > 0).sum() > 1 else 1))  # demand prob estimate
    for t in range(first_nz, n):
        if y[t] > 0:
            z = z + alpha * (y[t] - z)
            p = p + beta * (1.0 - p)
        else:
            p = p + beta * (0.0 - p)
    return float(p * z)


def croston_forecast(train: pd.DataFrame, future_weeks) -> pd.DataFrame:
    rows = []
    for p, g in train.groupby(KEY):
        rate = croston_tsb(g.sort_values(WK)[QTY].to_numpy())
        for w in future_weeks:
            rows.append((p, w, rate))
    return _frame(rows)


# --------------------------------------------------------------------------- #
# Global LightGBM
# --------------------------------------------------------------------------- #
CAT_FEATURES = [config.COL_BRAND, config.COL_RANGE, config.COL_CHANNEL,
                config.COL_REGION, "demand_class", "tier"]


def _prep_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in CAT_FEATURES:
        if c in df.columns:
            df[c] = df[c].astype("category")
    return df


def train_global_lgbm(feat_train: pd.DataFrame, params: dict = None):
    """Train a Tweedie-objective LightGBM on the training feature rows."""
    import lightgbm as lgb

    feats = fe.feature_columns(feat_train)
    X = _prep_categoricals(feat_train[feats])
    y = feat_train[QTY].clip(lower=0)

    default = dict(
        objective="tweedie", tweedie_variance_power=1.2,
        n_estimators=400, learning_rate=0.05, num_leaves=63,
        min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
        random_state=config.RANDOM_SEED, n_jobs=-1, verbose=-1,
    )
    if params:
        default.update(params)

    model = lgb.LGBMRegressor(**default)
    model.fit(X, y, categorical_feature=[c for c in CAT_FEATURES if c in feats])
    return model, feats


def recursive_forecast_lgbm(model, feats, hist_panel: pd.DataFrame,
                            static: pd.DataFrame, tiers: pd.DataFrame,
                            future_weeks, tail: int = 60) -> pd.DataFrame:
    """Forecast the horizon recursively, feeding predictions back as lags.

    hist_panel : weekly [Product9, week_start, Sales Qty] up to the cutoff.
    static     : per-product attributes (from features.product_static_attributes)
    Only the last `tail` weeks per product are kept in the working buffer — that
    is enough history for every lag/rolling feature and keeps each step fast.
    """
    static_t = tiers[[KEY, "tier", "demand_class"]].merge(static, on=KEY, how="left")
    panel = hist_panel[[KEY, WK, QTY]].copy()

    preds = []
    for w in future_weeks:
        # keep a rolling tail buffer per product
        panel = (panel.sort_values([KEY, WK])
                 .groupby(KEY, group_keys=False).tail(tail))
        # append the new (unknown) week for every product
        new = pd.DataFrame({KEY: panel[KEY].unique()})
        new[WK] = pd.Timestamp(w)
        new[QTY] = np.nan
        work = pd.concat([panel, new], ignore_index=True)

        # build features on the buffer, attach statics, predict the frontier
        work = fe.add_calendar_features(work)
        work = fe.add_lag_features(work, [KEY], QTY)
        work = fe.add_rolling_features(work, [KEY], QTY)
        work = fe.add_intermittency_features(work, [KEY], QTY)
        work = work.merge(static_t, on=KEY, how="left")

        frontier = work[work[WK] == pd.Timestamp(w)].copy()
        X = _prep_categoricals(frontier[feats])
        yhat = np.clip(model.predict(X), 0, None)
        frontier[QTY] = yhat

        # write predictions back into the panel so they feed the next step
        panel = pd.concat(
            [panel, frontier[[KEY, WK, QTY]]], ignore_index=True)
        for p, v in zip(frontier[KEY], yhat):
            preds.append((p, pd.Timestamp(w), float(v)))

    return pd.DataFrame(preds, columns=[KEY, WK, "forecast"])
