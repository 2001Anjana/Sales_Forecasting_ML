"""
run_step4.py
============
Step 4 — train models, evaluate them on a held-out 12-week horizon, and compare
per tier. Run like the others (Run / Ctrl+F5).

Pipeline
--------
1. Split: hold out the last 12 weeks as the test horizon.
2. Tier C  -> moving-average baseline.
3. Tier A+B-> baselines (naïve, seasonal-naïve, moving avg), Croston/TSB,
   and the global LightGBM (recursive 12-week forecast).
4. Evaluate every model per tier with WAPE / MASE / MAE / RMSE.
5. Save the trained model, a metrics table, and actual-vs-predicted charts.

Outputs: outputs/models/global_lgbm.txt
         outputs/model_comparison.csv
         outputs/figures/04_*.png
         outputs/step4_model_report.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config, features as fe, train as tr, evaluate as ev  # noqa: E402

KEY, WK, QTY = config.COL_PRODUCT9, config.COL_WEEK, config.COL_SALES_QTY


def section(t): 
    bar = "=" * 70
    return f"\n{bar}\n{t}\n{bar}"


def main() -> None:
    config.ensure_dirs()
    lines = []
    def out(t=""):
        print(t, flush=True); lines.append(str(t))

    out(section("STEP 4 — MODEL TRAINING & EVALUATION"))
    if not (config.PROCESSED_DIR / "weekly_product_sales.csv").exists():
        out("ERROR: run steps 2 and 3 first."); return

    # ---- load ------------------------------------------------------------- #
    out("Loading weekly panel, tiers, and feature table ...")
    weekly = fe.load_weekly_product()
    tiers = fe.load_tiers()
    feat = pd.read_parquet(config.PROCESSED_DIR / "feature_table.parquet") \
        if (config.PROCESSED_DIR / "feature_table.parquet").exists() \
        else pd.read_csv(config.PROCESSED_DIR / "feature_table.csv",
                         parse_dates=[WK])
    clean = fe.load_cleaned()
    static = fe.product_static_attributes(clean)

    # ---- split ------------------------------------------------------------ #
    cutoff = ev.train_test_cutoff(weekly)
    tw = ev.test_weeks(weekly)
    out(section("1. TIME-BASED SPLIT"))
    out(f"  cutoff (last train week): {pd.Timestamp(cutoff).date()}")
    out(f"  test horizon           : {len(tw)} weeks "
        f"({pd.Timestamp(tw[0]).date()} .. {pd.Timestamp(tw[-1]).date()})")

    train_hist = weekly[weekly[WK] <= cutoff]
    actuals = weekly[weekly[WK].isin(tw)][[KEY, WK, QTY]]

    a_b_ids = tiers.loc[tiers["tier"].isin(["A", "B"]), KEY]
    c_ids = tiers.loc[tiers["tier"] == "C", KEY]

    results = []

    # ---- 2. baselines (all tiers) ---------------------------------------- #
    out(section("2. BASELINES (all tiers)"))
    out("  computing naïve, seasonal-naïve, moving-average ...")
    fc_naive = tr.naive_forecast(train_hist, tw)
    fc_ma = tr.moving_average_forecast(train_hist, tw, window=8)
    fc_snaive = tr.seasonal_naive_forecast(weekly[weekly[WK] <= cutoff], tw)
    fc_zero = tr.predict_zero_forecast(train_hist, tw)
    for name, fc in [("Naive", fc_naive), ("SeasonalNaive", fc_snaive),
                     ("MovingAvg8", fc_ma), ("PredictZero", fc_zero)]:
        results.append(ev.evaluate_per_tier(fc, actuals, tiers, train_hist, name))

    # ---- 3. Croston/TSB (intermittent, evaluate on A+B) ------------------ #
    out(section("3. CROSTON / TSB (intermittent demand)"))
    ab_train = train_hist[train_hist[KEY].isin(a_b_ids)]
    fc_croston = tr.croston_forecast(ab_train, tw)
    results.append(ev.evaluate_per_tier(fc_croston, actuals, tiers,
                                        train_hist, "CrostonTSB"))
    out("  done.")

    # ---- 4. global LightGBM (A+B) ---------------------------------------- #
    out(section("4. GLOBAL LIGHTGBM (Tiers A+B)"))
    feat_train = feat[feat[WK] <= cutoff].copy()
    out(f"  training rows: {len(feat_train):,}  (Tweedie objective)")
    model, feats = tr.train_global_lgbm(feat_train)
    out("  trained. forecasting recursively over 12 weeks ...")
    ab_hist = train_hist[train_hist[KEY].isin(a_b_ids)]
    fc_lgbm = tr.recursive_forecast_lgbm(model, feats, ab_hist, static, tiers, tw)
    results.append(ev.evaluate_per_tier(fc_lgbm, actuals, tiers,
                                        train_hist, "LightGBM"))

    # ---- 5. comparison table --------------------------------------------- #
    out(section("5. MODEL COMPARISON (per tier; lower is better)"))
    comp = pd.concat(results, ignore_index=True)
    # Croston/TSB and LightGBM only serve Tiers A+B; drop their (all-zero) Tier-C rows
    ab_only = comp["model"].isin(["CrostonTSB", "LightGBM"])
    comp = comp[~(ab_only & (comp["tier"] == "C"))]
    comp = comp.sort_values(["tier", "WAPE"]).reset_index(drop=True)
    out(comp.to_string(index=False))
    comp.to_csv(config.OUTPUTS_DIR / "model_comparison.csv", index=False)

    # Two honest lenses. PredictZero is the trivial all-zero reference
    # (WAPE always = 1.0, 0% volume) — excluded from "best".
    out("\n  PredictZero is the trivial all-zero reference (WAPE=1.0, 0% volume).")
    out("  Among real models there are two lenses:")
    real = comp[comp["model"] != "PredictZero"]
    out("\n  (a) Best per-week point accuracy (lowest MASE):")
    for t, g in real.groupby("tier"):
        b = g.loc[g["MASE"].idxmin()]
        out(f"      Tier {t}: {b['model']}  (MASE={b['MASE']}, WAPE={b['WAPE']})")
    out("\n  (b) Best total-volume capture (closest to 1.0 — matters for restocking):")
    for t, g in real.groupby("tier"):
        g = g.copy(); g["gap"] = (g["vol_capture"] - 1.0).abs()
        b = g.loc[g["gap"].idxmin()]
        out(f"      Tier {t}: {b['model']}  (vol_capture={b['vol_capture']}, "
            f"MASE={b['MASE']})")
    out("\n  Takeaway: LightGBM minimises point error but under-forecasts volume")
    out("  (recursive shrinkage on intermittent demand); moving-average / Croston")
    out("  capture total volume better. The report should discuss this trade-off.")

    # ---- save model ------------------------------------------------------- #
    out(section("SAVING MODEL & CHARTS"))
    model_path = config.MODELS_DIR / "global_lgbm.txt"
    model.booster_.save_model(str(model_path))
    out(f"  saved model: {model_path}")

    # feature importance
    imp = pd.Series(model.feature_importances_, index=feats).sort_values()
    fig, ax = plt.subplots(figsize=(8, 9))
    imp.tail(25).plot(kind="barh", color="#7a4a2b", ax=ax)
    ax.set_title("LightGBM feature importance (top 25)")
    fig.tight_layout()
    p = config.FIGURES_DIR / "04_feature_importance.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); out(f"  saved: {p}")

    # actual vs predicted for a few Tier-A products
    top_a = (tiers[tiers["tier"] == "A"].sort_values("total_qty", ascending=False)
             [KEY].head(4).tolist())
    fig, axes = plt.subplots(2, 2, figsize=(13, 7))
    for ax, pid in zip(axes.ravel(), top_a):
        h = weekly[(weekly[KEY] == pid)].sort_values(WK).tail(40)
        f = fc_lgbm[fc_lgbm[KEY] == pid]
        ax.plot(h[WK], h[QTY], color="#444", lw=1, label="actual")
        ax.plot(f[WK], f["forecast"], color="#c44", lw=1.5, ls="--",
                label="LightGBM forecast")
        ax.axvline(pd.Timestamp(cutoff), color="grey", ls=":", lw=1)
        ax.set_title(pid, fontsize=9); ax.legend(fontsize=7)
    fig.suptitle("Actual vs 12-week forecast — top Tier-A products")
    fig.tight_layout()
    p = config.FIGURES_DIR / "04_actual_vs_forecast.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); out(f"  saved: {p}")

    report = config.OUTPUTS_DIR / "step4_model_report.txt"
    report.write_text("\n".join(lines), encoding="utf-8")
    out(section("DONE"))
    out(f"Report: {report}")
    out("Next: step 5 — generate the final 12-week forecasts for all products.")


if __name__ == "__main__":
    main()
