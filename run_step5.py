"""
run_step5.py
============
Step 5 — generate the deliverable 12-week-ahead forecasts and package the
outputs. Run like the others (Run / Ctrl+F5). Run steps 2 and 3 first.

Routing (best model per tier, from step 4):
    Tier A, B -> global LightGBM, volume-bias-corrected
    Tier C    -> moving-average baseline

Outputs: outputs/forecasts/product_12_week_forecast.csv
         outputs/forecasts/product_country_12_week_forecast.csv
         outputs/figures/05_forecast_overview.png
         outputs/step5_forecast_report.txt

NOTE: this trains LightGBM twice (once for the bias backtest, once on all data)
and runs two recursive forecasts, so it takes a few minutes. Progress prints.
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

from src import config, features as fe, forecast as fc  # noqa: E402

KEY, WK, QTY = config.COL_PRODUCT9, config.COL_WEEK, config.COL_SALES_QTY


def section(t):
    bar = "=" * 70
    return f"\n{bar}\n{t}\n{bar}"


def main() -> None:
    config.ensure_dirs()
    lines = []
    def out(t=""):
        print(t, flush=True); lines.append(str(t))

    out(section("STEP 5 — FINAL 12-WEEK FORECASTS"))
    if not (config.PROCESSED_DIR / "feature_table.parquet").exists() and \
       not (config.PROCESSED_DIR / "feature_table.csv").exists():
        out("ERROR: run steps 2 and 3 first."); return

    out("Loading inputs ...")
    weekly = fe.load_weekly_product()
    tiers = fe.load_tiers()
    feat = pd.read_parquet(config.PROCESSED_DIR / "feature_table.parquet") \
        if (config.PROCESSED_DIR / "feature_table.parquet").exists() \
        else pd.read_csv(config.PROCESSED_DIR / "feature_table.csv", parse_dates=[WK])
    clean = fe.load_cleaned()

    fweeks = fc.future_weeks(weekly)
    out(f"  forecasting {len(fweeks)} weeks: "
        f"{fweeks[0].date()} .. {fweeks[-1].date()}")

    # ---- product-level forecast ------------------------------------------ #
    out(section("TRAINING & FORECASTING (a few minutes) ..."))
    out("  estimating volume-bias correction on a 12-week backtest ...")
    product_fc, factors = fc.forecast_products(weekly, tiers, feat, clean)
    out("  bias-correction factors (multiplier applied to LightGBM):")
    for t, f in sorted(factors.items()):
        out(f"    Tier {t}: x{f:.2f}")

    out(section("PRODUCT-LEVEL FORECAST"))
    out(f"  rows: {len(product_fc):,}  "
        f"(products {product_fc[KEY].nunique():,} x {len(fweeks)} weeks)")
    out("  model used by tier:")
    for (t, m), n in product_fc.groupby(["tier", "model_used"]).size().items():
        out(f"    Tier {t}: {m}  ({n // len(fweeks):,} products)")
    out("  total forecast volume by tier (12-week sum):")
    vt = product_fc.groupby("tier")["forecast_qty"].sum()
    for t, v in vt.items():
        out(f"    Tier {t}: {v:,.0f} units")

    # ---- country disaggregation ------------------------------------------ #
    out(section("PRODUCT x COUNTRY FORECAST (top-down by recent share)"))
    country_fc = fc.disaggregate_to_country(product_fc, clean)
    out(f"  rows: {len(country_fc):,}  "
        f"(product-country pairs {country_fc[[KEY, config.COL_COUNTRY]].drop_duplicates().shape[0]:,})")
    # consistency check: country forecasts should sum back to product forecasts
    s_prod = product_fc["forecast_qty"].sum()
    s_ctry = country_fc["forecast_qty"].sum()
    out(f"  consistency: product total={s_prod:,.0f}  "
        f"country total={s_ctry:,.0f}  "
        f"({'OK' if abs(s_prod - s_ctry) / max(s_prod,1) < 0.02 else 'CHECK'})")

    # ---- save ------------------------------------------------------------ #
    out(section("SAVING FORECASTS"))
    paths = fc.save_forecasts(product_fc, country_fc)
    for k, p in paths.items():
        out(f"  {k}: {p}")

    # ---- overview chart -------------------------------------------------- #
    fig, ax = plt.subplots(figsize=(11, 5))
    hist = (weekly.groupby(WK)[QTY].sum().reset_index().tail(40))
    fut = (product_fc.groupby("week_start")["forecast_qty"].sum().reset_index())
    ax.plot(hist[WK], hist[QTY], color="#444", lw=1.2, label="actual (history)")
    ax.plot(fut["week_start"], fut["forecast_qty"], color="#c44", lw=2,
            ls="--", marker="o", ms=3, label="forecast (next 12 weeks)")
    ax.axvline(pd.Timestamp(weekly[WK].max()), color="grey", ls=":", lw=1)
    ax.set_title("Company-wide weekly Sales Qty — history and 12-week forecast")
    ax.set_xlabel("Week"); ax.set_ylabel("Total Sales Qty"); ax.legend()
    fig.tight_layout()
    p = config.FIGURES_DIR / "05_forecast_overview.png"
    fig.savefig(p, dpi=150, bbox_inches="tight"); out(f"  saved chart: {p}")

    report = config.OUTPUTS_DIR / "step5_forecast_report.txt"
    report.write_text("\n".join(lines), encoding="utf-8")
    out(section("DONE — project complete"))
    out(f"  Deliverables in: {config.FORECASTS_DIR}")
    out(f"  Report: {report}")


if __name__ == "__main__":
    main()
