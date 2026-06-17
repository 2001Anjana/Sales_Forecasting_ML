"""
run_step3.py
============
Plain-script version of Step 3 — Feature engineering. Run like the others:
open in VS Code and press Run (or Ctrl+F5).

It builds the supervised feature table for the products we will model with a
global / ML model (Tiers A and B), runs a leakage sanity-check, saves the table,
and produces a feature-correlation heatmap.

Inputs  (from step 2): data/processed/weekly_product_sales.csv,
                        data/processed/product_tiers.csv,
                        data/interim/cleaned_transactions.parquet (or .csv)
Outputs: data/processed/feature_table.parquet (or .csv)
         outputs/figures/03_feature_correlation.png
         outputs/step3_feature_report.txt
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

from src import config, features as fe  # noqa: E402


def section(title: str) -> str:
    bar = "=" * 70
    return f"\n{bar}\n{title}\n{bar}"


def main() -> None:
    config.ensure_dirs()
    lines: list[str] = []

    def out(text=""):
        print(text, flush=True)
        lines.append(str(text))

    out(section("STEP 3 — FEATURE ENGINEERING"))

    needed = config.PROCESSED_DIR / "weekly_product_sales.csv"
    if not needed.exists():
        out("ERROR: data/processed/weekly_product_sales.csv not found. "
            "Run run_step2.py first.")
        return

    out("Building feature table for Tiers A and B (this takes ~15-30s) ...")
    df = fe.build_feature_table(("A", "B"))
    feats = fe.feature_columns(df)

    out(f"  rows           : {len(df):,}")
    out(f"  products        : {df[config.COL_PRODUCT9].nunique():,}")
    out(f"  feature columns : {len(feats)}")
    out("  tier breakdown  : "
        + str(dict(df.drop_duplicates(config.COL_PRODUCT9)['tier'].value_counts())))

    out(section("FEATURE FAMILIES"))
    out("  calendar   : woy, month, quarter, year, week_of_month, "
        "woy_sin/cos, month_sin/cos")
    out("  seasonal   : is_quarter_end, is_peak_month")
    out(f"  lag        : {[c for c in feats if c.startswith('lag_')]}")
    out(f"  rolling    : {[c for c in feats if c.startswith('roll_')]}")
    out("  intermittency: weeks_since_last_sale, zero_frac_13, mean_nonzero_13, "
        "weeks_since_start")
    out("  static     : Brand Category, Product Range, Sales Channel, Region")

    # ---- leakage sanity-check ------------------------------------------- #
    out(section("LEAKAGE SANITY-CHECK"))
    one = df[df[config.COL_PRODUCT9] == df[config.COL_PRODUCT9].iloc[0]].head(5)
    out("  For one product, lag_1 should equal the previous week's Sales Qty:")
    out(one[[config.COL_WEEK, config.COL_SALES_QTY, "lag_1", "lag_2",
             "roll_mean_4"]].to_string(index=False))
    # programmatic check across a sample
    chk = df.copy()
    chk["prev"] = chk.groupby(config.COL_PRODUCT9)[config.COL_SALES_QTY].shift(1)
    ok = np.allclose(chk["lag_1"].dropna(),
                     chk.loc[chk["lag_1"].notna(), "prev"], equal_nan=True)
    out(f"  lag_1 == previous-week Sales Qty everywhere: {'PASS' if ok else 'FAIL'}")

    # ---- save ------------------------------------------------------------ #
    out(section("SAVING FEATURE TABLE"))
    p = fe.save_feature_table(df)
    out(f"  saved: {p}")

    # ---- correlation heatmap (required deliverable) --------------------- #
    out(section("SAVING FEATURE CORRELATION HEATMAP"))
    num = df[[config.COL_SALES_QTY] + [c for c in feats
             if df[c].dtype.kind in "fi"]].copy()
    # keep it readable: drop near-constant flags from the matrix
    corr = num.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr.columns, fontsize=7)
    ax.set_title("Feature correlation matrix (target = Sales Qty)")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    pcorr = config.FIGURES_DIR / "03_feature_correlation.png"
    fig.savefig(pcorr, dpi=150, bbox_inches="tight")
    out(f"  saved: {pcorr}")
    # top correlations with the target
    tgt = corr[config.COL_SALES_QTY].drop(config.COL_SALES_QTY).abs().sort_values(
        ascending=False)
    out("  Top 8 features by |correlation| with Sales Qty:")
    for name, v in tgt.head(8).items():
        out(f"    {name:>22}: {corr[config.COL_SALES_QTY][name]:+.3f}")

    report = config.OUTPUTS_DIR / "step3_feature_report.txt"
    report.write_text("\n".join(lines), encoding="utf-8")
    out(section("DONE"))
    out(f"Text report saved to: {report}")
    out("Next: step 4 — model training (baselines + global LightGBM by tier).")


if __name__ == "__main__":
    main()
