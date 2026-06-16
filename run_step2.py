"""
run_step2.py
============
Plain-script version of Step 2 — Preprocessing & cleaning. Run it exactly like
run_step1.py: open in VS Code and press the Run button (or Ctrl+F5).

It performs the three sub-steps from the plan and writes their outputs:
  2a. clean transactions      -> data/interim/cleaned_transactions.parquet
  2b. weekly aggregation      -> data/processed/weekly_product_sales.csv
                                 data/processed/weekly_product_country_sales.csv
  2c. product tiers           -> data/processed/product_tiers.csv

It also saves two diagnostic charts to outputs/figures/ and a text summary to
outputs/step2_preprocessing_report.txt.

NOTE: building the weekly panels creates ~1.5 million rows and writes two CSV
files (each tens of MB). Writing the CSVs is the slow part (up to a minute) —
this is normal, not a hang. Progress is printed as it goes.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src import config, data_loader, preprocessing as pp  # noqa: E402


def section(title: str) -> str:
    bar = "=" * 70
    return f"\n{bar}\n{title}\n{bar}"


def main() -> None:
    config.ensure_dirs()
    lines: list[str] = []

    def out(text=""):
        print(text, flush=True)
        lines.append(str(text))

    out(section("STEP 2 — PREPROCESSING & CLEANING"))
    if not config.RAW_FILE.exists():
        out("ERROR: data/raw/Cinnamon_export_sales.xlsx not found.")
        return

    # ---- load ------------------------------------------------------------- #
    out("Loading raw data ...")
    df = data_loader.load_raw()
    out(f"  raw rows: {len(df):,}")

    # ---- 2a. clean -------------------------------------------------------- #
    out(section("2a. CLEAN TRANSACTIONS"))
    clean, log = pp.clean_transactions(df)
    for k, v in log.items():
        out(f"  {k:>28}: {v:,}")
    out(f"  returns policy: '{config.RETURNS_POLICY}'  "
        f"(negatives kept as net demand)")

    # ---- 2b. weekly panels ----------------------------------------------- #
    out(section("2b. AGGREGATE TO WEEKLY SERIES (with zero-week fill)"))
    out("  building weekly product panel ...")
    weekly_product = pp.build_weekly_product(clean)
    out(f"    rows: {len(weekly_product):,}  "
        f"(products x weeks, zero-filled to global end)")

    out("  building weekly product x country panel ...")
    weekly_pc = pp.build_weekly_product_country(clean)
    out(f"    rows: {len(weekly_pc):,}")

    # integrity check: total quantity must be unchanged by reindexing
    q_clean = clean[config.COL_SALES_QTY].sum()
    q_panel = weekly_product[config.COL_SALES_QTY].sum()
    out(f"  integrity: total Sales Qty clean={q_clean:,.0f} "
        f"panel={q_panel:,.0f}  ({'OK' if abs(q_clean-q_panel)<1 else 'MISMATCH'})")

    # ---- 2c. tiers -------------------------------------------------------- #
    out(section("2c. PRODUCT TIERS & DEMAND CLASSIFICATION"))
    tiers = pp.build_product_tiers(weekly_product, clean)
    out("  Tier counts (routing key for which model each product gets):")
    for t, n in tiers["tier"].value_counts().sort_index().items():
        out(f"    Tier {t}: {n:,} products")
    out("  Volume share by tier (%):")
    vs = tiers.groupby("tier")["total_qty"].sum()
    for t, pct in (vs / vs.sum() * 100).round(1).sort_index().items():
        out(f"    Tier {t}: {pct}%")
    out("  Demand class (Syntetos-Boylan):")
    for c, n in tiers["demand_class"].value_counts().items():
        out(f"    {c:>13}: {n:,}")

    # ---- save processed files -------------------------------------------- #
    out(section("SAVING PROCESSED FILES (writing CSVs can take ~1 min) ..."))
    paths = pp.save_outputs(clean, weekly_product, weekly_pc, tiers)
    for name, p in paths.items():
        out(f"  {name}: {p}")

    # ---- diagnostic charts ----------------------------------------------- #
    out(section("SAVING DIAGNOSTIC CHARTS"))
    # tiers bar
    fig, ax = plt.subplots(figsize=(6, 4))
    tiers["tier"].value_counts().sort_index().plot(
        kind="bar", color="#7a4a2b", ax=ax)
    ax.set_title("Products per tier")
    ax.set_xlabel("Tier"); ax.set_ylabel("Number of products")
    ax.set_yscale("log")
    fig.tight_layout()
    p1 = config.FIGURES_DIR / "02_product_tiers.png"
    fig.savefig(p1, dpi=150, bbox_inches="tight"); out(f"  saved: {p1}")

    # zero-week distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(tiers["pct_zero_weeks"].dropna(), bins=40,
            color="#7a4a2b", edgecolor="white", linewidth=0.3)
    ax.set_title("Distribution of % zero-demand weeks per product")
    ax.set_xlabel("% of weeks with zero demand"); ax.set_ylabel("Number of products")
    fig.tight_layout()
    p2 = config.FIGURES_DIR / "02_zero_week_distribution.png"
    fig.savefig(p2, dpi=150, bbox_inches="tight"); out(f"  saved: {p2}")

    # ---- done ------------------------------------------------------------- #
    report = config.OUTPUTS_DIR / "step2_preprocessing_report.txt"
    report.write_text("\n".join(lines), encoding="utf-8")
    out(section("DONE"))
    out(f"Text report saved to: {report}")
    out("Next: step 3 — feature engineering, using these processed files.")


if __name__ == "__main__":
    main()
