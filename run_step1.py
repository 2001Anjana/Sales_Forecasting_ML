"""
run_step1.py
============
Plain-script version of notebooks/01_data_understanding.ipynb.

WHY THIS EXISTS
---------------
IDLE (and any plain Python setup) cannot run Jupyter notebooks (.ipynb files).
This script does exactly what the notebook does, but as ordinary Python you can
run from IDLE: open this file in IDLE and press F5  (Run > Run Module).

It will:
  * load the raw spreadsheet,
  * print the profiling reports to the screen,
  * save the three charts into outputs/figures/,
  * save a plain-text summary into outputs/step1_data_understanding_report.txt

You do NOT need Jupyter, Claude Code, Cursor, or any AI tool to run this.
You only need Python with the packages in requirements.txt installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

# --------------------------------------------------------------------------- #
# Make the project importable no matter where it is run from.
# This file lives at the project root, so its folder is the root.
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")  # save figures to file without needing a display window

from src import config, data_loader, profiling  # noqa: E402


def section(title: str) -> str:
    bar = "=" * 70
    return f"\n{bar}\n{title}\n{bar}"


def main() -> None:
    config.ensure_dirs()

    # Collect everything we print into one list so we can also save it to a file.
    lines: list[str] = []

    def out(text=""):
        print(text)
        lines.append(str(text))

    out(section("STEP 1 — DATA UNDERSTANDING"))
    out(f"Project root : {config.PROJECT_ROOT}")
    out(f"Raw file     : {config.RAW_FILE}")
    if not config.RAW_FILE.exists():
        out("\nERROR: raw file not found. Put Cinnamon_export_sales.xlsx in "
            "data/raw/ and run again.")
        return

    # ---- load -------------------------------------------------------------- #
    df = data_loader.load_raw()
    out(f"\nLoaded shape : {df.shape[0]:,} rows  x  {df.shape[1]} columns")

    # ---- 2. missing -------------------------------------------------------- #
    out(section("2. MISSING VALUES"))
    out(profiling.missing_report(df).to_string())

    # ---- 3. cardinality ---------------------------------------------------- #
    out(section("3. CARDINALITY (unique values per column)"))
    out(profiling.cardinality_report(df).to_string())

    # ---- 4. numeric summary ------------------------------------------------ #
    out(section("4. NUMERIC SUMMARY (note negatives / zeros / outliers)"))
    out(profiling.numeric_summary(df).to_string())
    out("\nTop 5 largest Sales Qty rows (candidate outliers):")
    cols = [config.COL_PRODUCT9, config.COL_COUNTRY, config.COL_ORDER_DATE,
            config.COL_SALES_QTY, config.COL_SALES_KG, config.COL_SALES_USD]
    out(df.nlargest(5, config.COL_SALES_QTY)[cols].to_string())

    # ---- 5. categoricals --------------------------------------------------- #
    out(section("5. CATEGORICAL FIELDS"))
    for col in [config.COL_CHANNEL, config.COL_REGION,
                config.COL_BRAND, config.COL_RANGE]:
        out(f"\n--- {col} (top 15) ---")
        out(profiling.category_value_counts(df, col, top=15).to_string())

    # ---- 6. sparsity / Pareto --------------------------------------------- #
    out(section("6. PRODUCT CONCENTRATION (the headline finding)"))
    for k, v in profiling.sparsity_headline(df).items():
        out(f"  {k:>28}: {v:,}" if isinstance(v, int) else f"  {k:>28}: {v}")

    # ---- 7. dates ---------------------------------------------------------- #
    out(section("7. DATE CONSISTENCY"))
    for k, v in profiling.date_consistency_report(df).items():
        out(f"  {k:>26}: {v}")
    out(f"\nDecision: bucket demand into weeks using '{config.DEMAND_DATE_COL}', "
        f"anchored {config.WEEK_ANCHOR}.")

    # ---- charts ------------------------------------------------------------ #
    out(section("SAVING CHARTS to outputs/figures/"))
    _, _, p1 = profiling.plot_pareto(df)
    _, _, p2 = profiling.plot_txn_distribution(df)
    _, _, p3 = profiling.plot_weekly_total(df)
    for p in (p1, p2, p3):
        out(f"  saved: {p}")

    # ---- save the text report --------------------------------------------- #
    report_path = config.OUTPUTS_DIR / "step1_data_understanding_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    out(section("DONE"))
    out(f"Text report saved to: {report_path}")
    out("Open the charts in outputs/figures/ to view them.")


if __name__ == "__main__":
    main()
