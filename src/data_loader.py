"""
data_loader.py
==============
Load the raw cinnamon export spreadsheet and apply *light* type coercion only.

STEP 1 RULE: this module does NOT clean the data. It does not drop rows, impute
values, cap outliers, or alter quantities. It only:

  * reads the xlsx,
  * coerces date and numeric columns to their proper dtypes (recording, not
    fixing, anything that fails to parse),
  * derives the 9-character product key.

All actual cleaning happens later in ``preprocessing.py`` (step 2). Keeping
loading and cleaning separate means the profiling in step 1 reflects the data
*as delivered*.
"""

from __future__ import annotations

import pandas as pd

from . import config


def load_raw(path=None, coerce: bool = True) -> pd.DataFrame:
    """Read the raw spreadsheet into a DataFrame.

    Parameters
    ----------
    path
        Override the default raw-file path (``config.RAW_FILE``).
    coerce
        If True (default), parse date and numeric columns and add the product
        key. If False, return the spreadsheet exactly as pandas reads it
        (every column as object/string) -- useful for inspecting raw values.

    Returns
    -------
    pandas.DataFrame
    """
    path = path or config.RAW_FILE
    df = pd.read_excel(path)

    if not coerce:
        return df

    return coerce_types(df)


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce dates -> datetime, sales fields -> numeric, add product key.

    Values that cannot be parsed become ``NaT``/``NaN`` (via ``errors='coerce'``)
    rather than raising, so the profiling step can *measure* how many bad values
    exist instead of crashing on them.
    """
    df = df.copy()

    for col in config.DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    for col in config.NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Customer Code is an identifier, not a quantity -- keep it as a string so
    # it is never accidentally summed or averaged.
    if config.COL_CUSTOMER_CODE in df.columns:
        df[config.COL_CUSTOMER_CODE] = df[config.COL_CUSTOMER_CODE].astype("string")

    # Product key = first N characters of the full Product Code.
    if config.COL_PRODUCT_CODE in df.columns:
        df[config.COL_PRODUCT9] = (
            df[config.COL_PRODUCT_CODE]
            .astype("string")
            .str[: config.PRODUCT_KEY_LEN]
        )

    return df
