"""
Data validation utilities for user-uploaded CSV market data.
Ensures friendly, actionable feedback for non-technical users.
"""

from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = {"timestamp", "symbol", "close"}


def validate_market_data(df: pd.DataFrame) -> None:
    """
    Validate that the uploaded market data contains all required fields
    and has correct formatting.
    Raises ValueError with a friendly, descriptive message if invalid.
    """

    # -----------------------------
    # Check required columns exist
    # -----------------------------
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"❌ Uploaded CSV is missing required columns: {', '.join(sorted(missing))}\n"
            f"Columns present: {list(df.columns)}"
        )

    # -----------------------------
    # Validate timestamp column
    # -----------------------------
    try:
        pd.to_datetime(df["timestamp"])
    except Exception:
        raise ValueError(
            "❌ Column 'timestamp' could not be parsed.\n"
            "Use formats such as '2023-01-01' or '2023-01-01 09:30:00'."
        )

    # -----------------------------
    # Validate symbols
    # -----------------------------
    if df["symbol"].isna().any():
        raise ValueError("❌ Column 'symbol' contains empty values.")

    # -----------------------------
    # Validate prices
    # -----------------------------
    if df["close"].isna().any():
        raise ValueError("❌ Column 'close' contains missing prices.")

    # Additional optional checks could go here:
    # - symbol character restrictions
    # - monotonic timestamp warnings
    # - duplicate timestamp+symbol rows
