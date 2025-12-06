"""
Enhanced data validation utilities for user-uploaded CSV market data.

Hard validations (raise errors):
    - Required columns
    - Timestamp parse failures
    - Missing symbol or price values

Soft validations (warnings):
    - Duplicate rows
    - Missing timestamps per symbol
    - Large outlier returns
    - Uneven symbol coverage
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"timestamp", "symbol", "close"}


class ValidationResult:
    """
    Container for validation results:
    - errors: fatal issues (raise)
    - warnings: soft issues (display but do not stop execution)
    """

    def __init__(self) -> None:
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def raise_if_errors(self) -> None:
        if self.errors:
            raise ValueError("\n".join(self.errors))


def _normalize_for_validation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure we always have 'timestamp' as a column for validation logic.

    If the user/data pipeline has already moved 'timestamp' to the index
    (index.name == 'timestamp'), we reset the index on a *copy* so that
    validation can see all REQUIRED_COLUMNS in df.columns.

    The original df passed into validate_market_data is never mutated.
    """
    if "timestamp" in df.columns:
        return df.copy()

    if df.index.name == "timestamp":
        return df.reset_index().copy()

    # Fall back: no visible timestamp; validation will catch this
    return df.copy()


def validate_market_data(df: pd.DataFrame) -> ValidationResult:
    """
    Validate the uploaded market data.

    Returns:
        ValidationResult with errors + warnings.

    Only fatal errors will stop execution. Warnings are informational.

    IMPORTANT:
        This function is *read-only* with respect to the input df.
        Any normalization (e.g. resetting index) is done on a copy.
    """
    result = ValidationResult()

    # Work on a normalized copy so 'timestamp' is always a column if possible
    df_check = _normalize_for_validation(df)

    # ============================================================
    # HARD VALIDATIONS
    # ============================================================

    # 1. Required columns
    present_cols = set(df_check.columns)
    missing = REQUIRED_COLUMNS - present_cols
    if missing:
        result.errors.append(
            f"❌ Missing required columns: {', '.join(sorted(missing))}.\n"
            f"Present columns: {list(df_check.columns)}"
        )
        result.raise_if_errors()

    # 2. Timestamp parse
    try:
        pd.to_datetime(df_check["timestamp"])
    except Exception:
        result.errors.append(
            "❌ Column 'timestamp' could not be parsed as a valid datetime.\n"
            "Use formats like '2023-01-01' or '2023-01-01 09:30:00'."
        )
        result.raise_if_errors()

    # 3. Missing symbol values
    if df_check["symbol"].isna().any():
        result.errors.append("❌ Column 'symbol' contains missing values.")
        result.raise_if_errors()

    # 4. Missing prices
    if df_check["close"].isna().any():
        n_missing = df_check["close"].isna().sum()
        result.errors.append(f"❌ 'close' contains {n_missing} missing price values.")
        result.raise_if_errors()

    # ============================================================
    # SOFT WARNINGS (non-fatal)
    # ============================================================

    # 5. Duplicate timestamp+symbol rows
    dup_count = df_check.duplicated(subset=["timestamp", "symbol"]).sum()
    if dup_count > 0:
        result.warnings.append(
            f"⚠️ Found {dup_count} duplicate (timestamp, symbol) rows. "
            "Only the first occurrence will be used."
        )

    # 6. Missing timestamps per symbol (irregular gaps)
    try:
        df_sorted = df_check.sort_values(["symbol", "timestamp"])
        gaps = {}

        for sym, g in df_sorted.groupby("symbol"):
            ts = pd.to_datetime(g["timestamp"])
            diffs = ts.diff().dt.total_seconds().dropna()
            # Consider gap if > 2× median gap
            if len(diffs) > 0:
                median_gap = float(np.median(diffs))
                large_gaps = (diffs > 2 * median_gap).sum()
                if large_gaps > 0:
                    gaps[sym] = int(large_gaps)

        if gaps:
            warn_msg = ", ".join([f"{sym}: {cnt} gaps" for sym, cnt in gaps.items()])
            result.warnings.append(f"⚠️ Irregular timestamp gaps detected → {warn_msg}")
    except Exception:
        # Never let soft diagnostics break validation
        pass

    # 7. Outlier return detection
    try:
        df_ret = df_check.sort_values(["symbol", "timestamp"]).copy()
        df_ret["return"] = df_ret.groupby("symbol")["close"].pct_change()
        outliers = df_ret["return"].abs() > 0.25  # >25% move
        outlier_count = int(outliers.sum())
        if outlier_count > 0:
            result.warnings.append(
                f"⚠️ Detected {outlier_count} large return outliers (>25%). "
                "Verify data quality."
            )
    except Exception:
        pass

    # 8. Symbol coverage (some symbols have fewer rows)
    try:
        counts = df_check.groupby("symbol")["timestamp"].count()
        if counts.max() - counts.min() > 5:  # arbitrary threshold
            symbol_counts = ", ".join([f"{sym}: {cnt}" for sym, cnt in counts.items()])
            result.warnings.append(
                f"⚠️ Uneven coverage across symbols → {symbol_counts}"
            )
    except Exception:
        pass

    return result
