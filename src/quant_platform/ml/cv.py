"""Time-series aware walk-forward / rolling cross-validation with embargo."""
from __future__ import annotations
from typing import Iterable, Tuple
import numpy as np
import pandas as pd


def _unique_sorted_dates(ts: pd.Series) -> pd.Series:
    """Return unique sorted dates normalized to midnight."""
    return (
        pd.to_datetime(ts)
        .dt.normalize()
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )


def generate_time_series_splits(
    timestamps: pd.Series | pd.DatetimeIndex,
    n_splits: int = 5,
    cv_type: str = "expanding",
    train_window: int | None = None,
    val_window: int | None = None,
    embargo_days: int = 0,
) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    """
    Yield (train_indices, val_indices) preserving temporal order.

    Parameters
    ----------
    timestamps : pd.Series or pd.DatetimeIndex
        Time index aligned with samples.
    n_splits : int
        Number of folds.
    cv_type : str
        'expanding' or 'rolling' (currently only expanding implemented).
    train_window : int | None
        Optional max training window length.
    val_window : int | None
        Validation window length. Default: roughly n_dates / (n_splits+1)
    embargo_days : int
        Number of days to remove from training immediately after validation.

    Yields
    ------
    train_idx, val_idx : Tuple[np.ndarray, np.ndarray]
        Integer indices for training and validation.
    """
    # normalize timestamps
    if isinstance(timestamps, pd.DatetimeIndex):
        times = pd.Series(timestamps)
    else:
        times = pd.to_datetime(timestamps).reset_index(drop=True)

    uniq = _unique_sorted_dates(times)
    n_dates = len(uniq)

    if val_window is None:
        val_window = max(1, n_dates // (n_splits + 1))

    # start index for validation windows
    start = train_window if train_window is not None else val_window

    for i in range(n_splits):
        val_start_idx = start + i * val_window
        val_end_idx = val_start_idx + val_window
        if val_end_idx > n_dates:
            break  # don't overflow

        val_dates = uniq.iloc[val_start_idx:val_end_idx]
        train_dates = uniq.iloc[:val_start_idx]

        # get integer positions
        train_idx = np.where(times.isin(train_dates))[0]
        val_idx = np.where(times.isin(val_dates))[0]

        # apply embargo
        if embargo_days > 0:
            max_val = max(val_idx)
            train_idx = train_idx[train_idx < max_val - embargo_days + 1]

        yield train_idx, val_idx
