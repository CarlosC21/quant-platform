# src/quant_platform/risk/metrics.py
from __future__ import annotations

from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd

from quant_platform.risk.schemas import DrawdownStats


def compute_drawdown(
    equity_curve: np.ndarray,
    timestamps: np.ndarray,
) -> Tuple[np.ndarray, DrawdownStats]:
    equity = np.asarray(equity_curve, dtype=float)
    if equity.ndim != 1 or equity.size == 0:
        raise ValueError("equity_curve must be a non-empty 1D array.")
    if np.any(equity <= 0.0):
        raise ValueError("equity_curve must contain strictly positive values.")

    ts = np.asarray(timestamps)
    if ts.shape[0] != equity.shape[0]:
        raise ValueError("timestamps length must match equity_curve length.")

    # Convert timestamps to Python datetime robustly
    ts_py = [
        t if isinstance(t, datetime) else pd.to_datetime(t).to_pydatetime() for t in ts
    ]

    running_max = np.maximum.accumulate(equity)
    drawdown = equity / running_max - 1.0

    max_dd = float(drawdown.min())
    dd_end_idx = int(np.argmin(drawdown))
    dd_end_time = ts_py[dd_end_idx]

    # Peak (start of drawdown)
    peak_idx = int(np.argmax(equity[: dd_end_idx + 1]))
    peak_time = ts_py[peak_idx]
    peak_value = equity[peak_idx]

    # Recovery detection
    recovery_idx = None
    for i in range(peak_idx + 1, equity.shape[0]):
        if equity[i] >= peak_value:
            recovery_idx = i
            break

    ts_np = ts.astype("datetime64[ns]")

    if recovery_idx is None:
        time_under_water_days = None
    else:
        delta_days = (ts_np[recovery_idx] - ts_np[peak_idx]) / np.timedelta64(1, "D")
        time_under_water_days = int(delta_days)

    stats = DrawdownStats(
        start=ts_py[0],
        end=ts_py[-1],
        max_drawdown=max_dd,
        max_drawdown_start=peak_time,
        max_drawdown_end=dd_end_time,
        time_under_water_days=time_under_water_days,
    )

    return drawdown, stats
