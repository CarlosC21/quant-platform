# src/quant_platform/trading/stat_arb/spreads/zscore.py
from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd

from quant_platform.trading.stat_arb.spreads.ou_model import OUParams
from quant_platform.trading.stat_arb.spreads.schemas import (
    StaticSpreadResult,
    KalmanSpreadResult,
    ZScoreResult,
)

SpreadResult = Union[StaticSpreadResult, KalmanSpreadResult]


def _as_series(spread_result: SpreadResult) -> pd.Series:
    """
    Convert spread result to a pandas Series indexed by timestamps.
    """
    return pd.Series(
        spread_result.spread,
        index=spread_result.timestamps,
        name=f"{spread_result.symbol_y}-{spread_result.symbol_x}-spread",
    )


def zscore_from_ou(
    spread_result: SpreadResult,
    ou_params: OUParams,
) -> ZScoreResult:
    """
    Compute OU-based stationary z-scores:

        z_t = (s_t - theta) / sigma_s,

    where:
        s_t        : spread_t,
        theta      : OU long-run mean,
        sigma_s    : OU stationary std = sigma / sqrt(2 * kappa)

    Parameters
    ----------
    spread_result : StaticSpreadResult | KalmanSpreadResult
        Spread and timestamps.
    ou_params : OUParams
        OU parameters for the spread.

    Returns
    -------
    ZScoreResult
    """
    s = np.asarray(spread_result.spread, dtype=float)
    if s.ndim != 1:
        raise ValueError("Spread must be 1D for z-score computation.")

    sigma_s = ou_params.stationary_std
    if sigma_s <= 0.0:
        raise ValueError("Stationary std is non-positive; cannot compute z-scores.")

    z = (s - ou_params.theta) / sigma_s

    return ZScoreResult(
        symbol_y=spread_result.symbol_y,
        symbol_x=spread_result.symbol_x,
        zscore=z,
        spread=s,
        timestamps=spread_result.timestamps,
        method="ou_stationary",
        window=None,
    )


def zscore_rolling(
    spread_result: SpreadResult,
    window: int,
    min_periods: Optional[int] = None,
) -> ZScoreResult:
    """
    Compute rolling z-scores using rolling mean and std:

        z_t = (s_t - m_t) / sd_t,

    where m_t and sd_t are windowed estimates.

    Parameters
    ----------
    spread_result : StaticSpreadResult | KalmanSpreadResult
        Spread and timestamps.
    window : int
        Rolling window length (in observations).
    min_periods : int, optional
        Minimum number of observations to compute stats. If None, defaults to window.

    Returns
    -------
    ZScoreResult
    """
    if window <= 1:
        raise ValueError("Rolling window must be > 1.")

    series = _as_series(spread_result)
    mp = min_periods if min_periods is not None else window

    rolling_mean = series.rolling(window=window, min_periods=mp).mean()
    rolling_std = series.rolling(window=window, min_periods=mp).std(ddof=1)

    # To avoid division by zero, where std==0 we set z = 0 (or NaN).
    std_values = rolling_std.to_numpy()
    mean_values = rolling_mean.to_numpy()
    s_values = series.to_numpy()

    z = np.full_like(s_values, fill_value=np.nan, dtype=float)

    mask = np.isfinite(std_values) & (std_values > 0.0)
    z[mask] = (s_values[mask] - mean_values[mask]) / std_values[mask]

    return ZScoreResult(
        symbol_y=spread_result.symbol_y,
        symbol_x=spread_result.symbol_x,
        zscore=z,
        spread=s_values,
        timestamps=spread_result.timestamps,
        method="rolling",
        window=window,
    )
