# src/quant_platform/trading/stat_arb/spreads/spread_builder.py
from __future__ import annotations

import pandas as pd

from quant_platform.trading.stat_arb.spreads.schemas import StaticSpreadResult


def build_static_spread(
    series_y: pd.Series,
    series_x: pd.Series,
    beta: float,
) -> StaticSpreadResult:
    """
    Construct spread using static hedge ratio:

        s_t = y_t - beta * x_t

    Parameters
    ----------
    series_y : pd.Series
        Dependent asset Y with datetime index.
    series_x : pd.Series
        Independent asset X with datetime index.
    beta : float
        Hedge ratio from Engle–Granger OLS regression.

    Returns
    -------
    StaticSpreadResult
    """
    # Align on timestamp intersection
    joined = pd.concat([series_y, series_x], axis=1, join="inner").dropna()
    if joined.shape[0] < 3:
        raise ValueError("Not enough aligned observations to compute spread.")

    y = joined.iloc[:, 0].to_numpy(dtype=float)
    x = joined.iloc[:, 1].to_numpy(dtype=float)
    timestamps = joined.index.to_numpy()

    spread = y - beta * x

    return StaticSpreadResult(
        symbol_y=str(series_y.name),
        symbol_x=str(series_x.name),
        beta=float(beta),
        spread=spread,
        timestamps=timestamps,
    )
