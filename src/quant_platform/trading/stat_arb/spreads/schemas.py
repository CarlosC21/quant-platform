# src/quant_platform/trading/stat_arb/spreads/schemas.py
from __future__ import annotations

from typing import Optional, Literal

import numpy as np
from pydantic import BaseModel, Field, ConfigDict


class StaticSpreadResult(BaseModel):
    """
    Spread series constructed using a static hedge ratio beta.

    spread_t = y_t - beta * x_t
    """

    symbol_y: str
    symbol_x: str
    beta: float = Field(..., description="Static hedge ratio from Engle–Granger.")
    spread: np.ndarray = Field(..., description="1D spread series.")
    timestamps: np.ndarray = Field(..., description="Aligned timestamps.")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class KalmanSpreadResult(BaseModel):
    """
    Spread series constructed using a dynamic hedge ratio beta_t
    estimated via a Kalman filter.

    spread_t = y_t - beta_t * x_t
    """

    symbol_y: str
    symbol_x: str
    beta_t: np.ndarray = Field(..., description="Time-varying hedge ratio path.")
    spread: np.ndarray = Field(..., description="1D spread series.")
    timestamps: np.ndarray = Field(..., description="Aligned timestamps.")
    alpha_t: Optional[np.ndarray] = Field(
        default=None, description="Optional time-varying intercept alpha_t."
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ZScoreResult(BaseModel):
    """
    Z-scores computed for a spread series.

    method:
        'ou_stationary' -> (spread_t - theta) / sigma_stationary
        'rolling'       -> (spread_t - rolling_mean) / rolling_std
    """

    symbol_y: str
    symbol_x: str
    zscore: np.ndarray = Field(..., description="1D array of z-scores.")
    spread: np.ndarray = Field(..., description="Original spread used.")
    timestamps: np.ndarray = Field(..., description="Aligned timestamps.")
    method: Literal["ou_stationary", "rolling"]
    window: Optional[int] = Field(
        default=None,
        description="Rolling window length for 'rolling' method.",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)
