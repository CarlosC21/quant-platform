# src/quant_platform/portfolio/schemas.py
from __future__ import annotations

from typing import List, Dict, Optional, Any

import numpy as np
from pydantic import BaseModel, Field, ConfigDict


class AssetWeight(BaseModel):
    """
    Weight assigned to a single asset in a portfolio.
    """

    symbol: str = Field(..., description="Asset identifier (e.g. 'AAPL').")
    weight: float = Field(..., description="Portfolio weight (can be negative).")


class PortfolioConfig(BaseModel):
    """
    Generic configuration for portfolio construction.
    """

    symbols: List[str] = Field(..., description="List of assets in the universe.")
    allow_short: bool = Field(
        False, description="If False, enforce non-negative weights."
    )
    leverage_limit: float = Field(
        1.0,
        gt=0.0,
        description="Maximum L1 norm of weights (e.g. 1.0 for fully invested).",
    )
    target_return: Optional[float] = Field(
        None,
        description="Optional target portfolio return for mean-variance optimization.",
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)


class RiskModelInput(BaseModel):
    """
    Input container for portfolio optimizers.
    """

    mu: np.ndarray = Field(..., description="Expected returns vector (N,).")
    cov: np.ndarray = Field(..., description="Covariance matrix (N, N).")
    symbols: List[str] = Field(..., description="Corresponding asset symbols.")
    model_config = ConfigDict(arbitrary_types_allowed=True)


class PortfolioResult(BaseModel):
    """
    Output of a portfolio optimizer.
    """

    weights: List[AssetWeight] = Field(..., description="Optimized weights.")
    expected_return: float = Field(..., description="w^T mu.")
    expected_vol: float = Field(..., description="sqrt(w^T Σ w).")
    sharpe: Optional[float] = Field(
        None, description="Optional Sharpe ratio (mu / sigma)."
    )
    meta: Dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(arbitrary_types_allowed=True)
