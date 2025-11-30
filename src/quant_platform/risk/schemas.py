# src/quant_platform/risk/schemas.py
from __future__ import annotations

from datetime import datetime
from typing import Optional, List

from pydantic import BaseModel, ConfigDict, Field


class DrawdownStats(BaseModel):
    """
    Summary statistics for drawdown analysis on an equity curve.
    """

    start: datetime = Field(..., description="Start timestamp of the series.")
    end: datetime = Field(..., description="End timestamp of the series.")
    max_drawdown: float = Field(
        ...,
        description="Maximum drawdown as a fraction (e.g., -0.25 for -25%).",
    )
    max_drawdown_start: datetime = Field(
        ..., description="Timestamp of equity peak before max drawdown."
    )
    max_drawdown_end: datetime = Field(
        ..., description="Timestamp when max drawdown is realized."
    )
    time_under_water_days: Optional[int] = Field(
        default=None,
        description="Days between peak and full recovery (if recovery occurs).",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


# ============================================================
# ADDITIONAL RISK SCHEMAS FOR WEEK 10 (NON-BREAKING)
# ============================================================


class VaRResult(BaseModel):
    """
    Container for Value-at-Risk results.
    """

    alpha: float = Field(..., description="Tail probability (e.g. 0.05 for 95% VaR).")
    var: float = Field(..., description="Positive loss value for VaR.")
    method: str = Field(..., description="Method name (historical, parametric, MC).")


class CVaRResult(BaseModel):
    """
    Container for Conditional VaR (Expected Shortfall).
    """

    alpha: float = Field(..., description="Tail probability.")
    cvar: float = Field(..., description="Expected loss beyond VaR.")
    method: str = Field(..., description="Method name (historical, parametric, MC).")


class FactorExposure(BaseModel):
    """
    Factor exposure for a single asset.

    betas: k-factor loadings from regression.
    intercept: estimated alpha for asset.
    """

    symbol: str
    betas: List[float]
    intercept: float


class FactorModelResult(BaseModel):
    """
    PCA-based factor model output.

    factor_returns: T x K matrix
    loadings: N x K matrix
    specific_var: N-length vector
    explained_variance_ratio: K-length vector
    """

    n_factors: int
    factor_returns: List[List[float]]
    loadings: List[List[float]]
    specific_var: List[float]
    explained_variance_ratio: List[float]
