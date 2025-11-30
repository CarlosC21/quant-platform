# src/quant_platform/trading/stat_arb/cointegration/schemas.py
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class PairDefinition(BaseModel):
    """
    Definition of a trading pair.

    symbol_y : dependent asset (e.g. stock to be hedged)
    symbol_x : independent asset (e.g. hedge asset)
    """

    symbol_y: str = Field(..., description="Dependent asset (Y).")
    symbol_x: str = Field(..., description="Independent asset (X).")


class EngleGrangerConfig(BaseModel):
    """
    Configuration for Engle–Granger cointegration test.

    We use a minimal ADF(1) test on residuals with a constant only.
    """

    significance_level: float = Field(
        0.05,
        gt=0.0,
        lt=1.0,
        description="Significance level used to decide cointegration.",
    )
    min_obs: int = Field(
        50,
        ge=20,
        description="Minimum number of observations required for the test.",
    )


class ADFResult(BaseModel):
    """
    Result of the minimal ADF(1) test on residuals.

    test_stat : t-statistic of gamma in Δe_t = gamma * e_{t-1} + ε_t
    crit_1, crit_5, crit_10 : critical values for 1%, 5%, 10%
    p_value : coarse p-value bucket (optional, approximate)
    """

    test_stat: float
    crit_1: float
    crit_5: float
    crit_10: float
    p_value: Optional[float] = Field(
        default=None,
        description="Approximate p-value (very coarse, for logging only).",
    )

    @property
    def stationary_at_5pct(self) -> bool:
        """
        Return True if the null of a unit root is rejected at 5% level.
        For ADF, more negative statistics indicate stronger rejection.
        """
        return self.test_stat < self.crit_5


class CointegrationResult(BaseModel):
    """
    Summary of Engle–Granger cointegration test for a pair (Y, X).
    """

    symbol_y: str
    symbol_x: str
    beta: float = Field(..., description="Static hedge ratio from OLS regression.")
    alpha: float = Field(..., description="Intercept from OLS regression.")
    adf_result: ADFResult
    coint: bool = Field(..., description="True if series are cointegrated.")
    method: Literal["engle_granger"] = "engle_granger"
    n_obs: int = Field(..., ge=1, description="Number of observations used.")
