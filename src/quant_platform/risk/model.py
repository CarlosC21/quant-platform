# src/quant_platform/risk/model.py
from __future__ import annotations

from typing import Literal, Optional

import numpy as np
from pydantic import BaseModel, Field

from quant_platform.risk.covariance import (
    compute_sample_covariance,
    compute_ewma_covariance,
    compute_ledoit_wolf_covariance,
)


class RiskModelConfig(BaseModel):
    """
    Configuration for covariance estimation.
    """

    method: Literal["sample", "ewma", "ledoit_wolf"] = Field(
        ..., description="Covariance estimation method."
    )

    # EWMA parameter
    lambda_decay: Optional[float] = Field(
        None, description="Decay factor for EWMA. Required if method='ewma'."
    )

    # Ledoit-Wolf parameter
    assume_centered: bool = Field(
        False, description="Assume data is centered for Ledoit-Wolf."
    )


class CovarianceRiskModel:
    """
    Unified risk model interface for portfolio optimization.

    Usage:
        rm = CovarianceRiskModel(config)
        cov = rm.compute_covariance(returns)
    """

    def __init__(self, config: RiskModelConfig):
        self.config = config

    def compute_covariance(self, returns: np.ndarray) -> np.ndarray:
        """
        Compute covariance matrix using the configured method.

        Parameters
        ----------
        returns : ndarray (T, N)
            Matrix of asset returns.

        Returns
        -------
        cov : ndarray (N, N)
            Covariance matrix.
        """
        method = self.config.method

        if method == "sample":
            return compute_sample_covariance(returns)

        elif method == "ewma":
            if self.config.lambda_decay is None:
                raise ValueError("lambda_decay must be set for EWMA risk model.")
            return compute_ewma_covariance(returns, self.config.lambda_decay)

        elif method == "ledoit_wolf":
            return compute_ledoit_wolf_covariance(
                returns,
                assume_centered=self.config.assume_centered,
            )

        else:
            raise ValueError(f"Unknown covariance method: {method}")
