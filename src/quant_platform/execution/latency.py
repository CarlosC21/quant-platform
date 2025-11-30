from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from quant_platform.execution.models import MarketDataSnapshot, Order


class BaseLatencyModel(BaseModel, ABC):
    """Abstract base class for latency models.

    A latency model returns a non-negative delay (in seconds) between
    order submission and effective time of execution.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @abstractmethod
    def sample_delay(
        self,
        order: Order,
        snapshot: MarketDataSnapshot | None = None,
        context: dict[str, Any] | None = None,
    ) -> float:
        """Sample a latency (in seconds) for the given order/market state.

        Parameters
        ----------
        order
            Order being submitted.
        snapshot
            Optional market snapshot at submission time.
        context
            Optional additional information (venue stats, network, etc.).

        Returns
        -------
        float
            Non-negative latency in seconds.
        """
        raise NotImplementedError


class FixedLatencyModel(BaseLatencyModel):
    """Deterministic, constant latency model."""

    delay_seconds: float = Field(..., ge=0.0)

    def sample_delay(
        self,
        order: Order,
        snapshot: MarketDataSnapshot | None = None,
        context: dict[str, Any] | None = None,
    ) -> float:
        return self.delay_seconds


class NormalLatencyModel(BaseLatencyModel):
    """Gaussian latency model with clipping at zero.

    Notes
    -----
    For now, tests use only deterministic cases (`std_seconds = 0.0`)
    to ensure fully reproducible behavior without having to manage RNG state.
    """

    mean_seconds: float = Field(..., description="Mean latency in seconds.")
    std_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Standard deviation of latency in seconds.",
    )

    def sample_delay(
        self,
        order: Order,
        snapshot: MarketDataSnapshot | None = None,
        context: dict[str, Any] | None = None,
    ) -> float:
        if self.std_seconds == 0.0:
            # Fully deterministic for tests
            delay = self.mean_seconds
        else:
            delay = float(
                np.random.normal(loc=self.mean_seconds, scale=self.std_seconds)
            )

        # Enforce non-negativity
        return max(delay, 0.0)
