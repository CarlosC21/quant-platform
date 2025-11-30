from __future__ import annotations

from abc import ABC, abstractmethod
from math import sqrt
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from quant_platform.execution.enums import Side


class BaseSlippageModel(BaseModel, ABC):
    """Abstract base class for slippage models used by the simulator."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @abstractmethod
    def get_execution_price(
        self,
        mid_price: float,
        side: Side,
        quantity: float,
        daily_volume: Optional[float] = None,
        volatility: Optional[float] = None,
    ) -> float:
        """Return an execution price given the inputs.

        Parameters
        ----------
        mid_price:
            Current mid price.
        side:
            BUY or SELL.
        quantity:
            Order quantity (absolute).
        daily_volume:
            Optional daily volume for impact scaling.
        volatility:
            Optional volatility for more elaborate models.

        Returns
        -------
        float
            The execution price.
        """
        raise NotImplementedError


class NoSlippageModel(BaseSlippageModel):
    """Execution at mid price with no impact."""

    def get_execution_price(
        self,
        mid_price: float,
        side: Side,  # noqa: ARG002 - kept for a consistent signature
        quantity: float,  # noqa: ARG002
        daily_volume: Optional[float] = None,  # noqa: ARG002
        volatility: Optional[float] = None,  # noqa: ARG002
    ) -> float:
        if mid_price <= 0.0:
            msg = "mid_price must be positive"
            raise ValueError(msg)
        return mid_price


class LinearSlippageModel(BaseSlippageModel):
    """Linear price impact model.

    P_exec = P_mid * (1 + sign * kappa * q / V)

    If `daily_volume` is not provided, we interpret q/V as q itself (i.e. use
    the raw quantity as an impact proxy). This keeps the model usable in
    backtests without volume data.
    """

    kappa: float = Field(
        default=1e-4,
        gt=0.0,
        description="Impact coefficient for q/V (dimensionless).",
    )

    def get_execution_price(
        self,
        mid_price: float,
        side: Side,
        quantity: float,
        daily_volume: Optional[float] = None,
        volatility: Optional[float] = None,  # noqa: ARG002
    ) -> float:
        if mid_price <= 0.0:
            msg = "mid_price must be positive"
            raise ValueError(msg)
        if quantity <= 0.0:
            msg = "quantity must be positive"
            raise ValueError(msg)

        if daily_volume is not None and daily_volume > 0.0:
            participation = quantity / daily_volume
        else:
            participation = quantity

        impact = side.sign * self.kappa * participation
        price = mid_price * (1.0 + impact)
        if price <= 0.0:
            msg = "Computed execution price must be positive"
            raise ValueError(msg)
        return price


class SquareRootSlippageModel(BaseSlippageModel):
    """Square-root impact model.

    P_exec = P_mid * (1 + sign * kappa * sqrt(q / V))

    If `daily_volume` is not provided, we use sqrt(q) instead, similar to the
    linear model's fallback.
    """

    kappa: float = Field(
        default=1e-3,
        gt=0.0,
        description="Impact coefficient for sqrt(q/V).",
    )

    def get_execution_price(
        self,
        mid_price: float,
        side: Side,
        quantity: float,
        daily_volume: Optional[float] = None,
        volatility: Optional[float] = None,  # noqa: ARG002
    ) -> float:
        if mid_price <= 0.0:
            msg = "mid_price must be positive"
            raise ValueError(msg)
        if quantity <= 0.0:
            msg = "quantity must be positive"
            raise ValueError(msg)

        if daily_volume is not None and daily_volume > 0.0:
            participation = quantity / daily_volume
        else:
            participation = quantity

        if participation < 0.0:
            msg = "participation must be non-negative"
            raise ValueError(msg)

        impact = side.sign * self.kappa * sqrt(participation)
        price = mid_price * (1.0 + impact)
        if price <= 0.0:
            msg = "Computed execution price must be positive"
            raise ValueError(msg)
        return price
