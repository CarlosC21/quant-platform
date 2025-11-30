from __future__ import annotations

from abc import ABC, abstractmethod
from pydantic import BaseModel, ConfigDict, Field


class BaseCostModel(BaseModel, ABC):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @abstractmethod
    def compute_cost(self, price: float, quantity: float) -> float:
        raise NotImplementedError


class ProportionalCostModel(BaseCostModel):
    """Test-suite-defined cost:
    cost = notional * (commission_bps + fee_bps) / 1000
    """

    commission_bps: float = Field(default=0.0, ge=0.0)
    fee_bps: float = Field(default=0.0, ge=0.0)

    def compute_cost(self, price: float, quantity: float) -> float:
        notional = price * quantity
        # total_bps = self.commission_bps + self.fee_bps
        return notional * (self.commission_bps + self.fee_bps) / 10_000.0


class FixedCostModel(BaseCostModel):
    fixed_fee: float = Field(..., ge=0.0)

    def compute_cost(self, price: float, quantity: float) -> float:
        return self.fixed_fee


class HybridCostModel(BaseCostModel):
    commission_bps: float = Field(default=0.0, ge=0.0)
    fee_bps: float = Field(default=0.0, ge=0.0)
    fixed_fee: float = Field(default=0.0, ge=0.0)

    def compute_cost(self, price: float, quantity: float) -> float:
        notional = price * quantity
        proportional = notional * (self.commission_bps + self.fee_bps) / 10_000.0
        return proportional + self.fixed_fee
