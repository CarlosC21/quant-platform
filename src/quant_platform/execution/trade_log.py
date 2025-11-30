from __future__ import annotations

from datetime import datetime
from typing import Dict, List

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from quant_platform.execution.enums import Side
from quant_platform.execution.models import Fill


class TradeLogEntry(BaseModel):
    """Canonical trade log record derived from a Fill."""

    model_config = ConfigDict(extra="forbid")

    order_id: str
    symbol: str
    side: Side
    quantity: float = Field(..., gt=0.0)
    price: float = Field(..., gt=0.0)
    cost: float = Field(default=0.0, ge=0.0)
    timestamp: datetime

    @classmethod
    def from_fill(cls, fill: Fill) -> "TradeLogEntry":
        return cls(
            order_id=fill.order_id,
            symbol=fill.symbol,
            side=fill.side,
            quantity=fill.quantity,
            price=fill.price,
            cost=fill.cost,
            timestamp=fill.timestamp,
        )


class PositionState(BaseModel):
    """Per-symbol position & realized PnL in average-cost convention."""

    model_config = ConfigDict(extra="forbid")

    symbol: str
    quantity: float = 0.0  # signed: >0 long, <0 short
    avg_cost: float = 0.0  # average entry price for current open position
    realized_pnl: float = 0.0

    def copy_update(
        self,
        *,
        quantity: float | None = None,
        avg_cost: float | None = None,
        realized_pnl: float | None = None,
    ) -> "PositionState":
        return PositionState(
            symbol=self.symbol,
            quantity=self.quantity if quantity is None else quantity,
            avg_cost=self.avg_cost if avg_cost is None else avg_cost,
            realized_pnl=self.realized_pnl if realized_pnl is None else realized_pnl,
        )


class TradeLog(BaseModel):
    """In-memory trade log + helper to export as DataFrame."""

    model_config = ConfigDict(extra="forbid")

    entries: List[TradeLogEntry] = Field(default_factory=list)

    def append(self, entry: TradeLogEntry) -> None:
        self.entries.append(entry)

    def to_dataframe(self) -> pd.DataFrame:
        if not self.entries:
            return pd.DataFrame(
                columns=[
                    "order_id",
                    "symbol",
                    "side",
                    "quantity",
                    "price",
                    "cost",
                    "timestamp",
                ]
            )
        return pd.DataFrame([e.model_dump() for e in self.entries])

    def by_symbol(self) -> Dict[str, List[TradeLogEntry]]:
        grouped: Dict[str, List[TradeLogEntry]] = {}
        for e in self.entries:
            grouped.setdefault(e.symbol, []).append(e)
        return grouped
