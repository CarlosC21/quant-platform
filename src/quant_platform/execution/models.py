from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from quant_platform.execution.enums import (
    LiquiditySide,
    OrderStatus,
    OrderType,
    Side,
    TimeInForce,
    Venue,
)


class Order(BaseModel):
    """Order intent submitted to the execution engine/simulator.

    This is the *requested* order. Dynamic state is tracked via ExecutionReport.
    """

    model_config = ConfigDict(extra="forbid")

    order_id: str = Field(..., description="Unique order identifier.")
    symbol: str = Field(..., min_length=1)
    side: Side
    quantity: float = Field(..., gt=0.0, description="Absolute quantity, > 0.")
    order_type: OrderType
    limit_price: Optional[float] = Field(
        default=None,
        description="Limit price for LIMIT orders. Must be set for LIMIT orders.",
    )
    time_in_force: TimeInForce = TimeInForce.DAY
    venue: Venue = Venue.SIMULATED
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Submission timestamp in UTC.",
    )
    tags: Dict[str, Any] = Field(
        default_factory=dict,
        description="Free-form metadata (strategy, signal_id, etc.).",
    )

    def validate_for_simulation(self) -> None:
        """Perform static validation for the simulator.

        Raises
        ------
        ValueError
            If the order configuration is invalid.
        """
        if self.order_type is OrderType.LIMIT and self.limit_price is None:
            msg = "limit_price must be set for LIMIT orders"
            raise ValueError(msg)


class Fill(BaseModel):
    """Represents a single fill event (partial or full)."""

    model_config = ConfigDict(extra="forbid")

    order_id: str
    symbol: str
    side: Side
    quantity: float = Field(..., gt=0.0)
    price: float = Field(..., gt=0.0)
    timestamp: datetime
    liquidity_side: LiquiditySide
    venue: Venue = Venue.SIMULATED
    cost: float = Field(default=0.0, ge=0.0)


class ExecutionReport(BaseModel):
    """Aggregated execution state for an order at a point in time."""

    model_config = ConfigDict(extra="forbid")

    order_id: str
    symbol: str
    side: Side
    status: OrderStatus
    requested_quantity: float = Field(..., gt=0.0)
    filled_quantity: float = Field(..., ge=0.0)
    avg_price: Optional[float] = Field(
        default=None,
        description="Volume-weighted average price across all fills.",
    )
    last_fill: Optional[Fill] = None
    venue: Venue = Venue.SIMULATED
    time_in_force: TimeInForce
    created_at: datetime
    updated_at: datetime

    @property
    def remaining_quantity(self) -> float:
        """Remaining quantity to be filled."""
        return max(self.requested_quantity - self.filled_quantity, 0.0)

    @property
    def is_terminal(self) -> bool:
        """Whether the order is in a terminal state."""
        return self.status in {
            OrderStatus.FILLED,
            OrderStatus.CANCELLED,
            OrderStatus.REJECTED,
        }


class MarketDataSnapshot(BaseModel):
    """Minimal market data snapshot required by the simulator.

    This is intentionally light-weight so we can drive the simulator from
    backtests using bar data (mid prices, spreads, etc.).
    """

    model_config = ConfigDict(extra="forbid")

    symbol: str
    timestamp: datetime
    mid_price: float = Field(..., gt=0.0)
    bid_price: Optional[float] = Field(default=None, gt=0.0)
    ask_price: Optional[float] = Field(default=None, gt=0.0)
    spread: Optional[float] = Field(
        default=None,
        description="Optional explicit spread; if None, infer from bid/ask if available.",
    )

    @classmethod
    def from_bar(
        cls,
        symbol: str,
        ts: datetime | pd.Timestamp,
        close: float,
        spread_bps: float | None = None,
    ) -> "MarketDataSnapshot":
        """Construct a snapshot from bar/close data + an assumed spread.

        Parameters
        ----------
        symbol:
            Instrument identifier.
        ts:
            Timestamp (datetime or pandas.Timestamp).
        close:
            Close or mid price.
        spread_bps:
            Optional spread in basis points; used only to populate `spread`.
        """
        dt = ts.to_pydatetime() if isinstance(ts, pd.Timestamp) else ts
        spread = None
        if spread_bps is not None:
            spread = close * spread_bps / 10_000.0

        return cls(
            symbol=symbol,
            timestamp=dt,
            mid_price=close,
            spread=spread,
        )
