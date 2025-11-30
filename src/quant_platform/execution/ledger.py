from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from quant_platform.execution.enums import Side, Venue
from quant_platform.execution.models import Fill


class TradeRecord(BaseModel):
    """Normalized trade record derived from a Fill.

    This is the atomic building block for the ledger and position book.
    """

    model_config = ConfigDict(extra="forbid")

    order_id: str
    symbol: str
    side: Side
    quantity: float = Field(..., gt=0.0)
    price: float = Field(..., gt=0.0)
    timestamp: datetime
    venue: Venue = Venue.SIMULATED
    cost: float = Field(default=0.0, ge=0.0)

    @property
    def signed_quantity(self) -> float:
        """Signed quantity (positive for BUY, negative for SELL)."""
        return self.side.sign * self.quantity

    @property
    def notional(self) -> float:
        """Trade notional (price * quantity)."""
        return self.price * self.quantity

    @classmethod
    def from_fill(cls, fill: Fill) -> "TradeRecord":
        """Convert an execution Fill into a TradeRecord."""
        return cls(
            order_id=fill.order_id,
            symbol=fill.symbol,
            side=fill.side,
            quantity=fill.quantity,
            price=fill.price,
            timestamp=fill.timestamp,
            venue=fill.venue,
            cost=fill.cost,
        )


class Position(BaseModel):
    """Represents the position state for a single symbol.

    Quantities are stored as signed:
      > 0  → long
      < 0  → short
    """

    model_config = ConfigDict(extra="forbid")

    symbol: str
    quantity: float = 0.0  # signed
    avg_price: float | None = None  # cost basis of OPEN position
    realized_pnl: float = 0.0  # gross, before fees
    fees: float = 0.0  # accumulated transaction costs/fees

    def apply_fill(self, trade: TradeRecord) -> None:
        """Update position given a new trade using FIFO-like cost basis.

        Logic:
        - If flat → new position opened at trade.price.
        - If same side → average in using notional-weighted average.
        - If opposite side → realize PnL on the closed portion; if trade
          crosses through flat, remaining opens a new position at trade.price.
        """
        # Update fee accumulator (kept separate from realized_pnl)
        self.fees += trade.cost

        q = self.quantity
        dq = trade.signed_quantity
        p = trade.price

        # Flat → new position
        if q == 0.0:
            self.quantity = dq
            self.avg_price = p
            return

        # Same direction (adding to existing position)
        if q * dq > 0.0:
            q_abs = abs(q)
            dq_abs = abs(dq)
            new_qty_abs = q_abs + dq_abs
            assert self.avg_price is not None  # should hold if q != 0

            new_avg = (self.avg_price * q_abs + p * dq_abs) / new_qty_abs
            self.quantity = q + dq
            self.avg_price = new_avg
            return

        # Opposite direction: closing or flipping
        # Existing position sign determines realized PnL direction
        sign_q = 1.0 if q > 0 else -1.0
        q_abs = abs(q)
        dq_abs = abs(dq)
        closed_qty = min(q_abs, dq_abs)

        assert self.avg_price is not None

        # Realized PnL on closed portion
        # Long (q>0):  (p - avg_price) * closed_qty
        # Short(q<0):  (avg_price - p) * closed_qty
        self.realized_pnl += (p - self.avg_price) * closed_qty * sign_q

        # Remaining signed quantity after applying trade
        new_q = q + dq

        if new_q == 0.0:
            # Fully closed
            self.quantity = 0.0
            self.avg_price = None
        else:
            # If abs(dq) < abs(q): partial close, keep old avg_price
            # If abs(dq) > abs(q): cross through zero and open new pos
            if dq_abs <= q_abs:
                # Partial close only
                self.quantity = new_q
                # avg_price unchanged
            else:
                # Crossed through flat → remaining opens at trade price
                self.quantity = new_q
                self.avg_price = p

    def market_value(self, last_price: float) -> float:
        """Current market value of the position at a given price."""
        return self.quantity * last_price

    def unrealized_pnl(self, last_price: float) -> float:
        """Unrealized PnL at the given price (gross, before fees)."""
        if self.quantity == 0.0 or self.avg_price is None:
            return 0.0
        return (last_price - self.avg_price) * self.quantity

    @property
    def net_realized_pnl(self) -> float:
        """Realized PnL after subtracting fees."""
        return self.realized_pnl - self.fees


class PositionBook(BaseModel):
    """Book of positions + cash account."""

    model_config = ConfigDict(extra="forbid")

    positions: Dict[str, Position] = Field(default_factory=dict)
    cash: float = 0.0

    def get_position(self, symbol: str) -> Position:
        """Get or create a Position for the given symbol."""
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        return self.positions[symbol]

    def apply_trade(self, trade: TradeRecord) -> None:
        """Apply trade to cash + position.

        Cash convention:
        - BUY  → cash -= notional + cost
        - SELL → cash += notional - cost

        Fees (cost) are also accumulated in the Position.
        """
        pos = self.get_position(trade.symbol)

        notional = trade.notional
        if trade.side is Side.BUY:
            self.cash -= notional
            self.cash -= trade.cost
        else:
            self.cash += notional
            self.cash -= trade.cost

        pos.apply_fill(trade)

    def portfolio_value(self, prices: Dict[str, float]) -> float:
        """Total portfolio value = cash + sum(position market values)."""
        total = self.cash
        for symbol, pos in self.positions.items():
            if symbol in prices:
                total += pos.market_value(prices[symbol])
        return total

    @property
    def total_realized_pnl(self) -> float:
        """Sum of realized PnL across all symbols (gross)."""
        return sum(pos.realized_pnl for pos in self.positions.values())

    @property
    def total_fees(self) -> float:
        """Sum of all transaction costs across all positions."""
        return sum(pos.fees for pos in self.positions.values())

    @property
    def total_net_realized_pnl(self) -> float:
        """Realized PnL after fees."""
        return sum(pos.net_realized_pnl for pos in self.positions.values())


class TradeLedger(BaseModel):
    """Execution ledger that consumes Fills and maintains a PositionBook."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    trades: List[TradeRecord] = Field(default_factory=list)
    position_book: PositionBook = Field(default_factory=PositionBook)

    def record_fill(self, fill: Fill) -> TradeRecord:
        """Record a new fill and update positions + cash."""
        trade = TradeRecord.from_fill(fill)
        self.trades.append(trade)
        self.position_book.apply_trade(trade)
        return trade

    def get_position(self, symbol: str) -> Optional[Position]:
        return self.position_book.positions.get(symbol)

    @property
    def cash(self) -> float:
        return self.position_book.cash

    @property
    def total_realized_pnl(self) -> float:
        return self.position_book.total_realized_pnl

    @property
    def total_fees(self) -> float:
        return self.position_book.total_fees

    @property
    def total_net_realized_pnl(self) -> float:
        return self.position_book.total_net_realized_pnl

    def portfolio_value(self, prices: Dict[str, float]) -> float:
        return self.position_book.portfolio_value(prices)
