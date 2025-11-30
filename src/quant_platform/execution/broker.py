from __future__ import annotations

from typing import Dict, List, Tuple, Optional

from pydantic import BaseModel, ConfigDict, Field

from quant_platform.execution.costs import BaseCostModel
from quant_platform.execution.models import (
    ExecutionReport,
    Fill,
    MarketDataSnapshot,
    Order,
)
from quant_platform.execution.slippage import BaseSlippageModel, NoSlippageModel
from quant_platform.execution.simulator import ExecutionSimulator
from quant_platform.execution.trade_log import PositionState, TradeLog, TradeLogEntry
from quant_platform.execution.ledger import TradeLedger  # <-- added import


class BrokerConfig(BaseModel):
    """Configuration for the Broker."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    default_slippage_model: BaseSlippageModel = Field(
        default_factory=NoSlippageModel,
        description="Default slippage model used when none provided explicitly.",
    )
    default_cost_model: BaseCostModel | None = Field(
        default=None,
        description="Optional default transaction cost model.",
    )


class Broker(BaseModel):
    """Simple in-memory broker abstraction on top of the execution simulator.

    Responsibilities:
    - Submit orders to ExecutionSimulator
    - Log fills into a TradeLog
    - Maintain per-symbol positions with average-cost realized PnL
    - Optionally write fills into a shared global ledger (Week 11 feature)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    config: BrokerConfig = Field(default_factory=BrokerConfig)
    simulator: ExecutionSimulator = Field(default_factory=ExecutionSimulator)
    trade_log: TradeLog = Field(default_factory=TradeLog)
    positions: Dict[str, PositionState] = Field(default_factory=dict)

    # NEW: optional shared ledger to support multi-venue environments
    ledger: Optional[TradeLedger] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute_order(
        self,
        order: Order,
        snapshot: MarketDataSnapshot,
        slippage_model: BaseSlippageModel | None = None,
        cost_model: BaseCostModel | None = None,
    ) -> Tuple[ExecutionReport, List[Fill]]:
        """Execute an order synchronously and update trade log & positions."""

        if slippage_model is None:
            slippage_model = self.config.default_slippage_model
        if cost_model is None:
            cost_model = self.config.default_cost_model

        report, fills = self.simulator.simulate_order(
            order=order,
            snapshot=snapshot,
            slippage_model=slippage_model,
            cost_model=cost_model,
        )

        for fill in fills:
            # Update trade log (existing)
            entry = TradeLogEntry.from_fill(fill)
            self.trade_log.append(entry)

            # Update per-symbol positions (existing)
            self._update_position_with_fill(fill)

            # NEW: record into shared ledger (optional)
            if self.ledger is not None:
                self.ledger.record_fill(fill)

        return report, fills

    def get_position(self, symbol: str) -> PositionState:
        """Return current position for a symbol (zeroed if unseen)."""
        if symbol not in self.positions:
            self.positions[symbol] = PositionState(symbol=symbol)
        return self.positions[symbol]

    def total_realized_pnl(self) -> float:
        """Total realized PnL aggregated across symbols."""
        return sum(p.realized_pnl for p in self.positions.values())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_position_with_fill(self, fill: Fill) -> None:
        """Average-cost position + realized PnL update from a new fill."""

        symbol = fill.symbol
        pos = self.get_position(symbol)

        trade_qty_signed = fill.quantity * fill.side.sign
        price = fill.price

        position_qty = pos.quantity
        avg_cost = pos.avg_cost
        realized_pnl = pos.realized_pnl

        # No existing position
        if position_qty == 0.0:
            new_pos = PositionState(
                symbol=symbol,
                quantity=trade_qty_signed,
                avg_cost=price if trade_qty_signed != 0 else 0.0,
                realized_pnl=realized_pnl,
            )
            self.positions[symbol] = new_pos
            return

        pos_sign = 1.0 if position_qty > 0 else -1.0
        trade_sign = 1.0 if trade_qty_signed > 0 else -1.0

        # Averaging in
        if pos_sign == trade_sign:
            new_qty = position_qty + trade_qty_signed
            if new_qty == 0.0:
                new_avg_cost = 0.0
            else:
                new_avg_cost = (
                    position_qty * avg_cost + trade_qty_signed * price
                ) / new_qty

            new_pos = PositionState(
                symbol=symbol,
                quantity=new_qty,
                avg_cost=new_avg_cost,
                realized_pnl=realized_pnl,
            )
            self.positions[symbol] = new_pos
            return

        # Closing/fllipping logic
        closed_qty = min(abs(position_qty), abs(trade_qty_signed))
        realized_pnl += (price - avg_cost) * closed_qty * pos_sign
        remaining_qty = position_qty + trade_qty_signed

        if remaining_qty == 0.0:
            new_pos = PositionState(
                symbol=symbol,
                quantity=0.0,
                avg_cost=0.0,
                realized_pnl=realized_pnl,
            )
        else:
            new_sign = 1.0 if remaining_qty > 0 else -1.0
            if new_sign == pos_sign:
                new_pos = PositionState(
                    symbol=symbol,
                    quantity=remaining_qty,
                    avg_cost=avg_cost,
                    realized_pnl=realized_pnl,
                )
            else:
                new_pos = PositionState(
                    symbol=symbol,
                    quantity=remaining_qty,
                    avg_cost=price,
                    realized_pnl=realized_pnl,
                )

        self.positions[symbol] = new_pos
