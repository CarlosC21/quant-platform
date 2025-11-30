from __future__ import annotations

from typing import Dict, List, Tuple

from pydantic import BaseModel, ConfigDict, Field

from quant_platform.execution.engine import ExecutionEngine, EngineEvent
from quant_platform.execution.ledger import TradeLedger
from quant_platform.execution.models import Order, MarketDataSnapshot
from quant_platform.execution.broker import Broker
from quant_platform.execution.routing import VenueRouter


class ExecutionContext(BaseModel):
    """
    High-level trading context bundling:

        • ExecutionEngine  (latency + routing + simulator)
        • Brokers (one or many)
        • TradeLedger (persistent ledger used across venues)
        • PositionBook  (merged view across brokers)

    This is the primary interface used by strategies, backtests,
    the portfolio layer, and Week 12 dashboards.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    engine: ExecutionEngine = Field(default_factory=ExecutionEngine)
    ledger: TradeLedger = Field(default_factory=TradeLedger)

    # ---------------------------------------------------------------
    # Optional multi-venue setup
    # ---------------------------------------------------------------
    router: VenueRouter | None = None
    venue_brokers: Dict[str, Broker] = Field(default_factory=dict)

    # ---------------------------------------------------------------
    # Initialization logic
    # ---------------------------------------------------------------
    def model_post_init(self, __context):
        """
        After initialization, wire the ledger into:

            • default engine broker
            • any venue-specific brokers (routing)
        """

        # Default engine broker
        self.engine.broker.ledger = self.ledger

        # Routing brokers (if router present)
        if self.router is not None:
            for venue_id, cfg in self.router.venues.items():
                # Attach ledger to each venue broker
                cfg.broker.ledger = self.ledger
                self.venue_brokers[venue_id] = cfg.broker

            # Tell the engine to use the router
            self.engine.router = self.router

    # ---------------------------------------------------------------
    # EXECUTE ORDER
    # ---------------------------------------------------------------
    def execute(
        self,
        order: Order,
        snapshot: MarketDataSnapshot | Dict[str, MarketDataSnapshot],
    ) -> Tuple[List[EngineEvent], object]:
        """
        Execute an order across either:

            • Single venue (no routing), snapshot = MarketDataSnapshot
            • Multi-venue (routing enabled), snapshot = {venue_id: snapshot}

        Returns:
            events      = list of EngineEvent
            position_book = merged positions from ledger
        """

        # Ensure engine always uses latest ledger reference
        self.engine.broker.ledger = self.ledger

        # Router case: ensure each venue broker uses the same ledger
        if isinstance(snapshot, dict) and self.router:
            for venue_id, cfg in self.router.venues.items():
                cfg.broker.ledger = self.ledger

        events = list(self.engine.process_order(order, snapshot))

        # Return high-level state (positions, realized PnL, etc.)
        return events, self.ledger.position_book

    # ---------------------------------------------------------------
    # Convenience Accessors
    # ---------------------------------------------------------------
    def positions(self):
        """Merged view of positions across all venues (ledger is authoritative)."""
        return self.ledger.position_book

    def realized_pnl(self) -> float:
        return self.ledger.realized_pnl_total()

    def last_report(self, order_id: str):
        return self.ledger.last_report(order_id)

    def last_fills(self, order_id: str):
        return self.ledger.last_fills(order_id)

    def position(self, symbol: str):
        return self.ledger.position_book.get(symbol)
