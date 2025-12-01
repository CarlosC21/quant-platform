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
    High-level trading context bundling execution engine + ledger + routing.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    # NEW: starting capital
    initial_cash: float = 100_000.0

    # Core components
    engine: ExecutionEngine = Field(default_factory=ExecutionEngine)
    ledger: TradeLedger = Field(default_factory=TradeLedger)

    # Optional router / multi-venue
    router: VenueRouter | None = None
    venue_brokers: Dict[str, Broker] = Field(default_factory=dict)

    # ---------------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------------
    def model_post_init(self, __context):
        # Inject initial cash
        if hasattr(self.ledger, "position_book"):
            self.ledger.position_book.cash = float(self.initial_cash)

        # Default engine broker uses ledger
        self.engine.broker.ledger = self.ledger

        # Routing/brokers
        if self.router is not None:
            for venue_id, cfg in self.router.venues.items():
                cfg.broker.ledger = self.ledger
                self.venue_brokers[venue_id] = cfg.broker
            self.engine.router = self.router

    # ---------------------------------------------------------------
    # EXECUTE
    # ---------------------------------------------------------------
    def execute(
        self,
        order: Order,
        snapshot: MarketDataSnapshot | Dict[str, MarketDataSnapshot],
    ) -> Tuple[List[EngineEvent], object]:
        # Always sync ledger
        self.engine.broker.ledger = self.ledger

        # Multi-venue snapshot dict
        if isinstance(snapshot, dict) and self.router:
            for venue_id, cfg in self.router.venues.items():
                cfg.broker.ledger = self.ledger

        events = list(self.engine.process_order(order, snapshot))

        # Return events + merged positions
        return events, self.ledger.position_book

    # ---------------------------------------------------------------
    # ACCESSORS
    # ---------------------------------------------------------------
    def positions(self):
        return self.ledger.position_book

    def realized_pnl(self) -> float:
        return self.ledger.total_realized_pnl

    def last_report(self, order_id: str):
        return self.ledger.last_report(order_id)

    def last_fills(self, order_id: str):
        return self.ledger.last_fills(order_id)

    def position(self, symbol: str):
        return self.ledger.position_book.positions.get(symbol)


# DEBUG: confirm path + method presence
print("LOADED ExecutionContext FROM:", __file__)
print("ExecutionContext has execute:", hasattr(ExecutionContext, "execute"))
