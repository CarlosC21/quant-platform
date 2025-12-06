from __future__ import annotations

from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, ConfigDict, Field

from quant_platform.execution.engine import ExecutionEngine, EngineEvent
from quant_platform.execution.ledger import TradeLedger
from quant_platform.execution.models import Order, MarketDataSnapshot
from quant_platform.execution.broker import Broker
from quant_platform.execution.routing import VenueRouter


class ExecutionContext(BaseModel):
    """
    High-level trading context bundling execution engine + ledger + routing.

    Week 12 note:
    -------------
    This object is the main interface between strategies / runners and the
    execution stack (engine, broker, ledger, routing).

    We extend it with convenience methods like `get_portfolio_snapshot`
    so that backtests and UIs can easily query the state of the book
    (cash, positions, equity, PnL) in a consistent way.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    # Starting capital
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
    def model_post_init(self, __context: Any) -> None:
        # Seed initial cash into the ledger's position book
        if hasattr(self.ledger, "position_book"):
            # Only set cash if ledger has not been initialized
            if getattr(self.ledger.position_book, "cash", None) in (0, 0.0, None):
                self.ledger.position_book.cash = float(self.initial_cash)

        # Default engine broker uses ledger
        self.engine.broker.ledger = self.ledger

        # Routing / multi-venue
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
        """
        Submit an order + market snapshot to the execution engine and
        update the shared ledger / positions.

        Returns
        -------
        events : list[EngineEvent]
            Engine lifecycle events (ACK, FILL, etc).
        position_book : object
            The underlying position book from the ledger. Callers that
            need detailed state should use `get_portfolio_snapshot`.
        """
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
        # TradeLedger exposes total realized PnL via its position book
        return self.ledger.position_book.total_realized_pnl

    def last_report(self, order_id: str):
        return self.ledger.last_report(order_id)

    def last_fills(self, order_id: str):
        return self.ledger.last_fills(order_id)

    def position(self, symbol: str):
        return self.ledger.position_book.positions.get(symbol)

    # ---------------------------------------------------------------
    # Week 12: Portfolio snapshot API
    # ---------------------------------------------------------------
    def get_portfolio_snapshot(self, prices: Dict[str, float]) -> Dict[str, Any]:
        """
        Compute a normalized portfolio snapshot from the ledger.

        Parameters
        ----------
        prices : dict[str, float]
            Mapping symbol -> last/mark price for this bar.

        Returns
        -------
        dict
            {
                "cash": float,
                "positions": { symbol: quantity },
                "market_value": float,
                "total_equity": float,
                "unrealized_pnl": float,
                "realized_pnl": float,
            }

        Notes
        -----
        - Uses the TradeLedger's PositionBook as the single source of truth.
        - Does not mutate state; pure read of current ledger.
        """
        pb = self.ledger.position_book

        # 1) Cash
        cash = float(pb.cash)

        # 2) Symbol -> quantity
        positions = {sym: float(pos.quantity) for sym, pos in pb.positions.items()}

        # 3) Market value & unrealized PnL
        market_value = 0.0
        unrealized = 0.0

        for sym, pos in pb.positions.items():
            if sym not in prices:
                continue
            px = float(prices[sym])
            market_value += pos.market_value(px)
            unrealized += pos.unrealized_pnl(px)

        # 4) Realized PnL (gross)
        realized = float(pb.total_realized_pnl)

        # 5) Total equity
        total_equity = cash + market_value

        return {
            "cash": cash,
            "positions": positions,
            "market_value": market_value,
            "total_equity": total_equity,
            "unrealized_pnl": unrealized,
            "realized_pnl": realized,
        }


# DEBUG: confirm path + method presence
print("LOADED ExecutionContext FROM:", __file__)
print("ExecutionContext has execute:", hasattr(ExecutionContext, "execute"))
print(
    "ExecutionContext has get_portfolio_snapshot:",
    hasattr(ExecutionContext, "get_portfolio_snapshot"),
)
