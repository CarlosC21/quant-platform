from __future__ import annotations

from typing import Dict, List, Tuple

from pydantic import BaseModel, ConfigDict, Field

from quant_platform.execution.broker import Broker
from quant_platform.execution.models import (
    Order,
    MarketDataSnapshot,
    ExecutionReport,
    Fill,
)


class VenueConfig(BaseModel):
    """Represents the configuration of a single venue."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    venue_id: str
    broker: Broker
    priority: int = 0  # lower = preferred


class RoutingDecision(BaseModel):
    """Chosen venue for an order."""

    model_config = ConfigDict(extra="forbid")

    venue_id: str
    reason: str


class VenueRouter(BaseModel):
    """
    Simple router implementing:

        - Best-price routing
        - Priority routing fallback
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    venues: Dict[str, VenueConfig] = Field(default_factory=dict)

    def add_venue(self, venue: VenueConfig) -> None:
        self.venues[venue.venue_id] = venue

    # ------------------------------------------------------------------
    # Decision logic
    # ------------------------------------------------------------------

    def choose_venue(
        self,
        order: Order,
        snapshots: Dict[str, MarketDataSnapshot],
    ) -> RoutingDecision:
        """
        Select the best venue based on market conditions.

        Logic:
        1. Filter venues that have snapshots for the symbol.
        2. Choose the best price depending on BUY vs SELL.
        3. If tie → choose lowest priority.
        """

        candidates: List[Tuple[str, MarketDataSnapshot, VenueConfig]] = []

        for venue_id, snap in snapshots.items():
            if venue_id in self.venues and snap.symbol == order.symbol:
                candidates.append((venue_id, snap, self.venues[venue_id]))

        if not candidates:
            raise ValueError("No valid venues for order")

        if order.side.is_buy():
            # Best ask → lowest mid_price
            best = min(candidates, key=lambda c: (c[1].mid_price, c[2].priority))
        else:
            # Best bid → highest mid_price
            best = max(candidates, key=lambda c: (c[1].mid_price, -c[2].priority))

        best_venue_id, _, _ = best

        return RoutingDecision(
            venue_id=best_venue_id,
            reason="best-price",
        )

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(
        self,
        order: Order,
        snapshots: Dict[str, MarketDataSnapshot],
    ) -> Tuple[ExecutionReport, List[Fill], RoutingDecision]:
        """
        Route an order to the chosen venue and execute it.

        Returns:
            (report, fills, routing_decision)
        """

        decision = self.choose_venue(order, snapshots)
        venue_id = decision.venue_id
        venue_config = self.venues[venue_id]

        broker = venue_config.broker
        snapshot = snapshots[venue_id]

        report, fills = broker.execute_order(order, snapshot)

        return report, fills, decision
