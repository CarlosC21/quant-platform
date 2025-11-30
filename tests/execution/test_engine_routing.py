from __future__ import annotations

from datetime import datetime, timezone

from quant_platform.execution.engine import ExecutionEngine
from quant_platform.execution.models import Order, MarketDataSnapshot
from quant_platform.execution.enums import Side, OrderType
from quant_platform.execution.latency import FixedLatencyModel
from quant_platform.execution.routing import (
    VenueRouter,
    VenueConfig,
)
from quant_platform.execution.broker import Broker


def _order():
    return Order(
        order_id="rt1",
        symbol="AAPL",
        side=Side.BUY,
        quantity=50,
        order_type=OrderType.MARKET,
        timestamp=datetime.now(timezone.utc),
    )


def _snap(mid):
    return MarketDataSnapshot(
        symbol="AAPL",
        timestamp=datetime.now(timezone.utc),
        mid_price=mid,
    )


def test_routing_selects_best_venue():
    order = _order()

    snapshots = {
        "V1": _snap(101.0),
        "V2": _snap(99.5),  # CHEAPEST → should be selected
        "V3": _snap(100.2),
    }

    router = VenueRouter(
        venues={
            "V1": VenueConfig(venue_id="V1", broker=Broker()),
            "V2": VenueConfig(venue_id="V2", broker=Broker()),
            "V3": VenueConfig(venue_id="V3", broker=Broker()),
        }
    )

    engine = ExecutionEngine(
        latency_model=FixedLatencyModel(delay_seconds=0.3),
        router=router,
    )

    events = list(engine.process_order(order, snapshots))

    # ROUTED event exists
    routed = [e for e in events if e.type == "ROUTED"]
    assert len(routed) == 1
    assert routed[0].metadata["venue"] == "V2"


def test_routing_event_sequence():
    order = _order()
    snapshots = {
        "X": _snap(100.0),
        "Y": _snap(99.0),
    }

    router = VenueRouter(
        venues={
            "X": VenueConfig(venue_id="X", broker=Broker()),
            "Y": VenueConfig(venue_id="Y", broker=Broker()),
        }
    )
    engine = ExecutionEngine(
        latency_model=FixedLatencyModel(delay_seconds=0.5),
        router=router,
    )

    events = list(engine.process_order(order, snapshots))
    types = [e.type for e in events]

    # sequence must be EXACTLY:
    # SUBMITTED → ROUTED → ACK → FILL → FINAL
    assert types == ["SUBMITTED", "ROUTED", "ACK", "FILL", "FINAL"]


def test_routed_fill_uses_correct_venue():
    order = _order()

    # VGOOD has best price → should be picked
    snapshots = {
        "VBAD": _snap(100.0),
        "VGOOD": _snap(95.0),
    }

    router = VenueRouter(
        venues={
            "VBAD": VenueConfig(venue_id="VBAD", broker=Broker()),
            "VGOOD": VenueConfig(venue_id="VGOOD", broker=Broker()),
        }
    )

    engine = ExecutionEngine(
        latency_model=FixedLatencyModel(delay_seconds=0.2),
        router=router,
    )

    events = list(engine.process_order(order, snapshots))

    # Grab fill event
    fill_event = next(e for e in events if e.type == "FILL")

    assert fill_event.metadata["venue"] == "VGOOD"

    # price must come from VGOOD snapshot
    assert fill_event.fills[0].price == 95.0


def test_routing_preserves_latency():
    order = _order()

    snapshots = {
        "A": _snap(100.0),
        "B": _snap(99.0),
    }

    router = VenueRouter(
        venues={
            "A": VenueConfig(venue_id="A", broker=Broker()),
            "B": VenueConfig(venue_id="B", broker=Broker()),
        }
    )

    engine = ExecutionEngine(
        latency_model=FixedLatencyModel(delay_seconds=0.5),
        router=router,
    )

    events = list(engine.process_order(order, snapshots))

    # SUBMITTED ts
    submitted_ts = order.timestamp.timestamp()

    ack_event = next(e for e in events if e.type == "ACK")
    expected_ts = submitted_ts + 0.5

    assert abs(ack_event.timestamp - expected_ts) < 1e-6
