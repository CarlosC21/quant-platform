from __future__ import annotations

from datetime import datetime, timezone


from quant_platform.execution.routing import VenueRouter, VenueConfig
from quant_platform.execution.broker import Broker
from quant_platform.execution.enums import Side, OrderType
from quant_platform.execution.models import Order, MarketDataSnapshot


def snapshot(symbol: str, price: float) -> MarketDataSnapshot:
    return MarketDataSnapshot(
        symbol=symbol,
        timestamp=datetime(2025, 1, 1, 9, 0, 0, tzinfo=timezone.utc),
        mid_price=price,
    )


def test_best_price_routing_buy():
    r = VenueRouter()

    r.add_venue(VenueConfig(venue_id="V1", broker=Broker(), priority=0))
    r.add_venue(VenueConfig(venue_id="V2", broker=Broker(), priority=0))

    order = Order(
        order_id="rt1",
        symbol="AAPL",
        side=Side.BUY,
        quantity=50,
        order_type=OrderType.MARKET,
        timestamp=datetime.now(timezone.utc),
    )

    snaps = {
        "V1": snapshot("AAPL", 101.0),
        "V2": snapshot("AAPL", 100.0),  # better for BUY
    }

    report, fills, decision = r.execute(order, snaps)

    assert decision.venue_id == "V2"


def test_best_price_routing_sell():
    r = VenueRouter()

    r.add_venue(VenueConfig(venue_id="V1", broker=Broker(), priority=0))
    r.add_venue(VenueConfig(venue_id="V2", broker=Broker(), priority=0))

    order = Order(
        order_id="rt2",
        symbol="AAPL",
        side=Side.SELL,
        quantity=50,
        order_type=OrderType.MARKET,
        timestamp=datetime.now(timezone.utc),
    )

    snaps = {
        "V1": snapshot("AAPL", 100.0),
        "V2": snapshot("AAPL", 101.0),  # better for SELL
    }

    report, fills, decision = r.execute(order, snaps)

    assert decision.venue_id == "V2"
