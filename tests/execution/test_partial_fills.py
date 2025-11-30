from __future__ import annotations

from datetime import datetime, timezone

import pytest

from quant_platform.execution.enums import Side, OrderType
from quant_platform.execution.models import Order
from quant_platform.execution.models_orderbook import OrderBookSnapshot, PriceLevel
from quant_platform.execution.simulator import ExecutionSimulator


def test_market_order_partial_fill_buy():
    """
    Market BUY 120 units, but book provides:

    ask levels:
      101 @ 50
      102 @ 50
      103 @ 50

    Should fill:
       50 @ 101
       50 @ 102
       20 @ 103
    """
    sim = ExecutionSimulator()

    order = Order(
        order_id="pf1",
        symbol="AAPL",
        side=Side.BUY,
        quantity=120,
        order_type=OrderType.MARKET,
        timestamp=datetime.now(timezone.utc),
    )

    book = OrderBookSnapshot(
        symbol="AAPL",
        timestamp=datetime.now(timezone.utc),
        bids=[PriceLevel(price=100, size=50)],
        asks=[
            PriceLevel(price=101, size=50),
            PriceLevel(price=102, size=50),
            PriceLevel(price=103, size=50),
        ],
    )

    report, fills = sim.simulate_order_with_depth(order, book)

    assert report.filled_quantity == 120
    assert report.remaining_quantity == 0
    assert len(fills) == 3

    prices = [f.price for f in fills]
    quantities = [f.quantity for f in fills]

    assert prices == [101, 102, 103]
    assert quantities == [50, 50, 20]

    avg_price = (101 * 50 + 102 * 50 + 103 * 20) / 120
    assert report.avg_price == pytest.approx(avg_price)


def test_limit_sell_partial_fill():
    """
    Limit SELL 80 @ 101

    Book:
       bid levels:
         100 @ 30
         101 @ 40
         102 @ 50  ← won't hit this because limit is 101

    Should fill:
        30 @ 100
        40 @ 101
    remaining = 10
    """
    sim = ExecutionSimulator()

    order = Order(
        order_id="pf2",
        symbol="AAPL",
        side=Side.SELL,
        quantity=80,
        order_type=OrderType.LIMIT,
        limit_price=101,
        timestamp=datetime.now(timezone.utc),
    )

    book = OrderBookSnapshot(
        symbol="AAPL",
        timestamp=datetime.now(timezone.utc),
        bids=[
            PriceLevel(price=100, size=30),
            PriceLevel(price=101, size=40),
            PriceLevel(price=102, size=50),
        ],
        asks=[],
    )

    report, fills = sim.simulate_order_with_depth(order, book)

    assert report.filled_quantity == 70
    assert report.remaining_quantity == 10

    prices = [f.price for f in fills]
    quantities = [f.quantity for f in fills]

    assert prices == [100, 101]
    assert quantities == [30, 40]

    avg_price = (100 * 30 + 101 * 40) / 70
    assert report.avg_price == pytest.approx(avg_price)
