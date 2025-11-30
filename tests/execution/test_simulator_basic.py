from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from quant_platform.execution.enums import OrderStatus, OrderType, Side
from quant_platform.execution.models import MarketDataSnapshot, Order
from quant_platform.execution.simulator import ExecutionSimulator
from quant_platform.execution.slippage import NoSlippageModel


def _ts(offset_seconds: int = 0) -> datetime:
    return datetime(2024, 1, 1, 10, 0, 0) + timedelta(seconds=offset_seconds)


def test_market_order_filled_at_mid_with_no_slippage():
    sim = ExecutionSimulator()
    order = Order(
        order_id="o1",
        symbol="AAPL",
        side=Side.BUY,
        quantity=100.0,
        order_type=OrderType.MARKET,
        timestamp=_ts(0),
    )
    snapshot = MarketDataSnapshot(
        symbol="AAPL",
        timestamp=_ts(1),
        mid_price=150.0,
    )

    report, fills = sim.simulate_order(
        order=order,
        snapshot=snapshot,
        slippage_model=NoSlippageModel(),
    )

    assert report.status == OrderStatus.FILLED
    assert report.filled_quantity == pytest.approx(100.0)
    assert report.avg_price == pytest.approx(150.0)
    assert len(fills) == 1
    fill = fills[0]
    assert fill.price == pytest.approx(150.0)
    assert fill.quantity == pytest.approx(100.0)


def test_limit_buy_not_filled_if_price_above_limit():
    sim = ExecutionSimulator()
    order = Order(
        order_id="o2",
        symbol="MSFT",
        side=Side.BUY,
        quantity=50.0,
        order_type=OrderType.LIMIT,
        limit_price=300.0,
        timestamp=_ts(0),
    )
    snapshot = MarketDataSnapshot(
        symbol="MSFT",
        timestamp=_ts(1),
        mid_price=305.0,
    )

    report, fills = sim.simulate_order(order=order, snapshot=snapshot)

    assert report.status in {OrderStatus.NEW, OrderStatus.CANCELLED}
    assert report.filled_quantity == pytest.approx(0.0)
    assert report.avg_price is None
    assert fills == []


def test_limit_buy_filled_when_mid_below_limit():
    sim = ExecutionSimulator()
    order = Order(
        order_id="o3",
        symbol="MSFT",
        side=Side.BUY,
        quantity=50.0,
        order_type=OrderType.LIMIT,
        limit_price=300.0,
        timestamp=_ts(0),
    )
    snapshot = MarketDataSnapshot(
        symbol="MSFT",
        timestamp=_ts(1),
        mid_price=295.0,
    )

    report, fills = sim.simulate_order(
        order=order,
        snapshot=snapshot,
        slippage_model=NoSlippageModel(),
    )

    assert report.status == OrderStatus.FILLED
    assert report.filled_quantity == pytest.approx(50.0)
    assert report.avg_price == pytest.approx(295.0)
    assert len(fills) == 1
    assert fills[0].price == pytest.approx(295.0)


def test_limit_sell_uses_bid_when_available():
    sim = ExecutionSimulator()
    order = Order(
        order_id="o4",
        symbol="SPY",
        side=Side.SELL,
        quantity=10.0,
        order_type=OrderType.LIMIT,
        limit_price=450.0,
        timestamp=_ts(0),
    )
    snapshot = MarketDataSnapshot(
        symbol="SPY",
        timestamp=_ts(1),
        mid_price=449.0,
        bid_price=451.0,
        ask_price=452.0,
    )

    report, fills = sim.simulate_order(
        order=order,
        snapshot=snapshot,
        slippage_model=NoSlippageModel(),
    )

    # We only care that the order is filled given bid >= limit.
    assert report.status == OrderStatus.FILLED
    assert len(fills) == 1
    assert fills[0].quantity == pytest.approx(10.0)


def test_symbol_mismatch_raises():
    sim = ExecutionSimulator()
    order = Order(
        order_id="o5",
        symbol="AAPL",
        side=Side.BUY,
        quantity=10.0,
        order_type=OrderType.MARKET,
        timestamp=_ts(0),
    )
    snapshot = MarketDataSnapshot(
        symbol="MSFT",
        timestamp=_ts(1),
        mid_price=100.0,
    )

    with pytest.raises(ValueError):
        sim.simulate_order(order=order, snapshot=snapshot)
