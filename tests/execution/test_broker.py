from __future__ import annotations

from datetime import datetime, timezone

import pytest

from quant_platform.execution.broker import Broker
from quant_platform.execution.enums import OrderType, Side
from quant_platform.execution.models import MarketDataSnapshot, Order


def _snapshot(
    symbol: str, mid: float, ts: datetime | None = None
) -> MarketDataSnapshot:
    return MarketDataSnapshot(
        symbol=symbol,
        timestamp=ts or datetime.now(timezone.utc),
        mid_price=mid,
    )


def test_broker_executes_and_logs_single_trade():
    broker = Broker()

    order = Order(
        order_id="b1",
        symbol="AAPL",
        side=Side.BUY,
        quantity=100.0,
        order_type=OrderType.MARKET,
        timestamp=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
    )
    snapshot = _snapshot("AAPL", mid=100.0, ts=order.timestamp)

    report, fills = broker.execute_order(order, snapshot)

    # Single full fill at 100
    assert report.filled_quantity == pytest.approx(100.0)
    assert report.avg_price == pytest.approx(100.0)

    # Trade log updated
    assert len(broker.trade_log.entries) == 1
    entry = broker.trade_log.entries[0]
    assert entry.symbol == "AAPL"
    assert entry.side is Side.BUY
    assert entry.quantity == pytest.approx(100.0)
    assert entry.price == pytest.approx(100.0)

    # Position updated
    pos = broker.get_position("AAPL")
    assert pos.quantity == pytest.approx(100.0)
    assert pos.avg_cost == pytest.approx(100.0)
    assert pos.realized_pnl == pytest.approx(0.0)


def test_broker_realized_pnl_and_position_avg_cost():
    broker = Broker()

    # Long 100 @ 100
    ts1 = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    buy_order = Order(
        order_id="b2_buy",
        symbol="AAPL",
        side=Side.BUY,
        quantity=100.0,
        order_type=OrderType.MARKET,
        timestamp=ts1,
    )
    buy_snapshot = _snapshot("AAPL", mid=100.0, ts=ts1)

    broker.execute_order(buy_order, buy_snapshot)

    # Sell 40 @ 110
    ts2 = datetime(2025, 1, 1, 12, 5, 0, tzinfo=timezone.utc)
    sell_order = Order(
        order_id="b2_sell",
        symbol="AAPL",
        side=Side.SELL,
        quantity=40.0,
        order_type=OrderType.MARKET,
        timestamp=ts2,
    )
    sell_snapshot = _snapshot("AAPL", mid=110.0, ts=ts2)

    broker.execute_order(sell_order, sell_snapshot)

    pos = broker.get_position("AAPL")

    # 100 long, then sell 40 → 60 remaining
    assert pos.quantity == pytest.approx(60.0)
    # Remaining avg cost stays at original 100
    assert pos.avg_cost == pytest.approx(100.0)
    # Realized PnL: (110 - 100) * 40 = 400
    assert pos.realized_pnl == pytest.approx(400.0)

    # Total realized PnL across broker matches
    assert broker.total_realized_pnl() == pytest.approx(400.0)


def test_broker_flip_position_crosses_through_flat():
    broker = Broker()

    # Long 50 @ 100
    ts1 = datetime(2025, 1, 1, 13, 0, 0, tzinfo=timezone.utc)
    buy_order = Order(
        order_id="b3_buy",
        symbol="AAPL",
        side=Side.BUY,
        quantity=50.0,
        order_type=OrderType.MARKET,
        timestamp=ts1,
    )
    buy_snapshot = _snapshot("AAPL", mid=100.0, ts=ts1)
    broker.execute_order(buy_order, buy_snapshot)

    # Sell 100 @ 90 → close 50 long at 90 (loss), open 50 short @ 90
    ts2 = datetime(2025, 1, 1, 13, 10, 0, tzinfo=timezone.utc)
    sell_order = Order(
        order_id="b3_sell",
        symbol="AAPL",
        side=Side.SELL,
        quantity=100.0,
        order_type=OrderType.MARKET,
        timestamp=ts2,
    )
    sell_snapshot = _snapshot("AAPL", mid=90.0, ts=ts2)
    broker.execute_order(sell_order, sell_snapshot)

    pos = broker.get_position("AAPL")

    # Now short 50
    assert pos.quantity == pytest.approx(-50.0)
    # New short opened at 90
    assert pos.avg_cost == pytest.approx(90.0)
    # Realized PnL on closed 50 long: (90 - 100) * 50 = -500
    assert pos.realized_pnl == pytest.approx(-500.0)
