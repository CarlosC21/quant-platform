from __future__ import annotations

from datetime import datetime, timezone, timedelta

from quant_platform.execution.engine import ExecutionEngine
from quant_platform.execution.latency import FixedLatencyModel
from quant_platform.execution.models import Order, MarketDataSnapshot
from quant_platform.execution.enums import Side, OrderType


def _order():
    return Order(
        order_id="e1",
        symbol="AAPL",
        side=Side.BUY,
        quantity=100,
        order_type=OrderType.MARKET,
        timestamp=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
    )


def _snapshot():
    return MarketDataSnapshot(
        symbol="AAPL",
        timestamp=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        mid_price=100.0,
    )


def test_engine_fixed_latency_order_of_events():
    order = _order()
    snapshot = _snapshot()

    engine = ExecutionEngine(
        latency_model=FixedLatencyModel(delay_seconds=0.5),
    )

    events = list(engine.process_order(order, snapshot))

    # Should emit SUBMITTED → ACK → FILL → FINAL
    assert [e.type for e in events] == ["SUBMITTED", "ACK", "FILL", "FINAL"]

    submitted_ts = order.timestamp
    expected_exec_ts = submitted_ts + timedelta(seconds=0.5)

    assert events[1].timestamp == expected_exec_ts  # ACK timestamp
    assert events[2].timestamp == expected_exec_ts  # FILL timestamp
    assert events[3].timestamp == expected_exec_ts  # FINAL timestamp
