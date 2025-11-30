import asyncio
from datetime import datetime, timezone, timedelta

import pytest

from quant_platform.execution.async_engine import AsyncExecutionEngine
from quant_platform.execution.latency import FixedLatencyModel
from quant_platform.execution.models import Order, MarketDataSnapshot
from quant_platform.execution.enums import Side, OrderType


@pytest.mark.asyncio
async def test_async_engine_basic_flow():
    engine = AsyncExecutionEngine(
        latency_model=FixedLatencyModel(delay_seconds=0.5),
    )

    # Snapshot provider returns a static snapshot per order
    def snapshot_provider(order):
        return MarketDataSnapshot(
            symbol=order.symbol,
            timestamp=order.timestamp,
            mid_price=100.0,
        )

    # Start engine loop (single iteration)
    async def engine_runner():
        await engine.run(snapshot_provider)

    task = asyncio.create_task(engine_runner())

    order = Order(
        order_id="ae1",
        symbol="AAPL",
        side=Side.BUY,
        quantity=100,
        order_type=OrderType.MARKET,
        timestamp=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
    )

    await engine.submit_order(order)

    # Collect 4 events: SUBMITTED, ACK, FILL, FINAL
    events = []
    for _ in range(4):
        evt = await asyncio.wait_for(engine.event_queue.get(), timeout=1.0)
        events.append(evt)

    assert [e.type for e in events] == ["SUBMITTED", "ACK", "FILL", "FINAL"]

    expected_ts = order.timestamp + timedelta(seconds=0.5)
    assert events[1].timestamp == expected_ts
    assert events[2].timestamp == expected_ts
    assert events[3].timestamp == expected_ts

    task.cancel()
