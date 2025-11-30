from __future__ import annotations

from datetime import datetime, timezone

import pytest

from quant_platform.execution.costs import ProportionalCostModel
from quant_platform.execution.enums import Side, OrderType
from quant_platform.execution.models import MarketDataSnapshot, Order
from quant_platform.execution.simulator import ExecutionSimulator

timestamp = datetime.now(timezone.utc)


def test_market_order_with_costs():
    sim = ExecutionSimulator()
    order = Order(
        order_id="tc1",
        symbol="AAPL",
        side=Side.BUY,
        quantity=100,
        order_type=OrderType.MARKET,
        timestamp=datetime.now(timezone.utc),
    )
    snapshot = MarketDataSnapshot(
        symbol="AAPL",
        timestamp=datetime.now(timezone.utc),
        mid_price=100.0,
    )
    cost_model = ProportionalCostModel(commission_bps=10.0)

    report, fills = sim.simulate_order(
        order=order,
        snapshot=snapshot,
        cost_model=cost_model,
    )

    fill = fills[0]
    assert fill.cost == pytest.approx(100 * 100 * 0.001)
