from __future__ import annotations

from datetime import datetime, timezone

import pytest

from quant_platform.execution.enums import OrderType, Side
from quant_platform.execution.latency import FixedLatencyModel, NormalLatencyModel
from quant_platform.execution.models import MarketDataSnapshot, Order


def _make_dummy_order() -> Order:
    return Order(
        order_id="lat1",
        symbol="AAPL",
        side=Side.BUY,
        quantity=100.0,
        order_type=OrderType.MARKET,
        timestamp=datetime.now(timezone.utc),
    )


def _make_dummy_snapshot() -> MarketDataSnapshot:
    return MarketDataSnapshot(
        symbol="AAPL",
        timestamp=datetime.now(timezone.utc),
        mid_price=100.0,
    )


def test_fixed_latency_model_returns_constant():
    order = _make_dummy_order()
    snapshot = _make_dummy_snapshot()

    model = FixedLatencyModel(delay_seconds=0.25)
    d1 = model.sample_delay(order, snapshot)
    d2 = model.sample_delay(order, snapshot)

    assert d1 == pytest.approx(0.25)
    assert d2 == pytest.approx(0.25)


def test_normal_latency_zero_std_is_deterministic():
    order = _make_dummy_order()
    snapshot = _make_dummy_snapshot()

    model = NormalLatencyModel(mean_seconds=0.1, std_seconds=0.0)
    d1 = model.sample_delay(order, snapshot)
    d2 = model.sample_delay(order, snapshot)

    assert d1 == pytest.approx(0.1)
    assert d2 == pytest.approx(0.1)


def test_normal_latency_clipped_at_zero():
    order = _make_dummy_order()
    snapshot = _make_dummy_snapshot()

    # Negative mean + zero std → deterministically negative, then clipped to 0.0
    model = NormalLatencyModel(mean_seconds=-0.05, std_seconds=0.0)
    d = model.sample_delay(order, snapshot)

    assert d == pytest.approx(0.0)
