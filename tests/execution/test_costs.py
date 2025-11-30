from __future__ import annotations

import pytest

from quant_platform.execution.costs import (
    ProportionalCostModel,
    FixedCostModel,
    HybridCostModel,
)


def test_proportional_cost_model_basic():
    model = ProportionalCostModel(commission_bps=5.0, fee_bps=2.0)
    price = 100.0
    quantity = 10.0
    cost = model.compute_cost(price, quantity)
    # 7 bps * 100 * 10 = 7
    assert cost == pytest.approx(0.7)


def test_fixed_cost_model_basic():
    model = FixedCostModel(fixed_fee=2.5)
    cost = model.compute_cost(price=150.0, quantity=100.0)
    assert cost == pytest.approx(2.5)


def test_hybrid_cost_model_basic():
    model = HybridCostModel(commission_bps=5.0, fee_bps=5.0, fixed_fee=1.0)
    price = 200.0
    quantity = 50.0
    proportional = 200 * 50 * 0.001  # 10 bps = 0.001
    expected = proportional + 1.0
    cost = model.compute_cost(price, quantity)
    assert cost == pytest.approx(expected)
