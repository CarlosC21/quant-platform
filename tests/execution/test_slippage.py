from __future__ import annotations

import pytest

from quant_platform.execution.enums import Side
from quant_platform.execution.slippage import (
    LinearSlippageModel,
    NoSlippageModel,
    SquareRootSlippageModel,
)


def test_no_slippage_returns_mid():
    model = NoSlippageModel()
    mid = 100.0
    price_buy = model.get_execution_price(mid_price=mid, side=Side.BUY, quantity=10.0)
    price_sell = model.get_execution_price(mid_price=mid, side=Side.SELL, quantity=10.0)
    assert price_buy == pytest.approx(mid)
    assert price_sell == pytest.approx(mid)


@pytest.mark.parametrize("side", [Side.BUY, Side.SELL])
def test_linear_slippage_scales_with_quantity(side: Side):
    model = LinearSlippageModel(kappa=1e-4)
    mid = 100.0
    q_small = 1_000.0
    q_large = 10_000.0
    V = 1_000_000.0

    p_small = model.get_execution_price(
        mid_price=mid,
        side=side,
        quantity=q_small,
        daily_volume=V,
    )
    p_large = model.get_execution_price(
        mid_price=mid,
        side=side,
        quantity=q_large,
        daily_volume=V,
    )

    if side is Side.BUY:
        assert p_large > p_small > mid
    else:
        assert p_large < p_small < mid


@pytest.mark.parametrize("side", [Side.BUY, Side.SELL])
def test_sqrt_slippage_and_linear_relative_to_mid(side: Side):
    mid = 100.0
    q = 10_000.0
    V = 1_000_000.0

    linear = LinearSlippageModel(kappa=1e-4)
    sqrt_model = SquareRootSlippageModel(kappa=1e-4)

    p_linear = linear.get_execution_price(
        mid_price=mid,
        side=side,
        quantity=q,
        daily_volume=V,
    )
    p_sqrt = sqrt_model.get_execution_price(
        mid_price=mid,
        side=side,
        quantity=q,
        daily_volume=V,
    )

    if side is Side.BUY:
        assert p_linear > mid
        assert p_sqrt > mid
    else:
        assert p_linear < mid
        assert p_sqrt < mid
