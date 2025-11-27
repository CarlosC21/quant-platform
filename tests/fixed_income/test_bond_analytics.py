import numpy as np

from src.quant_platform.fixed_income.yield_curve import (
    convexity,
    discount_factor,
    forward_rate,
    macaulay_duration,
    modified_duration,
    price_coupon_bond,
    price_zero_coupon,
)


def simple_zero_curve(t):
    return 5.0  # 5% constant rate


def test_discount_factor():
    df = discount_factor(2.0, simple_zero_curve)
    assert np.isclose(df, np.exp(-0.05 * 2.0))


def test_forward_rate():
    fwd = forward_rate(1.0, 2.0, simple_zero_curve)
    assert np.isclose(fwd, 0.05)


def test_zero_coupon_bond_price():
    price = price_zero_coupon(100, 1.0, simple_zero_curve)
    assert np.isclose(price, 100 * np.exp(-0.05 * 1.0))


def test_coupon_bond_price():
    price = price_coupon_bond(100, 6, [0.5, 1.0], simple_zero_curve)
    prev = 0.0
    expected = 0.0
    for t in [0.5, 1.0]:
        period_frac = t - prev
        prev = t
        cf = 100 * 6 * period_frac / 100
        if t == 1.0:
            cf += 100
        expected += cf * np.exp(-0.05 * t)
    assert np.isclose(price, expected)


def test_macaulay_duration():
    t = [0.5, 1.0]
    mac = macaulay_duration(100, 6, t, simple_zero_curve)
    assert mac > 0 and mac < 2


def test_modified_duration():
    t = [0.5, 1.0]
    mod = modified_duration(100, 6, t, simple_zero_curve)
    assert mod > 0 and mod < 2


def test_convexity():
    t = [0.5, 1.0]
    conv = convexity(100, 6, t, simple_zero_curve)
    assert conv > 0 and conv < 5
