# tests/fixed_income/test_bond_pricing.py
import numpy as np

from quant_platform.fixed_income.yield_curve import price_coupon_bond


# simple flat zero curve for testing
def simple_zero_curve(t):
    """Flat 5% annual continuous discounting"""
    return 0.05


def test_coupon_bond_fractional_payments():
    """Check coupon payments scaled by period fractions"""
    face = 100
    coupon_rate = 6  # annual rate %
    # maturities (in years), possibly non-integer periods
    maturities = [0.5, 1.0]  # semi-annual payments

    # compute coupons scaled by period fraction
    prev = 0.0
    coupons = []
    for t in maturities:
        period_frac = t - prev
        coupons.append(face * coupon_rate * period_frac / 100)
        prev = t
    # add face to last cash flow
    coupons[-1] += face

    # price bond
    price = price_coupon_bond(face, coupon_rate, maturities, simple_zero_curve)

    # discount each cash flow using zero curve
    expected = sum(
        cf * np.exp(-simple_zero_curve(t) * t) for cf, t in zip(coupons, maturities)
    )

    assert np.isclose(price, expected, rtol=1e-12)


def test_zero_coupon_bond():
    """Check that zero-coupon bond prices correctly"""
    face = 100
    maturities = [1.0]
    coupon_rate = 0.0

    price = price_coupon_bond(face, coupon_rate, maturities, simple_zero_curve)
    expected = face * np.exp(-0.05 * 1.0)

    assert np.isclose(price, expected, rtol=1e-12)


def test_irregular_coupon_bond():
    """Check bonds with irregular periods"""
    face = 100
    coupon_rate = 6
    maturities = [0.3, 0.9, 1.7]  # irregular

    prev = 0.0
    coupons = []
    for t in maturities:
        period_frac = t - prev
        coupons.append(face * coupon_rate * period_frac / 100)
        prev = t
    coupons[-1] += face

    price = price_coupon_bond(face, coupon_rate, maturities, simple_zero_curve)
    expected = sum(
        cf * np.exp(-simple_zero_curve(t) * t) for cf, t in zip(coupons, maturities)
    )

    assert np.isclose(price, expected, rtol=1e-12)
