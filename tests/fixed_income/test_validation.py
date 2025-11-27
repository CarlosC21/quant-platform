# tests/fixed_income/test_validation.py

from datetime import date

import pytest

from quant_platform.fixed_income.schemas import YieldPoint
from quant_platform.fixed_income.validation import validate_yield_points


def make_point(d, m, y):
    return YieldPoint(date=d, maturity_months=m, yield_pct=y)


def test_negative_yield_raises():
    pts = [
        make_point(date(2020, 1, 1), 1, 1.2),
        make_point(date(2020, 1, 1), 3, -0.5),  # negative
    ]
    with pytest.raises(ValueError):
        validate_yield_points(pts)


def test_duplicate_maturity_raises():
    pts = [
        make_point(date(2020, 1, 1), 12, 1.0),
        make_point(date(2020, 1, 1), 12, 1.1),  # duplicate maturity
    ]
    with pytest.raises(ValueError):
        validate_yield_points(pts)


def test_unsorted_maturities_raises():
    pts = [
        make_point(date(2020, 1, 1), 24, 1.8),
        make_point(date(2020, 1, 1), 6, 1.5),  # out of order
    ]
    with pytest.raises(ValueError):
        validate_yield_points(pts)


def test_valid_points_pass():
    pts = [
        make_point(date(2020, 1, 1), 6, 1.0),
        make_point(date(2020, 1, 1), 12, 1.2),
        make_point(date(2020, 1, 2), 6, 0.9),
        make_point(date(2020, 1, 2), 12, 1.1),
    ]

    # Should not raise
    validate_yield_points(pts)
