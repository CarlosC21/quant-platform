# tests/options/test_schema.py
from datetime import date

import pytest

from src.quant_platform.data.schemas.options import OptionRecord


def test_valid_option_record():
    rec = OptionRecord(
        symbol="ABC123",
        underlying="XYZ",
        trade_date=date(2024, 1, 1),
        expiry=date(2024, 2, 1),
        strike=100,
        option_type="call",
        bid=2.1,
        ask=2.3,
        last=2.2,
    )
    assert rec.option_type == "call"
    assert rec.strike == 100


def test_invalid_option_type():
    with pytest.raises(ValueError):
        OptionRecord(
            symbol="ABC",
            underlying="XYZ",
            trade_date=date(2024, 1, 1),
            expiry=date(2024, 2, 1),
            strike=100,
            option_type="invalid",
        )


def test_negative_strike_disallowed():
    with pytest.raises(ValueError):
        OptionRecord(
            symbol="ABC",
            underlying="XYZ",
            trade_date=date(2024, 1, 1),
            expiry=date(2024, 2, 1),
            strike=-5,
            option_type="call",
        )
