# src/quant_platform/data/validation/validators.py
from typing import List

from quant_platform.data.schemas.equities import EquityPriceSchema


class ValidationError(Exception):
    """Custom validation exception."""

    pass


def validate_equity_records(records: List[EquityPriceSchema]):
    """
    Domain-level validation for equities.
    Raises ValidationError if any record fails.
    """
    for r in records:
        # use the actual Python attribute names (with _)
        if r.close_ is not None and r.close_ < 0:
            raise ValidationError(
                f"{r.symbol} on {r.trade_date} has negative close price"
            )
        if r.volume is not None and r.volume < 0:
            raise ValidationError(f"{r.symbol} on {r.date} has negative volume")
