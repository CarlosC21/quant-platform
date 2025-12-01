from typing import List

from quant_platform.data.validation.validators import ValidationError
from quant_platform.options.schemas import OptionQuoteSchema


def validate_option_records(records: List[OptionQuoteSchema]):
    for r in records:
        if r.strike <= 0:
            raise ValidationError(f"{r.symbol} strike {r.strike} is <= 0")
        if r.expiry < r.trade_date:
            raise ValidationError(
                f"{r.symbol} expiry {r.expiry} before trade_date {r.trade_date}"
            )
        if r.bid is not None and r.ask is not None and r.bid > r.ask:
            raise ValidationError(f"{r.symbol} bid {r.bid} > ask {r.ask}")
        if r.last is not None and r.last < 0:
            raise ValidationError(f"{r.symbol} last {r.last} < 0")
        if r.option_type not in ("call", "put"):
            raise ValidationError(f"{r.symbol} option_type {r.option_type} invalid")
