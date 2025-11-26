from __future__ import annotations

from datetime import date, datetime
from typing import Literal, Optional

from pydantic import BaseModel


class OptionQuoteSchema(BaseModel):
    symbol: str
    underlying: str
    trade_date: date
    timestamp: Optional[datetime] = None
    strike: float
    expiry: date
    option_type: Literal["call", "put"]
    bid: Optional[float] = None
    ask: Optional[float] = None
    last: Optional[float] = None
    implied_vol: Optional[float] = None
    volume: Optional[int] = None
    open_interest: Optional[int] = None

    class Config:
        arbitrary_types_allowed = True
