# src/quant_platform/data/schemas/options.py

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class OptionRecord(BaseModel):
    """
    Basic option record schema used after ingestion and before
    feature engineering / IV computation.
    """

    symbol: str = Field(..., description="Option symbol")
    underlying: str = Field(..., description="Underlying asset symbol")
    trade_date: date = Field(..., description="Trade date (UTC)")
    expiry: date = Field(..., description="Expiration date")
    strike: float = Field(..., gt=0)
    option_type: str = Field(..., description="'call' or 'put'")
    bid: Optional[float] = Field(None, description="Bid price")
    ask: Optional[float] = Field(None, description="Ask price")
    last: Optional[float] = Field(None, description="Last traded price")
    volume: Optional[int] = None
    open_interest: Optional[int] = None

    @field_validator("option_type")
    def validate_option_type(cls, v):
        v = v.lower()
        if v not in {"call", "put"}:
            raise ValueError("option_type must be 'call' or 'put'")
        return v
