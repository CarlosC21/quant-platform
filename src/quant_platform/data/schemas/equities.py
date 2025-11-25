# src/quant_platform/data/schemas/equities.py
from datetime import date

from pydantic import BaseModel, Field


class EquityPriceSchema(BaseModel):
    symbol: str = Field(..., description="Ticker symbol of the equity")
    trade_date: date = Field(..., alias="date", description="Trading date")
    open_: float | None = Field(None, alias="open", description="Opening price")
    high: float | None = None
    low: float | None = None
    close_: float | None = Field(None, alias="close", description="Closing price")
    volume: int | None = None

    model_config = {"populate_by_name": True}
