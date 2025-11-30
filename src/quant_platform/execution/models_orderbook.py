from __future__ import annotations

from datetime import datetime
from typing import List

from pydantic import BaseModel, Field, ConfigDict


class PriceLevel(BaseModel):
    """
    Represents a single level of the order book.

    Attributes
    ----------
    price : float
        The price at this level.
    size : float
        Executable quantity at this level.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    price: float = Field(..., ge=0.0)
    size: float = Field(..., ge=0.0)


class OrderBookSnapshot(BaseModel):
    """
    Bid/ask ladder snapshot for simulating partial fills.

    Attributes
    ----------
    symbol : str
        Associated trading symbol.
    timestamp : datetime
        Timestamp of snapshot.
    bids : List[PriceLevel]
        Sorted descending by price.
    asks : List[PriceLevel]
        Sorted ascending by price.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    symbol: str
    timestamp: datetime
    bids: List[PriceLevel] = Field(default_factory=list)
    asks: List[PriceLevel] = Field(default_factory=list)

    def best_bid(self) -> float | None:
        if not self.bids:
            return None
        return max(self.bids, key=lambda x: x.price).price

    def best_ask(self) -> float | None:
        if not self.asks:
            return None
        return min(self.asks, key=lambda x: x.price).price
