from __future__ import annotations

from enum import Enum


class Side(str, Enum):
    """Trade direction."""

    BUY = "BUY"
    SELL = "SELL"

    @property
    def sign(self) -> int:
        """Return +1 for BUY, -1 for SELL."""
        return 1 if self is Side.BUY else -1

    # --- Added for simulator partial-fill logic ---
    def is_buy(self) -> bool:
        """Return True if side is BUY."""
        return self is Side.BUY

    def is_sell(self) -> bool:
        """Return True if side is SELL."""
        return self is Side.SELL


class OrderType(str, Enum):
    """High-level order types supported by the simulator."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"

    # Optional, but used by our simulator implementation
    def is_market(self) -> bool:
        return self is OrderType.MARKET

    def is_limit(self) -> bool:
        return self is OrderType.LIMIT


class TimeInForce(str, Enum):
    """Time-in-force semantics for limit orders."""

    DAY = "DAY"
    IOC = "IOC"
    FOK = "FOK"


class OrderStatus(str, Enum):
    """Lifecycle status of an order."""

    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class LiquiditySide(str, Enum):
    """Whether the fill is providing or taking liquidity."""

    MAKER = "MAKER"
    TAKER = "TAKER"


class Venue(str, Enum):
    """Simple venue abstraction for routing."""

    SIMULATED = "SIMULATED"
