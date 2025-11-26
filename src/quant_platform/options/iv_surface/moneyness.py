from __future__ import annotations

import math


def log_moneyness(spot: float, strike: float) -> float:
    """Return log(S/K)."""
    if spot <= 0 or strike <= 0:
        raise ValueError("spot and strike must be > 0")
    return math.log(spot / strike)


def normalized_strike(spot: float, strike: float) -> float:
    """Return normalized strike K/S."""
    if spot <= 0 or strike <= 0:
        raise ValueError("spot and strike must be > 0")
    return strike / spot
