# src/quant_platform/options/features.py

from __future__ import annotations

import math
from datetime import date
from typing import Optional

import polars as pl

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def year_fraction(start: date, end: date, basis: int = 365) -> float:
    """ACT/365 simple year fraction."""
    return (end - start).days / basis


def compute_mid(bid: float | None, ask: float | None) -> Optional[float]:
    """Return mid price if both bid and ask exist, otherwise None."""
    if bid is None or ask is None:
        return None
    return 0.5 * (bid + ask)


def compute_spread(bid: float | None, ask: float | None) -> Optional[float]:
    """Bid/ask spread."""
    if bid is None or ask is None:
        return None
    return ask - bid


def compute_moneyness(spot: float, strike: float, option_type: str) -> float:
    """
    Basic moneyness measure: S/K for calls, K/S for puts.

    For surface work, normalized log-moneyness will be used later.
    """
    if option_type.lower() == "call":
        return spot / strike
    if option_type.lower() == "put":
        return strike / spot
    raise ValueError(f"Invalid option_type: {option_type}")


# ---------------------------------------------------------------------------
# Black-Scholes implied volatility (Newton solver)
# ---------------------------------------------------------------------------


def _bs_price(
    spot: float, strike: float, t: float, r: float, vol: float, option_type: str
) -> float:
    """
    Standard Black-Scholes price with no dividends (we add q later if needed).
    """
    if vol <= 0:
        return 0.0

    d1 = (math.log(spot / strike) + (r + 0.5 * vol * vol) * t) / (vol * math.sqrt(t))
    d2 = d1 - vol * math.sqrt(t)

    from math import erf  # removed sqrt import per Ruff

    def norm_cdf(x):
        return 0.5 * (1 + erf(x / math.sqrt(2)))

    if option_type.lower() == "call":
        return spot * norm_cdf(d1) - strike * math.exp(-r * t) * norm_cdf(d2)
    else:
        return strike * math.exp(-r * t) * norm_cdf(-d2) - spot * norm_cdf(-d1)


def implied_vol_bs(
    spot: float,
    strike: float,
    t: float,
    option_type: str,
    price: float,
    r: float = 0.0,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> Optional[float]:
    """
    Newton-Raphson solver for Black-Scholes implied volatility.
    Returns None if solver fails to converge.
    """
    if t <= 0 or price <= 0:
        return None

    vol = 0.2  # initial guess

    for _ in range(max_iter):
        # Price and vega
        p = _bs_price(spot, strike, t, r, vol, option_type)

        # Numerical vega
        eps = 1e-6
        p_up = _bs_price(spot, strike, t, r, vol + eps, option_type)
        vega = (p_up - p) / eps

        if abs(vega) < 1e-8:
            return None

        diff = p - price
        if abs(diff) < tol:
            return vol

        vol = vol - diff / vega

        if vol <= 0 or vol > 5:
            return None

    return None


# ---------------------------------------------------------------------------
# Feature engineering main function
# ---------------------------------------------------------------------------


def compute_option_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add mid, spread, moneyness, year_fraction, and implied volatility.
    Expects df to already include:
    - spot: underlying price
    """
    df = df.with_columns(
        [
            pl.struct(["bid", "ask"])
            .map_elements(lambda s: compute_mid(s["bid"], s["ask"]))
            .alias("mid"),
            pl.struct(["bid", "ask"])
            .map_elements(lambda s: compute_spread(s["bid"], s["ask"]))
            .alias("spread"),
        ]
    )

    df = df.with_columns(
        [
            pl.struct(["underlying_price", "strike", "option_type"])
            .map_elements(
                lambda s: compute_moneyness(
                    float(s["underlying_price"]), float(s["strike"]), s["option_type"]
                )
            )
            .alias("moneyness")
        ]
    )

    df = df.with_columns(
        [
            pl.struct(["trade_date", "expiry"])
            .map_elements(lambda s: year_fraction(s["trade_date"], s["expiry"]))
            .alias("t")
        ]
    )

    df = df.with_columns(
        [
            pl.struct(["underlying_price", "strike", "t", "option_type", "mid"])
            .map_elements(
                lambda s: implied_vol_bs(
                    spot=float(s["underlying_price"]),
                    strike=float(s["strike"]),
                    t=float(s["t"]),
                    option_type=s["option_type"],
                    price=float(s["mid"]) if s["mid"] is not None else None,
                )
            )
            .alias("implied_vol")
        ]
    )

    return df
