# src/quant_platform/portfolio/position_sizing.py
from __future__ import annotations

from typing import Literal


def classical_kelly(mu: float, sigma: float) -> float:
    """
    Classical Kelly fraction for a single bet / asset:

        f* = mu / sigma^2

    where
        mu    = expected return (per period)
        sigma = volatility (per period, > 0)

    This is the fraction of capital to allocate (can be > 1 if very high edge).
    """
    if sigma <= 0.0:
        msg = "sigma must be positive for Kelly sizing."
        raise ValueError(msg)

    return float(mu / (sigma**2))


def fractional_kelly(mu: float, sigma: float, fraction: float = 0.5) -> float:
    """
    Fractional Kelly sizing:

        f = fraction * f*

    where fraction ∈ (0, 1].  In practice, 0.2–0.5 is common to reduce
    drawdown risk relative to full Kelly.
    """
    if not (0.0 < fraction <= 1.0):
        msg = "fraction must be in (0, 1]."
        raise ValueError(msg)

    f_star = classical_kelly(mu, sigma)
    return float(fraction * f_star)


def vol_target_scaler(realized_vol: float, target_vol: float) -> float:
    """
    Volatility targeting scaler:

        scale = target_vol / realized_vol

    so that if realized_vol > target_vol, scale < 1 and positions shrink.
    """
    if realized_vol <= 0.0:
        msg = "realized_vol must be positive."
        raise ValueError(msg)
    if target_vol <= 0.0:
        msg = "target_vol must be positive."
        raise ValueError(msg)

    return float(target_vol / realized_vol)


def size_stat_arb_position(
    side: Literal["long", "short", "flat"],
    zscore: float,
    z_entry: float,
    mu: float,
    sigma: float,
    target_vol: float,
    realized_vol: float,
    kelly_fraction: float = 0.5,
    leverage_limit: float = 1.0,
) -> float:
    """
    Position sizing for a stat-arb spread trade.

    Combines:
        - Fractional Kelly sizing based on (mu, sigma)
        - Volatility targeting (target_vol / realized_vol)
        - Z-score intensity (|z| / z_entry, capped at 1)
        - Leverage cap

    Parameters
    ----------
    side:
        "long", "short" or "flat" spread.
    zscore:
        Current spread z-score.
    z_entry:
        Entry threshold; intensity scales from 0 to 1 as |z| / z_entry.
    mu:
        Expected return of the trade (per period).
    sigma:
        Expected volatility of the trade (per period).
    target_vol:
        Target volatility (per period) for this position.
    realized_vol:
        Realized or estimated current volatility (per period).
    kelly_fraction:
        Fraction of full Kelly, in (0, 1].
    leverage_limit:
        Max |position| allowed (e.g. 1.0 for 100% notional).

    Returns
    -------
    position:
        Signed position size in [-leverage_limit, leverage_limit].
    """
    if side == "flat":
        return 0.0

    if z_entry <= 0.0:
        msg = "z_entry must be positive."
        raise ValueError(msg)
    if leverage_limit <= 0.0:
        msg = "leverage_limit must be positive."
        raise ValueError(msg)

    # Base Kelly fraction (could be > 1; we'll clip later)
    f_kelly = fractional_kelly(mu, sigma, kelly_fraction)

    # Intensity based on how far the zscore is beyond the entry level
    intensity = min(abs(zscore) / z_entry, 1.0)

    # Volatility targeting
    vol_scale = vol_target_scaler(realized_vol, target_vol)

    raw_position = f_kelly * intensity * vol_scale

    # Apply side
    sign = 1.0 if side == "long" else -1.0
    position = sign * raw_position

    # Enforce leverage cap
    if abs(position) > leverage_limit:
        position = sign * leverage_limit

    return float(position)
