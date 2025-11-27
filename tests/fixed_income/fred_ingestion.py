# src/quant_platform/fixed_income/yield_curve.py
from dataclasses import dataclass
from typing import Tuple, Sequence, Optional, Dict, Any, Callable

import numpy as np
from scipy.optimize import minimize


# ------------------------------------------------------------
# Nelson–Siegel parameters
# ------------------------------------------------------------
@dataclass
class NSParams:
    beta0: float
    beta1: float
    beta2: float
    tau: float


# ------------------------------------------------------------
# Nelson–Siegel yield function (vectorized)
# ------------------------------------------------------------
def ns_yield(maturities: np.ndarray | float, params: NSParams) -> np.ndarray | float:
    """
    Compute Nelson–Siegel zero yield(s) for given maturity or array of maturities.
    Returns yields in percent (to be consistent with previous tests that expect percent).
    """
    m = np.asarray(maturities, dtype=float)

    # handle tau <= 0 defensively
    tau = float(params.tau)
    if tau <= 0:
        raise ValueError("tau must be positive")

    # avoid division by zero: use limits for m == 0
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.where(m == 0, 0.0, m / tau)
        # term1 limit -> 1 at m==0
        term1 = np.where(m == 0, 1.0, (1 - np.exp(-x)) / x)
        term2 = term1 - np.exp(-x)

    y = params.beta0 + params.beta1 * term1 + params.beta2 * term2
    return float(y) if np.isscalar(maturities) else y


# ------------------------------------------------------------
# Fit NS model via least squares (returns params, opt_result)
# ------------------------------------------------------------
def fit_ns(
    maturities: np.ndarray,
    yields: np.ndarray,
    initial: Optional[Sequence[float]] = None,
) -> Tuple[NSParams, Dict[str, Any]]:
    """
    Fit Nelson–Siegel curve parameters using least squares.
    Returns (NSParams, optimization_result).
    """
    t = np.asarray(maturities, dtype=float)
    y_obs = np.asarray(yields, dtype=float)

    if initial is None:
        beta0_0 = float(y_obs[-1])
        beta1_0 = float(y_obs[0] - y_obs[-1])
        beta2_0 = 0.0
        tau_0 = max(0.5, float(np.median(t)))
        initial = [beta0_0, beta1_0, beta2_0, tau_0]

    def loss(theta):
        b0, b1, b2, tau = theta
        if tau <= 0:
            return 1e8 + abs(tau) * 1e4
        p = NSParams(b0, b1, b2, tau)
        y_pred = ns_yield(t, p)
        return float(np.sum((y_pred - y_obs) ** 2))

    bounds = [(None, None), (None, None), (None, None), (1e-6, None)]
    res = minimize(loss, x0=initial, bounds=bounds, method="L-BFGS-B")
    params = NSParams(*res.x)
    return params, res


# ------------------------------------------------------------
# Helper to normalize rate returned by zero_curve to decimal
# Accepts functions that return either percent (e.g. 5.0) or decimal (0.05).
# ------------------------------------------------------------
def _rate_to_decimal(rate_value: float) -> float:
    r = float(rate_value)
    # Heuristic: treat numbers > 1.0 as percent (e.g., 5.0 -> 0.05), numbers <=1 as decimal
    return r / 100.0 if abs(r) > 1.0 else r


# ------------------------------------------------------------
# Discount factor given a zero-curve callable (zero_curve(t) -> percent or decimal)
# ------------------------------------------------------------
def discount_factor(t: float, zero_curve: Callable[[float], float]) -> float:
    """
    Continuous discount factor using zero curve.
    zero_curve(t) may return yield in percent (5.0) or decimal (0.05).
    """
    r = _rate_to_decimal(zero_curve(t))
    return np.exp(-r * t)


# ------------------------------------------------------------
# Price a zero-coupon using zero_curve callable
# ------------------------------------------------------------
def price_zero_coupon(
    face: float, t: float, zero_curve: Callable[[float], float]
) -> float:
    """
    Price a zero-coupon bond using the provided zero_curve (percent or decimal).
    """
    if t < 0:
        raise ValueError("maturity must be non-negative")
    return face * discount_factor(t, zero_curve)


# ------------------------------------------------------------
# Price a coupon bond given list of payment times and zero_curve
# (supports fractional/irregular periods)
# ------------------------------------------------------------
def price_coupon_bond(
    face: float,
    coupon_rate: float,
    maturities: Sequence[float],
    zero_curve: Callable[[float], float],
) -> float:
    """
    Price a coupon-bearing bond via continuous discounting from a zero curve.
    coupon_rate is annual percent (e.g. 6 -> 6%).
    maturities is list of payment times in years (e.g. [0.5, 1.0, ...]).
    zero_curve(t) returns either percent or decimal.
    """
    price = 0.0
    prev = 0.0
    for t in maturities:
        period_frac = t - prev
        prev = t
        coupon = face * coupon_rate * period_frac / 100.0
        if t == maturities[-1]:
            coupon += face
        price += coupon * discount_factor(t, zero_curve)
    return float(price)


# ------------------------------------------------------------
# Macaulay / Modified duration and convexity (using zero_curve)
# ------------------------------------------------------------
def macaulay_duration(
    face: float,
    coupon_rate: float,
    maturities: Sequence[float],
    zero_curve: Callable[[float], float],
) -> float:
    """
    Macaulay duration computed from cash flows implied by face, coupon_rate and maturities.
    """
    prev = 0.0
    pv_weighted = 0.0
    pv_total = 0.0
    for t in maturities:
        period_frac = t - prev
        prev = t
        cf = face * coupon_rate * period_frac / 100.0
        if t == maturities[-1]:
            cf += face
        df = discount_factor(t, zero_curve)
        pv = cf * df
        pv_total += pv
        pv_weighted += t * pv

    if pv_total == 0:
        return 0.0
    return pv_weighted / pv_total


def modified_duration(
    face: float,
    coupon_rate: float,
    maturities: Sequence[float],
    zero_curve: Callable[[float], float],
) -> float:
    """
    Modified duration = Macaulay / (1 + y) using last-point zero as approximation for y.
    """
    D_mac = macaulay_duration(face, coupon_rate, maturities, zero_curve)
    y = _rate_to_decimal(zero_curve(maturities[-1]))
    return D_mac / (1.0 + y)


def convexity(
    face: float,
    coupon_rate: float,
    maturities: Sequence[float],
    zero_curve: Callable[[float], float],
) -> float:
    """
    Convexity computed from cash flows and zero-curve discounting.
    Uses continuous discounting consistent with discount_factor().
    """
    prev = 0.0
    conv_sum = 0.0
    pv_total = 0.0
    for t in maturities:
        period_frac = t - prev
        prev = t
        cf = face * coupon_rate * period_frac / 100.0
        if t == maturities[-1]:
            cf += face
        r = _rate_to_decimal(zero_curve(t))
        df = np.exp(-r * t)
        pv = cf * df
        pv_total += pv
        conv_sum += cf * (t**2) * df

    if pv_total == 0:
        return 0.0
    return conv_sum / pv_total


# ------------------------------------------------------------
# Forward rate between two times from zero curve (zero_curve returns percent or decimal)
# ------------------------------------------------------------
def forward_rate(t1: float, t2: float, zero_curve: Callable[[float], float]) -> float:
    """
    Compute simple forward rate between t1 and t2 using zero yields from zero_curve.
    Returns forward rate in decimal (e.g. 0.05 for 5%).
    """
    if t2 <= t1:
        raise ValueError("t2 must be greater than t1")

    y1 = _rate_to_decimal(zero_curve(t1))
    y2 = _rate_to_decimal(zero_curve(t2))
    return (y2 * t2 - y1 * t1) / (t2 - t1)
