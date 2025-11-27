# src/quant_platform/fixed_income/yield_curve.py
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

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
# Nelson–Siegel zero yield (vectorized). Returns yields in percent.
# ------------------------------------------------------------
def ns_yield(maturities: np.ndarray | float, params: NSParams) -> np.ndarray | float:
    """
    Nelson–Siegel yield(s). **Returns yields in percent** (e.g. 2.5 means 2.5%).
    Accepts scalar or array maturities (years).
    """
    m = np.asarray(maturities, dtype=float)
    tau = float(params.tau)
    if tau <= 0:
        raise ValueError("tau must be positive")

    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.where(m == 0, 1.0, m / tau)
        term1 = np.where(m == 0, 1.0, (1 - np.exp(-x)) / x)
        term2 = term1 - np.exp(-x)

    y = params.beta0 + params.beta1 * term1 + params.beta2 * term2
    return float(y) if np.isscalar(maturities) else y


# ------------------------------------------------------------
# Fit Nelson–Siegel model via least-squares (expects yields in percent).
# Returns (NSParams, optimization_result)
# ------------------------------------------------------------
def fit_ns(
    maturities: np.ndarray,
    yields: np.ndarray,
    initial: Optional[Sequence[float]] = None,
) -> Tuple[NSParams, Dict[str, Any]]:
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
# Discount factor given a zero-curve callable.
# zero_curve(t) -> yield either in percent (e.g. 5.0) or decimal (e.g. 0.05).
# This helper normalizes to decimal internally.
# ------------------------------------------------------------
def discount_factor(t: float, zero_curve: Callable[[float], float]) -> float:
    """
    Continuous discount factor using zero_curve.

    Accepts zero_curve(t) that returns either:
      - percent (e.g. 5.0)  -> converted to 0.05
      - decimal (e.g. 0.05) -> used directly

    Returns exp(-r * t) with r in decimal.
    """
    r_val = float(zero_curve(t))
    # if the supplied value looks like a percent (greater than 1), convert to decimal
    # threshold 1.0: anything >1 we'll treat as percent (5.0 -> 0.05). This covers
    # inputs like 5.0 (percent) vs 0.05 (decimal).
    if abs(r_val) > 1.0:
        r = r_val / 100.0
    else:
        r = r_val
    return np.exp(-r * t)


# ------------------------------------------------------------
# Price zero-coupon given zero_curve callable (percent or decimal).
# ------------------------------------------------------------
def price_zero_coupon(
    face: float, t: float, zero_curve: Callable[[float], float]
) -> float:
    if t < 0:
        raise ValueError("maturity must be non-negative")
    return face * discount_factor(t, zero_curve)


# ------------------------------------------------------------
# Price a coupon bond given list of payment times (supports fractional/irregular periods).
# coupon_rate is in percent (e.g. 6 => 6%).
# zero_curve(t) may return percent or decimal.
# ------------------------------------------------------------
def price_coupon_bond(
    face: float,
    coupon_rate: float,
    maturities: Sequence[float],
    zero_curve: Callable[[float], float],
) -> float:
    """
    Price a coupon-bearing bond via continuous discounting from a zero curve.
    - coupon_rate is annual percent (e.g. 6 -> 6%)
    - maturities is list of payment times in years (e.g. [0.5, 1.0, ...])
    - zero_curve(t) returns percent OR decimal (this function will normalize)
    """
    price = 0.0
    prev = 0.0
    for t in maturities:
        period_frac = t - prev
        prev = t
        coupon = face * coupon_rate * period_frac / 100.0
        if t == maturities[-1]:
            coupon += face
        df = discount_factor(t, zero_curve)
        price += coupon * df
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
    y_val = float(zero_curve(maturities[-1]))
    if abs(y_val) > 1.0:
        y = y_val / 100.0
    else:
        y = y_val
    return D_mac / (1.0 + y)


def convexity(
    face: float,
    coupon_rate: float,
    maturities: Sequence[float],
    zero_curve: Callable[[float], float],
) -> float:
    prev = 0.0
    conv_sum = 0.0
    pv_total = 0.0
    for t in maturities:
        period_frac = t - prev
        prev = t
        cf = face * coupon_rate * period_frac / 100.0
        if t == maturities[-1]:
            cf += face
        r_val = float(zero_curve(t))
        if abs(r_val) > 1.0:
            r = r_val / 100.0
        else:
            r = r_val
        df = np.exp(-r * t)
        pv = cf * df
        pv_total += pv
        conv_sum += cf * (t**2) * df

    if pv_total == 0:
        return 0.0
    return conv_sum / pv_total


# ------------------------------------------------------------
# Forward rate between t1 and t2 using zero_curve (zero_curve may return percent or decimal).
# Returns forward rate in decimal (e.g. 0.05 for 5%).
# ------------------------------------------------------------
def forward_rate(t1: float, t2: float, zero_curve: Callable[[float], float]) -> float:
    if t2 <= t1:
        raise ValueError("t2 must be greater than t1")
    y1_val = float(zero_curve(t1))
    y2_val = float(zero_curve(t2))
    if abs(y1_val) > 1.0:
        y1 = y1_val / 100.0
    else:
        y1 = y1_val
    if abs(y2_val) > 1.0:
        y2 = y2_val / 100.0
    else:
        y2 = y2_val
    return (y2 * t2 - y1 * t1) / (t2 - t1)
