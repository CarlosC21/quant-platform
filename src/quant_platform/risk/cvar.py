# src/quant_platform/risk/cvar.py
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.stats import norm


def compute_historical_cvar(
    returns: np.ndarray,
    alpha: float,
) -> float:
    """
    Compute historical CVaR (Expected Shortfall) on returns.

    Definition:
        VaR threshold q_alpha = quantile(returns, alpha)
        CVaR = - E[ returns | returns <= q_alpha ]

    Parameters
    ----------
    returns:
        1D array of portfolio returns.
    alpha:
        Tail probability in (0, 1).

    Returns
    -------
    cvar:
        Positive expected loss beyond the VaR threshold.
    """
    r = np.asarray(returns, dtype=float)
    if r.ndim != 1 or r.size == 0:
        msg = "returns must be a non-empty 1D array."
        raise ValueError(msg)
    if not (0.0 < alpha < 1.0):
        msg = "alpha must be in (0, 1)."
        raise ValueError(msg)

    q = np.quantile(r, alpha)
    tail = r[r <= q]
    if tail.size == 0:
        # No tail observations: CVaR collapses to VaR level
        return float(-q)

    cvar = -float(tail.mean())
    return cvar


def compute_parametric_cvar(
    mean: float,
    sigma: float,
    alpha: float,
) -> float:
    """
    Parametric CVaR (Expected Shortfall) under Normal(mu, sigma).

    For X ~ N(mu, sigma^2), left-tail ES at level alpha is:

        ES_alpha = mu - sigma * φ(z_alpha) / alpha

    where z_alpha = Phi^{-1}(alpha), φ is standard normal pdf.

    We return CVaR as positive loss:

        cvar = -ES_alpha

    Parameters
    ----------
    mean:
        Expected return (per period).
    sigma:
        Volatility (per period), must be > 0.
    alpha:
        Tail probability in (0, 1).

    Returns
    -------
    cvar:
        Positive expected shortfall beyond VaR level.
    """
    if sigma <= 0.0:
        msg = "sigma must be positive."
        raise ValueError(msg)
    if not (0.0 < alpha < 1.0):
        msg = "alpha must be in (0, 1)."
        raise ValueError(msg)

    z = norm.ppf(alpha)
    phi = norm.pdf(z)

    es_alpha = mean - sigma * phi / alpha
    cvar = -es_alpha
    return float(cvar)


def compute_parametric_cvar_from_series(
    returns: np.ndarray,
    alpha: float,
) -> Tuple[float, float, float]:
    """
    Convenience helper: estimate mean and sigma from a return series
    and compute parametric CVaR.

    Returns
    -------
    cvar, mean_hat, sigma_hat
    """
    r = np.asarray(returns, dtype=float)
    if r.ndim != 1 or r.size == 0:
        msg = "returns must be a non-empty 1D array."
        raise ValueError(msg)
    mu_hat = float(r.mean())
    sigma_hat = float(r.std(ddof=1))
    cvar = compute_parametric_cvar(mu_hat, sigma_hat, alpha)
    return cvar, mu_hat, sigma_hat
