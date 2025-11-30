# src/quant_platform/risk/var.py
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.stats import norm


def compute_historical_var(
    returns: np.ndarray,
    alpha: float,
) -> float:
    """
    Compute (left-tail) historical VaR on returns.

    Parameters
    ----------
    returns:
        1D array of portfolio returns (e.g. daily log or simple returns).
    alpha:
        Tail probability in (0, 1), e.g. 0.05 for 95% VaR.

    Returns
    -------
    var:
        Positive number representing the loss level such that
        P(loss > var) = alpha. For returns, this is defined as:

            var = -quantile(returns, alpha)

        so more negative returns correspond to larger VaR.
    """
    r = np.asarray(returns, dtype=float)
    if r.ndim != 1 or r.size == 0:
        msg = "returns must be a non-empty 1D array."
        raise ValueError(msg)
    if not (0.0 < alpha < 1.0):
        msg = "alpha must be in (0, 1)."
        raise ValueError(msg)

    q = np.quantile(r, alpha)
    return float(-q)


def compute_parametric_var(
    mean: float,
    sigma: float,
    alpha: float,
) -> float:
    """
    Parametric VaR under a Normal(mu, sigma) assumption.

    VaR is defined as the positive loss number corresponding
    to the alpha-quantile of returns:

        q_alpha = mu + sigma * z_alpha
        var = -q_alpha

    where z_alpha = Phi^{-1}(alpha).

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
    var:
        Positive loss corresponding to VaR level.
    """
    if sigma <= 0.0:
        msg = "sigma must be positive."
        raise ValueError(msg)
    if not (0.0 < alpha < 1.0):
        msg = "alpha must be in (0, 1)."
        raise ValueError(msg)

    z = norm.ppf(alpha)
    q_alpha = mean + sigma * z
    return float(-q_alpha)


def compute_parametric_var_from_series(
    returns: np.ndarray,
    alpha: float,
) -> Tuple[float, float, float]:
    """
    Convenience helper: estimate mean and sigma from a return series
    and compute parametric VaR.

    Returns
    -------
    var, mean_hat, sigma_hat
    """
    r = np.asarray(returns, dtype=float)
    if r.ndim != 1 or r.size == 0:
        msg = "returns must be a non-empty 1D array."
        raise ValueError(msg)
    mu_hat = float(r.mean())
    sigma_hat = float(r.std(ddof=1))
    var = compute_parametric_var(mu_hat, sigma_hat, alpha)
    return var, mu_hat, sigma_hat
