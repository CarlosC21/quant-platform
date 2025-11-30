# src/quant_platform/risk/stress.py
from __future__ import annotations

import numpy as np


def apply_return_shock(
    returns: np.ndarray,
    shock: np.ndarray,
) -> np.ndarray:
    """
    Apply a deterministic return shock to each asset.

    Parameters
    ----------
    returns:
        2D array (n_obs, n_assets) of base returns.
    shock:
        1D array (n_assets,) of shocks to be added to each asset's returns.

    Returns
    -------
    shocked_returns:
        2D array (n_obs, n_assets) with shock applied.
    """
    R = np.asarray(returns, dtype=float)
    s = np.asarray(shock, dtype=float)

    if R.ndim != 2:
        msg = "returns must be a 2D array."
        raise ValueError(msg)
    if s.ndim != 1:
        msg = "shock must be a 1D array."
        raise ValueError(msg)
    if R.shape[1] != s.shape[0]:
        msg = "shock length must match number of assets (columns of returns)."
        raise ValueError(msg)

    return R + s[None, :]


def scenario_pnl(
    weights: np.ndarray,
    scenario_returns: np.ndarray,
) -> float:
    """
    Compute portfolio PnL under a single scenario.

    Parameters
    ----------
    weights:
        1D array (n_assets,) of portfolio weights or notional exposures.
    scenario_returns:
        1D array (n_assets,) of scenario returns.

    Returns
    -------
    pnl:
        Scalar PnL = w^T r_scenario.
    """
    w = np.asarray(weights, dtype=float)
    r = np.asarray(scenario_returns, dtype=float)

    if w.ndim != 1 or r.ndim != 1:
        msg = "weights and scenario_returns must be 1D arrays."
        raise ValueError(msg)
    if w.shape[0] != r.shape[0]:
        msg = "weights and scenario_returns must have same length."
        raise ValueError(msg)

    return float(w @ r)


def historical_stress_pnls(
    weights: np.ndarray,
    returns: np.ndarray,
) -> np.ndarray:
    """
    Compute portfolio PnL across a matrix of historical returns.

    Useful for historical stress analysis (e.g., worst 10 days).

    Parameters
    ----------
    weights:
        1D array (n_assets,) of portfolio weights.
    returns:
        2D array (n_obs, n_assets) of historical returns.

    Returns
    -------
    pnls:
        1D array (n_obs,) of PnL per historical observation.
    """
    w = np.asarray(weights, dtype=float)
    R = np.asarray(returns, dtype=float)

    if w.ndim != 1 or R.ndim != 2:
        msg = "weights must be 1D and returns must be 2D."
        raise ValueError(msg)
    if R.shape[1] != w.shape[0]:
        msg = "weights length must match number of assets (columns of returns)."
        raise ValueError(msg)

    return R @ w
