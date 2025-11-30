# src/quant_platform/risk/covariance.py
from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.covariance import LedoitWolf


def compute_sample_covariance(
    returns: np.ndarray,
    ddof: int = 1,
) -> np.ndarray:
    """
    Compute sample covariance matrix of asset returns.

    Parameters
    ----------
    returns:
        2D array of shape (n_obs, n_assets) with row-wise observations.
    ddof:
        Delta degrees of freedom (1 for unbiased sample covariance).

    Returns
    -------
    cov:
        2D array (n_assets, n_assets).
    """
    r = np.asarray(returns, dtype=float)
    if r.ndim != 2:
        msg = "returns must be a 2D array of shape (n_obs, n_assets)."
        raise ValueError(msg)
    if r.shape[0] < 2:
        msg = "need at least 2 observations to compute covariance."
        raise ValueError(msg)

    # np.cov uses rowvar=False for features as columns
    cov = np.cov(r, rowvar=False, ddof=ddof)
    return np.asarray(cov, dtype=float)


def compute_ewma_covariance(
    returns: np.ndarray,
    lambda_decay: float,
) -> np.ndarray:
    """
    Compute EWMA (exponentially weighted) covariance matrix.

    Parameters
    ----------
    returns:
        2D array of shape (n_obs, n_assets).
    lambda_decay:
        Decay factor in (0, 1). Higher -> slower decay (longer memory).

    Returns
    -------
    cov_ewma:
        2D array (n_assets, n_assets).
    """
    r = np.asarray(returns, dtype=float)
    if r.ndim != 2:
        raise ValueError("returns must be 2D.")
    n_obs, n_assets = r.shape
    if not (0.0 < lambda_decay < 1.0):
        raise ValueError("lambda_decay must be in (0, 1).")
    if n_obs < 1:
        raise ValueError("need at least 1 observation for EWMA covariance.")

    # Center returns
    r_centered = r - r.mean(axis=0, keepdims=True)

    # Initialize with outer product of first observation
    cov = np.outer(r_centered[0], r_centered[0])

    lam = float(lambda_decay)
    one_minus_lam = 1.0 - lam

    for t in range(1, n_obs):
        x = r_centered[t]
        outer = np.outer(x, x)
        cov = lam * cov + one_minus_lam * outer

    # Normalize so that expected weight sum is 1
    # The effective weight sum = 1 - lam^n_obs, so divide by that:
    weight_sum = 1.0 - lam**n_obs
    if weight_sum > 0:
        cov = cov / weight_sum

    return cov.astype(float)


def compute_ledoit_wolf_covariance(
    returns: np.ndarray,
    assume_centered: bool = False,
    shrinkage: Literal["auto"] = "auto",
) -> np.ndarray:
    """
    Compute Ledoit-Wolf shrinkage covariance using scikit-learn.

    Parameters
    ----------
    returns:
        2D array of shape (n_obs, n_assets).
    assume_centered:
        If True, data is assumed to be centered.
    shrinkage:
        Only 'auto' is supported (standard Ledoit-Wolf estimator).

    Returns
    -------
    cov_lw:
        2D array (n_assets, n_assets).
    """
    r = np.asarray(returns, dtype=float)
    if r.ndim != 2:
        raise ValueError("returns must be 2D.")

    if shrinkage != "auto":
        msg = "Only 'auto' shrinkage is supported for Ledoit-Wolf."
        raise ValueError(msg)

    lw = LedoitWolf(assume_centered=assume_centered)
    lw.fit(r)
    return np.asarray(lw.covariance_, dtype=float)
