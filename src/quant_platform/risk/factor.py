# src/quant_platform/risk/factor.py
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

from quant_platform.risk.schemas import FactorModelResult, FactorExposure


def compute_pca_factor_model(
    returns: np.ndarray,
    n_factors: int,
) -> FactorModelResult:
    """
    Fit a PCA-based factor model to asset returns.

    Parameters
    ----------
    returns:
        2D array of shape (n_obs, n_assets) with time series of returns.
    n_factors:
        Number of principal components (factors) to extract.

    Returns
    -------
    FactorModelResult
        factor_returns: [n_obs x n_factors]
        loadings:       [n_assets x n_factors]
        specific_var:   [n_assets]
        explained_variance_ratio: [n_factors]
    """
    r = np.asarray(returns, dtype=float)
    if r.ndim != 2:
        msg = "returns must be a 2D array (n_obs, n_assets)."
        raise ValueError(msg)
    n_obs, n_assets = r.shape
    if n_obs < 2 or n_assets < 1:
        msg = "returns must have at least 2 observations and 1 asset."
        raise ValueError(msg)
    if not (1 <= n_factors <= n_assets):
        msg = "n_factors must be between 1 and n_assets."
        raise ValueError(msg)

    pca = PCA(n_components=n_factors)
    factor_returns = pca.fit_transform(r)  # (n_obs, n_factors)
    # components_: (n_factors, n_assets) → loadings: (n_assets, n_factors)
    loadings = pca.components_.T

    # Reconstruct and compute specific variance per asset
    r_hat = factor_returns @ loadings.T  # (n_obs, n_assets)
    residuals = r - r_hat
    specific_var = residuals.var(axis=0, ddof=1)

    return FactorModelResult(
        n_factors=int(n_factors),
        factor_returns=factor_returns.tolist(),
        loadings=loadings.tolist(),
        specific_var=specific_var.tolist(),
        explained_variance_ratio=pca.explained_variance_ratio_.tolist(),
    )


def compute_factor_exposures_regression(
    asset_returns: np.ndarray,
    factor_returns: np.ndarray,
    symbols: list[str],
) -> list[FactorExposure]:
    """
    Compute factor exposures via time-series regressions:

        r_i(t) = alpha_i + beta_i^T f(t) + eps_i(t)

    for each asset i.

    Parameters
    ----------
    asset_returns:
        2D array (n_obs, n_assets).
    factor_returns:
        2D array (n_obs, n_factors).
    symbols:
        List of asset symbols of length n_assets.

    Returns
    -------
    list[FactorExposure]
        One FactorExposure per asset.
    """
    R = np.asarray(asset_returns, dtype=float)
    F = np.asarray(factor_returns, dtype=float)

    if R.ndim != 2 or F.ndim != 2:
        msg = "asset_returns and factor_returns must be 2D arrays."
        raise ValueError(msg)

    n_obs, n_assets = R.shape
    n_obs_f, n_factors = F.shape

    if n_obs != n_obs_f:
        msg = "asset_returns and factor_returns must have the same number of rows (n_obs)."
        raise ValueError(msg)
    if len(symbols) != n_assets:
        msg = "symbols length must match number of assets (columns of asset_returns)."
        raise ValueError(msg)

    # Design matrix with intercept
    X = np.column_stack([np.ones(n_obs), F])  # (n_obs, 1 + n_factors)

    exposures: list[FactorExposure] = []
    for j in range(n_assets):
        y = R[:, j]
        # OLS: beta_hat = (X^T X)^(-1) X^T y via lstsq
        beta_hat, *_ = np.linalg.lstsq(X, y, rcond=None)
        intercept = float(beta_hat[0])
        betas = beta_hat[1:].astype(float).tolist()

        exposures.append(
            FactorExposure(
                symbol=symbols[j],
                intercept=intercept,
                betas=betas,
            ),
        )

    return exposures
