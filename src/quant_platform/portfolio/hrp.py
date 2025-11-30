# src/quant_platform/portfolio/hrp.py
from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram

from quant_platform.portfolio.schemas import (
    PortfolioConfig,
    RiskModelInput,
    PortfolioResult,
    AssetWeight,
)


def _correlation_distance(corr: np.ndarray) -> np.ndarray:
    """
    Convert correlation matrix to distance matrix:
        d_ij = sqrt(0.5 * (1 - corr_ij))
    """
    return np.sqrt(0.5 * (1.0 - corr))


def _quasi_diag(link: np.ndarray) -> list[int]:
    """
    Quasi-diagonalization: returns leaf order after hierarchical clustering.
    Follows López de Prado's HRP algorithm.
    """
    sort_idx = dendrogram(link, no_plot=True)["leaves"]
    return list(sort_idx)


def _get_cluster_var(cov: np.ndarray, indices: list[int]) -> float:
    """
    Compute variance of a cluster: w' Σ w with inverse-variance weights.
    """
    sub_cov = cov[np.ix_(indices, indices)]
    ivp = 1.0 / np.diag(sub_cov)
    w = ivp / ivp.sum()
    return float(w @ sub_cov @ w)


def solve_hrp(
    rmi: RiskModelInput,
    config: PortfolioConfig,
) -> PortfolioResult:
    """
    Hierarchical Risk Parity (HRP) optimizer.

    Steps:
    1. Convert covariance -> correlation
    2. Hierarchical clustering on distance matrix
    3. Quasi-diagonalize correlation
    4. Recursive bisection to assign weights
    """

    if config.allow_short:
        raise ValueError("HRP currently supports only long-only portfolios.")

    cov = np.asarray(rmi.cov, dtype=float)
    corr = np.corrcoef(cov) if cov.shape[0] == cov.shape[1] else None
    corr = cov / np.sqrt(np.outer(np.diag(cov), np.diag(cov)))

    # 1) Correlation -> Distance matrix
    dist = _correlation_distance(corr)

    # 2) Hierarchical clustering
    link = linkage(dist[np.triu_indices_from(dist, k=1)], method="single")

    # 3) Quasi-diagonalize
    sort_idx = _quasi_diag(link)

    # Reorder covariance
    cov_sorted = cov[np.ix_(sort_idx, sort_idx)]

    # 4) Recursive bisection
    n = len(sort_idx)
    w = np.ones(n)
    clusters = [list(range(n))]

    while clusters:
        cluster = clusters.pop(0)
        if len(cluster) <= 1:
            continue

        split = len(cluster) // 2
        cluster_left = cluster[:split]
        cluster_right = cluster[split:]

        var_left = _get_cluster_var(cov_sorted, cluster_left)
        var_right = _get_cluster_var(cov_sorted, cluster_right)

        alpha = 1.0 - var_left / (var_left + var_right)

        w[cluster_left] *= alpha
        w[cluster_right] *= 1 - alpha

        clusters.append(cluster_left)
        clusters.append(cluster_right)

    # Normalize final weights
    w = w / w.sum()

    # Map back to original asset ordering
    w_final = np.zeros_like(w)
    for i, idx in enumerate(sort_idx):
        w_final[idx] = w[i]

    ret = float(w_final @ rmi.mu)
    vol = float(np.sqrt(w_final @ cov @ w_final))
    sharpe = ret / vol if vol > 0 else None

    weights = [
        AssetWeight(symbol=symbol, weight=float(w_final[i]))
        for i, symbol in enumerate(rmi.symbols)
    ]

    return PortfolioResult(
        weights=weights,
        expected_return=ret,
        expected_vol=vol,
        sharpe=sharpe,
        meta={"method": "hrp"},
    )
