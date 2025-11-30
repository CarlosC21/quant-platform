# src/quant_platform/portfolio/risk_parity.py
from __future__ import annotations

import numpy as np

from quant_platform.portfolio.schemas import (
    PortfolioConfig,
    RiskModelInput,
    PortfolioResult,
    AssetWeight,
)


def _compute_portfolio_vol(w: np.ndarray, cov: np.ndarray) -> float:
    return float(np.sqrt(w.T @ cov @ w))


def _risk_contributions(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """
    RC_i = w_i * (Σ w)_i
    """
    sigma_w = cov @ w
    return w * sigma_w


def solve_risk_parity(
    rmi: RiskModelInput,
    config: PortfolioConfig,
    tol: float = 1e-10,
    max_iter: int = 10_000,
) -> PortfolioResult:
    """
    Equal Risk Contribution (Risk Parity) optimizer.

    Solves:
        w_i * (Σ w)_i = constant  for all i
    subject to:
        w_i >= 0, sum(w_i) = 1

    Uses cyclic coordinate descent (CCD) for stability.
    """

    cov = np.asarray(rmi.cov, dtype=float)
    n = cov.shape[0]

    if config.allow_short:
        raise ValueError("Risk parity currently supports long-only portfolios only.")

    # --- initialize weights uniformly ---
    w = np.full(n, 1.0 / n)

    # Precompute diagonal for speed
    diag_cov = np.diag(cov)

    for iteration in range(max_iter):
        w_old = w.copy()

        for i in range(n):
            # Compute the target risk contribution:
            # All RC_i equal => RC_i = total_var / n
            sigma_w = cov @ w
            total_var = w @ sigma_w
            target = total_var / n

            # Solve quadratic equation for w_i
            # w_i * (Sigma w)_i = target  (where w_i affects (Sigma w)_i linearly)
            a = diag_cov[i]
            b = sigma_w[i] - a * w[i]
            # Solve: a * w_i^2 + b * w_i - target = 0
            disc = b * b + 4 * a * target
            w_i_new = (-b + np.sqrt(disc)) / (2 * a)

            # Enforce positivity
            w[i] = max(w_i_new, 1e-16)

        # Renormalize to sum to 1
        w = w / w.sum()

        if np.linalg.norm(w - w_old, ord=1) < tol:
            break
    else:
        raise RuntimeError("Risk parity failed to converge.")

    # Final portfolio stats
    ret = float(w @ rmi.mu)
    vol = _compute_portfolio_vol(w, cov)
    sharpe = ret / vol if vol > 0 else None

    weights = [
        AssetWeight(symbol=symbol, weight=float(wi))
        for symbol, wi in zip(rmi.symbols, w)
    ]

    return PortfolioResult(
        weights=weights,
        expected_return=ret,
        expected_vol=vol,
        sharpe=sharpe,
        meta={
            "method": "risk_parity",
            "iterations": iteration,
            "converged": iteration < max_iter - 1,
        },
    )
