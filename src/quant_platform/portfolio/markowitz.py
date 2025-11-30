# src/quant_platform/portfolio/markowitz.py
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from functools import partial

from quant_platform.portfolio.schemas import (
    PortfolioConfig,
    RiskModelInput,
    PortfolioResult,
    AssetWeight,
)


# ---------------------------------------------------------------------
# Objective functions (required to avoid lambda assignment: E731)
# ---------------------------------------------------------------------
def objective_min_variance(w: np.ndarray, cov: np.ndarray) -> float:
    return float(w.T @ cov @ w)


def objective_target_return(
    w: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
    target: float,
) -> float:
    """
    Penalize deviations from target return + variance.
    Ensures optimizer hits target while remaining stable.
    """
    ret = w @ mu
    var = w.T @ cov @ w
    return float(1000.0 * (ret - target) ** 2 + var)


# ---------------------------------------------------------------------
# Constraints (converted from lambda → functions to satisfy ruff)
# ---------------------------------------------------------------------
def constraint_sum_to_one(w: np.ndarray) -> float:
    return float(np.sum(w) - 1.0)


def constraint_leverage_limit(w: np.ndarray, L: float) -> float:
    return float(L - np.sum(np.abs(w)))


# ---------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------
def solve_markowitz(
    rmi: RiskModelInput,
    config: PortfolioConfig,
) -> PortfolioResult:
    mu = np.asarray(rmi.mu, dtype=float)
    cov = np.asarray(rmi.cov, dtype=float)
    n = len(mu)

    # ----------------------------------------------------
    # Case 1: Closed-form min-variance (no target-return, no-shorts)
    # ----------------------------------------------------
    if (
        config.target_return is None
        and not config.allow_short
        and (config.leverage_limit is None or np.isclose(config.leverage_limit, 1.0))
    ):
        # w_i ∝ 1 / σ_i^2
        inv_var = 1.0 / np.diag(cov)
        w = inv_var / inv_var.sum()

        ret = float(w @ mu)
        vol = float(np.sqrt(w.T @ cov @ w))
        sharpe = ret / vol if vol > 0 else None

        weights = [
            AssetWeight(symbol=sym, weight=float(wi)) for sym, wi in zip(rmi.symbols, w)
        ]

        return PortfolioResult(
            weights=weights,
            expected_return=ret,
            expected_vol=vol,
            sharpe=sharpe,
            meta={"closed_form": True},
        )

    # ----------------------------------------------------
    # Case 2: SLSQP-based Markowitz (target-return or allow_short)
    # ----------------------------------------------------

    x0 = np.full(n, 1.0 / n)

    # Constraints: sum(w) = 1
    constraints = [{"type": "eq", "fun": constraint_sum_to_one}]

    # Leverage constraint: ∑|w| ≤ L
    if config.leverage_limit is not None:
        L = config.leverage_limit
        constraints.append(
            {"type": "ineq", "fun": partial(constraint_leverage_limit, L=L)}
        )

    # Bounds: if no shorts → enforce w_i ∈ [0,1]
    bounds = None
    if not config.allow_short:
        bounds = [(0.0, 1.0) for _ in range(n)]

    # Objective
    if config.target_return is None:
        objective = partial(objective_min_variance, cov=cov)
    else:
        tar = float(config.target_return)
        objective = partial(
            objective_target_return,
            mu=mu,
            cov=cov,
            target=tar,
        )

    # Optimization
    result = minimize(
        fun=objective,
        x0=x0,
        bounds=bounds,
        constraints=constraints,
        method="SLSQP",
        options={"ftol": 1e-12, "disp": False, "maxiter": 200},
    )

    if not result.success:
        raise RuntimeError(f"Markowitz optimization failed: {result.message}")

    w = result.x
    ret = float(w @ mu)
    vol = float(np.sqrt(w.T @ cov @ w))
    sharpe = ret / vol if vol > 0 else None

    weights = [
        AssetWeight(symbol=sym, weight=float(wi)) for sym, wi in zip(rmi.symbols, w)
    ]

    return PortfolioResult(
        weights=weights,
        expected_return=ret,
        expected_vol=vol,
        sharpe=sharpe,
        meta={"closed_form": False},
    )
