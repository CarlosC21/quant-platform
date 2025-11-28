# src/quant_platform/sde/gbm.py
from __future__ import annotations
import numpy as np
import math
from typing import Tuple

from quant_platform.sde.schemas import GBMConfig, SimConfig
from quant_platform.sde.integrators import rng_with_seed


def gbm_exact(config: GBMConfig, sim: SimConfig) -> np.ndarray:
    """
    Exact vectorized GBM simulation.

    Returns:
        S: np.ndarray shaped (n_paths, n_steps + 1)
    """
    rng = rng_with_seed(sim.seed)
    n_paths = sim.n_paths
    n_steps = sim.n_steps
    dt = sim.dt

    # generate standard normal increments of shape (n_paths, n_steps)
    z = rng.normal(size=(n_paths, n_steps)) * math.sqrt(dt)

    # log increments
    drift = (config.mu - 0.5 * config.sigma**2) * dt
    log_incs = drift + config.sigma * z

    # prepend log(S0)
    logS0 = math.log(config.s0)
    # cumulative sum along time axis; produce shape (n_paths, n_steps+1)
    logS = np.concatenate(
        [np.full((n_paths, 1), logS0), np.cumsum(log_incs, axis=1) + logS0], axis=1
    )
    return np.exp(logS)


def gbm_euler(config: GBMConfig, sim: SimConfig) -> np.ndarray:
    """
    Euler-Maruyama discretization for dS = mu S dt + sigma S dW.
    Returns array of shape (n_paths, n_steps+1).
    """
    rng = rng_with_seed(sim.seed)
    n_paths = sim.n_paths
    n_steps = sim.n_steps
    dt = sim.dt

    S = np.empty((n_paths, n_steps + 1), dtype=float)
    S[:, 0] = config.s0

    sqrt_dt = math.sqrt(dt)
    for t in range(n_steps):
        z = rng.normal(size=n_paths)
        S[:, t + 1] = (
            S[:, t] + config.mu * S[:, t] * dt + config.sigma * S[:, t] * sqrt_dt * z
        )

    return S


def estimate_gbm_from_prices(prices: np.ndarray, dt: float) -> Tuple[float, float]:
    """
    Estimate (mu, sigma) from equally spaced price observations (1D array length N+1).
    Uses log-returns; returns (mu_hat, sigma_hat).
    """
    if prices.ndim != 1:
        raise ValueError("prices must be 1D array for single path estimation")
    r = np.diff(np.log(prices))
    mean_r = float(np.mean(r))
    var_r = float(np.var(r, ddof=1))
    sigma_hat = math.sqrt(var_r / dt)
    mu_hat = (mean_r / dt) + 0.5 * sigma_hat**2
    return mu_hat, sigma_hat
