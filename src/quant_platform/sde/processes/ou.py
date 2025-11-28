# src/quant_platform/sde/processes/ou.py
from __future__ import annotations
import math
import numpy as np
from typing import Tuple

from quant_platform.sde.schemas import OUConfig, SimConfig
from quant_platform.sde.integrators import rng_with_seed


def ou_exact(cfg: OUConfig, sim: SimConfig) -> np.ndarray:
    """Exact OU simulation (unchanged)."""
    rng = rng_with_seed(sim.seed)
    n_paths, n_steps, dt = sim.n_paths, sim.n_steps, sim.dt
    X = np.empty((n_paths, n_steps + 1), dtype=float)
    X[:, 0] = cfg.x0

    a = math.exp(-cfg.kappa * dt)
    mean_const = cfg.theta * (1.0 - a)
    var_const = max(cfg.sigma**2 * (1 - a**2) / (2.0 * cfg.kappa), 0.0)
    sqrt_var = math.sqrt(var_const)

    for t in range(n_steps):
        z = rng.normal(size=n_paths)
        X[:, t + 1] = X[:, t] * a + mean_const + sqrt_var * z
    return X


def ou_euler(config: OUConfig, sim: SimConfig) -> np.ndarray:
    """
    Euler-Maruyama discretization for OU:
        X_{t+dt} = X_t + kappa (theta - X_t) dt + sigma sqrt(dt) z
    """
    rng = rng_with_seed(sim.seed)
    n_paths = sim.n_paths
    n_steps = sim.n_steps
    dt = sim.dt

    X = np.empty((n_paths, n_steps + 1), dtype=float)
    X[:, 0] = config.x0
    sqrt_dt = math.sqrt(dt)

    for t in range(n_steps):
        z = rng.normal(size=n_paths)
        drift = config.kappa * (config.theta - X[:, t])
        X[:, t + 1] = X[:, t] + drift * dt + config.sigma * sqrt_dt * z

    return X


def estimate_ou_from_path(x: np.ndarray, dt: float) -> Tuple[float, float, float]:
    """Unchanged."""
    if x.ndim != 1:
        raise ValueError("x must be 1D")
    y, X = x[1:], x[:-1]
    n = X.size
    A = np.vstack([X, np.ones_like(X)]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    a_hat, b_hat = float(coef[0]), float(coef[1])
    a_hat = max(min(a_hat, 0.999999), -0.999999)
    kappa_hat = -math.log(a_hat) / dt if a_hat > 0 else 0.0
    theta_hat = b_hat / (1 - a_hat) if abs(1 - a_hat) > 1e-12 else float("nan")
    resid = y - (a_hat * X + b_hat)
    sigma_eta2 = float(np.sum(resid**2) / (n - 2))
    sigma_hat = math.sqrt(
        max(sigma_eta2 * 2 * max(kappa_hat, 1e-12) / (1 - a_hat**2), 0.0)
    )
    return kappa_hat, theta_hat, sigma_hat
