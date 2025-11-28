# src/quant_platform/sde/processes/vasicek.py
from __future__ import annotations
import math
import numpy as np

from quant_platform.sde.schemas import VasicekConfig, SimConfig
from quant_platform.sde.integrators import rng_with_seed


def vasicek_exact(cfg: VasicekConfig, sim: SimConfig) -> np.ndarray:
    """Exact Vasicek (unchanged)."""
    kappa, theta, sigma, r0 = cfg.kappa, cfg.theta, cfg.sigma, cfg.r0
    n_paths, n_steps, dt = sim.n_paths, sim.n_steps, sim.dt
    rng = np.random.default_rng(sim.seed)
    phi = math.exp(-kappa * dt)
    noise_std = math.sqrt(sigma**2 * (1 - phi**2) / (2 * kappa))

    R = np.zeros((n_paths, n_steps + 1), dtype=float)
    R[:, 0] = r0
    for t in range(n_steps):
        Z = rng.normal(size=n_paths)
        R[:, t + 1] = R[:, t] * phi + theta * (1 - phi) + noise_std * Z
    return R


def vasicek_euler(cfg: VasicekConfig, sim: SimConfig) -> np.ndarray:
    """
    Euler-Maruyama discretization for Vasicek short-rate model:
        r_{t+dt} = r_t + kappa (theta - r_t) dt + sigma sqrt(dt) z
    """
    rng = rng_with_seed(sim.seed)
    n_paths = sim.n_paths
    n_steps = sim.n_steps
    dt = sim.dt
    sqrt_dt = math.sqrt(dt)

    R = np.empty((n_paths, n_steps + 1), dtype=float)
    R[:, 0] = cfg.r0

    for t in range(n_steps):
        z = rng.normal(size=n_paths)
        drift = cfg.kappa * (cfg.theta - R[:, t])
        R[:, t + 1] = R[:, t] + drift * dt + cfg.sigma * sqrt_dt * z

    return R


def estimate_vasicek_from_path(r: np.ndarray, dt: float):
    """Unchanged."""
    x, y = r[:-1], r[1:]
    phi_hat, c = np.polyfit(x, y, 1)
    phi_hat = float(np.clip(phi_hat, 1e-10, 1 - 1e-10))
    kappa_hat = -np.log(phi_hat) / dt
    theta_hat = c / (1 - phi_hat)
    residuals = y - (phi_hat * x + c)
    var_eps = np.var(residuals, ddof=0)
    sigma_hat = np.sqrt(2 * kappa_hat * var_eps / (1 - phi_hat**2))
    return float(kappa_hat), float(theta_hat), float(sigma_hat)
