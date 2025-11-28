# src/quant_platform/sde/processes/cir.py
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np

from quant_platform.sde.schemas import SimConfig
from quant_platform.sde.integrators import rng_with_seed


@dataclass
class CIRConfig:
    kappa: float
    theta: float
    sigma: float
    r0: float = 0.01


def cir_exact(cfg: CIRConfig, sim: SimConfig) -> np.ndarray:
    """Exact CIR sampler using non-central chi-square (unchanged)."""
    rng = rng_with_seed(sim.seed)
    k, theta, s = cfg.kappa, cfg.theta, cfg.sigma
    n_paths, n_steps, dt = sim.n_paths, sim.n_steps, sim.dt

    out = np.zeros((n_paths, n_steps + 1), dtype=float)
    out[:, 0] = cfg.r0

    for t in range(n_steps):
        r_t = out[:, t]
        phi = math.exp(-k * dt)
        c = (s**2) * (1 - phi) / (4.0 * k)
        df = 4.0 * k * theta / (s**2)
        lam = np.clip(4.0 * k * phi * r_t / (s**2 * (1 - phi)), 0.0, None)
        X = rng.noncentral_chisquare(df, lam, size=n_paths)
        out[:, t + 1] = np.maximum(c * X, 0.0)
    return out


def cir_euler(
    cfg: CIRConfig, sim: SimConfig, full_truncation: bool = True
) -> np.ndarray:
    """
    Euler-Maruyama discretization for CIR:
      r_{t+dt} = r_t + kappa*(theta - r_t) dt + sigma * sqrt(r_t) * sqrt(dt) * Z
    """
    rng = rng_with_seed(sim.seed)
    n_paths = sim.n_paths
    n_steps = sim.n_steps
    dt = sim.dt
    sqrt_dt = math.sqrt(dt)

    r = np.empty((n_paths, n_steps + 1), dtype=float)
    r[:, 0] = cfg.r0

    for t in range(n_steps):
        z = rng.normal(size=n_paths)
        r_prev = r[:, t]
        sq = (
            np.sqrt(np.maximum(r_prev, 0.0))
            if full_truncation
            else np.sqrt(np.clip(r_prev, 0.0, None))
        )
        drift = cfg.kappa * (cfg.theta - r_prev) * dt
        diff = cfg.sigma * sq * sqrt_dt * z
        r_next = r_prev + drift + diff
        if full_truncation:
            r_next = np.maximum(r_next, 0.0)
        r[:, t + 1] = r_next

    return r
