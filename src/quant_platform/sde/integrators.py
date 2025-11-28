# src/quant_platform/sde/integrators.py
from __future__ import annotations
import numpy as np
from typing import Tuple
import math


def rng_with_seed(seed: int | None) -> np.random.Generator:
    """Create a numpy Generator deterministically from seed if provided."""
    return np.random.default_rng(seed)


def euler_maruyama_step(
    x: float, drift: float, diffusion: float, dt: float, dW: float
) -> float:
    """
    Single Euler-Maruyama step:
    X_{t+dt} = X_t + a(X_t)*dt + b(X_t)*dW
    Here we pass precomputed drift and diffusion scalars for speed.
    """
    return x + drift * dt + diffusion * dW


def milstein_step(
    x: float,
    drift: float,
    diffusion: float,
    diff_derivative: float,
    dt: float,
    dW: float,
) -> float:
    """
    Single Milstein step:
    X_{t+dt} = X_t + a dt + b dW + 0.5 b b' (dW^2 - dt)
    where diff_derivative is b'(X_t)
    """
    return (
        x
        + drift * dt
        + diffusion * dW
        + 0.5 * diffusion * diff_derivative * (dW * dW - dt)
    )


def gaussian_increments(rng: np.random.Generator, shape: Tuple[int, int], dt: float):
    """
    Return normal increments with variance dt, shape = (n_paths, n_steps)
    """
    return rng.normal(size=shape) * math.sqrt(dt)
