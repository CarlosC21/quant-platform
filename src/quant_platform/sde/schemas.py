# src/quant_platform/sde/schemas.py
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional


class SimConfig(BaseModel):
    """
    Generic simulation configuration.

    n_paths: number of Monte-Carlo paths
    n_steps: number of time steps (path length will be n_steps + 1 including t=0)
    dt: timestep size (in years)
    seed: Optional RNG seed for deterministic runs
    """

    n_paths: int = Field(..., ge=1)
    n_steps: int = Field(..., ge=1)
    dt: float = Field(..., gt=0.0)
    seed: Optional[int] = None


class GBMConfig(BaseModel):
    """
    Geometric Brownian Motion parameters.
    S_{t+dt} = S_t * exp((mu - 0.5*sigma^2) dt + sigma dW)
    """

    mu: float
    sigma: float = Field(..., gt=0.0)
    s0: float = Field(..., gt=0.0)


class OUConfig(BaseModel):
    """
    Ornstein-Uhlenbeck / mean-reverting process parameters.

    dX_t = kappa * (theta - X_t) dt + sigma dW_t
    """

    kappa: float = Field(..., gt=0.0)
    theta: float
    sigma: float = Field(..., gt=0.0)
    x0: float


class VasicekConfig(BaseModel):
    """
    Vasicek short-rate model:

        dr_t = kappa (theta - r_t) dt + sigma dW_t
    """

    kappa: float = Field(..., gt=0.0)
    theta: float
    sigma: float = Field(..., gt=0.0)
    r0: float
