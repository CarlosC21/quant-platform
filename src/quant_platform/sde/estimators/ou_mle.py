# src/quant_platform/sde/estimators/ou_mle.py

from __future__ import annotations

from math import log
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class OUParams(BaseModel):
    """
    Maximum likelihood estimate of OU parameters.

    Continuous-time OU:
        dX_t = kappa * (theta - X_t) dt + sigma dW_t
    """

    kappa: float = Field(..., gt=0.0, description="Mean reversion speed.")
    theta: float = Field(..., description="Long-run mean level.")
    sigma: float = Field(..., gt=0.0, description="Diffusion volatility.")
    dt: float = Field(..., gt=0.0, description="Sampling interval in years.")
    method: Literal["discrete_mle"] = Field(
        "discrete_mle", description="Estimation method identifier."
    )


class OUMLEResult(OUParams):
    """
    Extended OU MLE result including discrete-time quantities.
    """

    phi: float = Field(..., description="AR(1) coefficient: phi = exp(-kappa * dt).")
    sigma_eps: float = Field(
        ..., gt=0.0, description="Std dev of discrete-time innovations."
    )
    n_obs: int = Field(..., gt=0, description="Number of observations used.")


def _to_1d_array(x: Sequence[float] | np.ndarray | pd.Series) -> np.ndarray:
    """
    Convert input series to a 1D numpy array of dtype float.

    Parameters
    ----------
    x : Sequence[float] | np.ndarray | pd.Series
        Input series of OU observations.

    Returns
    -------
    np.ndarray
        1D numpy array of floats.

    Raises
    ------
    ValueError
        If array is not 1-dimensional or has fewer than 3 points.
    """
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError("OU MLE expects a 1D series of observations.")
    if arr.size < 3:
        raise ValueError(
            "OU MLE requires at least 3 observations to estimate parameters."
        )
    if not np.isfinite(arr).all():
        raise ValueError("OU MLE input contains non-finite values.")
    return arr


def estimate_ou_mle(
    x: Sequence[float] | np.ndarray | pd.Series,
    dt: float,
) -> OUMLEResult:
    """
    Estimate OU parameters using discrete-time MLE via AR(1) mapping.

    We model the sampled OU as:
        X_{t+1} = theta + phi * (X_t - theta) + eps_t,
    or equivalently:
        X_{t+1} = alpha + phi * X_t + eps_t,

    with:
        phi = exp(-kappa * dt),
        alpha = theta * (1 - phi),
        Var(eps_t) = sigma^2 / (2 * kappa) * (1 - phi^2).

    The MLE for (alpha, phi) coincides with OLS for the AR(1) regression.

    Parameters
    ----------
    x : Sequence[float] | np.ndarray | pd.Series
        1D time series of OU observations, sampled at fixed interval dt.
    dt : float
        Sampling interval in years (e.g., 1/252 for daily data).

    Returns
    -------
    OUMLEResult
        Pydantic model with continuous-time parameters (kappa, theta, sigma)
        and discrete-time counterparts (phi, sigma_eps).

    Raises
    ------
    ValueError
        If dt <= 0, variance of x_t is zero, or AR(1) coefficient invalid.
    """
    if dt <= 0.0:
        raise ValueError("dt must be positive for OU MLE estimation.")

    arr = _to_1d_array(x)
    x_t = arr[:-1]
    x_tp1 = arr[1:]
    n = x_t.size

    # Means of predictor and response.
    x_bar = float(x_t.mean())
    y_bar = float(x_tp1.mean())

    # AR(1) OLS estimates: y = alpha + phi * x + eps
    x_centered = x_t - x_bar
    y_centered = x_tp1 - y_bar

    s_xx = float(np.dot(x_centered, x_centered))
    if s_xx <= 0.0:
        raise ValueError("Variance of x_t is zero; cannot estimate OU parameters.")

    s_xy = float(np.dot(x_centered, y_centered))
    phi_hat = s_xy / s_xx

    # OU implies phi = exp(-kappa * dt) > 0.
    if phi_hat <= 0.0 or phi_hat >= 1.0:
        raise ValueError(
            f"Estimated AR(1) coefficient phi={phi_hat:.4f} not in (0, 1); "
            "OU model may not be appropriate for this series."
        )

    alpha_hat = y_bar - phi_hat * x_bar
    theta_hat = alpha_hat / (1.0 - phi_hat)

    # Residuals and discrete-time innovation variance.
    residuals = x_tp1 - (alpha_hat + phi_hat * x_t)
    sigma_eps_sq = float(np.dot(residuals, residuals) / n)
    if sigma_eps_sq <= 0.0:
        raise ValueError("Non-positive innovation variance; cannot estimate sigma.")

    kappa_hat = -log(phi_hat) / dt
    # Map discrete innovation variance to continuous-time sigma.
    sigma_hat = (sigma_eps_sq * 2.0 * kappa_hat / (1.0 - phi_hat**2)) ** 0.5

    return OUMLEResult(
        kappa=kappa_hat,
        theta=theta_hat,
        sigma=sigma_hat,
        dt=dt,
        phi=phi_hat,
        sigma_eps=sigma_eps_sq**0.5,
        n_obs=n + 1,
    )


__all__ = ["OUParams", "OUMLEResult", "estimate_ou_mle"]
