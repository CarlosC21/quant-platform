from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from quant_platform.sde.processes.ou import estimate_ou_from_path


class OUParams(BaseModel):
    """
    OU parameter estimates for a spread process.

    Returned by the stat-arb OU fit module, wrapping
    quant_platform.sde.processes.ou.estimate_ou_from_path.

    Attributes
    ----------
    kappa : float
        Mean-reversion speed.
    theta : float
        Long-run equilibrium level of the spread.
    sigma : float
        Diffusion volatility.
    dt : float
        Sampling interval (years).
    """

    kappa: float = Field(..., gt=0.0)
    theta: float
    sigma: float = Field(..., gt=0.0)
    dt: float = Field(..., gt=0.0)

    @property
    def half_life(self) -> float:
        """OU half-life: t_{1/2} = ln(2) / kappa."""
        return np.log(2.0) / self.kappa

    @property
    def stationary_std(self) -> float:
        """
        Stationary standard deviation of OU:

            sigma_s = sqrt(sigma^2 / (2*kappa))
        """
        return self.sigma / np.sqrt(2.0 * self.kappa)


def fit_ou_to_spread(
    spread: np.ndarray,
    dt: float,
) -> OUParams:
    """
    Fit OU parameters to a spread series using existing OU estimator.

    Parameters
    ----------
    spread : np.ndarray
        1D array of spread levels: s_t = y_t - beta * x_t.
    dt : float
        Sampling interval in years (e.g. 1/252 for daily data).

    Returns
    -------
    OUParams
        Model of OU dynamics for the spread.
    """
    if spread.ndim != 1:
        raise ValueError("Spread series must be 1D.")

    kappa, theta, sigma = estimate_ou_from_path(spread, dt)

    return OUParams(
        kappa=float(kappa),
        theta=float(theta),
        sigma=float(sigma),
        dt=float(dt),
    )
