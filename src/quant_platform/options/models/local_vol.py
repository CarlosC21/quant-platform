# src/quant_platform/options/models/local_vol.py
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from typing import Union


class LocalVolSurface:
    def __init__(self, strikes, maturities, vol_matrix):
        """
        strikes: 1D array-like of strikes (length M)
        maturities: 1D array-like of maturities (length N)
        vol_matrix: 2D array-like shape (M, N) corresponding to strikes x maturities
        """
        self._strikes = np.asarray(strikes, dtype=float)
        self._maturities = np.asarray(maturities, dtype=float)
        self._vol_matrix = np.asarray(vol_matrix, dtype=float)

        if self._vol_matrix.shape != (self._strikes.size, self._maturities.size):
            raise ValueError("vol_matrix shape must be (len(strikes), len(maturities))")

        # RegularGridInterpolator works well for a structured grid and supports extrapolation
        # when fill_value=None (it will extrapolate).
        self._interp = RegularGridInterpolator(
            (self._strikes, self._maturities),
            self._vol_matrix,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

    def _nearest_vol(self, K_val: float, T_val: float) -> float:
        """Nearest-grid fallback: return vol at nearest grid point."""
        ik = np.abs(self._strikes - K_val).argmin()
        it = np.abs(self._maturities - T_val).argmin()
        return float(self._vol_matrix[ik, it])

    def vol(
        self, K: Union[float, np.ndarray], T: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Return local vol for given strike(s) K and maturity(s) T.
        - If scalar inputs are provided, returns a Python float.
        - If array inputs are provided, returns a numpy array of the same shape.
        """
        K_arr = np.atleast_1d(K).astype(float)
        T_arr = np.atleast_1d(T).astype(float)

        if K_arr.shape != T_arr.shape:
            # allow broadcasting of scalars
            if K_arr.size == 1:
                K_arr = np.full_like(T_arr, K_arr.item(), dtype=float)
            elif T_arr.size == 1:
                T_arr = np.full_like(K_arr, T_arr.item(), dtype=float)
            else:
                # try to broadcast shapes
                K_arr, T_arr = np.broadcast_arrays(K_arr, T_arr)

        pts = np.column_stack([K_arr.ravel(), T_arr.ravel()])
        vols = self._interp(pts)  # shape (len(pts),)

        # Replace any NaNs from interpolation with nearest-grid fallback
        if np.any(np.isnan(vols)):
            nan_idx = np.nonzero(np.isnan(vols))[0]
            for idx in nan_idx:
                kv = pts[idx, 0]
                tv = pts[idx, 1]
                vols[idx] = self._nearest_vol(kv, tv)

        vols = vols.reshape(K_arr.shape)

        if vols.size == 1:
            return float(vols.item())
        return vols


class LocalVolOption:
    def __init__(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        local_vol_surface: LocalVolSurface,
        option_type: str = "call",
    ):
        self.S = float(S)
        self.K = float(K)
        self.T = float(T)
        self.r = float(r)
        self.option_type = option_type.lower()
        self.local_vol_surface = local_vol_surface

        # get scalar sigma at (K,T); floor at tiny epsilon to avoid division by zero
        sigma_val = local_vol_surface.vol(self.K, self.T)
        if isinstance(sigma_val, np.ndarray):
            # should be scalar, but handle defensively
            if sigma_val.size == 1:
                sigma_val = sigma_val.item()
            else:
                raise ValueError(
                    "Expected scalar sigma for given (K,T), received array."
                )
        self.sigma = float(max(sigma_val, 1e-8))
        if self.sigma <= 0:
            raise ValueError("Local volatility must be positive.")

    def price(self) -> float:
        """Placeholder Black-Scholes price using local vol at (K,T)."""
        from math import exp, log, sqrt
        from scipy.stats import norm

        if self.T == 0:
            return max(
                0.0,
                (self.S - self.K) if self.option_type == "call" else (self.K - self.S),
            )

        sigma_sqrt_T = sqrt(self.T) * self.sigma
        denom = max(sigma_sqrt_T, 1e-16)
        d1 = (log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / denom
        d2 = d1 - sigma_sqrt_T

        if self.option_type == "call":
            return self.S * norm.cdf(d1) - self.K * exp(-self.r * self.T) * norm.cdf(d2)
        else:
            return self.K * exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(
                -d1
            )

    def delta(self) -> float:
        """Analytic Black-Scholes-like delta using local vol (placeholder)."""
        from math import sqrt, log
        from scipy.stats import norm

        if self.T == 0:
            if self.option_type == "call":
                return 1.0 if self.S > self.K else 0.0
            else:
                return -1.0 if self.S < self.K else 0.0

        sigma_sqrt_T = sqrt(self.T) * self.sigma
        denom = max(sigma_sqrt_T, 1e-16)
        d1 = (log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / denom
        return (
            float(norm.cdf(d1))
            if self.option_type == "call"
            else float(norm.cdf(d1) - 1.0)
        )


def delta_hedge_simulator(
    option_cls, S_path, r, K, option_type="call", local_vol_surface=None
):
    """
    Delta-hedge a local vol option along a given spot path.

    Parameters
    ----------
    option_cls : class
        Option class (LocalVolOption)
    S_path : array-like
        Spot price path
    r : float
        Risk-free rate
    K : float
        Strike
    option_type : str
        'call' or 'put'
    local_vol_surface : LocalVolSurface
        Required for local vol option

    Returns
    -------
    pnl : float
        Profit and loss from delta hedging
    """
    S_path = np.asarray(S_path)
    N = len(S_path)
    dt = 1.0 / N  # normalized timestep; matches local vol Euler paths
    cash = 0.0
    delta_prev = 0.0

    # initialize option object with current spot and remaining T=1 (normalized)
    opt = option_cls(
        S=S_path[0],
        K=K,
        T=1.0,
        r=r,
        option_type=option_type,
        local_vol_surface=local_vol_surface,
    )

    for i in range(N - 1):
        t = (N - i - 1) * dt  # remaining time fraction
        # update spot and T
        opt.S = S_path[i]
        opt.T = t
        delta = opt.delta()
        d_delta = delta - delta_prev

        # adjust cash by buying/selling underlying
        cash -= d_delta * S_path[i]
        # accrue risk-free interest
        cash *= np.exp(r * dt)
        delta_prev = delta

    # final payoff
    if option_type == "call":
        payoff = max(S_path[-1] - K, 0)
    else:
        payoff = max(K - S_path[-1], 0)

    pnl = payoff + cash - delta_prev * S_path[-1]
    return pnl
