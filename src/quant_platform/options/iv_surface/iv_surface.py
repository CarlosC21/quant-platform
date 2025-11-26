# src/quant_platform/options/iv_surface/iv_surface.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import polars as pl

try:
    from scipy.interpolate import Akima1DInterpolator
except Exception:
    Akima1DInterpolator = None

from src.quant_platform.options.greeks.greeks import GreeksCalculator


@dataclass
class IVGrid:
    strikes: np.ndarray
    taus: np.ndarray
    iv_grid: np.ndarray


class IVSurface:
    """
    IVSurface using sequential Akima interpolation.
    """

    def __init__(self, strikes: np.ndarray, taus: np.ndarray, iv_grid: np.ndarray):
        strikes = np.asarray(strikes, dtype=float)
        taus = np.asarray(taus, dtype=float)
        iv_grid = np.asarray(iv_grid, dtype=float)

        if strikes.ndim != 1 or taus.ndim != 1:
            raise ValueError("strikes and taus must be 1D arrays")
        if iv_grid.ndim != 2 or iv_grid.shape != (len(taus), len(strikes)):
            raise ValueError("iv_grid must be 2D shape (len(taus), len(strikes))")

        sort_k_idx = np.argsort(strikes)
        strikes = strikes[sort_k_idx]
        iv_grid = iv_grid[:, sort_k_idx]

        sort_t_idx = np.argsort(taus)
        taus = taus[sort_t_idx]
        iv_grid = iv_grid[sort_t_idx, :]

        self.strikes = strikes
        self.taus = taus
        self.iv_grid = iv_grid

        self._smile_interp = [None] * len(self.taus)
        self._build_smile_interpolators()
        self._time_interp_cache: Dict[float, object] = {}

    @classmethod
    def from_grid(
        cls,
        strikes: Sequence[float],
        taus: Sequence[float],
        iv_grid: Sequence[Sequence[float]],
    ):
        return cls(
            np.asarray(strikes, dtype=float),
            np.asarray(taus, dtype=float),
            np.asarray(iv_grid, dtype=float),
        )

    def _build_smile_interpolators(self):
        for i, row in enumerate(self.iv_grid):
            ks, ivs = self.strikes, row
            mask = np.isfinite(ivs) & (ivs > 0) & (ks > 0)
            ks_m, ivs_m = ks[mask], ivs[mask]

            if ks_m.size == 0:

                def f_empty(K, _ks=ks):
                    return np.full_like(np.atleast_1d(K), np.nan, dtype=float)

                self._smile_interp[i] = f_empty
                continue

            if (Akima1DInterpolator is not None) and ks_m.size >= 3:
                try:
                    ak = Akima1DInterpolator(ks_m, ivs_m)

                    def f_ak(
                        K,
                        ak=ak,
                        kmin=ks_m[0],
                        kmax=ks_m[-1],
                        left=ivs_m[0],
                        right=ivs_m[-1],
                    ):
                        K_arr = np.asarray(K)
                        return np.where(
                            K_arr < kmin,
                            left,
                            np.where(K_arr > kmax, right, ak(K_arr)),
                        )

                    self._smile_interp[i] = f_ak
                    continue
                except Exception:
                    pass

            def make_linear(ks_m_local, ivs_m_local):
                def _f(K):
                    K_arr = np.atleast_1d(K).astype(float)
                    out = np.interp(
                        K_arr,
                        ks_m_local,
                        ivs_m_local,
                        left=ivs_m_local[0],
                        right=ivs_m_local[-1],
                    )
                    return out

                return _f

            self._smile_interp[i] = make_linear(ks_m, ivs_m)

    def iv(self, K: float, tau: float) -> float:
        K, tau = float(K), float(tau)
        tau = max(min(tau, self.taus[-1]), self.taus[0])
        ivs_at_k = np.array([interp(K) for interp in self._smile_interp], dtype=float)
        if not np.all(np.isfinite(ivs_at_k)):
            finite_idx = np.where(np.isfinite(ivs_at_k))[0]
            if finite_idx.size == 0:
                return float("nan")
            nearest = finite_idx[np.argmin(np.abs(self.taus[finite_idx] - tau))]
            return float(ivs_at_k[nearest])

        idx = np.searchsorted(self.taus, tau)
        if idx < len(self.taus) and np.isclose(self.taus[idx], tau):
            return float(ivs_at_k[idx])

        cache_key = float(round(K, 8))
        time_interp = self._time_interp_cache.get(cache_key)
        if time_interp is None:
            mask = np.isfinite(ivs_at_k)
            taus_m, ivs_m = self.taus[mask], ivs_at_k[mask]
            if taus_m.size == 0:
                return float("nan")

            if (Akima1DInterpolator is not None) and taus_m.size >= 3:
                try:
                    ak_t = Akima1DInterpolator(taus_m, ivs_m)

                    def _time_interp(
                        qtau,
                        ak=ak_t,
                        tmin=taus_m[0],
                        tmax=taus_m[-1],
                        left=ivs_m[0],
                        right=ivs_m[-1],
                    ):
                        q = np.asarray(qtau)
                        return np.where(
                            q < tmin,
                            left,
                            np.where(q > tmax, right, ak(q)),
                        )

                    time_interp = _time_interp
                except Exception:

                    def _time_interp(q):
                        return np.interp(
                            q, taus_m, ivs_m, left=ivs_m[0], right=ivs_m[-1]
                        )

                    time_interp = _time_interp
            else:

                def _time_interp(q):
                    return np.interp(q, taus_m, ivs_m, left=ivs_m[0], right=ivs_m[-1])

                time_interp = _time_interp

            self._time_interp_cache[cache_key] = time_interp

        return float(np.asarray(time_interp(tau)).item())

    def compute_greeks(self, df: pl.DataFrame, r: float = 0.01) -> pl.DataFrame:
        required_cols = {"strike", "ttm", "option_type", "underlying_price"}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")

        strikes = df["strike"].to_numpy()
        ttm = df["ttm"].to_numpy()
        ivs = np.array([self.iv(K, tau) for K, tau in zip(strikes, ttm)])
        df_with_iv = df.with_columns(
            [pl.Series("iv", ivs), pl.Series("r", np.full(len(df), r))]
        )

        calc = GreeksCalculator(df_with_iv)
        return calc.compute()

    def smile(self, tau: float) -> Tuple[np.ndarray, np.ndarray]:
        ks = self.strikes.copy()
        ivs = np.array([self.iv(k, tau) for k in ks], dtype=float)
        return ks, ivs

    def term_structure(self, K: float) -> Tuple[np.ndarray, np.ndarray]:
        taus = self.taus.copy()
        ivs = np.array([self.iv(K, t) for t in taus], dtype=float)
        return taus, ivs

    def dsigma_dK(self, K: float, tau: float, eps: Optional[float] = None) -> float:
        K, tau = float(K), float(tau)
        if eps is None:
            eps = max(1e-4 * max(abs(K), 1.0), 1e-6)
        return (self.iv(K + eps, tau) - self.iv(K - eps, tau)) / (2 * eps)

    def d2sigma_dK2(self, K: float, tau: float, eps: Optional[float] = None) -> float:
        K, tau = float(K), float(tau)
        if eps is None:
            eps = max(1e-3 * max(abs(K), 1.0), 1e-6)
        return (self.iv(K + eps, tau) - 2 * self.iv(K, tau) + self.iv(K - eps, tau)) / (
            eps**2
        )

    def dsigma_dT(self, K: float, tau: float, eps: Optional[float] = None) -> float:
        K, tau = float(K), float(tau)
        if eps is None:
            eps = max(1e-4 * max(tau, 1e-3), 1e-6)
        return (self.iv(K, tau + eps) - self.iv(K, max(tau - eps, 1e-12))) / (2 * eps)

    def grid(self) -> IVGrid:
        return IVGrid(self.strikes.copy(), self.taus.copy(), self.iv_grid.copy())

    def to_dict(self) -> Dict:
        return {
            "strikes": self.strikes.copy(),
            "taus": self.taus.copy(),
            "iv_grid": self.iv_grid.copy(),
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "IVSurface":
        return cls(
            np.asarray(d["strikes"], dtype=float),
            np.asarray(d["taus"], dtype=float),
            np.asarray(d["iv_grid"], dtype=float),
        )
