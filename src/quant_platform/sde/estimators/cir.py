# src/quant_platform/sde/estimators/cir.py
from __future__ import annotations
from typing import Tuple

import math
import numpy as np
from scipy.optimize import minimize
from scipy.stats import ncx2


def _cir_negloglik_logparams(z: np.ndarray, r: np.ndarray, dt: float) -> float:
    log_kappa, theta, log_sigma = float(z[0]), float(z[1]), float(z[2])
    if theta <= 0.0:
        return 1e18
    kappa = float(math.exp(log_kappa))
    sigma = float(math.exp(log_sigma))
    if not (kappa > 0 and sigma > 0):
        return 1e18

    n = r.size - 1
    phi = math.exp(-kappa * dt)
    denom = sigma * sigma * (1.0 - phi)
    if denom <= 0:
        return 1e18
    c = (sigma * sigma * (1.0 - phi)) / (4.0 * kappa)
    df = 4.0 * kappa * theta / (sigma * sigma)
    if df <= 0:
        return 1e18

    total = 0.0
    for i in range(n):
        rt = float(r[i])
        rt1 = float(r[i + 1])
        lam = (4.0 * kappa * phi * rt) / (sigma * sigma * (1.0 - phi))
        lam = max(lam, 0.0)
        if c <= 0:
            return 1e18
        x = rt1 / c
        try:
            pdf_val = ncx2.pdf(x, df, lam) / c
        except Exception:
            return 1e18
        if not np.isfinite(pdf_val) or pdf_val <= 0.0:
            return 1e18
        total += -math.log(pdf_val)
    return float(total)


def _regression_initial_guess(r: np.ndarray, dt: float) -> Tuple[float, float, float]:
    r = np.asarray(r, dtype=float)
    x = r[:-1]
    y = r[1:]
    n = x.size
    if n < 3:
        raise ValueError("path too short")

    xm = float(x.mean())
    ym = float(y.mean())
    Sxy = float(np.sum((x - xm) * (y - ym)))
    Sxx = float(np.sum((x - xm) ** 2))

    if Sxx <= 0:
        phi = 0.99
    else:
        phi = Sxy / Sxx
    phi = float(np.clip(phi, 1e-8, 0.9999999999))
    alpha = ym - phi * xm
    one_minus_phi = 1.0 - phi
    if one_minus_phi <= 0:
        kappa = 1.0 / dt
    else:
        kappa = max(-math.log(phi) / dt, 1e-8)
    theta = float(alpha / max(one_minus_phi, 1e-12))

    resid = y - (alpha + phi * x)
    sigma_eps2 = float(np.sum(resid**2) / max(n, 1))
    denom = max((1.0 - phi * phi), 1e-12)
    sigma_sq = max(sigma_eps2 * (2.0 * kappa) / denom, 1e-12)
    sigma = float(math.sqrt(sigma_sq))
    kappa = max(kappa, 1e-8)
    theta = max(theta, 1e-12)
    sigma = max(sigma, 1e-8)
    return kappa, theta, sigma


def _conditional_theta_sigma_given_kappa(
    r: np.ndarray, dt: float, kappa: float
) -> Tuple[float, float]:
    r = np.asarray(r, dtype=float)
    x = r[:-1]
    y = r[1:]
    n = x.size

    phi = math.exp(-kappa * dt)
    xm = float(x.mean())
    ym = float(y.mean())
    alpha = ym - phi * xm
    one_minus_phi = 1.0 - phi
    theta = float(alpha / max(one_minus_phi, 1e-12))

    resid = y - (alpha + phi * x)
    sigma_eps2 = float(np.sum(resid**2) / max(n, 1))
    denom = max((1.0 - phi * phi), 1e-12)
    sigma_sq = max(sigma_eps2 * (2.0 * kappa) / denom, 1e-12)
    sigma = float(math.sqrt(sigma_sq))
    return theta, sigma


def _nll_given_kappa(r: np.ndarray, dt: float, kappa: float) -> float:
    theta, sigma = _conditional_theta_sigma_given_kappa(r, dt, kappa)
    z = np.array(
        [
            math.log(max(kappa, 1e-12)),
            float(max(theta, 1e-12)),
            math.log(max(sigma, 1e-12)),
        ],
        dtype=float,
    )
    return _cir_negloglik_logparams(z, r, dt)


def _grid_search_kappa(
    r: np.ndarray, dt: float, kmin: float = 0.01, kmax: float = 5.0, ngrid: int = 400
) -> Tuple[float, float]:
    kappas = np.linspace(kmin, kmax, ngrid)
    best_k = None
    best_nll = float("inf")

    for k in kappas:
        try:
            nll = _nll_given_kappa(r, dt, float(k))
        except Exception:
            nll = float("inf")
        if nll < best_nll:
            best_nll = nll
            best_k = float(k)
    if best_k is None:
        return float(kmin), float("inf")
    return float(best_k), float(best_nll)


def estimate_cir_mle(
    r: np.ndarray, dt: float, initial: Tuple[float, float, float] | None = None
) -> Tuple[float, float, float]:
    r = np.asarray(r, dtype=float)
    if r.ndim != 1:
        raise ValueError("r must be 1D array")
    n = r.size - 1
    if n < 10:
        raise ValueError("Path too short for estimation")

    # regression guess only used if needed
    try:
        k0, th0, s0 = _regression_initial_guess(r, dt)
    except Exception:
        _ = 0.5, float(r.mean()), 0.1

    # --- IMPORTANT: use a broad fixed grid to find the best kappa region ---
    best_k, best_nll = _grid_search_kappa(r, dt, kmin=0.01, kmax=5.0, ngrid=400)

    # conditional theta/sigma at best_k
    theta0, sigma0 = _conditional_theta_sigma_given_kappa(r, dt, best_k)
    z0 = np.array(
        [
            math.log(max(best_k, 1e-12)),
            float(max(theta0, 1e-12)),
            math.log(max(sigma0, 1e-12)),
        ],
        dtype=float,
    )

    # local refinement in log-parameter space
    try:
        res = minimize(
            lambda z: _cir_negloglik_logparams(z, r, dt),
            z0,
            method="L-BFGS-B",
            bounds=[(math.log(1e-12), None), (1e-12, None), (math.log(1e-12), None)],
            options={"maxiter": 800, "ftol": 1e-12},
        )
    except Exception:
        res = None

    if res is None or not res.success:
        k_hat = float(best_k)
        theta_hat = float(theta0)
        sigma_hat = float(sigma0)
    else:
        k_hat = float(math.exp(res.x[0]))
        theta_hat = float(res.x[1])
        sigma_hat = float(math.exp(res.x[2]))

    # Clip plausibility (do NOT fallback to biased regression)
    k_hat = float(np.clip(k_hat, 1e-8, 1e3))
    theta_hat = float(max(theta_hat, 1e-12))
    sigma_hat = float(np.clip(sigma_hat, 1e-12, 1e3))

    return float(k_hat), float(theta_hat), float(sigma_hat)
