# tests/trading/stat_arb/spreads/test_kalman_filter.py

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_platform.trading.stat_arb.spreads.kalman_filter import (
    KalmanHedgeConfig,
    kalman_hedge_filter,
)


def _simulate_time_varying_beta_pair(
    n: int,
    rng: np.random.Generator,
) -> tuple[pd.Series, pd.Series, np.ndarray]:
    """
    Simulate a pair (Y, X) where:

        X_t: random walk
        beta_t: slow random walk
        Y_t = alpha + beta_t * X_t + noise_t
    """
    idx = pd.date_range("2020-01-01", periods=n, freq="D")

    # Random walk for X
    eps_x = rng.normal(scale=1.0, size=n)
    x = np.cumsum(eps_x)

    # Time-varying beta_t (slow random walk)
    beta = np.empty(n, dtype=float)
    beta[0] = 1.0
    for t in range(1, n):
        beta[t] = beta[t - 1] + rng.normal(scale=0.01)

    alpha_true = 0.5
    noise = rng.normal(scale=0.5, size=n)
    y = alpha_true + beta * x + noise

    series_x = pd.Series(x, index=idx, name="X_kf")
    series_y = pd.Series(y, index=idx, name="Y_kf")
    return series_y, series_x, beta


def test_kalman_hedge_filter_tracks_beta():
    rng = np.random.default_rng(2025)
    n = 300

    series_y, series_x, beta_true = _simulate_time_varying_beta_pair(n=n, rng=rng)

    cfg = KalmanHedgeConfig(
        q_alpha=1e-6,
        q_beta=1e-4,
        r=0.25,
        init_alpha=0.0,
        init_beta=1.0,
        init_var_scale=1e3,
        min_obs=50,
    )

    result = kalman_hedge_filter(series_y, series_x, config=cfg)

    assert result.beta_t.shape == (n,)
    assert result.spread.shape == (n,)
    assert result.alpha_t is not None
    assert result.timestamps.shape[0] == n

    # Correlation between true beta and estimated beta_t should be reasonably high
    corr = np.corrcoef(beta_true, result.beta_t)[0, 1]
    assert corr > 0.7


def test_kalman_hedge_filter_min_obs():
    rng = np.random.default_rng(2026)
    n = 20

    series_y, series_x, _ = _simulate_time_varying_beta_pair(n=n, rng=rng)

    cfg = KalmanHedgeConfig(min_obs=30)

    try:
        kalman_hedge_filter(series_y, series_x, config=cfg)
        raised = False
    except ValueError:
        raised = True

    assert raised is True
