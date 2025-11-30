# tests/trading/stat_arb/cointegration/test_engle_granger.py

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_platform.trading.stat_arb.cointegration.engle_granger import (
    EngleGrangerTester,
)
from quant_platform.trading.stat_arb.cointegration.schemas import EngleGrangerConfig


def _simulate_cointegrated_pair(
    n: int,
    beta: float,
    rng: np.random.Generator,
) -> tuple[pd.Series, pd.Series]:
    """
    Simulate a simple cointegrated pair:

        X_t: random walk
        Y_t: alpha + beta * X_t + stationary noise

    The noise is AR(1) with |rho| < 1 so that Y - beta X is stationary.
    """
    idx = pd.date_range("2020-01-01", periods=n, freq="D")

    # Random walk for X
    eps_x = rng.normal(scale=1.0, size=n)
    x = np.cumsum(eps_x)

    # Stationary AR(1) noise
    rho = 0.5
    eps_y = rng.normal(scale=0.5, size=n)
    noise = np.zeros(n, dtype=float)
    for t in range(1, n):
        noise[t] = rho * noise[t - 1] + eps_y[t]

    alpha_true = 2.0
    y = alpha_true + beta * x + noise

    series_x = pd.Series(x, index=idx, name="X")
    series_y = pd.Series(y, index=idx, name="Y")
    return series_y, series_x


def _simulate_non_cointegrated_pair(
    n: int,
    rng: np.random.Generator,
) -> tuple[pd.Series, pd.Series]:
    """
    Simulate two independent random walks (non-cointegrated).
    """
    idx = pd.date_range("2020-01-01", periods=n, freq="D")

    eps_x = rng.normal(scale=1.0, size=n)
    eps_y = rng.normal(scale=1.0, size=n)

    x = np.cumsum(eps_x)
    y = np.cumsum(eps_y)

    series_x = pd.Series(x, index=idx, name="X_nc")
    series_y = pd.Series(y, index=idx, name="Y_nc")
    return series_y, series_x


def test_engle_granger_detects_cointegration():
    rng = np.random.default_rng(42)
    n = 500
    beta_true = 1.5

    series_y, series_x = _simulate_cointegrated_pair(n=n, beta=beta_true, rng=rng)
    tester = EngleGrangerTester(config=EngleGrangerConfig(min_obs=100))

    result = tester.test_pair(series_y, series_x)

    assert result.coint is True
    assert result.symbol_y == "Y"
    assert result.symbol_x == "X"
    # Hedge ratio should be close to true beta
    assert abs(result.beta - beta_true) < 0.2
    # ADF statistic should be more negative than 5% critical value
    assert result.adf_result.test_stat < result.adf_result.crit_5


def test_engle_granger_rejects_non_cointegrated_pair():
    rng = np.random.default_rng(123)
    n = 500

    series_y, series_x = _simulate_non_cointegrated_pair(n=n, rng=rng)
    tester = EngleGrangerTester(config=EngleGrangerConfig(min_obs=100))

    result = tester.test_pair(series_y, series_x)

    # Typically, independent random walks should not be cointegrated
    assert (
        result.coint is False or result.adf_result.test_stat > result.adf_result.crit_5
    )
    assert result.n_obs >= 100
