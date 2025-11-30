# tests/risk/test_var.py
import numpy as np

from quant_platform.risk.var import (
    compute_historical_var,
    compute_parametric_var,
    compute_parametric_var_from_series,
)


def test_compute_historical_var_basic():
    # Construct returns where the 20% quantile is clearly -0.10
    returns = np.array([-0.10, -0.10, -0.05, 0.0, 0.0], dtype=float)
    alpha = 0.2

    var = compute_historical_var(returns, alpha)

    # VaR defined as positive loss: -q_alpha
    assert np.isclose(var, 0.10, atol=1e-12)


def test_compute_parametric_var_basic():
    mean = 0.0
    sigma = 0.02
    alpha = 0.05

    var = compute_parametric_var(mean, sigma, alpha)

    # Expected: sigma * |z_alpha| with z_0.05 ≈ -1.64485
    expected = 0.02 * 1.6448536269514722
    assert np.isclose(var, expected, atol=1e-6)


def test_compute_parametric_var_from_series():
    rng = np.random.default_rng(42)
    returns = rng.normal(loc=0.001, scale=0.02, size=500)

    alpha = 0.05
    var, mu_hat, sigma_hat = compute_parametric_var_from_series(returns, alpha)

    # Sanity checks
    assert sigma_hat > 0.0
    assert var > 0.0
