# tests/risk/test_cvar.py
import numpy as np

from quant_platform.risk.cvar import (
    compute_historical_cvar,
    compute_parametric_cvar,
    compute_parametric_cvar_from_series,
)


def test_compute_historical_cvar_basic():
    # Tail is clearly -0.10 values so CVaR should be 0.10
    returns = np.array([-0.10, -0.10, -0.10, 0.0, 0.0], dtype=float)
    alpha = 0.4

    cvar = compute_historical_cvar(returns, alpha)

    assert np.isclose(cvar, 0.10, atol=1e-12)


def test_compute_parametric_cvar_basic():
    mean = 0.0
    sigma = 0.02
    alpha = 0.05

    cvar = compute_parametric_cvar(mean, sigma, alpha)

    # ES for Normal is > VaR; just sanity check it's bigger than parametric VaR
    from quant_platform.risk.var import compute_parametric_var

    var = compute_parametric_var(mean, sigma, alpha)

    assert cvar > var
    assert cvar > 0.0


def test_compute_parametric_cvar_from_series():
    rng = np.random.default_rng(7)
    returns = rng.normal(loc=0.0, scale=0.03, size=1000)

    alpha = 0.01
    cvar, mu_hat, sigma_hat = compute_parametric_cvar_from_series(returns, alpha)

    assert sigma_hat > 0.0
    assert cvar > 0.0
