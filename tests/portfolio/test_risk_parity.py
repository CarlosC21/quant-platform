# tests/portfolio/test_risk_parity.py

import numpy as np

from quant_platform.portfolio.risk_parity import solve_risk_parity
from quant_platform.portfolio.schemas import PortfolioConfig, RiskModelInput


def _risk_contrib(w, cov):
    sigma_w = cov @ w
    return w * sigma_w


def test_risk_parity_basic():
    # Covariance: asset 0 low vol, asset 1 high vol
    cov = np.array(
        [
            [0.04, 0.01],
            [0.01, 0.16],
        ]
    )

    mu = np.array([0.1, 0.1])  # irrelevant for RP
    symbols = ["A", "B"]

    rmi = RiskModelInput(mu=mu, cov=cov, symbols=symbols)
    cfg = PortfolioConfig(symbols=symbols, allow_short=False)

    result = solve_risk_parity(rmi, cfg)

    w = np.array([wt.weight for wt in result.weights])
    rc = _risk_contrib(w, cov)

    # RC_A approximately equals RC_B
    assert np.isclose(rc[0], rc[1], rtol=1e-3, atol=1e-3)

    # Long-only + sum to 1
    assert np.all(w >= 0)
    assert np.isclose(w.sum(), 1.0)


def test_risk_parity_three_assets():
    cov = np.array(
        [
            [0.04, 0.01, 0.00],
            [0.01, 0.09, 0.02],
            [0.00, 0.02, 0.16],
        ]
    )

    mu = np.array([0.05, 0.06, 0.07])
    symbols = ["A", "B", "C"]

    rmi = RiskModelInput(mu=mu, cov=cov, symbols=symbols)
    cfg = PortfolioConfig(symbols=symbols, allow_short=False)

    result = solve_risk_parity(rmi, cfg)

    w = np.array([wt.weight for wt in result.weights])
    rc = _risk_contrib(w, cov)

    # All risk contributions roughly equal
    assert np.max(rc) - np.min(rc) < 1e-3
