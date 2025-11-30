# tests/portfolio/test_hrp.py

import numpy as np

from quant_platform.portfolio.hrp import solve_hrp
from quant_platform.portfolio.schemas import PortfolioConfig, RiskModelInput


def _assert_psd(cov):
    eigvals = np.linalg.eigvalsh(cov)
    assert np.all(eigvals >= -1e-8)


def test_hrp_weights_sum_to_one():
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

    result = solve_hrp(rmi, cfg)

    w = np.array([asset.weight for asset in result.weights])

    assert np.all(w >= 0)
    assert np.isclose(w.sum(), 1.0)


def test_hrp_diversification_not_equal_weights():
    # A simple case where assets have clearly different volatilities
    cov = np.array(
        [
            [0.04, 0.01],
            [0.01, 0.16],
        ]
    )

    mu = np.array([0.1, 0.1])
    symbols = ["A", "B"]

    rmi = RiskModelInput(mu=mu, cov=cov, symbols=symbols)
    cfg = PortfolioConfig(symbols=symbols, allow_short=False)

    result = solve_hrp(rmi, cfg)
    w = np.array([a.weight for a in result.weights])

    # HRP should allocate more to low-vol asset A
    assert w[0] > w[1]
