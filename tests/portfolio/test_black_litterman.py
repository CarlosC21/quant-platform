# tests/portfolio/test_black_litterman.py
import numpy as np

from quant_platform.portfolio.black_litterman import solve_black_litterman
from quant_platform.portfolio.schemas import RiskModelInput, PortfolioConfig


def test_black_litterman_basic():
    cov = np.array(
        [
            [0.04, 0.01],
            [0.01, 0.09],
        ]
    )
    mu = np.array([0.05, 0.05])  # equal priors
    symbols = ["A", "B"]

    rmi = RiskModelInput(mu=mu, cov=cov, symbols=symbols)
    cfg = PortfolioConfig(symbols=symbols, allow_short=False)

    # View: A expected to outperform B by 5%
    P = np.array([[1, -1]])
    Q = np.array([0.05])

    result = solve_black_litterman(rmi, cfg, P=P, Q=Q)

    wA = result.weights[0].weight
    wB = result.weights[1].weight

    # BL should tilt toward A
    assert wA > wB
    assert np.isclose(wA + wB, 1.0)


def test_black_litterman_no_views_matches_prior():
    cov = np.array(
        [
            [0.04, 0.01],
            [0.01, 0.09],
        ]
    )
    mu = np.array([0.05, 0.06])
    symbols = ["A", "B"]

    rmi = RiskModelInput(mu=mu, cov=cov, symbols=symbols)
    cfg = PortfolioConfig(symbols=symbols, allow_short=False)

    # No views → P,Q empty
    P = np.zeros((0, 2))
    Q = np.zeros(0)

    result = solve_black_litterman(rmi, cfg, P=P, Q=Q)

    # Should behave like Markowitz on original mu
    w = np.array([w.weight for w in result.weights])
    assert np.isclose(w.sum(), 1.0)
    # Higher return asset B should get higher allocation
    assert w[1] > w[0]
