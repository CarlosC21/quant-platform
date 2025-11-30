# tests/portfolio/test_markowitz.py

import numpy as np

from quant_platform.portfolio.schemas import PortfolioConfig, RiskModelInput
from quant_platform.portfolio.markowitz import solve_markowitz


def test_markowitz_min_variance_basic():
    mu = np.array([0.1, 0.2])
    cov = np.array([[0.04, 0.0], [0.0, 0.09]])  # asset 0 has lower vol

    rmi = RiskModelInput(mu=mu, cov=cov, symbols=["A", "B"])

    config = PortfolioConfig(
        symbols=["A", "B"],
        allow_short=False,
        leverage_limit=1.0,
    )

    result = solve_markowitz(rmi, config)

    # Should put more weight on lower-vol asset A
    assert result.weights[0].weight > result.weights[1].weight

    # Fully invested
    total_w = sum(w.weight for w in result.weights)
    assert np.isclose(total_w, 1.0, atol=1e-9)

    # Vol is computed correctly
    assert result.expected_vol > 0.0


def test_markowitz_target_return():
    mu = np.array([0.1, 0.2])
    cov = np.eye(2)

    rmi = RiskModelInput(mu=mu, cov=cov, symbols=["A", "B"])

    config = PortfolioConfig(
        symbols=["A", "B"],
        allow_short=False,
        leverage_limit=1.0,
        target_return=0.15,
    )

    result = solve_markowitz(rmi, config)

    assert result.expected_return > 0.14
    assert result.expected_return < 0.16
