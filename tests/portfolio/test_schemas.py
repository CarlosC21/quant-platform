# tests/portfolio/test_schemas.py
import numpy as np

from quant_platform.portfolio.schemas import (
    AssetWeight,
    PortfolioConfig,
    RiskModelInput,
    PortfolioResult,
)


def test_portfolio_schemas_basic():
    config = PortfolioConfig(
        symbols=["AAPL", "MSFT"],
        allow_short=False,
        leverage_limit=1.0,
    )

    assert config.symbols == ["AAPL", "MSFT"]
    assert config.allow_short is False

    mu = np.array([0.1, 0.2])
    cov = np.eye(2)
    rmi = RiskModelInput(mu=mu, cov=cov, symbols=["AAPL", "MSFT"])

    assert rmi.mu.shape == (2,)
    assert rmi.cov.shape == (2, 2)

    weights = [
        AssetWeight(symbol="AAPL", weight=0.6),
        AssetWeight(symbol="MSFT", weight=0.4),
    ]

    pr = PortfolioResult(
        weights=weights,
        expected_return=0.6 * 0.1 + 0.4 * 0.2,
        expected_vol=0.15,
        sharpe=1.0,
        meta={"test": 123},
    )

    assert len(pr.weights) == 2
    assert pr.meta["test"] == 123
