# tests/options/test_option_models.py
import numpy as np
from quant_platform.options.models.black_scholes_option import BlackScholesOption
from quant_platform.options.models.american_option import AmericanOption
from quant_platform.options.models.delta_hedge import delta_hedge_simulator


def test_delta_hedge_pnl_call():
    S0, K, _T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    np.random.seed(42)
    S_path = S0 * np.exp(
        np.cumsum(
            (r - 0.5 * sigma**2) * 0.01 + sigma * np.sqrt(0.01) * np.random.randn(100)
        )
    )

    # Black-Scholes call
    def option_factory(S, T):
        return BlackScholesOption(S, K, _T, r, sigma, option_type="call")

    pnl = delta_hedge_simulator(option_factory, S_path)
    assert isinstance(pnl, float)

    # American call
    def option_factory_amer(S, T):
        return AmericanOption(S, K, _T, r, sigma, steps=100, option_type="call")

    pnl_amer = delta_hedge_simulator(option_factory_amer, S_path)
    assert isinstance(pnl_amer, float)


def test_delta_hedge_pnl_put():
    S0, K, _T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    np.random.seed(42)
    S_path = S0 * np.exp(
        np.cumsum(
            (r - 0.5 * sigma**2) * 0.01 + sigma * np.sqrt(0.01) * np.random.randn(100)
        )
    )

    # Black-Scholes put
    def option_factory(S, T):
        return BlackScholesOption(S, K, _T, r, sigma, option_type="put")

    pnl = delta_hedge_simulator(option_factory, S_path)
    assert isinstance(pnl, float)

    # American put
    def option_factory_amer(S, T):
        return AmericanOption(S, K, _T, r, sigma, steps=100, option_type="put")

    pnl_amer = delta_hedge_simulator(option_factory_amer, S_path)
    assert isinstance(pnl_amer, float)
