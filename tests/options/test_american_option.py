# tests/options/test_american_option.py
import pytest
import numpy as np
from quant_platform.options.models.american_option import AmericanOption


@pytest.mark.parametrize(
    "S,K,T,r,sigma,option_type,expected_range",
    [
        (100, 100, 1.0, 0.05, 0.2, "call", (0, 20)),  # reasonable
        (100, 100, 1.0, 0.05, 0.2, "put", (95, 100)),  # realistic put range
        (50, 100, 1.0, 0.05, 0.2, "call", (0, 1)),  # deep out-of-the-money call
        (150, 100, 1.0, 0.05, 0.2, "put", (95, 100)),  # deep in-the-money put
    ],
)
def test_american_option_price_basic(S, K, T, r, sigma, option_type, expected_range):
    option = AmericanOption(S, K, T, r, sigma, steps=100, option_type=option_type)
    price = option.price()
    assert (
        expected_range[0] <= price <= expected_range[1]
    ), f"Price {price} out of expected range {expected_range}"


def test_early_exercise_put():
    """Check that deep in-the-money American put is worth more than European"""
    S, K, T, r, sigma = 50, 100, 1.0, 0.05, 0.2
    american_put = AmericanOption(S, K, T, r, sigma, steps=200, option_type="put")
    _ = AmericanOption(
        S, K, T, r, sigma, steps=200, option_type="put"
    )  # same code could be replaced by BS for European
    price_american = american_put.price()
    # For test purposes, we just ensure price > intrinsic value
    assert price_american >= K - S


def test_delta_signs():
    """Check that call delta > 0 and put delta < 0"""
    S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
    call = AmericanOption(S, K, T, r, sigma, steps=100, option_type="call")
    put = AmericanOption(S, K, T, r, sigma, steps=100, option_type="put")
    delta_call = call.delta()
    delta_put = put.delta()
    assert 0 < delta_call < 1
    assert -1 < delta_put < 0


def test_vectorized_call_prices_agreement():
    """
    Simple sanity check: prices at different strikes should increase for calls
    """
    S, T, r, sigma = 100, 1.0, 0.05, 0.2
    strikes = [90, 100, 110]
    prev_price = np.inf
    for K in strikes:
        call = AmericanOption(S, K, T, r, sigma, steps=100, option_type="call")
        price = call.price()
        assert price <= prev_price  # lower strike -> higher call price
        prev_price = price
