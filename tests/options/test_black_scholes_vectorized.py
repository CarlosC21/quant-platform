# tests/options/test_black_scholes_vectorized.py
import numpy as np
from quant_platform.options.models.black_scholes import bs_price, bs_delta


def test_vectorized_price_and_delta():
    # Batch of spot prices, strikes, maturities
    S = np.array([90, 100, 110])
    K = np.array([100, 100, 100])
    T = np.array([1.0, 1.0, 1.0])
    r = np.array([0.05, 0.05, 0.05])
    sigma = np.array([0.2, 0.2, 0.2])

    # Compute call prices vectorized
    call_prices = bs_price(S, K, T, r, sigma, option_type="call")
    put_prices = bs_price(S, K, T, r, sigma, option_type="put")

    # Compute deltas vectorized
    call_deltas = bs_delta(S, K, T, r, sigma, option_type="call")
    put_deltas = bs_delta(S, K, T, r, sigma, option_type="put")

    # Assertions
    # Call prices increase with spot price
    assert np.all(call_prices[1:] >= call_prices[:-1])

    # Put prices decrease with spot price
    assert np.all(put_prices[1:] <= put_prices[:-1])

    # Delta sanity checks
    assert np.all((call_deltas >= 0) & (call_deltas <= 1))
    assert np.all((put_deltas <= 0) & (put_deltas >= -1))

    # Optional: basic ranges (rough expected values)
    _ = np.array([2, 10, 20])
    assert np.all(call_prices >= 0)
    assert np.all(put_prices >= 0)
