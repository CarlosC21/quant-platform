# tests/options/test_black_scholes.py
import numpy as np
from quant_platform.options.models.black_scholes import (
    bs_price,
    bs_delta,
    delta_hedge_simulator,
)


def test_vectorized_price_and_delta():
    S = np.array([90, 100, 110])
    K = np.array([100, 100, 100])
    T = np.array([1.0, 1.0, 1.0])
    r = np.array([0.05, 0.05, 0.05])
    sigma = np.array([0.2, 0.2, 0.2])

    call_prices = bs_price(S, K, T, r, sigma, option_type="call")
    put_prices = bs_price(S, K, T, r, sigma, option_type="put")

    call_deltas = bs_delta(S, K, T, r, sigma, option_type="call")
    put_deltas = bs_delta(S, K, T, r, sigma, option_type="put")

    assert np.all(call_prices[1:] >= call_prices[:-1])
    assert np.all(put_prices[1:] <= put_prices[:-1])
    assert np.all((call_deltas >= 0) & (call_deltas <= 1))
    assert np.all((put_deltas <= 0) & (put_deltas >= -1))


def test_delta_hedge_simulator_basic():
    np.random.seed(42)
    S0 = 100
    K = 100
    T = 1.0
    r = 0.05
    sigma = 0.2
    N = 252  # daily steps

    # simulate geometric Brownian motion
    dt = T / N
    S_path = [S0]
    for _ in range(N):
        S_prev = S_path[-1]
        dS = S_prev * (r * dt + sigma * np.sqrt(dt) * np.random.randn())
        S_path.append(S_prev + dS)

    pnl_call = delta_hedge_simulator(S_path, K, T, r, sigma, option_type="call", dt=dt)
    pnl_put = delta_hedge_simulator(S_path, K, T, r, sigma, option_type="put", dt=dt)

    # just check we get a float and finite number
    assert isinstance(pnl_call, float) and np.isfinite(pnl_call)
    assert isinstance(pnl_put, float) and np.isfinite(pnl_put)
