import numpy as np
from scipy.optimize import brentq

from src.quant_platform.options.greeks.greeks import bs_vega
from src.quant_platform.options.models.black_scholes import bs_price


def implied_vol(price, S, K, T, r, option_type="call", tol=1e-6, max_iter=100):
    """
    Stable implied vol solver with:
    - Newton iterations
    - Brent fallback
    - Guards for no-solution cases
    """

    # Initial guess: classic approximation
    sigma = 0.2

    for _ in range(max_iter):
        price_est = bs_price(S, K, T, r, sigma, option_type)
        vega = bs_vega(S, K, T, r, sigma)

        # Avoid division issues
        if vega < 1e-10:
            break

        diff = price_est - price
        sigma = sigma - diff / vega

        if abs(diff) < tol:
            return float(sigma)

    # Brent fallback in case Newton fails
    try:
        return float(
            brentq(
                lambda vol: bs_price(S, K, T, r, vol, option_type) - price,
                1e-9,
                5.0,
                maxiter=200,
            )
        )
    except ValueError:
        return np.nan
