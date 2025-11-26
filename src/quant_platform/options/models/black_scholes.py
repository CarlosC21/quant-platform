import numpy as np
from scipy.stats import norm


def _d1(S, K, T, r, sigma):
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def _d2(S, K, T, r, sigma):
    return _d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


def bs_price(S, K, T, r, sigma, option_type="call"):
    """
    Black-Scholes price for a European option.
    Vectorized: S,K,T,r,sigma may be numpy arrays.
    """
    S = np.asarray(S)
    K = np.asarray(K)
    T = np.asarray(T)
    r = np.asarray(r)
    sigma = np.asarray(sigma)

    d1 = _d1(S, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    elif option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    else:
        raise ValueError("option_type must be 'call' or 'put'")
