# src/quant_platform/options/models/black_scholes.py
import numpy as np
from scipy.stats import norm


def _d1(S, K, T, r, sigma, q=0.0):
    """Helper function for Black-Scholes d1, supports dividends"""
    return (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def _d2(S, K, T, r, sigma, q=0.0):
    """Helper function for Black-Scholes d2, supports dividends"""
    return _d1(S, K, T, r, sigma, q) - sigma * np.sqrt(T)


def bs_price(S, K, T, r, sigma, option_type="call", q=0.0):
    """
    Black-Scholes European option price.
    Vectorized for arrays: S, K, T, r, sigma
    """
    S, K, T, r, sigma, q = map(np.asarray, (S, K, T, r, sigma, q))
    d1 = _d1(S, K, T, r, sigma, q)
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


def bs_delta(S, K, T, r, sigma, option_type="call", q=0.0):
    """
    Black-Scholes delta for European option.
    Vectorized for arrays.
    """
    S, K, T, r, sigma, q = map(np.asarray, (S, K, T, r, sigma, q))
    d1 = _d1(S, K, T, r, sigma, q)

    if option_type == "call":
        return np.exp(-q * T) * norm.cdf(d1)
    elif option_type == "put":
        return np.exp(-q * T) * (norm.cdf(d1) - 1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


class BlackScholes:
    """
    Black-Scholes European option class supporting price and delta.
    """

    def __init__(self, K, T, r=0.0, sigma=0.2, q=0.0, option_type="call"):
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q
        self.option_type = option_type.lower()

    def _d1(self, S, T):
        return (np.log(S / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * T) / (
            self.sigma * np.sqrt(T)
        )

    def _d2(self, S, T):
        return self._d1(S, T) - self.sigma * np.sqrt(T)

    def price(self, S, T=None):
        T = self.T if T is None else T
        d1, d2 = self._d1(S, T), self._d2(S, T)
        if self.option_type == "call":
            return S * np.exp(-self.q * T) * norm.cdf(d1) - self.K * np.exp(
                -self.r * T
            ) * norm.cdf(d2)
        else:
            return self.K * np.exp(-self.r * T) * norm.cdf(-d2) - S * np.exp(
                -self.q * T
            ) * norm.cdf(-d1)

    def delta(self, S, T=None):
        T = self.T if T is None else T
        d1 = self._d1(S, T)
        if self.option_type == "call":
            return np.exp(-self.q * T) * norm.cdf(d1)
        else:
            return np.exp(-self.q * T) * (norm.cdf(d1) - 1)


def delta_hedge_simulator(S_path, K, T, r, sigma, option_type="call", q=0.0, dt=None):
    """
    Delta-hedge P&L simulation for a single European option along a spot path.

    Parameters
    ----------
    S_path : array-like
        Spot price path (discrete steps)
    K : float
        Strike price
    T : float
        Time to maturity
    r : float
        Risk-free rate
    sigma : float
        Volatility
    option_type : str
        'call' or 'put'
    q : float
        Continuous dividend yield
    dt : float, optional
        Time step size (default: T / len(S_path))

    Returns
    -------
    pnl : float
        Profit and loss from delta hedging
    """
    S_path = np.asarray(S_path)
    N = len(S_path)
    dt = dt if dt is not None else T / N

    bs = BlackScholes(K=K, T=T, r=r, sigma=sigma, q=q, option_type=option_type)
    cash = 0.0
    delta_prev = 0.0

    for i in range(N - 1):
        t_remain = T - i * dt
        delta = bs.delta(S_path[i], t_remain)
        d_delta = delta - delta_prev
        cash -= d_delta * S_path[i]
        cash *= np.exp(r * dt)
        delta_prev = delta

    # final option payoff
    option_payoff = (
        max(S_path[-1] - K, 0) if option_type == "call" else max(K - S_path[-1], 0)
    )
    pnl = option_payoff + cash - delta_prev * S_path[-1]
    return pnl
