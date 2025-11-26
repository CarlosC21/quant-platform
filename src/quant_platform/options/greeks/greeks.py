# src/quant_platform/options/greeks/greeks.py
from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl
from scipy.stats import norm


# --------------------------
# Black-Scholes core functions
# --------------------------
def _d1_d2(S, K, T, r, sigma):
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return d1, d2


def bs_price(S, K, T, r, sigma, option_type: Literal["call", "put"] = "call"):
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_delta(S, K, T, r, sigma, option_type: Literal["call", "put"] = "call"):
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return norm.cdf(d1) if option_type == "call" else norm.cdf(d1) - 1.0


def bs_gamma(S, K, T, r, sigma):
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def bs_vega(S, K, T, r, sigma):
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T)


def bs_theta(S, K, T, r, sigma, option_type: Literal["call", "put"] = "call"):
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    pdf_d1 = norm.pdf(d1)
    if option_type == "call":
        return -S * pdf_d1 * sigma / (2 * np.sqrt(T)) - r * K * np.exp(
            -r * T
        ) * norm.cdf(d2)
    else:
        return -S * pdf_d1 * sigma / (2 * np.sqrt(T)) + r * K * np.exp(
            -r * T
        ) * norm.cdf(-d2)


def bs_rho(S, K, T, r, sigma, option_type: Literal["call", "put"] = "call"):
    _, d2 = _d1_d2(S, K, T, r, sigma)
    return (
        K * T * np.exp(-r * T) * norm.cdf(d2)
        if option_type == "call"
        else -K * T * np.exp(-r * T) * norm.cdf(-d2)
    )


# --------------------------
# Implied vol solver
# --------------------------
def solve_iv(S, K, T, r, price, option_type="call", tol=1e-6, max_iter=100):
    sigma = 0.2
    for _ in range(max_iter):
        price_est = bs_price(S, K, T, r, sigma, option_type)
        d1, _ = _d1_d2(S, K, T, r, sigma)
        vega = S * norm.pdf(d1) * np.sqrt(T)
        sigma -= (price_est - price) / (vega + 1e-8)
        if abs(price_est - price) < tol:
            return max(sigma, 1e-8)
    return max(sigma, 1e-8)


# --------------------------
# Vectorized Greeks Calculator
# --------------------------
@dataclass
class GreeksCalculator:
    df: pl.DataFrame  # columns: underlying_price, strike, ttm, r, iv, option_type

    def compute(self) -> pl.DataFrame:
        S = self.df["underlying_price"].to_numpy().astype(float)
        K = self.df["strike"].to_numpy().astype(float)
        T = self.df["ttm"].to_numpy().astype(float)
        r = self.df["r"].to_numpy().astype(float)
        sigma = self.df["iv"].to_numpy().astype(float)
        option_type = self.df["option_type"].to_numpy()

        price = np.array(
            [
                bs_price(s, k, t, rr, sig, ot)
                for s, k, t, rr, sig, ot in zip(S, K, T, r, sigma, option_type)
            ]
        )
        delta = np.array(
            [
                bs_delta(s, k, t, rr, sig, ot)
                for s, k, t, rr, sig, ot in zip(S, K, T, r, sigma, option_type)
            ]
        )
        gamma = np.array(
            [bs_gamma(s, k, t, rr, sig) for s, k, t, rr, sig in zip(S, K, T, r, sigma)]
        )
        vega = np.array(
            [bs_vega(s, k, t, rr, sig) for s, k, t, rr, sig in zip(S, K, T, r, sigma)]
        )
        theta = np.array(
            [
                bs_theta(s, k, t, rr, sig, ot)
                for s, k, t, rr, sig, ot in zip(S, K, T, r, sigma, option_type)
            ]
        )
        rho = np.array(
            [
                bs_rho(s, k, t, rr, sig, ot)
                for s, k, t, rr, sig, ot in zip(S, K, T, r, sigma, option_type)
            ]
        )

        return self.df.with_columns(
            [
                pl.Series("price", price),
                pl.Series("delta", delta),
                pl.Series("gamma", gamma),
                pl.Series("vega", vega),
                pl.Series("theta", theta),
                pl.Series("rho", rho),
            ]
        )
