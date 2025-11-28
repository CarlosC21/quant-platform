# src/quant_platform/options/hedging/delta_hedge.py
import numpy as np
from ..models.black_scholes import BlackScholesOption


def delta_hedge(option: BlackScholesOption, S_path: np.ndarray, dt: float):
    """
    Simulate delta hedging PnL for an option along a given underlying path
    """
    cash = 0.0
    position = 0.0
    for t in range(len(S_path) - 1):
        delta = option.delta()
        # Adjust position
        delta_change = delta - position
        cash -= delta_change * S_path[t]
        position = delta
        # Discount cash (optional)
        cash *= np.exp(option.r * dt)
    # Liquidate position at end
    cash += position * S_path[-1] - option.price()
    return cash
