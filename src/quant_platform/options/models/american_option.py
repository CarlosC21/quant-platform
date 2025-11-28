# src/quant_platform/options/models/american_option.py
from __future__ import annotations
from typing import Literal
import numpy as np

OptionType = Literal["call", "put"]


class AmericanOption:
    """
    American option pricing using the Cox-Ross-Rubinstein (CRR) binomial tree.

    Attributes
    ----------
    S : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to maturity (in years)
    r : float
        Risk-free interest rate (annual)
    sigma : float
        Volatility (annual)
    steps : int
        Number of steps in the binomial tree
    option_type : str
        "call" or "put"
    """

    def __init__(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        steps: int = 100,
        option_type: OptionType = "call",
    ):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.steps = steps
        self.option_type = option_type

        self.dt = T / steps
        self.u = np.exp(sigma * np.sqrt(self.dt)) if sigma > 0 else 1.0
        self.d = 1 / self.u if sigma > 0 else 1.0

        # Safe risk-neutral probability
        if self.u != self.d:
            self.p = (np.exp(r * self.dt) - self.d) / (self.u - self.d)
            self.p = np.clip(self.p, 0.0, 1.0)  # clamp to [0,1]
        else:
            # fallback if sigma == 0 or u == d
            self.p = 0.5

    def price(self) -> float:
        """
        Price the American option using backward induction.
        """
        # Step 1: initialize asset prices at maturity
        ST = np.array(
            [
                self.S * (self.u**j) * (self.d ** (self.steps - j))
                for j in range(self.steps + 1)
            ]
        )

        # Step 2: initialize option values at maturity
        if self.option_type == "call":
            option_values = np.maximum(ST - self.K, 0)
        else:
            option_values = np.maximum(self.K - ST, 0)

        # Step 3: backward induction
        discount = np.exp(-self.r * self.dt)
        for i in range(self.steps - 1, -1, -1):
            option_values = discount * (
                self.p * option_values[1 : i + 2]
                + (1 - self.p) * option_values[0 : i + 1]
            )
            ST = ST[0 : i + 1] / self.u  # step back asset prices
            if self.option_type == "call":
                option_values = np.maximum(option_values, ST - self.K)  # early exercise
            else:
                option_values = np.maximum(option_values, self.K - ST)

        return option_values[0]

    def delta(self) -> float:
        """
        Approximate delta using the first step of the binomial tree.
        """
        up_price = AmericanOption(
            S=self.S * self.u,
            K=self.K,
            T=self.T - self.dt,
            r=self.r,
            sigma=self.sigma,
            steps=self.steps - 1,
            option_type=self.option_type,
        ).price()
        down_price = AmericanOption(
            S=self.S * self.d,
            K=self.K,
            T=self.T - self.dt,
            r=self.r,
            sigma=self.sigma,
            steps=self.steps - 1,
            option_type=self.option_type,
        ).price()
        return (up_price - down_price) / (self.S * (self.u - self.d))
