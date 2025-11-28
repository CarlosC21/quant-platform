# src/quant_platform/options/models/option_base.py
from __future__ import annotations
from typing import Literal

OptionType = Literal["call", "put"]


class Option:
    def __init__(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: OptionType = "call",
    ):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type

    def price(self) -> float:
        raise NotImplementedError

    def delta(self) -> float:
        raise NotImplementedError
