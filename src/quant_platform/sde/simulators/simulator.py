# src/quant_platform/sde/simulators/simulator.py
from abc import ABC, abstractmethod
import numpy as np


class BaseSDE(ABC):
    @abstractmethod
    def drift(self, x: np.ndarray, t: float) -> np.ndarray:
        ...

    @abstractmethod
    def diffusion(self, x: np.ndarray, t: float) -> np.ndarray:
        ...


class Simulator:
    def __init__(
        self,
        sde: BaseSDE,
        dt: float,
        n_steps: int,
        n_paths: int,
        seed: int | None = None,
    ):
        self.sde = sde
        self.dt = dt
        self.n_steps = n_steps
        self.n_paths = n_paths
        self.rng = np.random.default_rng(seed)

    def euler(self, x0: float | np.ndarray) -> np.ndarray:
        X = np.zeros((self.n_paths, self.n_steps + 1), dtype=float)
        X[:, 0] = x0
        sqrt_dt = np.sqrt(self.dt)

        for i in range(self.n_steps):
            dW = self.rng.normal(size=self.n_paths) * sqrt_dt
            X[:, i + 1] = (
                X[:, i]
                + self.sde.drift(X[:, i], i * self.dt) * self.dt
                + self.sde.diffusion(X[:, i], i * self.dt) * dW
            )
        return X
