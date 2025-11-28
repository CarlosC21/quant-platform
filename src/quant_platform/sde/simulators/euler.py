# src/quant_platform/sde/simulators/euler.py
import numpy as np


def euler_simulate(
    process_fn,
    x0: float,
    dt: float,
    n_steps: int,
    n_paths: int = 1,
    seed: int | None = None,
    nonneg: bool = False,
):
    """
    Generic Euler-Maruyama simulator for an SDE.

    process_fn: callable(x, dt, dw) -> dx increment
    x0: initial value
    dt: time step
    n_steps: number of steps
    n_paths: number of independent paths
    seed: random seed
    nonneg: clip negative values (for CIR, interest rates, etc.)
    """
    rng = np.random.default_rng(seed)
    X = np.empty((n_paths, n_steps + 1), dtype=float)
    X[:, 0] = x0

    for t in range(n_steps):
        dw = rng.normal(0.0, np.sqrt(dt), size=n_paths)
        X[:, t + 1] = X[:, t] + np.array(
            [process_fn(x, dt, dwi) for x, dwi in zip(X[:, t], dw)]
        )
        if nonneg:
            X[:, t + 1] = np.maximum(X[:, t + 1], 0.0)
    return X
