# src/quant_platform/sde/simulators/milstein.py
import numpy as np


def milstein_simulate(
    process_fn,
    diff_fn,
    x0: float,
    dt: float,
    n_steps: int,
    n_paths: int = 1,
    seed: int | None = None,
    nonneg: bool = False,
):
    """
    Generic Milstein simulator for SDE: dX = mu dt + sigma dW
    process_fn: drift function mu(x)
    diff_fn: diffusion function sigma(x)
    """
    rng = np.random.default_rng(seed)
    X = np.empty((n_paths, n_steps + 1), dtype=float)
    X[:, 0] = x0

    for t in range(n_steps):
        dw = rng.normal(0.0, np.sqrt(dt), size=n_paths)
        X[:, t + 1] = X[:, t] + np.array(
            [
                process_fn(x) * dt
                + diff_fn(x) * dwi
                + 0.5 * diff_fn(x) * diff_fn(x) * (dwi**2 - dt)
                for x, dwi in zip(X[:, t], dw)
            ]
        )
        if nonneg:
            X[:, t + 1] = np.maximum(X[:, t + 1], 0.0)
    return X
