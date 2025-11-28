# tests/sde/test_ou.py
import numpy as np

from quant_platform.sde.schemas import OUConfig, SimConfig
from quant_platform.sde.processes.ou import ou_exact, ou_euler, estimate_ou_from_path


def test_ou_shapes_and_reproducibility():
    cfg = OUConfig(kappa=1.2, theta=0.5, sigma=0.08, x0=0.2)
    sim = SimConfig(n_paths=4, n_steps=100, dt=0.01, seed=2025)

    X1 = ou_exact(cfg, sim)
    X2 = ou_exact(cfg, sim)
    assert X1.shape == (4, 101)
    assert X2.shape == (4, 101)
    assert np.array_equal(X1, X2)

    XE = ou_euler(cfg, sim)
    assert XE.shape == (4, 101)


def test_ou_estimation_recovery():
    cfg = OUConfig(kappa=0.8, theta=0.2, sigma=0.07, x0=0.05)
    sim = SimConfig(n_paths=1, n_steps=4000, dt=0.005, seed=42)
    X = ou_exact(cfg, sim)
    x = X[0]  # single long path

    kappahat, thetahat, sigmahat = estimate_ou_from_path(x, sim.dt)

    # loose tolerances (estimation is noisy); good enough for sanity tests
    assert abs(kappahat - cfg.kappa) / max(cfg.kappa, 1e-6) < 0.25
    assert abs(thetahat - cfg.theta) < 0.05
    assert abs(sigmahat - cfg.sigma) < 0.05
