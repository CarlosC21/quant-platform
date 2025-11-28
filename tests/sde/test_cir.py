# tests/sde/test_cir.py
import numpy as np

from quant_platform.sde.processes.cir import CIRConfig, cir_exact, cir_euler
from quant_platform.sde.estimators.cir import estimate_cir_mle


def test_cir_shapes_and_nonnegativity():
    cfg = CIRConfig(kappa=1.2, theta=0.04, sigma=0.12, r0=0.03)
    from quant_platform.sde.schemas import SimConfig

    sim = SimConfig(n_paths=3, n_steps=120, dt=1.0 / 252.0, seed=2021)
    R1 = cir_exact(cfg, sim)
    R2 = cir_exact(cfg, sim)
    assert R1.shape == (3, 121)
    assert R2.shape == (3, 121)
    assert np.array_equal(R1, R2)
    assert np.all(R1 >= 0.0)

    RE = cir_euler(cfg, sim)
    assert RE.shape == (3, 121)
    assert np.all(RE >= 0.0)


def test_cir_mle_smoke():
    """Quick smoke test: estimator runs and returns finite positive parameters."""
    cfg = CIRConfig(kappa=0.9, theta=0.03, sigma=0.11, r0=0.02)
    from quant_platform.sde.schemas import SimConfig

    # small n_steps for fast test
    sim = SimConfig(n_paths=1, n_steps=500, dt=1.0 / 252.0, seed=2025)
    R = cir_exact(cfg, sim)
    r = R[0]

    k_hat, th_hat, s_hat = estimate_cir_mle(r, sim.dt)

    assert np.isfinite(k_hat) and k_hat > 0
    assert np.isfinite(th_hat) and th_hat >= 0
    assert np.isfinite(s_hat) and s_hat > 0
