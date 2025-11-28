# tests/sde/test_vasicek.py
import numpy as np

from quant_platform.sde.schemas import VasicekConfig, SimConfig
from quant_platform.sde.processes.vasicek import (
    vasicek_exact,
    vasicek_euler,
    estimate_vasicek_from_path,
)


def test_vasicek_shapes_and_reproducibility():
    cfg = VasicekConfig(kappa=0.9, theta=0.03, sigma=0.01, r0=0.02)
    sim = SimConfig(n_paths=3, n_steps=120, dt=1.0 / 252.0, seed=2026)

    R1 = vasicek_exact(cfg, sim)
    R2 = vasicek_exact(cfg, sim)
    assert R1.shape == (3, 121)
    assert R2.shape == (3, 121)
    assert np.array_equal(R1, R2)

    RE = vasicek_euler(cfg, sim)
    assert RE.shape == (3, 121)


# tests/sde/test_vasicek.py (replace the test_vasicek_estimation_recovery function)


def test_vasicek_mle_smoke():
    cfg = VasicekConfig(kappa=0.7, theta=0.02, sigma=0.015, r0=0.01)
    sim = SimConfig(n_paths=1, n_steps=500, dt=0.002, seed=2023)  # shorter path
    R = vasicek_exact(cfg, sim)
    r = R[0]

    k_hat, th_hat, s_hat = estimate_vasicek_from_path(r, sim.dt)

    assert np.isfinite(k_hat) and k_hat > 0
    assert np.isfinite(th_hat) and th_hat >= 0
    assert np.isfinite(s_hat) and s_hat > 0
