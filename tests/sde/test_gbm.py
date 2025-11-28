# tests/sde/test_gbm.py
import numpy as np

from quant_platform.sde.schemas import GBMConfig, SimConfig
from quant_platform.sde.gbm import gbm_exact, gbm_euler, estimate_gbm_from_prices


def test_gbm_shapes_and_reproducibility():
    cfg = GBMConfig(mu=0.03, sigma=0.15, s0=100.0)
    sim = SimConfig(n_paths=3, n_steps=50, dt=1.0 / 252.0, seed=777)

    S1 = gbm_exact(cfg, sim)
    S2 = gbm_exact(cfg, sim)  # reproducible
    assert S1.shape == (3, 51)
    assert S2.shape == (3, 51)
    assert np.array_equal(S1, S2)

    SE = gbm_euler(cfg, sim)
    assert SE.shape == (3, 51)


def test_gbm_estimation_close_to_true():
    cfg = GBMConfig(mu=0.05, sigma=0.2, s0=100.0)
    sim = SimConfig(n_paths=1, n_steps=2000, dt=1.0 / 252.0, seed=1234)

    S = gbm_exact(cfg, sim)
    prices = S[0]

    mu_hat, sigma_hat = estimate_gbm_from_prices(prices, sim.dt)

    assert abs(sigma_hat - cfg.sigma) < 0.05
    assert abs(mu_hat - cfg.mu) < 0.05
