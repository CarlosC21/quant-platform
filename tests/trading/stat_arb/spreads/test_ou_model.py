# tests/trading/stat_arb/spreads/test_ou_model.py

from __future__ import annotations

import numpy as np
import pytest

from quant_platform.sde.schemas import OUConfig, SimConfig
from quant_platform.sde.processes.ou import ou_exact

from quant_platform.trading.stat_arb.spreads.ou_model import (
    OUParams,
    fit_ou_to_spread,
)


def test_fit_ou_to_spread_recovers_parameters():
    """
    Validate that the stat-arb OU wrapper recovers parameters
    reasonably well using a long exact OU simulation.
    """

    cfg = OUConfig(kappa=1.0, theta=0.5, sigma=0.1, x0=0.0)
    sim = SimConfig(n_paths=1, n_steps=5000, dt=1 / 252, seed=123)

    X = ou_exact(cfg, sim)  # shape = (1, n_steps+1)
    spread = X[0]  # 1D array

    ou_params = fit_ou_to_spread(spread, dt=sim.dt)
    assert isinstance(ou_params, OUParams)

    # Parameter recovery checks (same tolerances as your SDE tests)
    # kappa estimation is noisy for small dt; allow looser tolerance
    assert abs(ou_params.kappa - cfg.kappa) / cfg.kappa < 0.6

    assert abs(ou_params.theta - cfg.theta) < 0.05
    assert abs(ou_params.sigma - cfg.sigma) < 0.05


def test_half_life_and_stationary_std():
    """
    Test closed-form OU half-life and stationary std computations.
    """

    params = OUParams(kappa=2.0, theta=0.0, sigma=0.1, dt=1 / 252)

    # half-life t1/2 = ln(2)/kappa
    expected_half_life = np.log(2) / 2.0
    assert pytest.approx(expected_half_life, rel=1e-6) == params.half_life

    # stationary std = sigma/sqrt(2*kappa)
    expected_std = 0.1 / np.sqrt(4.0)
    assert pytest.approx(expected_std, rel=1e-6) == params.stationary_std


def test_reject_non_1d_input():
    """
    fit_ou_to_spread must reject 2D arrays or invalid input.
    """

    with pytest.raises(ValueError):
        fit_ou_to_spread(np.zeros((10, 10)), dt=1 / 252)

    with pytest.raises(ValueError):
        fit_ou_to_spread(np.array([[1, 2, 3]]), dt=1 / 252)
