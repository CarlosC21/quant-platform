# tests/risk/test_stress.py
import numpy as np

from quant_platform.risk.stress import (
    apply_return_shock,
    scenario_pnl,
    historical_stress_pnls,
)


def test_apply_return_shock_basic():
    R = np.array(
        [
            [0.01, 0.02],
            [0.00, -0.01],
        ],
    )
    shock = np.array([-0.02, 0.01])

    shocked = apply_return_shock(R, shock)

    expected = R + shock[None, :]
    assert np.allclose(shocked, expected)


def test_scenario_pnl_basic():
    w = np.array([0.6, 0.4])
    r = np.array([-0.10, 0.05])

    pnl = scenario_pnl(w, r)

    expected = 0.6 * (-0.10) + 0.4 * 0.05
    assert np.isclose(pnl, expected, atol=1e-12)


def test_historical_stress_pnls_basic():
    w = np.array([0.5, 0.5])
    R = np.array(
        [
            [0.01, 0.03],
            [-0.02, 0.00],
            [0.00, -0.01],
        ],
    )

    pnls = historical_stress_pnls(w, R)

    expected = R @ w
    assert np.allclose(pnls, expected)
    # worst day must match min expected PnL
    assert pnls.min() == expected.min()
