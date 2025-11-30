# tests/trading/stat_arb/spreads/test_zscore.py

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_platform.sde.schemas import OUConfig, SimConfig
from quant_platform.sde.processes.ou import ou_exact

from quant_platform.trading.stat_arb.spreads.ou_model import (
    fit_ou_to_spread,
)
from quant_platform.trading.stat_arb.spreads.schemas import (
    StaticSpreadResult,
)
from quant_platform.trading.stat_arb.spreads.zscore import (
    zscore_from_ou,
    zscore_rolling,
)


def _make_static_spread_from_ou(
    cfg: OUConfig,
    sim: SimConfig,
) -> StaticSpreadResult:
    X = ou_exact(cfg, sim)  # shape (n_paths, n_steps+1)
    spread = X[0]  # 1D

    idx = pd.date_range("2020-01-01", periods=spread.size, freq="D")

    return StaticSpreadResult(
        symbol_y="Y_spread",
        symbol_x="X_spread",
        beta=1.0,
        spread=spread,
        timestamps=idx.to_numpy(),
    )


def test_zscore_from_ou_is_approximately_standard_normal():
    cfg = OUConfig(kappa=1.0, theta=0.0, sigma=0.1, x0=0.0)
    sim = SimConfig(n_paths=1, n_steps=5000, dt=1 / 252, seed=7)

    spread_result = _make_static_spread_from_ou(cfg, sim)

    ou_params = fit_ou_to_spread(spread_result.spread, dt=sim.dt)
    z_res = zscore_from_ou(spread_result, ou_params)

    z = z_res.zscore
    assert z.shape == spread_result.spread.shape

    # Remove initial few points (burn-in)
    z_burned = z[100:]

    mean_z = float(np.nanmean(z_burned))
    std_z = float(np.nanstd(z_burned))

    # In simulated OU world with estimated parameters, z-scores should be
    # roughly standard normal: mean ~ 0, std ~ 1.
    assert abs(mean_z) < 0.1
    assert 0.8 < std_z < 1.2


def test_zscore_rolling_basic_properties():
    idx = pd.date_range("2020-01-01", periods=10, freq="D")
    spread = np.arange(10, dtype=float)  # 0..9

    spread_result = StaticSpreadResult(
        symbol_y="Y_r",
        symbol_x="X_r",
        beta=1.0,
        spread=spread,
        timestamps=idx.to_numpy(),
    )

    window = 3
    z_res = zscore_rolling(spread_result, window=window)

    assert z_res.zscore.shape == spread.shape
    assert z_res.method == "rolling"
    assert z_res.window == window

    # For the first window-1 points, we expect NaNs due to min_periods == window
    assert np.isnan(z_res.zscore[: window - 1]).all()
    # After that, at least some finite values
    assert np.isfinite(z_res.zscore[window:]).any()
