# tests/trading/stat_arb/pipeline/test_stat_arb_pipeline.py

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_platform.trading.stat_arb.pipeline.stat_arb_pipeline import (
    StatArbPairPipeline,
)
from quant_platform.trading.stat_arb.schemas import (
    StatArbPairConfig,
    StatArbPipelineResult,
)
from quant_platform.trading.stat_arb.spreads.regime_filter import RegimeFilterConfig


def _simulate_cointegrated_pair(
    n: int,
    beta: float,
    rng: np.random.Generator,
) -> tuple[pd.Series, pd.Series]:
    """
    Simulate a simple cointegrated pair:

        X_t: random walk
        Y_t: alpha + beta * X_t + stationary AR(1) noise
    """
    idx = pd.date_range("2020-01-01", periods=n, freq="D")

    eps_x = rng.normal(scale=1.0, size=n)
    x = np.cumsum(eps_x)

    rho = 0.5
    eps_noise = rng.normal(scale=0.5, size=n)
    noise = np.zeros(n, dtype=float)
    for t in range(1, n):
        noise[t] = rho * noise[t - 1] + eps_noise[t]

    alpha_true = 1.0
    y = alpha_true + beta * x + noise

    series_x = pd.Series(x, index=idx, name="X_pipe")
    series_y = pd.Series(y, index=idx, name="Y_pipe")
    return series_y, series_x


def _simulate_non_cointegrated_pair(
    n: int,
    rng: np.random.Generator,
) -> tuple[pd.Series, pd.Series]:
    """
    Simulate two independent random walks (non-cointegrated).
    """
    idx = pd.date_range("2020-01-01", periods=n, freq="D")

    eps_x = rng.normal(scale=1.0, size=n)
    eps_y = rng.normal(scale=1.0, size=n)

    x = np.cumsum(eps_x)
    y = np.cumsum(eps_y)

    series_x = pd.Series(x, index=idx, name="X_nc_pipe")
    series_y = pd.Series(y, index=idx, name="Y_nc_pipe")
    return series_y, series_x


def _make_regime_df_all_good(n: int) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "date": idx,
            "regime_hmm": [0] * n,
            "prob_hmm_0": [0.9] * n,
        }
    )


def test_stat_arb_pipeline_cointegrated_pair_static_beta():
    rng = np.random.default_rng(123)
    n = 400
    beta_true = 1.5

    series_y, series_x = _simulate_cointegrated_pair(n=n, beta=beta_true, rng=rng)
    regime_df = _make_regime_df_all_good(n)

    pair_cfg = StatArbPairConfig(
        symbol_y="Y_pipe",
        symbol_x="X_pipe",
        dt=1 / 252,
        z_entry=2.0,
        z_exit=0.5,
        use_kalman=False,
        fail_if_not_coint=True,
    )

    regime_cfg = RegimeFilterConfig(
        regime_name="hmm",
        allowed_regimes=[0],
        min_regime_prob=0.5,
    )

    pipeline = StatArbPairPipeline(
        pair_config=pair_cfg,
        regime_config=regime_cfg,
    )

    result = pipeline.run(series_y=series_y, series_x=series_x, regime_df=regime_df)

    assert isinstance(result, StatArbPipelineResult)
    assert result.cointegrated is True
    assert len(result.signals) == n

    sides = {s.side for s in result.signals if s.tradable}
    # We expect at least some non-flat signals
    assert "long" in sides or "short" in sides


def test_stat_arb_pipeline_non_cointegrated_pair_returns_flat():
    rng = np.random.default_rng(2024)
    n = 300

    series_y, series_x = _simulate_non_cointegrated_pair(n=n, rng=rng)
    regime_df = _make_regime_df_all_good(n)

    pair_cfg = StatArbPairConfig(
        symbol_y="Y_nc_pipe",
        symbol_x="X_nc_pipe",
        dt=1 / 252,
        z_entry=2.0,
        z_exit=0.5,
        use_kalman=False,
        fail_if_not_coint=False,
    )

    regime_cfg = RegimeFilterConfig(
        regime_name="hmm",
        allowed_regimes=[0],
        min_regime_prob=0.5,
    )

    pipeline = StatArbPairPipeline(
        pair_config=pair_cfg,
        regime_config=regime_cfg,
    )

    result = pipeline.run(series_y=series_y, series_x=series_x, regime_df=regime_df)

    assert result.cointegrated is False
    assert len(result.signals) == n
    assert all(s.side == "flat" for s in result.signals)
    assert all(s.tradable is False for s in result.signals)
    assert all(s.reason == "not_cointegrated" for s in result.signals)
