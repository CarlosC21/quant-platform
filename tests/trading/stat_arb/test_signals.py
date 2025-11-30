# tests/trading/stat_arb/test_signals.py

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_platform.trading.stat_arb.schemas import StatArbSignal
from quant_platform.trading.stat_arb.spreads.schemas import ZScoreResult
from quant_platform.trading.stat_arb.spreads.regime_filter import RegimeFilterConfig
from quant_platform.trading.stat_arb.signals import build_signals_from_zscores


def _make_simple_zscore_result() -> ZScoreResult:
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    z = np.array([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=float)
    spread = np.zeros_like(z)

    return ZScoreResult(
        symbol_y="Y",
        symbol_x="X",
        zscore=z,
        spread=spread,
        timestamps=idx.to_numpy(),
        method="rolling",
        window=3,
    )


def test_build_signals_from_zscores_basic():
    z_res = _make_simple_zscore_result()

    # Simple regime: always allowed
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    df_regime = pd.DataFrame(
        {
            "date": idx,
            "regime_hmm": [0, 0, 0, 0, 0],
            "prob_hmm_0": [0.9] * 5,
        }
    )

    cfg = RegimeFilterConfig(
        regime_name="hmm", allowed_regimes=[0], min_regime_prob=0.5
    )

    z_entry = 2.0
    z_exit = 0.5

    signals = build_signals_from_zscores(
        zscore_result=z_res,
        z_entry=z_entry,
        z_exit=z_exit,
        regime_df=df_regime,
        regime_config=cfg,
    )

    assert len(signals) == 5
    assert all(isinstance(s, StatArbSignal) for s in signals)

    sides = [s.side for s in signals]
    # z = [-3, -1, 0, 1, 3]
    # with z_entry=2 -> t0: long, t4: short
    assert sides[0] == "long"
    assert sides[-1] == "short"
    # mid points should be flat or exit
    assert sides[2] == "flat"


def test_build_signals_regime_block():
    z_res = _make_simple_zscore_result()

    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    # Regime 1 is blocked, only 0 is allowed
    df_regime = pd.DataFrame(
        {
            "date": idx,
            "regime_hmm": [1, 1, 1, 1, 1],
            "prob_hmm_1": [0.9] * 5,
        }
    )

    cfg = RegimeFilterConfig(
        regime_name="hmm", allowed_regimes=[0], min_regime_prob=0.5
    )

    z_entry = 1.0
    z_exit = 0.5

    signals = build_signals_from_zscores(
        zscore_result=z_res,
        z_entry=z_entry,
        z_exit=z_exit,
        regime_df=df_regime,
        regime_config=cfg,
    )

    # All blocked by regime
    assert all(s.tradable is False for s in signals)
    assert all(s.side == "flat" for s in signals)
    assert all(s.reason == "regime_block" for s in signals)
