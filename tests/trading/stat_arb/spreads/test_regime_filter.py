# tests/trading/stat_arb/spreads/test_regime_filter.py

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_platform.trading.stat_arb.spreads.schemas import ZScoreResult
from quant_platform.trading.stat_arb.spreads.regime_filter import (
    RegimeFilterConfig,
    build_regime_mask,
)


def _dummy_zscore_result(n: int) -> ZScoreResult:
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    spread = np.zeros(n, dtype=float)
    z = np.linspace(-2.0, 2.0, n)

    return ZScoreResult(
        symbol_y="Y",
        symbol_x="X",
        zscore=z,
        spread=spread,
        timestamps=idx.to_numpy(),
        method="rolling",
        window=10,
    )


def test_regime_filter_label_only():
    n = 10
    z_res = _dummy_zscore_result(n)

    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    df_regime = pd.DataFrame(
        {
            "date": idx,
            "regime_hmm": [0, 0, 1, 1, 0, 2, 2, 0, 0, 1],
        }
    )

    cfg = RegimeFilterConfig(regime_name="hmm", allowed_regimes=[0])

    mask = build_regime_mask(z_res, df_regime, cfg)

    # Only points where regime_hmm == 0 are tradable
    expected = np.array(
        [True, True, False, False, True, False, False, True, True, False]
    )
    assert mask.shape == (n,)
    assert np.array_equal(mask, expected)


def test_regime_filter_with_prob_threshold():
    n = 5
    z_res = _dummy_zscore_result(n)

    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    df_regime = pd.DataFrame(
        {
            "date": idx,
            "regime_hmm": [0, 0, 0, 0, 0],
            "prob_hmm_0": [0.9, 0.4, 0.8, 0.2, 0.95],
        }
    )

    cfg = RegimeFilterConfig(
        regime_name="hmm", allowed_regimes=[0], min_regime_prob=0.7
    )

    mask = build_regime_mask(z_res, df_regime, cfg)

    expected = np.array([True, False, True, False, True])
    assert np.array_equal(mask, expected)
