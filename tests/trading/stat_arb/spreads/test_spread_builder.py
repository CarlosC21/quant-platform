# tests/trading/stat_arb/spreads/test_spread_builder.py

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_platform.trading.stat_arb.spreads.spread_builder import (
    build_static_spread,
)
from quant_platform.trading.stat_arb.spreads.schemas import StaticSpreadResult


def test_static_spread_builder():
    idx = pd.date_range("2020-01-01", periods=5, freq="D")

    x = pd.Series([1, 2, 3, 4, 5], index=idx, name="X")
    y = pd.Series([2, 4, 6, 8, 10], index=idx, name="Y")

    beta = 2.0  # true hedge ratio

    result = build_static_spread(y, x, beta)

    assert isinstance(result, StaticSpreadResult)
    assert result.beta == beta
    assert result.spread.shape == (5,)
    np.testing.assert_array_almost_equal(result.spread, np.zeros(5))


def test_static_spread_alignment():
    idx1 = pd.date_range("2020-01-01", periods=5, freq="D")
    idx2 = pd.date_range("2020-01-02", periods=5, freq="D")  # shifted index

    y = pd.Series([10, 11, 13, 14, 15], index=idx1, name="Y")
    x = pd.Series([1, 1, 2, 2, 2], index=idx2, name="X")

    beta = 1.0

    result = build_static_spread(y, x, beta)

    # aligned on intersection: idx2[0:4]
    expected_y = y.loc[idx2[:-1]].to_numpy()
    expected_x = x[:-1].to_numpy()

    expected_spread = expected_y - beta * expected_x

    np.testing.assert_array_almost_equal(result.spread, expected_spread)
    assert len(result.timestamps) == 4
