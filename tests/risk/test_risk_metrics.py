# tests/risk/test_metrics.py
from datetime import datetime, timedelta

import numpy as np

from quant_platform.risk.metrics import compute_drawdown


def test_compute_drawdown_basic():
    dates = np.array(
        [datetime(2025, 1, 1) + timedelta(days=i) for i in range(5)],
        dtype="datetime64[ns]",
    )
    # Equity: 100 -> 120 (peak) -> 90 (drawdown) -> 120 (recovery)
    equity = np.array([100.0, 110.0, 120.0, 90.0, 120.0], dtype=float)

    drawdown, stats = compute_drawdown(equity, dates)

    assert np.isclose(drawdown[2], 0.0, atol=1e-12)
    assert np.isclose(drawdown[3], -0.25, atol=1e-12)
    assert np.isclose(stats.max_drawdown, -0.25, atol=1e-12)
    assert stats.time_under_water_days == 2


def test_compute_drawdown_invalid_inputs():
    dates = np.array([datetime(2025, 1, 1)], dtype="datetime64[ns]")

    # Empty equity
    try:
        compute_drawdown(
            np.array([], dtype=float), np.array([], dtype="datetime64[ns]")
        )
    except ValueError as exc:
        assert "non-empty" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty equity curve.")

    # Non-positive equity
    try:
        compute_drawdown(np.array([0.0], dtype=float), dates)
    except ValueError as exc:
        assert "strictly positive" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-positive equity.")
