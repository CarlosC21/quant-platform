from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant_platform.runner.core import (
    BacktestResult,
    RunContext,
    Strategy,
    run_backtest,
)


# ---------------------------------------------------------------------------
# Dummy ledger & execution context for testing runner logic only.
# We do NOT re-test the real ExecutionEngine/Broker here.
# ---------------------------------------------------------------------------


class DummyLedger:
    """
    Minimal ledger with a portfolio_value(prices) method.

    For testing, portfolio value is simply the sum of all prices
    (so as prices increase, equity must increase).
    """

    def __init__(self) -> None:
        self.calls: int = 0

    def portfolio_value(self, prices: dict[str, float]) -> float:
        self.calls += 1
        return float(sum(prices.values()))


@dataclass
class DummyExecutionContext:
    ledger: DummyLedger


class DummyStrategy(Strategy):
    """
    Strategy that does nothing in on_bar.

    We only test that the runner:
      - iterates timestamps correctly
      - computes equity using ledger.portfolio_value
      - returns a BacktestResult with sensible shapes.
    """

    def on_start(self, context: RunContext) -> None:  # noqa: D401
        # Could stash warmup data in context.extra if needed.
        return

    def on_bar(self, context: RunContext, bar_data: pd.DataFrame) -> None:  # noqa: D401
        # No orders submitted; ledger's portfolio_value will just see current prices.
        return

    def on_end(self, context: RunContext) -> None:  # noqa: D401
        return


def _make_multiindex_market_data() -> pd.DataFrame:
    """
    Build a simple MultiIndex[timestamp, symbol] DataFrame with
    monotonically increasing 'close' prices across time.
    """
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    symbols = ["AAA", "BBB"]
    idx = pd.MultiIndex.from_product([dates, symbols], names=["timestamp", "symbol"])

    # Close prices: increase over time for each symbol
    closes = []
    for i, ts in enumerate(dates):
        for _sym in symbols:
            closes.append(100.0 + i)

    df = pd.DataFrame({"close": closes}, index=idx)
    return df


def test_run_backtest_with_dummy_ledger() -> None:
    market_data = _make_multiindex_market_data()
    ledger = DummyLedger()
    exec_ctx = DummyExecutionContext(ledger=ledger)
    strategy = DummyStrategy()

    result: BacktestResult = run_backtest(
        strategy=strategy,
        market_data=market_data,
        execution_context=exec_ctx,  # type: ignore[arg-type]
    )

    # Basic shape checks
    assert isinstance(result, BacktestResult)
    # 5 unique timestamps → 5 equity points
    assert result.equity_curve.shape[0] == 5

    # Equity should be strictly increasing since prices strictly increase.
    assert result.equity_curve.is_monotonic_increasing

    # Drawdowns: min drawdown <= 0
    assert result.drawdowns.min() <= 0.0

    # Risk metrics must contain these keys
    for key in ("cumulative_return", "max_drawdown", "sharpe", "volatility"):
        assert key in result.risk_metrics

    # Ledger portfolio_value should have been called once per bar
    assert ledger.calls == 5
