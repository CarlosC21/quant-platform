from __future__ import annotations

import pandas as pd

from quant_platform.execution.context import ExecutionContext
from quant_platform.runner.core import Strategy, run_backtest, RunContext


class BuyAndHoldUsingHelpers(Strategy):
    """
    Simple strategy exercising the new RunContext convenience methods:

    - On first bar: buy 1 share of AAA via ctx.order_market(...)
    - On each bar: we can optionally inspect ctx.snapshot()
    """

    def on_start(self, context: RunContext) -> None:
        # No-op for this simple test
        pass

    def on_bar(self, context: RunContext, bar_data: pd.DataFrame) -> None:
        # Only act on the first bar
        if context.bar_index == 0:
            # Buy 1 share of AAA using the helper
            context.order_market("AAA", signed_qty=1.0)

            # Ensure snapshot() can be called without error
            snap = context.snapshot()
            assert "total_equity" in snap
            assert "cash" in snap
            assert "positions" in snap

    def on_end(self, context: RunContext) -> None:
        # No-op for this simple test
        pass


def test_run_backtest_with_convenience_api() -> None:
    """
    Integration test to confirm that:

    - RunContext.order_market routes through ExecutionContext / engine.
    - Equity curve reflects the buy-and-hold of 1 share of AAA.
    - RunContext.snapshot is usable inside the strategy.
    """
    # Simple 3-bar price series for AAA
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=3, freq="D"),
            "symbol": ["AAA"] * 3,
            "close": [100.0, 110.0, 120.0],
        }
    ).set_index(["timestamp", "symbol"])

    # Start with 1,000 in cash
    exec_ctx = ExecutionContext(initial_cash=1_000.0)

    strat = BuyAndHoldUsingHelpers()

    result = run_backtest(
        strategy=strat,
        market_data=data,
        execution_context=exec_ctx,
    )

    eq = result.equity_curve

    # After first bar:
    # - We buy 1 share at 100
    # - Cash ~ 900, position = 1 share
    #
    # For this test, we mainly care that the equity
    # evolves with the AAA price:
    #   bar 0: 1000 (all cash before marking to market)
    #   bar 1: 900 + 110 = 1010
    #   bar 2: 900 + 120 = 1020

    assert len(eq) == 3

    # First point may reflect initial state depending on exact timing,
    # but we assert the *relative* changes are correct.
    start_equity = eq.iloc[0]
    assert eq.iloc[1] == start_equity + 10.0
    assert eq.iloc[2] == start_equity + 20.0

    # positions_ts should show a non-zero AAA position at the end
    assert result.positions_ts is not None
    pts = result.positions_ts
    # Find last timestamp row for AAA
    last_ts = pts["timestamp"].max()
    last_row = pts[(pts["timestamp"] == last_ts) & (pts["symbol"] == "AAA")].iloc[0]
    assert last_row["quantity"] == 1.0
