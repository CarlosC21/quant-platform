from __future__ import annotations

from typing import Dict


from quant_platform.execution.context import ExecutionContext
from quant_platform.execution.ledger import Position, PositionBook, TradeLedger


def test_get_portfolio_snapshot_basic() -> None:
    # Create an ExecutionContext with a custom ledger / position book
    ledger = TradeLedger()
    ledger.position_book = PositionBook(
        cash=1_000.0,
        positions={
            "AAA": Position(symbol="AAA", quantity=10.0, avg_price=100.0),
            "BBB": Position(symbol="BBB", quantity=-5.0, avg_price=50.0),
        },
    )

    ctx = ExecutionContext(initial_cash=0.0, ledger=ledger)

    prices: Dict[str, float] = {"AAA": 110.0, "BBB": 40.0}

    snap = ctx.get_portfolio_snapshot(prices)

    # Cash is 1,000
    assert snap["cash"] == 1_000.0

    # Position quantities preserved
    assert snap["positions"]["AAA"] == 10.0
    assert snap["positions"]["BBB"] == -5.0

    # Market value = 10*110 + (-5)*40 = 1,100 - 200 = 900
    assert snap["market_value"] == 900.0

    # Total equity = cash + market_value
    assert snap["total_equity"] == 1_900.0
