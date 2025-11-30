from __future__ import annotations

from datetime import datetime, timezone

import pytest

from quant_platform.execution.enums import Side, LiquiditySide, Venue
from quant_platform.execution.ledger import TradeLedger, PositionBook, TradeRecord
from quant_platform.execution.models import Fill


def _ts() -> datetime:
    return datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _fill(
    order_id: str,
    symbol: str,
    side: Side,
    quantity: float,
    price: float,
    cost: float = 0.0,
) -> Fill:
    return Fill(
        order_id=order_id,
        symbol=symbol,
        side=side,
        quantity=quantity,
        price=price,
        timestamp=_ts(),
        liquidity_side=LiquiditySide.TAKER,
        venue=Venue.SIMULATED,
        cost=cost,
    )


def test_long_position_partial_sell_realized_pnl_and_fees_and_cash():
    ledger = TradeLedger()

    # Buy 100 @ 10, cost = 1.0
    f1 = _fill("o1", "AAPL", Side.BUY, 100.0, 10.0, cost=1.0)
    ledger.record_fill(f1)

    # Sell 40 @ 12, cost = 0.5
    f2 = _fill("o2", "AAPL", Side.SELL, 40.0, 12.0, cost=0.5)
    ledger.record_fill(f2)

    pos = ledger.get_position("AAPL")
    assert pos is not None
    assert pos.quantity == pytest.approx(60.0)
    assert pos.avg_price == pytest.approx(10.0)

    # Realized PnL on 40 shares: (12 - 10) * 40 = 80
    assert pos.realized_pnl == pytest.approx(80.0)
    # Fees accumulated
    assert pos.fees == pytest.approx(1.5)

    # Cash:
    #   after buy:  cash = -100*10 - 1.0 = -1001.0
    #   after sell: cash += 40*12 - 0.5 = 480 - 0.5 = 479.5
    #   final:     -1001 + 479.5 = -521.5
    assert ledger.cash == pytest.approx(-521.5)

    # Gross realized PnL and total fees via ledger
    assert ledger.total_realized_pnl == pytest.approx(80.0)
    assert ledger.total_fees == pytest.approx(1.5)
    assert ledger.total_net_realized_pnl == pytest.approx(80.0 - 1.5)


def test_cross_through_zero_opens_new_position_at_new_price():
    ledger = TradeLedger()

    # Buy 100 @ 10
    ledger.record_fill(_fill("o1", "AAPL", Side.BUY, 100.0, 10.0))
    # Sell 150 @ 11 → close 100 long, open 50 short at 11
    ledger.record_fill(_fill("o2", "AAPL", Side.SELL, 150.0, 11.0))

    pos = ledger.get_position("AAPL")
    assert pos is not None

    # 100 long + (-150) = -50 → net short 50
    assert pos.quantity == pytest.approx(-50.0)
    # Crossed zero, new short opened at last trade price
    assert pos.avg_price == pytest.approx(11.0)

    # Realized PnL: closed 100 shares at (11 - 10) = 1 * 100 = 100
    assert pos.realized_pnl == pytest.approx(100.0)


def test_short_then_partial_cover():
    ledger = TradeLedger()

    # Sell short 50 @ 20 → position = -50 @ 20
    ledger.record_fill(_fill("o1", "MSFT", Side.SELL, 50.0, 20.0))

    # Buy to cover 20 @ 18:
    #   For short, PnL = (avg_price - trade_price) * closed_qty
    #   = (20 - 18) * 20 = 40
    ledger.record_fill(_fill("o2", "MSFT", Side.BUY, 20.0, 18.0))

    pos = ledger.get_position("MSFT")
    assert pos is not None
    assert pos.quantity == pytest.approx(-30.0)  # still short 30
    assert pos.avg_price == pytest.approx(20.0)  # cost basis unchanged
    assert pos.realized_pnl == pytest.approx(40.0)


def test_portfolio_value_uses_cash_plus_positions():
    book = PositionBook()

    # Simulate trades directly through PositionBook + TradeRecord
    t1 = TradeRecord(
        order_id="o1",
        symbol="AAPL",
        side=Side.BUY,
        quantity=100.0,
        price=10.0,
        timestamp=_ts(),
        cost=1.0,
    )
    book.apply_trade(t1)

    t2 = TradeRecord(
        order_id="o2",
        symbol="AAPL",
        side=Side.SELL,
        quantity=40.0,
        price=12.0,
        timestamp=_ts(),
        cost=0.5,
    )
    book.apply_trade(t2)

    # Same situation as earlier test:
    # - position: 60 @ 10
    # - cash: -521.5
    pos = book.get_position("AAPL")
    assert pos.quantity == pytest.approx(60.0)
    assert pos.avg_price == pytest.approx(10.0)

    prices = {"AAPL": 11.0}
    # portfolio value = cash + 60 * 11 = -521.5 + 660 = 138.5
    assert book.portfolio_value(prices) == pytest.approx(138.5)
