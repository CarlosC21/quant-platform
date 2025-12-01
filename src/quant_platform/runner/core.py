from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

import numpy as np
import pandas as pd

from quant_platform.execution.context import ExecutionContext

LOGGER = logging.getLogger(__name__)


# =====================================================================
# Strategy protocol
# =====================================================================


class Strategy(Protocol):
    def on_start(self, context: "RunContext") -> None:
        ...

    def on_bar(self, context: "RunContext", bar_data: pd.DataFrame) -> None:
        ...

    def on_end(self, context: "RunContext") -> None:
        ...


# =====================================================================
# Run context + result containers
# =====================================================================


@dataclass
class RunContext:
    timestamp: pd.Timestamp
    bar_index: int
    execution_context: ExecutionContext
    config: Any | None
    extra: Dict[str, Any]
    market_data: Optional[pd.DataFrame] = None
    current_bar: Optional[pd.DataFrame] = None


@dataclass
class BacktestResult:
    """
    Container for full backtest results — Week 12 compatible.

    positions_ts:
        Optional DataFrame of position snapshots per bar.
    """

    equity_curve: pd.Series
    drawdowns: pd.Series
    risk_metrics: Dict[str, float]
    prices_last: pd.DataFrame
    config: Any | None
    positions_ts: Optional[pd.DataFrame] = None


# =====================================================================
# Helpers
# =====================================================================


def _iterate_timestamps(market_data: pd.DataFrame) -> pd.Index:
    idx = market_data.index
    if isinstance(idx, pd.MultiIndex):
        ts = idx.get_level_values(0)
    else:
        ts = idx
    return ts.unique().sort_values()


def _slice_bar(market_data: pd.DataFrame, ts: pd.Timestamp) -> pd.DataFrame:
    idx = market_data.index
    if isinstance(idx, pd.MultiIndex):
        return market_data.loc[(ts,)]
    return market_data.loc[[ts]]


def _extract_prices_for_ledger(
    bar_data: pd.DataFrame,
    price_col: str = "close",
) -> Dict[str, float]:
    # Index = symbol
    if bar_data.index.name == "symbol":
        return {str(sym): float(row[price_col]) for sym, row in bar_data.iterrows()}

    # MultiIndex
    if isinstance(bar_data.index, pd.MultiIndex):
        symbols = bar_data.index.get_level_values(-1)
        return {
            str(sym): float(bar_data.xs(sym, level=-1)[price_col])
            for sym in symbols.unique()
        }

    # Flat with 'symbol' column
    if "symbol" in bar_data.columns:
        return {
            str(row["symbol"]): float(row[price_col]) for _, row in bar_data.iterrows()
        }

    raise KeyError(
        "Cannot extract prices: bar_data must have symbol index or symbol column."
    )


def _compute_drawdowns(equity: pd.Series) -> pd.Series:
    if equity.empty:
        return pd.Series(dtype=float, name="drawdown")
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    dd.name = "drawdown"
    return dd


def _compute_risk_metrics(
    equity: pd.Series,
    periods_per_year: int = 252,
) -> Dict[str, float]:
    if equity.empty or len(equity) < 2:
        return {
            "cumulative_return": float("nan"),
            "max_drawdown": float("nan"),
            "sharpe": float("nan"),
            "volatility": float("nan"),
        }

    ret = equity.pct_change().dropna()
    if ret.empty:
        return {
            "cumulative_return": float("nan"),
            "max_drawdown": float("nan"),
            "sharpe": float("nan"),
            "volatility": float("nan"),
        }

    start = float(equity.iloc[0])
    end = float(equity.iloc[-1])

    if start == 0.0:
        cumulative_return = float("nan")
    else:
        cumulative_return = end / start - 1.0

    drawdowns = _compute_drawdowns(equity)
    max_drawdown = float(drawdowns.min())

    vol = float(ret.std(ddof=1)) * np.sqrt(periods_per_year)
    mean = float(ret.mean()) * periods_per_year
    sharpe = mean / vol if vol > 0 else float("nan")

    return {
        "cumulative_return": cumulative_return,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "volatility": vol,
    }


# =====================================================================
# Main runner
# =====================================================================


def run_backtest(
    strategy: Strategy,
    market_data: pd.DataFrame,
    execution_context: ExecutionContext,
    *,
    config: Any | None = None,
    price_col: str = "close",
) -> BacktestResult:
    timestamps = _iterate_timestamps(market_data)
    if len(timestamps) == 0:
        raise ValueError("run_backtest received empty market_data.")

    LOGGER.info("Starting backtest with %d bars.", len(timestamps))

    equity_values: List[float] = []
    equity_index: List[pd.Timestamp] = []
    last_prices: Optional[pd.DataFrame] = None

    # Position snapshots per bar
    positions_records: List[Dict[str, Any]] = []

    ctx = RunContext(
        timestamp=pd.Timestamp(timestamps[0]),
        bar_index=0,
        execution_context=execution_context,
        config=config,
        extra={},
        market_data=market_data,
        current_bar=None,
    )

    strategy.on_start(ctx)

    for i, ts in enumerate(timestamps):
        ts = pd.Timestamp(ts)
        bar_data = _slice_bar(market_data, ts)
        last_prices = bar_data.copy()

        ctx.timestamp = ts
        ctx.bar_index = i
        ctx.current_bar = bar_data

        strategy.on_bar(ctx, bar_data)

        # Portfolio valuation
        prices_for_ledger = _extract_prices_for_ledger(bar_data, price_col=price_col)
        equity = execution_context.ledger.portfolio_value(prices_for_ledger)

        equity_values.append(float(equity))
        equity_index.append(ts)

        # Snapshot cumulative positions
        ledger = execution_context.ledger
        try:
            position_book = ledger.position_book.positions  # PositionBook.positions
            for sym, pos in position_book.items():
                positions_records.append(
                    {
                        "timestamp": ts,
                        "symbol": sym,
                        "quantity": pos.quantity,
                        "avg_price": pos.avg_price,
                    }
                )
        except AttributeError:
            # Fallback for simpler ledgers without PositionBook
            pass

    strategy.on_end(ctx)

    equity_series = pd.Series(
        equity_values,
        index=pd.DatetimeIndex(equity_index, name="timestamp"),
        name="equity",
    ).sort_index()

    drawdowns = _compute_drawdowns(equity_series)
    risk_metrics = _compute_risk_metrics(equity_series)

    if last_prices is None:
        last_prices = pd.DataFrame()

    positions_ts = pd.DataFrame(positions_records) if positions_records else None

    result = BacktestResult(
        equity_curve=equity_series,
        drawdowns=drawdowns,
        risk_metrics=risk_metrics,
        prices_last=last_prices,
        config=config,
        positions_ts=positions_ts,
    )

    LOGGER.info(
        "Completed backtest; final equity=%.2f; max_drawdown=%.4f; sharpe=%.4f",
        equity_series.iloc[-1],
        risk_metrics.get("max_drawdown", float("nan")),
        risk_metrics.get("sharpe", float("nan")),
    )

    return result
