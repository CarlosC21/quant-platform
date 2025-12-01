from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Protocol, Optional

import numpy as np
import pandas as pd

from quant_platform.execution.context import ExecutionContext

LOGGER = logging.getLogger(__name__)


# ===============================================================
# Strategy Protocol
# ===============================================================


class Strategy(Protocol):
    """
    Minimal interface that strategies must implement to be usable with the runner.
    """

    def on_start(self, context: RunContext) -> None:
        ...

    def on_bar(self, context: RunContext, bar_data: pd.DataFrame) -> None:
        ...

    def on_end(self, context: RunContext) -> None:
        ...


# ===============================================================
# Runtime Context
# ===============================================================


@dataclass
class RunContext:
    """
    Runtime context passed to strategies during a backtest.
    """

    timestamp: pd.Timestamp
    bar_index: int
    execution_context: ExecutionContext
    config: Any | None
    extra: Dict[str, Any]

    # NEW FIELDS (your strategy requires these)
    market_data: Optional[pd.DataFrame] = None
    current_bar: Optional[pd.DataFrame] = None


# ===============================================================
# Results Container
# ===============================================================


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    drawdowns: pd.Series
    risk_metrics: Dict[str, float]
    prices_last: pd.DataFrame
    config: Any | None


# ===============================================================
# Internal helpers
# ===============================================================


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
    # Symbol index case
    if bar_data.index.name == "symbol":
        return {
            str(symbol): float(row[price_col]) for symbol, row in bar_data.iterrows()
        }

    # MultiIndex
    if isinstance(bar_data.index, pd.MultiIndex):
        syms = bar_data.index.get_level_values(-1)
        return {
            str(sym): float(bar_data.xs(sym, level=-1)[price_col])
            for sym in syms.unique()
        }

    # Flat DataFrame with symbol column
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

    cumulative_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
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


# ===============================================================
# Main Backtest Runner
# ===============================================================


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

    equity_values: list[float] = []
    equity_index: list[pd.Timestamp] = []
    last_prices: Optional[pd.DataFrame] = None

    # ----------------------------------------------------------
    # Initialize the RunContext
    # ----------------------------------------------------------
    ctx = RunContext(
        timestamp=pd.Timestamp(timestamps[0]),
        bar_index=0,
        execution_context=execution_context,
        config=config,
        extra={},
        market_data=market_data,  # <-- FIX: make data accessible in on_start()
        current_bar=None,
    )

    # Strategy startup
    strategy.on_start(ctx)

    # ----------------------------------------------------------
    # Main iteration
    # ----------------------------------------------------------
    for i, ts in enumerate(timestamps):
        ts = pd.Timestamp(ts)
        bar_data = _slice_bar(market_data, ts)
        last_prices = bar_data.copy()

        # Update context
        ctx.timestamp = ts
        ctx.bar_index = i
        ctx.current_bar = bar_data  # <-- FIX: current bar provided each step

        # Strategy handles the bar
        strategy.on_bar(ctx, bar_data)

        # Compute valuation
        prices_for_ledger = _extract_prices_for_ledger(bar_data, price_col=price_col)
        equity = execution_context.ledger.portfolio_value(prices_for_ledger)

        equity_values.append(float(equity))
        equity_index.append(ts)

    # Finalize strategy
    strategy.on_end(ctx)

    # Build results
    equity_series = pd.Series(
        equity_values,
        index=pd.DatetimeIndex(equity_index, name="timestamp"),
        name="equity",
    ).sort_index()

    drawdowns = _compute_drawdowns(equity_series)
    risk_metrics = _compute_risk_metrics(equity_series)

    if last_prices is None:
        last_prices = pd.DataFrame()

    result = BacktestResult(
        equity_curve=equity_series,
        drawdowns=drawdowns,
        risk_metrics=risk_metrics,
        prices_last=last_prices,
        config=config,
    )

    LOGGER.info(
        "Completed backtest; final equity=%.2f; max_drawdown=%.4f; sharpe=%.4f",
        equity_series.iloc[-1],
        risk_metrics.get("max_drawdown", float("nan")),
        risk_metrics.get("sharpe", float("nan")),
    )

    return result
