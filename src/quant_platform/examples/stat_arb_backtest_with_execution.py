"""
Self-contained stat-arb backtest example integrating:

    • StatArbPairPipeline (signals)
    • StatArb position sizing (portfolio layer)
    • ExecutionContext (execution engine)
    • Runner Core (orchestration)
    • Simple synthetic bar data for AAA/BBB

This module is intentionally minimal and reproducible.
"""

from __future__ import annotations

from typing import Optional, Dict, Any, List
import logging

import numpy as np
import pandas as pd

from quant_platform.trading.stat_arb.pipeline.stat_arb_pipeline import (
    StatArbPairPipeline,
)
from quant_platform.trading.stat_arb.schemas import (
    StatArbPairConfig,
    StatArbPipelineResult,
    StatArbSignal,
)
from quant_platform.trading.stat_arb.spreads.regime_filter import RegimeFilterConfig

from quant_platform.portfolio.position_sizing import stat_arb_position_size
from quant_platform.execution.models import MarketDataSnapshot, Order
from quant_platform.execution.enums import Side, OrderType, TimeInForce, Venue
from quant_platform.runner.core import Strategy, RunContext

LOGGER = logging.getLogger(__name__)


# =====================================================================
# Strategy
# =====================================================================


class StatArbExecutionStrategy(Strategy):
    """
    Execution-aware stat-arb strategy that:

      1. Runs the stat-arb pipeline once in `on_start`
      2. Receives bars in `on_bar`
      3. Looks up the signal for each timestamp
      4. Converts signal → target positions using portfolio sizing
      5. Submits market orders via the ExecutionContext
    """

    def __init__(
        self,
        y_symbol: str,
        x_symbol: str,
        vol_target: float = 0.10,  # annualized vol target
        dollar_neutral: bool = True,
    ) -> None:
        self.y_symbol = y_symbol
        self.x_symbol = x_symbol
        self.vol_target = vol_target
        self.dollar_neutral = dollar_neutral

        self.pipeline_result: Optional[StatArbPipelineResult] = None
        self.signal_by_ts: dict[pd.Timestamp, StatArbSignal] = {}
        self._order_seq: int = 0  # simple counter for order_ids

    # ----------------------------------------------------------
    # Start
    # ----------------------------------------------------------
    def on_start(self, context: RunContext) -> None:
        df = context.market_data  # MultiIndex[timestamp, symbol]
        if df is None:
            raise ValueError("RunContext.market_data is None in on_start.")

        # Extract each series
        y = df.xs(self.y_symbol, level="symbol")["close"]
        x = df.xs(self.x_symbol, level="symbol")["close"]

        # Minimal regime DataFrame compatible with RegimeFilter
        regime_df = pd.DataFrame(
            {
                "date": y.index,
                "regime_hmm": 0,  # single dummy regime
            }
        )

        regime_cfg = RegimeFilterConfig(
            regime_name="hmm",
            allowed_regimes=[0],
            min_regime_prob=0.0,
        )

        pair_cfg = StatArbPairConfig(
            symbol_y=self.y_symbol,
            symbol_x=self.x_symbol,
            dt=1 / 252,  # daily data
            z_entry=1.0,
            z_exit=0.2,
            use_kalman=False,
            fail_if_not_coint=False,
        )

        pipe = StatArbPairPipeline(
            pair_config=pair_cfg,
            regime_config=regime_cfg,
        )

        # Run the pipeline: returns StatArbPipelineResult with StatArbSignal list
        self.pipeline_result = pipe.run(y, x, regime_df)

        # Log (instead of printing) basic diagnostics at DEBUG level
        if self.pipeline_result and self.pipeline_result.signals:
            zs = [s.zscore for s in self.pipeline_result.signals]
            LOGGER.debug(
                "StatArb pipeline produced %d signals. "
                "Z-score stats: min=%.3f max=%.3f mean=%.3f",
                len(self.pipeline_result.signals),
                min(zs),
                max(zs),
                sum(zs) / len(zs),
            )
            LOGGER.debug(
                "First 5 signals: %s",
                self.pipeline_result.signals[:5],
            )

        # IMPORTANT: key by pd.Timestamp, not raw datetime
        self.signal_by_ts = {
            pd.Timestamp(s.timestamp): s for s in self.pipeline_result.signals
        }

    # ----------------------------------------------------------
    # Helper: convert dict → Order model
    # ----------------------------------------------------------
    def _dict_to_order(self, ts: pd.Timestamp, od: Dict[str, Any]) -> Order:
        """
        Adapter so this example can work even if StatArbSignal.to_orders()
        returns plain dicts.

        Expected keys in od:
            - symbol: str
            - quantity: float (signed, + for buy / - for sell)
            - optional: tag / reason, etc.
        """
        symbol = str(od["symbol"])
        qty_signed = float(od.get("quantity", 0.0))

        if qty_signed == 0.0:
            # Should not be called, but guard anyway
            qty_signed = 0.0

        side = Side.BUY if qty_signed > 0 else Side.SELL
        quantity = abs(qty_signed)

        self._order_seq += 1
        order_id = od.get(
            "order_id",
            f"stat_arb_{symbol}_{ts.strftime('%Y%m%d')}_{self._order_seq}",
        )

        return Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=OrderType.MARKET,
            limit_price=None,
            time_in_force=TimeInForce.DAY,
            venue=Venue.SIMULATED,
            timestamp=ts.to_pydatetime(),
            tags={"strategy": "stat_arb_exec", **od.get("tags", {})},
        )

    # ----------------------------------------------------------
    # Per-bar execution
    # ----------------------------------------------------------
    def on_bar(self, context: RunContext, bar_data: pd.DataFrame) -> None:
        # After _slice_bar, bar_data is indexed by symbol
        ts = context.timestamp

        signal = self.signal_by_ts.get(ts)
        if signal is None:
            return  # no signal for this timestamp

        # Convert signal → target position (shares)
        price_y = float(bar_data.loc[self.y_symbol, "close"])
        price_x = float(bar_data.loc[self.x_symbol, "close"])

        size_y, size_x = stat_arb_position_size(
            zscore=signal.zscore,
            vol_target=self.vol_target,
            hedge_ratio=signal.hedge_ratio,
            dollar_neutral=self.dollar_neutral,
            price_y=price_y,
            price_x=price_x,
        )

        # If both legs zero → nothing to do
        if size_y == 0 and size_x == 0:
            return

        ctx = context.execution_context

        # Let the signal class translate target sizes -> *dict* orders
        orders_raw: List[Dict[str, Any]] = signal.to_orders(
            ts=ts.to_pydatetime(),
            y_symbol=self.y_symbol,
            x_symbol=self.x_symbol,
            target_y=size_y,
            target_x=size_x,
        )

        # Adapt dict orders → Order models and execute
        for od in orders_raw:
            symbol = str(od["symbol"])

            # Use current bar close as mid
            mid = float(bar_data.loc[symbol, "close"])
            snap = MarketDataSnapshot.from_bar(
                symbol=symbol,
                ts=ts,
                close=mid,
                spread_bps=2.0,
            )

            order_model = self._dict_to_order(ts, od)
            ctx.execute(order_model, snap)

    # ----------------------------------------------------------
    # End
    # ----------------------------------------------------------
    def on_end(self, context: RunContext) -> None:
        # Could log summary, risk, etc. For now keep it minimal.
        pass


# =====================================================================
# Helper to create a synthetic AAA/BBB CSV dataset
# =====================================================================


def generate_synthetic_pair_csv(path: str) -> None:
    """
    Creates a simple stationary synthetic Y/X pair so Engle–Granger detects cointegration.
    """
    n = 250
    dates = pd.date_range("2023-01-01", periods=n, freq="D")

    x = np.cumsum(np.random.normal(scale=1, size=n)) + 100
    noise = np.random.normal(scale=0.5, size=n)
    y = 2.0 * x + noise  # strongly cointegrated with beta≈2

    df = pd.DataFrame(
        {
            "timestamp": list(dates) * 2,
            "symbol": ["AAA"] * n + ["BBB"] * n,
            "close": list(y) + list(x),
        }
    )
    df.to_csv(path, index=False)


# =====================================================================
# Standalone runner
# =====================================================================


def run_example() -> None:
    """
    For manual interactive use:

        python -m quant_platform.examples.stat_arb_backtest_with_execution
    """
    from quant_platform.runner.run import run_from_config
    from quant_platform.runner.strategy_factory import register_strategy
    import json

    # Generate data
    csv_path = "synthetic_pair.csv"
    generate_synthetic_pair_csv(csv_path)

    # Build example config
    cfg = {
        "name": "stat_arb_demo",
        "strategy": {
            "params": {
                "name": "stat_arb_exec",
                "y_symbol": "AAA",
                "x_symbol": "BBB",
                "vol_target": 0.1,
                "dollar_neutral": True,
            }
        },
        "data_source": csv_path,
        "seeds": {"global_seed": 123},
        "execution": {
            "latency_seconds": 0.0,
            "slippage_model": "default",
            "cost_model": "default",
        },
    }

    cfg_path = "stat_arb_example.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    # Register strategy
    register_strategy("stat_arb_exec", StatArbExecutionStrategy)

    result = run_from_config(cfg_path, save_dir="runs/stat_arb_demo")
    print("Final equity:", result.equity_curve.iloc[-1])
