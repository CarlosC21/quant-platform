from __future__ import annotations

import logging
from typing import Dict, Any, Optional

import pandas as pd

# ALL imports at top — required for ruff E402
from quant_platform.runner.core import Strategy, RunContext
from quant_platform.execution.enums import Side, OrderType, TimeInForce, Venue
from quant_platform.execution.models import MarketDataSnapshot, Order

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

# Register strategy at import time (allowed ONLY at the top)
from quant_platform.runner.strategy_factory import register_strategy


LOGGER = logging.getLogger(__name__)


class StatArbExecutionStrategy(Strategy):
    PARAM_SCHEMA = {
        "y_symbol": {"type": "str", "required": True},
        "x_symbol": {"type": "str", "required": True},
        "vol_target": {"type": "float", "default": 0.10},
        "dollar_neutral": {"type": "bool", "default": True},
    }

    def __init__(
        self,
        y_symbol: str,
        x_symbol: str,
        vol_target: float = 0.10,
        dollar_neutral: bool = True,
    ) -> None:
        self.y_symbol = y_symbol
        self.x_symbol = x_symbol
        self.vol_target = vol_target
        self.dollar_neutral = dollar_neutral

        self.pipeline_result: Optional[StatArbPipelineResult] = None
        self.signal_by_ts: dict[pd.Timestamp, StatArbSignal] = {}
        self._order_seq = 0

    def on_start(self, context: RunContext) -> None:
        df = context.market_data
        if df is None:
            raise ValueError("market_data is None in on_start")

        try:
            y = df.xs(self.y_symbol, level="symbol")["close"]
            x = df.xs(self.x_symbol, level="symbol")["close"]
        except Exception as e:
            LOGGER.error("Price extraction failed: %s", e)
            self.pipeline_result = StatArbPipelineResult(
                pair_config=StatArbPairConfig(
                    symbol_y=self.y_symbol,
                    symbol_x=self.x_symbol,
                ),
                cointegrated=False,
                signals=[],
            )
            return

        regime_df = pd.DataFrame({"date": y.index, "regime_hmm": 0})

        pair_cfg = StatArbPairConfig(
            symbol_y=self.y_symbol,
            symbol_x=self.x_symbol,
            dt=1 / 252,
            z_entry=1.0,
            z_exit=0.2,
            use_kalman=False,
            fail_if_not_coint=False,
        )

        regime_cfg = RegimeFilterConfig(
            regime_name="hmm",
            allowed_regimes=[0],
            min_regime_prob=0.0,
        )

        pipe = StatArbPairPipeline(pair_config=pair_cfg, regime_config=regime_cfg)

        try:
            self.pipeline_result = pipe.run(y, x, regime_df)
        except Exception as e:
            print(f"Pipeline failed ({e}). Falling back to NO SIGNALS.")
            self.pipeline_result = StatArbPipelineResult(
                pair_config=pair_cfg,
                cointegrated=False,
                signals=[],
            )

        try:
            self.signal_by_ts = {
                pd.Timestamp(s.timestamp): s for s in self.pipeline_result.signals
            }
        except Exception:
            self.signal_by_ts = {}

        LOGGER.info("Pipeline ready: %d signals", len(self.signal_by_ts))

    def _dict_to_order(self, ts: pd.Timestamp, od: Dict[str, Any]) -> Order:
        symbol = str(od["symbol"])
        qty = float(od["quantity"])
        side = Side.BUY if qty > 0 else Side.SELL

        self._order_seq += 1

        return Order(
            order_id=f"ord_{self._order_seq}",
            symbol=symbol,
            side=side,
            quantity=abs(qty),
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY,
            venue=Venue.SIMULATED,
            timestamp=ts.to_pydatetime(),
        )

    def on_bar(self, context: RunContext, bar_data: pd.DataFrame) -> None:
        ts = context.timestamp
        signal = self.signal_by_ts.get(ts)

        if signal is None:
            return

        py = float(bar_data.loc[self.y_symbol, "close"])
        px = float(bar_data.loc[self.x_symbol, "close"])

        try:
            size_y, size_x = stat_arb_position_size(
                zscore=signal.zscore,
                vol_target=self.vol_target,
                hedge_ratio=signal.hedge_ratio,
                dollar_neutral=self.dollar_neutral,
                price_y=py,
                price_x=px,
            )
        except Exception as e:
            LOGGER.warning("Position sizing failed: %s", e)
            return

        if size_y == 0 and size_x == 0:
            return

        try:
            raw_orders = signal.to_orders(
                ts=ts.to_pydatetime(),
                y_symbol=self.y_symbol,
                x_symbol=self.x_symbol,
                target_y=size_y,
                target_x=size_x,
            )
        except Exception as e:
            LOGGER.warning("Signal→order conversion failed: %s", e)
            return

        ctx = context.execution_context

        for od in raw_orders:
            sym = od["symbol"]
            mid = float(bar_data.loc[sym, "close"])

            snap = MarketDataSnapshot.from_bar(
                symbol=sym,
                ts=ts,
                close=mid,
                spread_bps=2.0,
            )

            ctx.execute(self._dict_to_order(ts, od), snap)

    def on_end(self, context: RunContext) -> None:
        pass


# Register immediately when module is imported
register_strategy("stat_arb_exec", StatArbExecutionStrategy)
