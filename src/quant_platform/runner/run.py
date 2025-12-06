from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from quant_platform.runner.config.loader import load_config, seed_everything
from quant_platform.runner.config.models import BacktestConfig
from quant_platform.runner.core import run_backtest, BacktestResult
from quant_platform.runner.strategy_factory import create_strategy
from quant_platform.execution.context import ExecutionContext
from quant_platform.ui.data_validation import validate_market_data

LOGGER = logging.getLogger(__name__)


# ======================================================================
# Load flat market data
# ======================================================================


def _load_market_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Market data path does not exist: {path}")

    df = pd.read_csv(path)

    required = {"timestamp", "symbol", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV missing required columns: {missing}\nPresent: {df.columns.tolist()}"
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
    df = df.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    return df


# ======================================================================
# Execution context builder
# ======================================================================


def _build_execution_context(cfg: BacktestConfig) -> ExecutionContext:
    return ExecutionContext(initial_cash=cfg.initial_cash)


# ======================================================================
# Main entrypoint
# ======================================================================


def run_from_config(
    path: str | Path,
    save_dir: str | Path | None = None,
) -> BacktestResult:
    LOGGER.info("Loading config: %s", path)
    cfg = load_config(path)

    if save_dir is not None:
        cfg.save.directory = str(save_dir)

    LOGGER.info("Applying random seeds…")
    seed_everything(cfg)

    if cfg.data_source is None:
        raise ValueError("Config must provide `data_source`")

    LOGGER.info("Loading market data…")
    market_data = _load_market_data(cfg.data_source)

    # Validate flat structure first
    validate_market_data(market_data)

    # ======================================================================
    # 🔥 REQUIRED FIX #1 — Convert to MultiIndex for strategies like stat-arb
    # ======================================================================
    market_data = market_data.set_index(["timestamp", "symbol"]).sort_index()

    LOGGER.info("Market data converted to MultiIndex (timestamp, symbol)")

    # ======================================================================
    # Build strategy
    # ======================================================================

    params = dict(cfg.strategy.params)
    strategy_name = params.pop("name", None)
    if strategy_name is None:
        raise ValueError("strategy.params.name must specify a registered strategy")

    LOGGER.info("Creating strategy '%s'…", strategy_name)
    strategy = create_strategy(strategy_name, params=params)

    # ======================================================================
    # Build execution context
    # ======================================================================

    LOGGER.info("Building execution context…")
    exec_ctx = _build_execution_context(cfg)

    # ======================================================================
    # Run backtest
    # ======================================================================

    LOGGER.info("Running backtest…")

    result: BacktestResult = run_backtest(
        strategy=strategy,
        market_data=market_data,
        execution_context=exec_ctx,
        config=cfg,
    )

    # ======================================================================
    # Attach trade log
    # ======================================================================

    result.trade_log = exec_ctx.ledger.trades.copy()

    # ======================================================================
    # Portfolio snapshots (MultiIndex-safe)
    # ======================================================================

    snapshots: list[dict[str, Any]] = []

    # Iterate through timestamps (first level only)
    for ts in market_data.index.get_level_values("timestamp").unique():
        try:
            bars = market_data.loc[ts]
        except KeyError:
            continue

        # bars is a frame indexed by symbol
        if isinstance(bars, pd.Series):
            bars = bars.to_frame().T

        price_map = {str(sym): float(row["close"]) for sym, row in bars.iterrows()}

        snap = exec_ctx.get_portfolio_snapshot(price_map)
        snap["timestamp"] = ts
        snapshots.append(snap)

    result.portfolio_snapshots = snapshots

    # ======================================================================
    # Save results
    # ======================================================================

    if cfg.save.directory:
        _persist_results(cfg, result)

    return result


# ======================================================================
# Save outputs
# ======================================================================


def _persist_results(cfg: BacktestConfig, result: BacktestResult) -> None:
    out_dir = Path(cfg.save.directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Saving results to: %s", out_dir)

    if cfg.save.save_equity_curve:
        result.equity_curve.to_csv(out_dir / "equity_curve.csv")

    if cfg.save.save_positions and result.positions_ts is not None:
        result.positions_ts.to_csv(out_dir / "positions.csv", index=False)

    if hasattr(result, "trade_log"):
        import json

        with open(out_dir / "trades.json", "w") as f:
            json.dump(result.trade_log, f, indent=2)

    if hasattr(result, "portfolio_snapshots"):
        pd.DataFrame(result.portfolio_snapshots).to_csv(
            out_dir / "portfolio_snapshots.csv",
            index=False,
        )
