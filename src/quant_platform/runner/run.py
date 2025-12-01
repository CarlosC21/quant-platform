from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from quant_platform.execution.context import ExecutionContext
from quant_platform.runner.config.loader import load_config, seed_everything
from quant_platform.runner.config.models import BacktestConfig
from quant_platform.runner.core import BacktestResult, run_backtest
from quant_platform.runner.strategy_factory import create_strategy


# ======================================================================
# Market data loader
# ======================================================================


def load_market_data(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Market data file not found: {path}")

    if p.suffix.lower() == ".parquet":
        try:
            df = pd.read_parquet(p)
        except Exception as exc:  # pragma: no cover - env specific
            raise RuntimeError(
                "Parquet requires pyarrow/fastparquet; install them or use CSV."
            ) from exc
    elif p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
    else:
        raise ValueError("Market data must be CSV or Parquet")

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    if not isinstance(df.index, pd.MultiIndex):
        if "timestamp" not in df.columns or "symbol" not in df.columns:
            raise ValueError("Data must contain 'timestamp' and 'symbol'")
        df = df.set_index(["timestamp", "symbol"]).sort_index()

    return df


# ======================================================================
# Saving utilities
# ======================================================================


def _cfg_save(cfg: BacktestConfig):
    """
    Safe accessor for cfg.save — handles older BacktestConfig
    without .save if needed.
    """
    return getattr(cfg, "save", None)


def _save_outputs(
    result: BacktestResult,
    cfg: BacktestConfig,
    exec_ctx: ExecutionContext,
    outdir: Path,
) -> None:
    """
    Save Week-12 artifacts (equity, drawdowns, positions, trades, config).
    """
    outdir.mkdir(parents=True, exist_ok=True)

    save_cfg = _cfg_save(cfg)

    # Equity & drawdowns
    if save_cfg and getattr(save_cfg, "save_equity_curve", False):
        result.equity_curve.to_csv(outdir / "equity.csv")
        result.equity_curve.to_csv(outdir / "equity_curve.csv")
        result.drawdowns.to_csv(outdir / "drawdown.csv")
        result.drawdowns.to_csv(outdir / "drawdowns.csv")

    # Last prices (always safe)
    result.prices_last.to_csv(outdir / "prices_last.csv")

    # Positions
    if save_cfg and getattr(save_cfg, "save_positions", False):
        pt = getattr(result, "positions_ts", None)
        if pt is not None and not pt.empty:
            pt.to_csv(outdir / "positions.csv", index=False)

    # Trades
    if save_cfg and getattr(save_cfg, "save_trades", False):
        trades = getattr(exec_ctx.ledger, "trades", None)
        if trades:
            tdf = pd.DataFrame([t.model_dump() for t in trades])
            tdf.to_csv(outdir / "trades.csv", index=False)

    # Risk metrics + config snapshot
    (outdir / "risk_metrics.json").write_text(json.dumps(result.risk_metrics, indent=2))
    (outdir / "config.json").write_text(json.dumps(cfg.model_dump(), indent=2))


# ======================================================================
# High-level runner
# ======================================================================


def run_from_config(
    config_path: str | Path,
    save_dir: Optional[str | Path] = None,
) -> BacktestResult:
    """
    High-level orchestration entrypoint:

        load config → seed RNG → load strategy + data →
        construct ExecutionContext → run_backtest → save outputs.
    """
    cfg: BacktestConfig = load_config(config_path)
    seed_everything(cfg)

    # Strategy resolution
    params = cfg.strategy.params or {}
    name = params.get("name") or params.get("strategy")
    if name is None:
        raise ValueError("Strategy params must include 'name'")

    strategy = create_strategy(name, params)

    if cfg.data_source is None:
        raise ValueError("BacktestConfig must specify data_source")

    market_data = load_market_data(cfg.data_source)

    # ExecutionContext (Week 11 engine + ledger)
    exec_ctx = ExecutionContext()

    # --- NEW: seed initial cash into the ledger ---
    # Our TradeLedger -> PositionBook -> cash structure
    try:
        exec_ctx.ledger.position_book.cash = cfg.initial_cash
    except AttributeError:
        # Fallback if someone swaps in a different ledger implementation
        pass

    # Run the backtest
    result = run_backtest(
        strategy=strategy,
        market_data=market_data,
        execution_context=exec_ctx,
        config=cfg,
    )

    # Decide output directory:
    # CLI/Streamlit save_dir overrides config.save.directory
    save_cfg = _cfg_save(cfg)

    if save_dir is not None:
        final_dir = Path(save_dir)
    elif save_cfg and getattr(save_cfg, "directory", None):
        final_dir = Path(save_cfg.directory)
    else:
        final_dir = None

    if final_dir is not None:
        _save_outputs(result, cfg, exec_ctx, final_dir)

    return result
