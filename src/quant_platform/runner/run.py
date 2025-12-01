from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from quant_platform.runner.config.loader import load_config, seed_everything
from quant_platform.runner.config.models import BacktestConfig
from quant_platform.runner.core import run_backtest, BacktestResult
from quant_platform.runner.strategy_factory import create_strategy
from quant_platform.execution.context import ExecutionContext


def load_market_data(path: str) -> pd.DataFrame:
    """
    Minimal loader for CSV/Parquet with MultiIndex[timestamp, symbol].
    CSV is preferred for environments without pyarrow/fastparquet.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Market data file not found: {path}")

    if p.suffix.lower() == ".parquet":
        try:
            df = pd.read_parquet(p)
        except Exception as exc:
            raise RuntimeError(
                "Parquet file requires pyarrow or fastparquet. "
                "Please install them or use CSV instead."
            ) from exc
    elif p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
    else:
        raise ValueError("Market data must be CSV or Parquet")

    # Ensure timestamp is datetime if present
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Convert to MultiIndex if needed
    if not isinstance(df.index, pd.MultiIndex):
        if "timestamp" not in df.columns or "symbol" not in df.columns:
            raise ValueError("Data must contain 'timestamp' and 'symbol' columns")
        df = df.set_index(["timestamp", "symbol"]).sort_index()

    return df


def run_from_config(
    config_path: str | Path, save_dir: Optional[str | Path] = None
) -> BacktestResult:
    """
    High-level orchestration entrypoint.
    Loads config → seeds RNG → loads strategy + data → constructs ExecutionContext → runs backtest.
    """
    cfg: BacktestConfig = load_config(config_path)
    seed_everything(cfg)

    if cfg.strategy is None:
        raise ValueError("BacktestConfig.strategy missing!")

    # Strategy name is inside strategy.params.name
    params = cfg.strategy.params or {}
    strategy_name = params.get("name") or params.get("strategy")
    if strategy_name is None:
        raise ValueError("Strategy params must include 'name'")

    strategy = create_strategy(strategy_name, params)

    if cfg.data_source is None:
        raise ValueError("BacktestConfig must specify data_source")

    df = load_market_data(cfg.data_source)

    context = ExecutionContext()
    result = run_backtest(
        strategy=strategy,
        execution_context=context,
        market_data=df,
        config=cfg,
    )

    # Save results
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        result.equity_curve.to_csv(save_dir / "equity.csv")
        result.drawdowns.to_csv(save_dir / "drawdown.csv")
        result.prices_last.to_csv(save_dir / "prices_last.csv")

        (save_dir / "risk_metrics.json").write_text(
            json.dumps(result.risk_metrics, indent=2)
        )

        (save_dir / "config.json").write_text(json.dumps(cfg.model_dump(), indent=2))

    return result
