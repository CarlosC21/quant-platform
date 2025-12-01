from __future__ import annotations

from typing import Optional, Dict, Any

from pydantic import BaseModel, Field, ConfigDict


# ============================================================
# Random seeds
# ============================================================


class RandomSeedConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    python: Optional[int] = None
    numpy: Optional[int] = None
    global_seed: Optional[int] = Field(
        default=None,
        description="If set, overrides python + numpy seeds.",
    )


# ============================================================
# Execution Settings (Week 11 engine compatible)
# ============================================================


class ExecutionSettings(BaseModel):
    """
    Execution controls for the backtest.
    """

    model_config = ConfigDict(extra="allow")

    latency_seconds: float = 0.0
    slippage_bps: Optional[float] = None
    cost_bps: Optional[float] = None

    # for compatibility with older files
    slippage_model: Optional[str] = None
    cost_model: Optional[str] = None


# ============================================================
# Strategy settings
# ============================================================


class StrategySettings(BaseModel):
    model_config = ConfigDict(extra="allow")
    params: Dict[str, Any] = Field(default_factory=dict)


# ============================================================
# Save Settings
# ============================================================


class SaveSettings(BaseModel):
    model_config = ConfigDict(extra="allow")

    directory: Optional[str] = None
    save_equity_curve: bool = True
    save_positions: bool = True
    save_trades: bool = True


# ============================================================
# Top-level BacktestConfig
# ============================================================


class BacktestConfig(BaseModel):
    """
    Global backtest configuration.
    """

    # 🔥 ALLOW extra fields so nothing breaks
    model_config = ConfigDict(extra="allow")

    name: str = "default_run"

    strategy: StrategySettings = Field(default_factory=StrategySettings)
    execution: ExecutionSettings = Field(default_factory=ExecutionSettings)
    seeds: RandomSeedConfig = Field(default_factory=RandomSeedConfig)

    data_source: Optional[str] = None

    # 🔥 Fully supported top-level starting cash
    initial_cash: float = Field(
        default=100_000.0, description="Starting portfolio cash."
    )

    save: SaveSettings = Field(default_factory=SaveSettings)
