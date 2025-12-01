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
        default=None, description="If set, overrides python + numpy seeds."
    )


# ============================================================
# Execution Settings (supports Week 11 engine)
# ============================================================


class ExecutionSettings(BaseModel):
    """
    Execution controls for backtest.
    These map cleanly to ExecutionContext + cost/slippage models.
    """

    model_config = ConfigDict(extra="forbid")

    latency_seconds: float = 0.0

    # Added for compatibility with Week 11/12 workflow
    slippage_bps: Optional[float] = None
    cost_bps: Optional[float] = None

    # Optional advanced models
    slippage_model: Optional[str] = None
    cost_model: Optional[str] = None


# ============================================================
# Strategy settings
# ============================================================


class StrategySettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    params: Dict[str, Any] = Field(default_factory=dict)


# ============================================================
# Save Settings (NEW)
# ============================================================


class SaveSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    directory: Optional[str] = None
    save_equity_curve: bool = True
    save_positions: bool = True
    save_trades: bool = True


# ============================================================
# Top-level BacktestConfig
# ============================================================


class BacktestConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "default_run"
    strategy: StrategySettings = Field(default_factory=StrategySettings)
    execution: ExecutionSettings = Field(default_factory=ExecutionSettings)
    seeds: RandomSeedConfig = Field(default_factory=RandomSeedConfig)

    data_source: Optional[str] = None

    # NEW
    save: SaveSettings = Field(default_factory=SaveSettings)
