# src/quant_platform/trading/stat_arb/schemas.py
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional, Dict, Any, List

from pydantic import BaseModel, Field, ConfigDict


class StatArbSignal(BaseModel):
    """
    Point-in-time stat-arb signal for a pair (Y, X).

    side:
        'long'  -> long spread (long Y, short X)
        'short' -> short spread (short Y, long X)
        'flat'  -> no spread position

    tradable:
        False if regime filters block trading, even if z-score suggests entry.
    """

    timestamp: datetime
    symbol_y: str
    symbol_x: str

    zscore: float = Field(..., description="Point-in-time z-score of the spread.")
    spread: float = Field(..., description="Point-in-time spread value.")

    side: Literal["long", "short", "flat"]
    z_entry: float = Field(..., gt=0.0, description="Entry threshold |z| >= z_entry.")
    z_exit: float = Field(
        ..., gt=0.0, description="Exit/flatten threshold |z| <= z_exit."
    )

    regime: Optional[int] = Field(
        default=None,
        description="Optional regime label from HMM / RegimeStore.",
    )
    tradable: bool = Field(
        True,
        description="If False, trading is blocked (e.g., bad regime).",
    )

    reason: str = Field(
        "",
        description="Short reason code: 'entry_long', 'exit_flat', 'regime_block', 'not_cointegrated', etc.",
    )
    meta: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary extra diagnostics.",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)


class StatArbPairConfig(BaseModel):
    """
    Configuration for a single stat-arb pair pipeline.
    """

    symbol_y: str = Field(..., description="Dependent asset (Y).")
    symbol_x: str = Field(..., description="Independent asset (X).")

    dt: float = Field(
        ...,
        gt=0.0,
        description="Sampling interval in years (e.g. 1/252 for daily data).",
    )

    z_entry: float = Field(
        2.0,
        gt=0.0,
        description="Entry threshold: open positions when |z| >= z_entry.",
    )
    z_exit: float = Field(
        0.5,
        gt=0.0,
        description="Exit/flatten threshold: |z| <= z_exit.",
    )

    use_kalman: bool = Field(
        False,
        description="If True, use Kalman dynamic hedge; otherwise static beta from Engle–Granger.",
    )
    fail_if_not_coint: bool = Field(
        False,
        description="If True, raise if pair is not cointegrated; otherwise return flat, non-tradable signals.",
    )


class StatArbPipelineResult(BaseModel):
    """
    Result of running the stat-arb pair pipeline.
    """

    pair_config: StatArbPairConfig
    signals: List[StatArbSignal]
    cointegrated: bool = Field(
        ...,
        description="True if Engle–Granger found cointegration.",
    )
    meta: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional diagnostics (e.g. EG stats, OU params).",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)
