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
    """

    timestamp: datetime
    symbol_y: str
    symbol_x: str

    zscore: float = Field(..., description="Point-in-time z-score of the spread.")
    spread: float = Field(..., description="Point-in-time spread value.")

    # NEW: we attach hedge_ratio directly into the signal object
    hedge_ratio: float = Field(1.0, description="Hedge ratio (beta): Y = beta * X.")

    side: Literal["long", "short", "flat"]
    z_entry: float = Field(..., gt=0.0, description="Entry threshold |z| >= z_entry.")
    z_exit: float = Field(
        ..., gt=0.0, description="Exit/flatten threshold |z| <= z_exit."
    )

    regime: Optional[int] = Field(default=None, description="Optional regime label.")

    tradable: bool = Field(
        True, description="If False, trading is blocked (e.g., regime)."
    )

    reason: str = Field(
        "", description="Reason code: entry_long, exit_flat, regime_block, etc."
    )

    meta: Dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary diagnostics."
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # =====================================================================
    # NEW: Convert signal + target sizes → simple executable order dicts
    # =====================================================================
    def to_orders(
        self,
        ts,
        y_symbol: str,
        x_symbol: str,
        target_y: float,
        target_x: float,
    ) -> List[Dict[str, Any]]:
        """
        Convert target position sizes into execution-ready order dicts.
        These order dicts match ExecutionContext.execute(order, snapshot).

        The ExecutionContext in your repo accepts:
            - dict with { "timestamp": ts, "symbol": ..., "quantity": ..., "type": "market" }
        """

        orders = []

        # --- Order for Y ---
        orders.append(
            {
                "timestamp": ts,
                "symbol": y_symbol,
                "quantity": float(target_y),
                "type": "market",
            }
        )

        # --- Order for X ---
        orders.append(
            {
                "timestamp": ts,
                "symbol": x_symbol,
                "quantity": float(target_x),
                "type": "market",
            }
        )

        return orders


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
        description="Exit threshold: |z| <= z_exit.",
    )

    use_kalman: bool = Field(False, description="Use Kalman dynamic hedge ratio.")
    fail_if_not_coint: bool = Field(
        False, description="Raise if Engle–Granger does not confirm cointegration."
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
        description="Pipeline diagnostics (EG stats, OU params, etc.).",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)
