# src/quant_platform/trading/stat_arb/signals.py
from __future__ import annotations

from typing import List
import pandas as pd

from quant_platform.trading.stat_arb.schemas import StatArbSignal
from quant_platform.trading.stat_arb.spreads.schemas import ZScoreResult
from quant_platform.trading.stat_arb.spreads.regime_filter import (
    RegimeFilterConfig,
    build_regime_mask,
)


def _target_side_from_zscore(
    z: float,
    z_entry: float,
    z_exit: float,
) -> str:
    """
    Stateless mapping from z-score to target side.
    """
    if abs(z) >= z_entry:
        return "short" if z > 0 else "long"
    if abs(z) <= z_exit:
        return "flat"
    return "flat"


def _resolve_beta(z: ZScoreResult, i: int) -> float:
    """
    Resolve hedge ratio for timestamp i.
    Supports:
        - z.beta      (static beta)
        - z.beta_ts   (array-like dynamic beta)
    """
    # dynamic beta_ts takes precedence
    if hasattr(z, "beta_ts") and z.beta_ts is not None:
        return float(z.beta_ts[i])

    # fallback: static beta
    if hasattr(z, "beta") and z.beta is not None:
        return float(z.beta)

    # final fallback
    return 1.0


def build_signals_from_zscores(
    zscore_result: ZScoreResult,
    z_entry: float,
    z_exit: float,
    regime_df,
    regime_config: RegimeFilterConfig,
) -> List[StatArbSignal]:
    """
    Build StatArbSignal list from z-scores and regime information.
    """
    if z_entry <= 0.0 or z_exit <= 0.0:
        raise ValueError("z_entry and z_exit must be positive.")
    if z_exit >= z_entry:
        raise ValueError("z_exit should be < z_entry.")

    tradable_mask = build_regime_mask(zscore_result, regime_df, regime_config)

    timestamps = zscore_result.timestamps
    spread = zscore_result.spread
    zscores = zscore_result.zscore

    signals: List[StatArbSignal] = []

    for i in range(len(timestamps)):
        ts = timestamps[i]
        ts_py = pd.Timestamp(ts).to_pydatetime()

        z = float(zscores[i])
        s = float(spread[i])
        tradable = bool(tradable_mask[i])

        # hedge ratio for this timestamp
        beta_i = _resolve_beta(zscore_result, i)

        # Determine side + reason
        if not tradable:
            side = "flat"
            reason = "regime_block"
        else:
            side = _target_side_from_zscore(z, z_entry=z_entry, z_exit=z_exit)
            if side == "long":
                reason = "entry_long" if abs(z) >= z_entry else "flat_zone"
            elif side == "short":
                reason = "entry_short" if abs(z) >= z_entry else "flat_zone"
            else:
                reason = "exit_flat" if abs(z) <= z_exit else "flat_zone"

        signal = StatArbSignal(
            timestamp=ts_py,
            symbol_y=zscore_result.symbol_y,
            symbol_x=zscore_result.symbol_x,
            zscore=z,
            spread=s,
            hedge_ratio=beta_i,  # << 🔥 KEY FIX
            side=side,
            z_entry=z_entry,
            z_exit=z_exit,
            regime=None,
            tradable=tradable,
            reason=reason,
            meta={},
        )
        signals.append(signal)

    return signals
