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

    Rules:
      - |z| >= z_entry:
            z > 0  -> 'short' (short spread)
            z < 0  -> 'long'
      - |z| <= z_exit: 'flat'
      - else: 'flat' (no hysteresis here; pipeline can add stateful logic).
    """
    if abs(z) >= z_entry:
        return "short" if z > 0 else "long"
    if abs(z) <= z_exit:
        return "flat"
    return "flat"


def build_signals_from_zscores(
    zscore_result: ZScoreResult,
    z_entry: float,
    z_exit: float,
    regime_df,
    regime_config: RegimeFilterConfig,
) -> List[StatArbSignal]:
    """
    Build StatArbSignal list from z-scores and regime information.

    Parameters
    ----------
    zscore_result : ZScoreResult
        Z-scores and spreads for a given pair.
    z_entry : float
        Entry threshold |z| >= z_entry.
    z_exit : float
        Exit/flatten threshold |z| <= z_exit.
    regime_df :
        Regime DataFrame with 'date', 'regime_<name>', 'prob_<name>_<state>'.
        Typically obtained via RegimeFeatureStore.load_regime(..., as_pandas=True).
    regime_config : RegimeFilterConfig
        Configuration of allowed regimes and probability threshold.

    Returns
    -------
    signals : List[StatArbSignal]
        One signal per timestamp.
    """
    if z_entry <= 0.0 or z_exit <= 0.0:
        raise ValueError("z_entry and z_exit must be positive.")
    if z_exit >= z_entry:
        # Usually z_exit < z_entry to provide hysteresis cushion.
        # We do not enforce strict inequality, but warn via error for now.
        raise ValueError("z_exit should be < z_entry for sensible trading logic.")

    tradable_mask = build_regime_mask(zscore_result, regime_df, regime_config)

    timestamps = zscore_result.timestamps
    spread = zscore_result.spread
    zscores = zscore_result.zscore

    signals: List[StatArbSignal] = []

    for i in range(len(timestamps)):
        ts = timestamps[i]
        # convert numpy.datetime64 → python datetime
        ts_py = pd.Timestamp(ts).to_pydatetime()

        z = float(zscores[i])
        s = float(spread[i])
        tradable = bool(tradable_mask[i])

        # Determine regime label if present
        regime_label = None
        # Regime label alignment logic can live in pipeline; for now we
        # leave it as None to avoid double work. It can be injected later.

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
            side=side,  # type: ignore[arg-type]
            z_entry=z_entry,
            z_exit=z_exit,
            regime=regime_label,
            tradable=tradable,
            reason=reason,
            meta={},
        )
        signals.append(signal)

    return signals
