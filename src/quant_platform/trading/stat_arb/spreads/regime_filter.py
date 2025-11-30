# src/quant_platform/trading/stat_arb/spreads/regime_filter.py
from __future__ import annotations


import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from quant_platform.trading.stat_arb.spreads.schemas import ZScoreResult


class RegimeFilterConfig(BaseModel):
    """
    Configuration for regime-aware trading filter.

    allowed_regimes:
        Set of integer regime labels where trading is allowed.
        Example: [0] for "normal/mean-reverting" regime.

    min_regime_prob:
        If > 0, require that the probability of the active regime be at least
        this value using columns prob_<regime_name>_<state>. If such a column
        does not exist, the probability check is skipped.
    """

    regime_name: str = Field(
        "hmm",
        description="Base regime name used in column names: regime_<name>, prob_<name>_<state>.",
    )
    allowed_regimes: list[int] = Field(
        default_factory=lambda: [0],
        description="List of allowed integer regimes for trading.",
    )
    min_regime_prob: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Minimum probability threshold for active regime.",
    )


def build_regime_mask(
    zscore_result: ZScoreResult,
    regime_df: pd.DataFrame,
    config: RegimeFilterConfig,
) -> np.ndarray:
    """
    Build a boolean mask indicating whether each timestamp in zscore_result
    is tradable under the given regime configuration.

    Parameters
    ----------
    zscore_result : ZScoreResult
        Z-scores and timestamps for a given pair.
    regime_df : pd.DataFrame
        Regime information with columns:
            - 'date'
            - 'regime_<regime_name>'
            - optional 'prob_<regime_name>_<state>' columns
        Typically obtained via RegimeFeatureStore.load_regime(..., as_pandas=True).
    config : RegimeFilterConfig
        Configuration of allowed regimes and probability threshold.

    Returns
    -------
    mask : np.ndarray
        Boolean array of shape (n,) aligned with zscore_result.timestamps.
        True => trading allowed, False => trading blocked.
    """
    if "date" not in regime_df.columns:
        raise ValueError("regime_df must contain a 'date' column.")

    regime_col = f"regime_{config.regime_name}"
    if regime_col not in regime_df.columns:
        raise ValueError(f"regime_df is missing column '{regime_col}'.")

    df = regime_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    df = df.set_index("date").sort_index()

    # Align regime info with z-score timestamps
    ts = pd.to_datetime(zscore_result.timestamps)
    # Reindex regime on zscore timestamps
    active_regime = df[regime_col].reindex(ts)

    mask = np.full(active_regime.shape[0], fill_value=False, dtype=bool)

    # Determine tradable points
    for i, (timestamp, r) in enumerate(zip(ts, active_regime.to_numpy())):
        if pd.isna(r):
            mask[i] = False
            continue

        r_int = int(r)
        if r_int not in config.allowed_regimes:
            mask[i] = False
            continue

        # If no probability filter, we are done.
        if config.min_regime_prob <= 0.0:
            mask[i] = True
            continue

        # Try to obtain prob_<regime_name>_<state> for this regime.
        prob_col = f"prob_{config.regime_name}_{r_int}"
        if prob_col not in df.columns:
            # Probabilities not stored; fall back to label-only rule.
            mask[i] = True
            continue

        prob_series = df[prob_col].reindex(ts)
        p = prob_series.iloc[i]
        if pd.isna(p) or p < config.min_regime_prob:
            mask[i] = False
        else:
            mask[i] = True

    return mask
