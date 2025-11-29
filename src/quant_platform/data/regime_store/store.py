# src/quant_platform/data/regime_store/store.py

from __future__ import annotations
from typing import Optional, Dict
import numpy as np
import pandas as pd
import polars as pl

from quant_platform.data.feature_store.store import FeatureStore


# ------------------------------------------------------------------
# Local conversion helpers (same pattern used in ML pipelines)
# ------------------------------------------------------------------


def _polars_to_pandas(df):
    """
    Safe Polars → Pandas conversion without requiring pyarrow.
    Uses df.to_numpy() and df.columns.
    """
    if isinstance(df, pl.DataFrame):
        return pd.DataFrame(df.to_numpy(), columns=df.columns)
    return df


def _pandas_to_polars(df):
    """
    Safe Pandas → Polars conversion.
    """
    if isinstance(df, pd.DataFrame):
        return pl.from_pandas(df)
    return df


# ------------------------------------------------------------------
# Regime Feature Store
# ------------------------------------------------------------------


class RegimeFeatureStore:
    """
    Manages storage of regime labels + probabilities, optionally in sync
    with a FeatureStore.

    Features:
      - save_regime(): persist labels, probabilities (merge or in-memory)
      - load_regime(): retrieve them
      - attach_to_features(): horizontal merge
    """

    def __init__(self, feature_store: Optional[FeatureStore] = None):
        self.feature_store = feature_store
        # fallback local memory if no FeatureStore is provided
        self._memory: Dict[str, pl.DataFrame] = {}

    # ------------------------------------------------------------------
    def _build_regime_df(
        self,
        labels: np.ndarray,
        probs: np.ndarray,
        regime_name: str,
        dates: Optional[np.ndarray] = None,
    ) -> pl.DataFrame:
        n, k = probs.shape
        df = pd.DataFrame()

        if dates is not None:
            df["date"] = pd.to_datetime(dates)

        df[f"regime_{regime_name}"] = labels.astype(int)

        for i in range(k):
            df[f"prob_{regime_name}_{i}"] = probs[:, i]

        return _pandas_to_polars(df)

    # ------------------------------------------------------------------
    def save_regime(
        self,
        feature_name: str,
        regime_name: str,
        labels: np.ndarray,
        probs: np.ndarray,
        dates: Optional[np.ndarray] = None,
    ) -> None:
        """
        Save regime information.

        If FeatureStore is provided:
            → Merge regime columns into the existing feature table.
        Otherwise:
            → Store in internal memory.
        """

        regime_df = self._build_regime_df(labels, probs, regime_name, dates)

        if self.feature_store is not None:
            # Load existing feature table
            base = self.feature_store.get_features(feature_name)

            base_pd = _polars_to_pandas(base).reset_index(drop=True)
            regime_pd = _polars_to_pandas(regime_df).reset_index(drop=True)

            # Drop date column from regime (FeatureStore already has date)
            merged = pd.concat(
                [base_pd, regime_pd.drop(columns=["date"], errors="ignore")],
                axis=1,
            )

            self.feature_store.save_features(feature_name, _pandas_to_polars(merged))
            return

        # In-memory fallback
        key = f"{feature_name}:{regime_name}"
        self._memory[key] = regime_df

    # ------------------------------------------------------------------
    def load_regime(
        self,
        feature_name: str,
        regime_name: str,
        as_pandas: bool = False,
    ):
        """
        Retrieve stored regime information.
        """
        if self.feature_store is not None:
            df = self.feature_store.get_features(feature_name)
            df_pd = _polars_to_pandas(df)

            regime_cols = [
                c
                for c in df_pd.columns
                if c.startswith(f"regime_{regime_name}")
                or c.startswith(f"prob_{regime_name}")
            ]

            cols = ["date"] + regime_cols if "date" in df_pd.columns else regime_cols
            out = df_pd[cols]

            return out if as_pandas else _pandas_to_polars(out)

        # No FeatureStore → load from memory
        key = f"{feature_name}:{regime_name}"
        df = self._memory.get(key)
        if df is None:
            raise KeyError(f"No regime stored for {feature_name}/{regime_name}")

        return df if not as_pandas else _polars_to_pandas(df)

    # ------------------------------------------------------------------
    @staticmethod
    def attach_to_features(
        features: pl.DataFrame, regime_df: pl.DataFrame
    ) -> pl.DataFrame:
        """
        Horizontally merge regimes into an existing feature table.
        Assumes row alignment.
        """
        f = _polars_to_pandas(features).reset_index(drop=True)
        r = _polars_to_pandas(regime_df).reset_index(drop=True)

        merged = pd.concat(
            [f, r.drop(columns=["date"], errors="ignore")],
            axis=1,
        )
        return _pandas_to_polars(merged)
