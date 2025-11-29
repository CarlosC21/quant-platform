# src/quant_platform/data/feature_store/store.py
from typing import Dict
import polars as pl


class FeatureStore:
    """
    Simple in-memory feature store (versioned) with optional
    rolling/lag feature utilities.
    """

    def __init__(self):
        self.store: Dict[str, pl.DataFrame] = {}

    def save_features(self, name: str, df: pl.DataFrame):
        self.store[name] = df

    def get_features(self, name: str) -> pl.DataFrame:
        df = self.store.get(name)
        if df is None:
            raise ValueError(f"No feature set named '{name}' found in FeatureStore.")
        return df

    def list_features(self):
        return list(self.store.keys())

    # --- Feature utilities ---
    def add_lag_feature(self, name: str, base_feature: str, lag: int):
        df = self.get_features(name)
        if base_feature not in df.columns:
            raise ValueError(
                f"Base feature '{base_feature}' not found in feature set '{name}'"
            )
        df = df.with_columns(
            [df[base_feature].shift(lag).alias(f"{base_feature}_lag{lag}")]
        )
        self.save_features(name, df)

    def add_rolling_feature(
        self, name: str, base_feature: str, window: int, func: str = "mean"
    ):
        df = self.get_features(name)
        if base_feature not in df.columns:
            raise ValueError(
                f"Base feature '{base_feature}' not found in feature set '{name}'"
            )
        if func == "mean":
            rolled = (
                df[base_feature]
                .rolling_mean(window)
                .alias(f"{base_feature}_roll{window}_mean")
            )
        elif func == "std":
            rolled = (
                df[base_feature]
                .rolling_std(window)
                .alias(f"{base_feature}_roll{window}_std")
            )
        else:
            raise ValueError("func must be 'mean' or 'std'")
        df = df.with_columns([rolled])
        self.save_features(name, df)
