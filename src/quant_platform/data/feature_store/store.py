# src/quant_platform/data/feature_store/store.py
from typing import Dict

import polars as pl


class FeatureStore:
    """
    Simple in-memory feature store (versioned).
    """

    def __init__(self):
        self.store: Dict[str, pl.DataFrame] = {}

    def save_features(self, name: str, df: pl.DataFrame):
        self.store[name] = df

    def get_features(self, name: str) -> pl.DataFrame:
        return self.store.get(name)

    def list_features(self):
        return list(self.store.keys())
