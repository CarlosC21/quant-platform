# tests/test_features.py
import polars as pl

from src.quant_platform.data.feature_store.features import (
    compute_equity_features,
    compute_macro_features,
)


def test_equity_features():
    df = pl.DataFrame({"close": [100, 105, 110, 120, 115, 125]})
    df_feat = compute_equity_features(df)
    assert "returns" in df_feat.columns
    assert "log_returns" in df_feat.columns
    assert "ma_5" in df_feat.columns
    assert "ma_20" in df_feat.columns


def test_macro_features():
    df = pl.DataFrame({"value": [10, 20, 30, 40, 50]})
    df_feat = compute_macro_features(df)
    assert "normalized_value" in df_feat.columns
    assert abs(df_feat["normalized_value"].mean()) < 1e-6  # approx 0 mean
