# src/quant_platform/data/feature_store/features.py
import polars as pl

__all__ = ["compute_equity_features", "compute_macro_features"]


def compute_equity_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute equity features: returns, log_returns, ma_5, ma_20.
    Works with Polars DataFrame that has a 'close' or 'close_' column.
    """
    if df.is_empty():
        return df

    # support both 'close' (raw CSV key) and 'close_' (schema alias populated)
    if "close" in df.columns and "close_" not in df.columns:
        close_col = "close"
    elif "close_" in df.columns:
        close_col = "close_"
    else:
        raise ValueError(
            "DataFrame must contain 'close' or 'close_' column for compute_equity_features"
        )

    # Use Polars expressions for vectorized ops
    df = df.with_columns(
        [
            (pl.col(close_col).pct_change().fill_null(0)).alias("returns"),
            (pl.col(close_col).log().diff().fill_null(0)).alias("log_returns"),
            pl.col(close_col).rolling_mean(5).alias("ma_5"),
            pl.col(close_col).rolling_mean(20).alias("ma_20"),
        ]
    )
    return df


def compute_macro_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Normalize macro 'value' column to zero mean, unit variance.
    Adds 'normalized_value' column. Handles constant/std=0 safely.
    """
    if df.is_empty():
        return df

    if "value" not in df.columns:
        raise ValueError(
            "DataFrame must contain 'value' column for compute_macro_features"
        )

    mean_val = df["value"].mean()
    std_val = df["value"].std()

    if std_val in (0, None):
        df = df.with_columns([pl.lit(0.0).alias("normalized_value")])
    else:
        df = df.with_columns(
            [((pl.col("value") - mean_val) / std_val).alias("normalized_value")]
        )

    return df
