# src/quant_platform/options/data_ingestion.py
import polars as pl


def ingest_option_chain(df_raw: pl.DataFrame) -> pl.DataFrame:
    """
    Normalize raw option chain to canonical schema for IVSurface:
        ["symbol", "underlying_price", "trade_date", "expiry", "strike",
         "option_type", "bid", "ask", "implied_vol"]
    """
    required_cols = [
        "symbol",
        "underlying_price",
        "trade_date",
        "expiry",
        "strike",
        "option_type",
        "bid",
        "ask",
        "implied_vol",
    ]

    # Keep only required columns if present
    df = df_raw.select([c for c in required_cols if c in df_raw.columns])

    # Cast types
    df = df.with_columns(
        [
            pl.col("strike").cast(pl.Float64),
            pl.col("underlying_price").cast(pl.Float64),
            pl.col("bid").cast(pl.Float64),
            pl.col("ask").cast(pl.Float64),
            pl.col("implied_vol").cast(pl.Float64),
            pl.col("expiry").cast(pl.Date),
            pl.col("trade_date").cast(pl.Date),
        ]
    )

    # Filter invalid rows
    df = df.filter(
        (pl.col("strike") > 0)
        & (pl.col("underlying_price") > 0)
        & (pl.col("bid") >= 0)
        & (pl.col("ask") >= 0)
    )

    return df
