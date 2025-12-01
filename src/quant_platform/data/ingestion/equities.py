# src/quant_platform/data/ingestion/equities.py
import asyncio
from typing import List

import polars as pl

from quant_platform.data.ingestion.base import BaseIngestor
from quant_platform.data.schemas.equities import EquityPriceSchema
from quant_platform.data.validation.validators import validate_equity_records


class EquitiesIngestor(BaseIngestor):
    """
    Async ingestor for equities CSV or API.
    """

    def __init__(self, source: str):
        self.source = source

    async def fetch_data(self) -> pl.DataFrame:
        """
        Async fetch CSV data. Placeholder for API ingestion in the future.
        """
        df = pl.read_csv(self.source)
        await asyncio.sleep(0)  # async placeholder for compatibility
        return df

    async def transform(self, raw_df: pl.DataFrame) -> List[EquityPriceSchema]:
        """
        Convert raw Polars DataFrame to list of Pydantic schemas,
        then apply domain validation.
        """
        records = [EquityPriceSchema(**r) for r in raw_df.to_dicts()]
        validate_equity_records(records)
        return records

    async def run_pipeline(self) -> pl.DataFrame:
        """
        Full ingestion pipeline: fetch -> transform -> convert to Polars with features.
        """
        records = await self.run()
        # Convert back to Polars for feature computation
        df = pl.DataFrame([r.model_dump() for r in records])
        return df
