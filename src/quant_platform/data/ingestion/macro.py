# src/quant_platform/data/ingestion/macro.py
import asyncio
from typing import List

import polars as pl

from quant_platform.data.ingestion.base import BaseIngestor
from quant_platform.data.schemas.macro import MacroIndicatorSchema

# Optional: reuse or create validators
from quant_platform.data.validation.validators import ValidationError


class MacroIngestor(BaseIngestor):
    """
    Async ingestor for macroeconomic CSV/API.
    """

    def __init__(self, source: str):
        self.source = source

    async def fetch_data(self) -> pl.DataFrame:
        df = pl.read_csv(self.source)
        await asyncio.sleep(0)  # async placeholder
        return df

    async def transform(self, raw_df: pl.DataFrame) -> List[MacroIndicatorSchema]:
        """
        Convert raw Polars DataFrame to Pydantic schemas and validate.
        """
        records = [MacroIndicatorSchema(**r) for r in raw_df.to_dicts()]

        # Simple domain-level validation example
        for r in records:
            if r.value < 0:
                raise ValidationError(f"{r.indicator} on {r.date} has negative value")

        return records

    async def run_pipeline(self) -> pl.DataFrame:
        """
        Full ingestion pipeline: fetch -> transform -> convert to Polars.
        """
        records = await self.run()
        df = pl.DataFrame([r.model_dump() for r in records])
        return df
