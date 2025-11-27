import asyncio
from typing import List

import polars as pl

from src.quant_platform.data.ingestion.base import BaseIngestor
from src.quant_platform.data.validation.options_validators import (
    validate_option_records,
)
from src.quant_platform.options.schemas import OptionQuoteSchema


class OptionsIngestor(BaseIngestor):
    def __init__(self, source: str):
        self.source = source

    async def fetch_data(self) -> pl.DataFrame:
        df = pl.read_csv(self.source)
        await asyncio.sleep(0)
        return df

    async def transform(self, raw_df: pl.DataFrame) -> List[OptionQuoteSchema]:
        records = [OptionQuoteSchema(**r) for r in raw_df.to_dicts()]
        validate_option_records(records)
        return records

    async def run_pipeline(self) -> pl.DataFrame:
        records = await self.run()
        df = pl.DataFrame([r.model_dump() for r in records])
        return df
