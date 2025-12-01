# scripts/run_pipeline.py
import asyncio

from quant_platform.data.ingestion.pipelines import DataPipeline


async def main():
    pipeline = DataPipeline(
        equities_source="data/raw/equities.csv", macro_source="data/raw/macro.csv"
    )
    feature_store = await pipeline.run()

    print("Available feature sets:", feature_store.list_features())


asyncio.run(main())
