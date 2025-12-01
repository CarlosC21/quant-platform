# src/quant_platform/data/ingestion/pipelines.py
import logging

from quant_platform.data.feature_store.features import (
    compute_equity_features,
    compute_macro_features,
)
from quant_platform.data.feature_store.store import FeatureStore

from .equities import EquitiesIngestor
from .macro import MacroIngestor

logger = logging.getLogger(__name__)


class DataPipeline:
    def __init__(self, equities_source: str, macro_source: str):
        self.equities_ingestor = EquitiesIngestor(equities_source)
        self.macro_ingestor = MacroIngestor(macro_source)
        self.feature_store = FeatureStore()

    async def run(self):
        logger.info("Starting data pipeline...")

        # Equities stage
        equities_df = await self.equities_ingestor.run_pipeline()
        equities_features = compute_equity_features(equities_df)
        self.feature_store.save_features("equities_features", equities_features)
        logger.info("Equities stage completed.")

        # Macro stage
        macro_df = await self.macro_ingestor.run_pipeline()
        macro_features = compute_macro_features(macro_df)
        self.feature_store.save_features("macro_features", macro_features)
        logger.info("Macro stage completed.")

        return self.feature_store
