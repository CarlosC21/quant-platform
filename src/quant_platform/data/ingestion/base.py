# src/quant_platform/data/ingestion/base.py
from abc import ABC, abstractmethod


class BaseIngestor(ABC):
    """Abstract base class for async ingestion pipelines."""

    @abstractmethod
    async def fetch_data(self, *args, **kwargs):
        """Fetch raw data asynchronously."""
        pass

    @abstractmethod
    async def transform(self, raw_data):
        """Transform raw data into validated schema."""
        pass

    async def run(self, *args, **kwargs):
        raw_data = await self.fetch_data(*args, **kwargs)
        return await self.transform(raw_data)
