# src/quant_platform/fixed_income/ingestion.py

# Re-export symbols to satisfy tests expecting this module path.
from .fred_ingestion import FREDIngestor

__all__ = ["FREDIngestor"]
