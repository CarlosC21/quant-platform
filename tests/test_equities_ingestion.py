# tests/test_equities_ingestion.py
from datetime import date

import polars as pl
import pytest

from src.quant_platform.data.ingestion.equities import EquitiesIngestor
from src.quant_platform.data.schemas.equities import EquityPriceSchema
from src.quant_platform.data.validation.validators import (
    ValidationError, validate_equity_records)

# Sample CSV data
sample_data = pl.DataFrame(
    {
        "symbol": ["AAPL", "MSFT"],
        "date": [date(2025, 11, 25), date(2025, 11, 25)],
        "open": [150, 300],
        "high": [152, 305],
        "low": [149, 298],
        "close": [151, 303],
        "volume": [10000, 15000],
    }
)


@pytest.mark.asyncio
async def test_equities_ingestor_pipeline(tmp_path):
    # Write sample CSV
    csv_file = tmp_path / "equities.csv"
    sample_data.write_csv(csv_file)

    ingestor = EquitiesIngestor(str(csv_file))
    df = await ingestor.run_pipeline()

    # Check dataframe shape
    assert df.shape[0] == 2
    assert "symbol" in df.columns
    assert df["close_"].to_list() == [151, 303]


@pytest.mark.parametrize(
    "invalid_record",
    [
        {
            "symbol": "AAPL",
            "date": date(2025, 11, 25),
            "open": 150,
            "high": 152,
            "low": 149,
            "close": -1,
            "volume": 10000,
        }
    ],
)
def test_validate_equity_records_raises(invalid_record):
    schema = EquityPriceSchema(**invalid_record)
    with pytest.raises(ValidationError):
        validate_equity_records([schema])
