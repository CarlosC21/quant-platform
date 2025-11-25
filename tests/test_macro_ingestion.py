# tests/test_macro_ingestion.py
from datetime import date

import polars as pl
import pytest

from src.quant_platform.data.ingestion.macro import MacroIngestor
from src.quant_platform.data.schemas.macro import MacroIndicatorSchema
from src.quant_platform.data.validation.validators import ValidationError

sample_macro = pl.DataFrame(
    {
        "indicator": ["CPI", "GDP"],
        "date": [date(2025, 11, 25), date(2025, 11, 25)],
        "value": [300.5, 2100.0],
    }
)


@pytest.mark.asyncio
async def test_macro_ingestor_pipeline(tmp_path):
    csv_file = tmp_path / "macro.csv"
    sample_macro.write_csv(csv_file)

    ingestor = MacroIngestor(str(csv_file))
    df = await ingestor.run_pipeline()

    assert df.shape[0] == 2
    assert "indicator" in df.columns


@pytest.mark.parametrize(
    "invalid_record", [{"indicator": "CPI", "date": date(2025, 11, 25), "value": -5}]
)
def test_macro_validator_raises(invalid_record):
    schema = MacroIndicatorSchema(**invalid_record)

    with pytest.raises(ValidationError):
        if schema.value < 0:
            raise ValidationError("Negative macro value")
