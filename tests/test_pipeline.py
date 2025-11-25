# tests/test_pipeline.py

import pytest

from src.quant_platform.data.ingestion.pipelines import DataPipeline


@pytest.mark.asyncio
async def test_pipeline_end_to_end(tmp_path):
    # Create sample CSVs
    equities_csv = tmp_path / "equities.csv"
    equities_csv.write_text(
        "symbol,date,open,high,low,close,volume\n"
        "AAPL,2025-11-25,150,152,149,151,10000\n"
        "MSFT,2025-11-25,300,305,298,303,15000\n"
    )
    macro_csv = tmp_path / "macro.csv"
    macro_csv.write_text(
        "indicator,date,value\n" "CPI,2025-11-25,300.5\n" "GDP,2025-11-25,2100.0\n"
    )

    pipeline = DataPipeline(str(equities_csv), str(macro_csv))
    fs = await pipeline.run()

    # Check features exist
    assert "equities_features" in fs.list_features()
    assert "macro_features" in fs.list_features()
