# tests/fixed_income/fred_ingestion.py
from datetime import date

import polars as pl
import pytest

from src.quant_platform.fixed_income.fred_ingestion import FREDIngestor


# Minimal fake response helper
class FakeResponse:
    def __init__(self, observations):
        self._data = {"observations": observations}
        self.status = 200

    async def json(self):
        return self._data

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")


class DummySession:
    def __init__(self, mapping):
        """
        mapping: series_id -> list of observations (date/value string)
        """
        self.mapping = mapping
        self.closed = False

    async def get(self, url, params=None):
        series_id = params.get("series_id")
        obs = self.mapping.get(series_id, [])
        return FakeResponse(obs)

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_fetch_and_pivot_basic():
    mapping = {
        "DGS1": [
            {"date": "2020-01-02", "value": "1.23"},
            {"date": "2020-01-03", "value": "1.25"},
        ],
        "DGS2": [
            {"date": "2020-01-02", "value": "1.50"},
            {"date": "2020-01-03", "value": "."},
        ],
    }
    session = DummySession(mapping)
    ing = FREDIngestor(api_key="fake", session=session)
    df = await ing.fetch_and_pivot(
        series_ids=["DGS1", "DGS2"], start="2020-01-01", end="2020-01-05"
    )
    assert "date" in df.columns
    assert "DGS1" in df.columns and "DGS2" in df.columns
    pdf = df.to_pandas()
    row = pdf.loc[pdf["date"] == date(2020, 1, 2)].iloc[0]
    assert float(row["DGS1"]) == pytest.approx(1.23)
    assert float(row["DGS2"]) == pytest.approx(1.50)
    row2 = pdf.loc[pdf["date"] == date(2020, 1, 3)].iloc[0]
    assert float(row2["DGS1"]) == pytest.approx(1.25)
    assert (
        (pl.Series([row2["DGS2"]]).is_null().to_list()[0]) or str(row2["DGS2"]) == "nan"
    )
