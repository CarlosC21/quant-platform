# src/quant_platform/fixed_income/fred_ingestion.py
"""Minimal FRED ingestor used by tests (accepts injected session)."""

from typing import Any, Dict, List, Optional

import polars as pl

FRED_SERIES = {
    "DGS1": 1.0,
    "DGS2": 2.0,
    "DGS3": 3.0,
    "DGS5": 5.0,
    "DGS7": 7.0,
    "DGS10": 10.0,
    "DGS20": 20.0,
    "DGS30": 30.0,
    # add others if needed by tests
}


class FREDIngestor:
    """
    Minimal FRED ingestor. Tests inject a DummySession implementing:
      async def get(self, url, params=None) -> FakeResponse
    where FakeResponse has async def json(self) -> dict.

    Only implements fetch_and_pivot(series_ids, start, end) used by tests.
    """

    def __init__(self, api_key: str, session: Optional[Any] = None):
        self.api_key = api_key
        self.session = session

    async def _fetch_series(
        self, series_id: str, start: Optional[str] = None, end: Optional[str] = None
    ) -> List[Dict]:
        """
        Use injected session (tests provide a DummySession). Return observations list.
        Each observation is expected to be dict with keys "date" and "value".
        """
        params = {"series_id": series_id, "api_key": self.api_key, "file_type": "json"}
        if start is not None:
            params["observation_start"] = start
        if end is not None:
            params["observation_end"] = end

        # test session implements async get(series_id) and returns object with .json()
        if self.session is None:
            raise RuntimeError(
                "No session available for FRED requests in test environment"
            )

        resp = await self.session.get("", params=params)
        data = await resp.json()
        return data.get("observations", [])

    async def fetch_and_pivot(
        self,
        series_ids: List[str],
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pl.DataFrame:
        """
        Fetch multiple series and pivot to a wide Polars DataFrame with 'date' column
        and one column per series id. Missing values become None/Null.
        """
        # Gather series -> list of observations per series
        series_obs = {}
        for sid in series_ids:
            obs = await self._fetch_series(sid, start=start, end=end)
            series_obs[sid] = obs

        # Collect all unique dates
        dates = set()
        for obs in series_obs.values():
            for o in obs:
                dates.add(o["date"])
        dates = sorted(dates)

        # Build rows: one dict per date
        rows = []
        for d in dates:
            row = {"date": d}
            for sid, obs in series_obs.items():
                # find obs for date d if present
                val = None
                for o in obs:
                    if o.get("date") == d:
                        v = o.get("value")
                        if v == ".":
                            val = None
                        else:
                            try:
                                val = float(v)
                            except Exception:
                                val = None
                        break
                row[sid] = val
            rows.append(row)

        df = pl.DataFrame(rows)
        # convert 'date' strings to Date type if possible
        if "date" in df.columns:
            df = df.with_columns(pl.col("date").str.to_date().alias("date"))
        return df
