# src/quant_platform/fixed_income/schemas.py
from __future__ import annotations

import datetime as _dt

from pydantic import BaseModel, Field


class YieldPoint(BaseModel):
    """
    Single point on the yield curve (standardized from FRED ingestion).

    - date: observation date (datetime.date)
    - maturity_months: integer months to maturity (e.g., 60 for 5y)
    - yield_pct: yield in percent (e.g., 2.50 means 2.50%)
    """

    date: _dt.date = Field(..., description="Observation date")
    maturity_months: int = Field(..., ge=1, description="Maturity in months")
    yield_pct: float = Field(
        ..., description="Yield percentage (e.g., 4.25)"
    )  # removed ge=0.0
