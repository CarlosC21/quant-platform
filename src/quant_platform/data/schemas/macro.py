# src/quant_platform/data/schemas/macro.py
from datetime import date

from pydantic import BaseModel, Field


class MacroIndicatorSchema(BaseModel):
    indicator: str = Field(..., description="Name of the macroeconomic indicator")
    obs_date: date = Field(..., alias="date", description="Date of observation")
    value: float | None = None
    unit_: str | None = Field(None, alias="unit", description="Unit of measurement")

    model_config = {"populate_by_name": True}
