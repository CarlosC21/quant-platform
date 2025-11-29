from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import date


class FeatureSpec(BaseModel):
    name: str
    dtype: Literal["float", "int", "bool", "category", "datetime"] = "float"
    description: Optional[str] = None


class TrainConfig(BaseModel):
    """
    Training configuration for supervised models.
    Use model_dump() / model_validate() for pydantic v2 compatibility.
    """

    experiment_name: str = Field(..., description="Experiment id")
    target_col: str = Field(..., description="Name of target column")
    timestamp_col: str = Field("timestamp", description="Name of timestamp column")
    entity_col: Optional[str] = Field(
        None, description="Optional entity id column (ticker)"
    )
    features: List[str] = Field(..., description="List of feature column names")
    model_type: Literal["lgb", "ridge", "logistic"] = "lgb"
    random_seed: int = 42
    n_splits: int = 5
    cv_type: Literal["expanding", "rolling"] = "expanding"
    train_window: Optional[int] = Field(
        None, description="Window (in days) for rolling training"
    )
    val_window: Optional[int] = Field(
        None, description="Window (in days) for validation"
    )
    embargo_days: int = 0
    metrics: List[str] = Field(default_factory=lambda: ["sharpe", "ic"])
    output_dir: str = "artifacts/"
    lightgbm_params: dict = Field(default_factory=dict)
    start_date: Optional[date] = None
    end_date: Optional[date] = None

    class Config:
        extra = "forbid"
