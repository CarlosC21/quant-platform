from __future__ import annotations
from typing import Any
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score
from quant_platform.data.feature_store.store import FeatureStore


class BasePipeline:
    """
    Base class for ML pipelines.
    Handles feature fetching, training, prediction, and scoring.
    """

    def __init__(self, model: Any, feature_store: FeatureStore):
        """
        Parameters
        ----------
        model: Any
            A sklearn-like estimator implementing fit/predict.
        feature_store: FeatureStore
            Feature store to fetch training features from.
        """
        self.model = model
        self.feature_store = feature_store

    def fit(self, X: pd.DataFrame, y: pd.Series) -> BasePipeline:
        """Fit the model."""
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained model."""
        return self.model.predict(X)

    def score(self, X: pd.DataFrame, y: pd.Series, metric: str = "default") -> float:
        """Evaluate predictions using the specified metric."""
        preds = self.predict(X)
        if metric == "rmse" or metric == "default":
            return np.sqrt(mean_squared_error(y, preds))
        elif metric == "accuracy":
            return accuracy_score(y, np.round(preds))
        else:
            raise ValueError(f"Unsupported metric: {metric}")


class RegressionPipeline(BasePipeline):
    """Regression-specific pipeline."""

    def score(self, X: pd.DataFrame, y: pd.Series, metric: str = "rmse") -> float:
        """Regression metrics."""
        return super().score(X, y, metric)


class ClassificationPipeline(BasePipeline):
    """Classification-specific pipeline."""

    def score(self, X: pd.DataFrame, y: pd.Series, metric: str = "accuracy") -> float:
        """Classification metrics."""
        return super().score(X, y, metric)
