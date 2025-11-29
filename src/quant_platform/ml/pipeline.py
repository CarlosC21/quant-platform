from __future__ import annotations
from typing import Optional, Dict, Any
import pandas as pd
from sklearn.base import RegressorMixin
from quant_platform.ml.walk_forward import WalkForwardRegression
from quant_platform.data.feature_store.store import FeatureStore


class MLWalkForwardPipeline:
    """
    High-level pipeline to run walk-forward regression using features from FeatureStore.
    Handles:
      - model initialization
      - walk-forward execution
      - metrics aggregation
      - predictions logging
    """

    def __init__(
        self,
        model: RegressorMixin,
        feature_store: FeatureStore,
        feature_name: str,
        target: str,
        val_window: int = 5,
        embargo_days: int = 1,
        n_splits: Optional[int] = None,
    ):
        self.pipeline = WalkForwardRegression(
            model=model,
            feature_store=feature_store,
            feature_name=feature_name,
            target=target,
            val_window=val_window,
            embargo_days=embargo_days,
            n_splits=n_splits,
        )
        self.results: list = []
        self.metrics: dict[str, float] = {}
        self.predictions: pd.DataFrame = pd.DataFrame()

    def run(self) -> Dict[str, Any]:
        """Run the walk-forward pipeline and store results, metrics, and predictions."""
        self.results = self.pipeline.fit_predict_score()
        self.metrics = self.pipeline.aggregate_metrics()
        self.predictions = self.pipeline.predictions_df()
        return {
            "results": self.results,
            "metrics": self.metrics,
            "predictions": self.predictions,
        }

    def report(self) -> str:
        """Return a simple human-readable report of metrics."""
        report_lines = ["Walk-Forward Regression Metrics:"]
        for k, v in self.metrics.items():
            report_lines.append(f"{k}: {v:.4f}")
        return "\n".join(report_lines)

    def save_predictions(self, path: str) -> None:
        """Save predictions DataFrame to CSV."""
        if self.predictions.empty:
            raise ValueError("No predictions to save. Run the pipeline first.")
        self.predictions.to_csv(path, index=False)
