from __future__ import annotations
from typing import List, Dict, Any
from sklearn.base import RegressorMixin, clone
from quant_platform.ml.pipeline import MLWalkForwardPipeline
from quant_platform.data.feature_store.store import FeatureStore


class MultiModelWalkForwardPipeline:
    """
    Run multiple models over one or more feature sets in walk-forward fashion.
    Stores metrics and predictions for comparison.
    """

    def __init__(
        self,
        models: Dict[str, RegressorMixin],
        feature_store: FeatureStore,
        feature_sets: List[str],
        target: str,
        val_window: int = 5,
        embargo_days: int = 1,
    ):
        self.models = models
        self.feature_store = feature_store
        self.feature_sets = feature_sets
        self.target = target
        self.val_window = val_window
        self.embargo_days = embargo_days
        self.results: Dict[
            str, Dict[str, Any]
        ] = {}  # {feature_set: {model_name: pipeline_output}}

    def run(self) -> Dict[str, Dict[str, Any]]:
        """Run all models across all feature sets."""
        for fs_name in self.feature_sets:
            self.results[fs_name] = {}
            for model_name, model in self.models.items():
                # clone model to avoid cross-fit contamination
                pipeline = MLWalkForwardPipeline(
                    model=clone(model),
                    feature_store=self.feature_store,
                    feature_name=fs_name,
                    target=self.target,
                    val_window=self.val_window,
                    embargo_days=self.embargo_days,
                )
                output = pipeline.run()
                self.results[fs_name][model_name] = output
        return self.results

    def best_model_per_feature_set(self) -> Dict[str, str]:
        """
        Return the best model name per feature set using RMSE as primary metric.
        """
        best_models = {}
        for fs_name, models_out in self.results.items():
            best_model = min(models_out.items(), key=lambda x: x[1]["metrics"]["rmse"])[
                0
            ]
            best_models[fs_name] = best_model
        return best_models
