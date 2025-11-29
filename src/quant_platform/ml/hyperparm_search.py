from typing import Dict
from sklearn.base import RegressorMixin, clone
from quant_platform.ml.walk_forward import WalkForwardRegression
from quant_platform.data.feature_store.store import FeatureStore
import itertools


class WalkForwardHyperparamSearch:
    """
    Simple grid-search hyperparameter tuning for walk-forward pipelines.
    """

    def __init__(
        self,
        base_model: RegressorMixin,
        param_grid: Dict[str, list],
        feature_store: FeatureStore,
        feature_name: str,
        target: str,
        val_window: int = 5,
        embargo_days: int = 1,
    ):
        self.base_model = base_model
        self.param_grid = param_grid
        self.feature_store = feature_store
        self.feature_name = feature_name
        self.target = target
        self.val_window = val_window
        self.embargo_days = embargo_days
        self.results = []

    def run(self):
        """Run walk-forward CV for all hyperparameter combinations."""
        keys, values = zip(*self.param_grid.items())
        for combination in itertools.product(*values):
            params = dict(zip(keys, combination))
            model = clone(self.base_model).set_params(**params)
            pipeline = WalkForwardRegression(
                model=model,
                feature_store=self.feature_store,
                feature_name=self.feature_name,
                target=self.target,
                val_window=self.val_window,
                embargo_days=self.embargo_days,
            )
            splits = pipeline.fit_predict_score()
            avg_rmse = sum([s[2]["rmse"] for s in splits]) / len(splits)
            self.results.append(
                {"params": params, "avg_rmse": avg_rmse, "splits": splits}
            )
        # sort best first
        self.results.sort(key=lambda x: x["avg_rmse"])
        return self.results

    def best_params(self):
        return self.results[0]["params"] if self.results else None
