# src/quant_platform/ml/walk_forward.py
from __future__ import annotations
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from quant_platform.ml.cv import generate_time_series_splits
from quant_platform.ml.metrics import (
    rmse,
    directional_accuracy,
    sharpe_ratio,
    information_coefficient,
)
from quant_platform.data.feature_store.store import FeatureStore


def _to_scalar(x) -> float:
    """
    Convert metric output to a Python float robustly.

    Accepts:
      - Python scalars
      - numpy arrays
      - pandas Series

    If multiple values are present, returns their numeric mean (nan-safe).
    If conversion fails, returns float('nan').
    """
    try:
        # Try fast path for float-like scalars
        if isinstance(x, (float, int, np.floating, np.integer)):
            return float(x)
        arr = np.asarray(x)

        if arr.size == 0:
            return float("nan")
        if arr.size == 1:
            return float(arr.item())

        # multiple values -> coerce to numeric then mean
        numeric = pd.to_numeric(pd.Series(arr.ravel()), errors="coerce")
        return float(numeric.mean(skipna=True))
    except Exception:
        return float("nan")


class WalkForwardRegression:
    """
    Walk-forward regression pipeline.

    Pulls features from FeatureStore, performs time-series CV, trains model for each split,
    computes metrics per-split and exposes aggregation/predictions utilities.
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
        self.model = model
        self.feature_store = feature_store
        self.feature_name = feature_name
        self.target = target
        self.val_window = val_window
        self.embargo_days = embargo_days
        self.n_splits = n_splits

        # runtime state
        self.scores: List[float] = []
        self.metrics_per_split: List[dict] = []
        self.results: List[Tuple[np.ndarray, np.ndarray, dict, np.ndarray]] = []
        self.df: Optional[pd.DataFrame] = None

    def fit_predict_score(
        self,
    ) -> list[tuple[np.ndarray, np.ndarray, dict, np.ndarray]]:
        """
        Run walk-forward training/prediction/scoring.

        Returns:
            list of tuples: (train_idx, val_idx, metrics_dict, preds_array)
        """
        df = self.feature_store.get_features(self.feature_name)

        # Support polars DataFrame or pandas DataFrame
        try:
            import polars as pl  # type: ignore

            if isinstance(df, pl.DataFrame):
                df = df.to_pandas()
        except Exception:
            # If polars not available or other error, assume df is pandas or let type check below raise
            pass

        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"FeatureStore returned unsupported type {type(df)}")

        # Reset index so positional indices from CV align with df.iloc
        df = df.reset_index(drop=True)
        self.df = df

        timestamps = (
            df["date"] if "date" in df.columns else pd.Series(np.arange(len(df)))
        )

        n_splits = (
            self.n_splits
            if (self.n_splits and self.n_splits > 0)
            else max(1, len(timestamps) // self.val_window)
        )

        splits_iter = generate_time_series_splits(
            timestamps,
            n_splits=n_splits,
            val_window=self.val_window,
            embargo_days=self.embargo_days,
        )

        # Strictly enforce embargo gap between train end and validation start:
        # validation indices must satisfy val_idx[0] >= train_idx[-1] + embargo_days
        adjusted_splits: List[Tuple[np.ndarray, np.ndarray]] = []
        for train_idx, val_idx in splits_iter:
            # if no train indices, keep val as-is (no embargo enforcement possible)
            if len(train_idx) == 0:
                adjusted_splits.append((train_idx, val_idx))
                continue

            min_allowed_idx = int(train_idx[-1]) + int(self.embargo_days)
            # keep validation indices that start at or after min_allowed_idx
            val_idx_filtered = val_idx[val_idx >= min_allowed_idx]

            if val_idx_filtered.size == 0:
                # skip split if embargo removes entire validation window
                continue

            adjusted_splits.append((train_idx, val_idx_filtered))

        results: List[Tuple[np.ndarray, np.ndarray, dict, np.ndarray]] = []

        for train_idx, val_idx in adjusted_splits:
            X_train = df.iloc[train_idx].drop(
                columns=[self.target, "date"], errors="ignore"
            )
            y_train = df.iloc[train_idx][self.target]

            X_val = df.iloc[val_idx].drop(
                columns=[self.target, "date"], errors="ignore"
            )
            y_val = df.iloc[val_idx][self.target]

            # Fit & predict (sklearn API)
            self.model.fit(X_train, y_train)
            preds = self.model.predict(X_val)
            preds_arr = np.asarray(preds)

            # Compute metrics and coerce to scalars
            rmse_val = _to_scalar(rmse(y_val, preds_arr))
            dir_acc = _to_scalar(directional_accuracy(preds_arr, y_val))

            # Use residuals (y - pred) for Sharpe stability
            residuals = np.asarray(y_val) - preds_arr
            sr_val = _to_scalar(sharpe_ratio(residuals))

            ic_val = _to_scalar(information_coefficient(preds_arr, y_val))

            metrics = {
                "rmse": float(rmse_val if not np.isnan(rmse_val) else 0.0),
                "directional_accuracy": float(
                    dir_acc if not np.isnan(dir_acc) else 0.0
                ),
                "sharpe_ratio": float(sr_val if not np.isnan(sr_val) else 0.0),
                "information_coefficient": float(
                    ic_val if not np.isnan(ic_val) else 0.0
                ),
            }

            # record
            self.scores.append(metrics["rmse"])
            self.metrics_per_split.append(metrics)
            results.append((train_idx, val_idx, metrics, preds_arr))

            # brief logging for debugging
            print(
                f"Split {len(results)} - RMSE: {metrics['rmse']:.4f}, "
                f"DirAcc: {metrics['directional_accuracy']:.4f}, "
                f"SR: {metrics['sharpe_ratio']:.4f}, "
                f"IC: {metrics['information_coefficient']:.4f}"
            )

        self.results = results
        return results

    def aggregate_metrics(self) -> dict[str, float]:
        """Aggregate numeric metrics across splits and return mean metrics."""
        if not self.metrics_per_split:
            raise ValueError("No metrics recorded. Run fit_predict_score first.")

        df_metrics = pd.DataFrame(self.metrics_per_split)
        numeric_metrics = df_metrics.select_dtypes(include=[np.number])
        return numeric_metrics.mean().fillna(0.0).to_dict()

    def predictions_df(self) -> pd.DataFrame:
        """Return a DataFrame of all validation predictions with split index and true targets."""
        if not self.results or self.df is None:
            raise ValueError("No results recorded. Run fit_predict_score first.")

        records = []
        for i, (_, val_idx, _, preds) in enumerate(self.results):
            for idx, pred in zip(val_idx, preds):
                records.append(
                    {
                        "split": int(i + 1),
                        "index": int(idx),
                        "prediction": float(pred),
                        "target": float(self.df.iloc[int(idx)][self.target]),
                    }
                )
        return pd.DataFrame(records)
