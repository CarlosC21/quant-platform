# src/quant_platform/ml/unsupervised/time_series_clustering.py
from __future__ import annotations

from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)

from quant_platform.data.feature_store.store import FeatureStore
from quant_platform.ml.cv import generate_time_series_splits
from quant_platform.ml.unsupervised.base import UnsupervisedModel


def _polars_to_pandas(df):
    """
    Convert Polars → Pandas with a safe fallback when pyarrow is missing.

    Works for:
      - Polars DataFrame (primary path uses .to_pandas())
      - Polars DataFrame without pyarrow (fallback: to_dict → Pandas)
      - Pandas DataFrame (returned unchanged)
    """
    # Polars-like case: has .to_pandas()
    if hasattr(df, "to_pandas"):
        try:
            # primary path, may require pyarrow
            return df.to_pandas()
        except ModuleNotFoundError:
            # fallback: pure-Python conversion
            try:
                return pd.DataFrame(df.to_dict(as_series=False))
            except Exception:
                pass

    # Already Pandas or unknown type; return as is
    return df


class TimeSeriesClusteringPipeline:
    """
    Time-series aware clustering pipeline.

    Steps:
      - Pull features from FeatureStore
      - Convert Polars → Pandas if needed
      - Generate walk-forward time splits
      - Fit clustering model on training window
      - Predict labels on validation window
      - Compute internal cluster metrics
      - Aggregate metrics and expose labels_
      - (Optionally) write regime labels back into FeatureStore
    """

    def __init__(
        self,
        model: UnsupervisedModel,
        feature_store: FeatureStore,
        feature_name: str,
        val_window: int = 5,
        embargo_days: int = 1,
        n_splits: Optional[int] = None,
        regime_name: Optional[str] = None,
    ):
        self.model = model
        self.feature_store = feature_store
        self.feature_name = feature_name
        self.val_window = val_window
        self.embargo_days = embargo_days
        self.n_splits = n_splits

        # regime_<name>, e.g. regime_kmeans, regime_hmm
        inferred = self._infer_model_name()
        base_name = regime_name or inferred
        self.regime_col = f"regime_{base_name}"

        self.df: Optional[pd.DataFrame] = None
        self.results: List[
            Tuple[np.ndarray, np.ndarray, Dict[str, float], np.ndarray]
        ] = []
        self.metrics_per_split: List[Dict[str, float]] = []
        self.labels_: Optional[np.ndarray] = None

    def _infer_model_name(self) -> str:
        """Infer a simple model name from the model class."""
        name = self.model.__class__.__name__
        return name.lower()

    def run(self) -> Dict[str, Any]:
        """Execute the full walk-forward clustering pipeline."""
        df = self.feature_store.get_features(self.feature_name)
        df = _polars_to_pandas(df)

        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"Expected Pandas DataFrame after conversion, got {type(df)}"
            )

        df = df.reset_index(drop=True)
        self.df = df

        # build timestamps
        timestamps = (
            df["date"] if "date" in df.columns else pd.Series(np.arange(len(df)))
        )

        # determine number of splits
        n_splits = (
            self.n_splits
            if self.n_splits and self.n_splits > 0
            else max(1, len(timestamps) // self.val_window)
        )

        splits = generate_time_series_splits(
            timestamps,
            n_splits=n_splits,
            val_window=self.val_window,
            embargo_days=self.embargo_days,
        )

        # enforce embargo
        adjusted_splits: List[Tuple[np.ndarray, np.ndarray]] = []
        for train_idx, val_idx in splits:
            if len(train_idx) == 0:
                continue

            min_allowed = int(train_idx[-1]) + int(self.embargo_days)
            val_filtered = val_idx[val_idx >= min_allowed]

            if val_filtered.size == 0:
                continue

            adjusted_splits.append((train_idx, val_filtered))

        results: List[Tuple[np.ndarray, np.ndarray, Dict[str, float], np.ndarray]] = []

        for train_idx, val_idx in adjusted_splits:
            X_train = df.iloc[train_idx].drop(columns=["date"], errors="ignore")
            X_val = df.iloc[val_idx].drop(columns=["date"], errors="ignore")

            self.model.fit(X_train)
            labels_val = self.model.predict(X_val)
            labels_val_arr = np.asarray(labels_val)

            # compute internal metrics if >1 cluster
            if len(np.unique(labels_val_arr)) > 1:
                try:
                    s = float(silhouette_score(X_val, labels_val_arr))
                except Exception:
                    s = 0.0
                try:
                    ch = float(calinski_harabasz_score(X_val, labels_val_arr))
                except Exception:
                    ch = 0.0
                try:
                    db = float(davies_bouldin_score(X_val, labels_val_arr))
                except Exception:
                    db = 0.0
            else:
                s = ch = db = 0.0

            metrics = {
                "silhouette": s,
                "calinski_harabasz": ch,
                "davies_bouldin": db,
            }

            self.metrics_per_split.append(metrics)
            results.append((train_idx, val_idx, metrics, labels_val_arr))

        self.results = results

        # Build per-row prediction vector aligned to df index
        all_labels = np.full(len(df), fill_value=-1, dtype=int)
        for _, val_idx, _, labels in results:
            all_labels[val_idx] = labels

        self.labels_ = all_labels

        return {
            "results": results,
            "metrics": self.aggregate_metrics(),
            "labels": self.labels_,
        }

    def aggregate_metrics(self) -> Dict[str, float]:
        """Aggregate numeric metrics across splits."""
        if not self.metrics_per_split:
            return {
                "silhouette": 0.0,
                "calinski_harabasz": 0.0,
                "davies_bouldin": 0.0,
            }

        df = pd.DataFrame(self.metrics_per_split)
        return df.mean(numeric_only=True).to_dict()

    def store_regime_labels(self) -> None:
        """
        Write the inferred regime labels back into the FeatureStore as a new column.

        Column name:
            self.regime_col, e.g. 'regime_kmeans', 'regime_hmm'
        """
        if self.labels_ is None:
            raise ValueError(
                "No labels_ found. Run the pipeline before storing regimes."
            )

        # Get original Polars frame from FeatureStore
        df_pl = self.feature_store.get_features(self.feature_name)

        try:
            import polars as pl
        except Exception as exc:  # pragma: no cover - polars is available in tests
            raise RuntimeError("Polars is required to store regime labels.") from exc

        if len(df_pl) != len(self.labels_):
            raise ValueError(
                f"Length mismatch between FeatureStore frame ({len(df_pl)}) "
                f"and labels_ ({len(self.labels_)})"
            )

        labels_series = pl.Series(self.regime_col, self.labels_.tolist())
        df_pl = df_pl.with_columns(labels_series)

        self.feature_store.save_features(self.feature_name, df_pl)
