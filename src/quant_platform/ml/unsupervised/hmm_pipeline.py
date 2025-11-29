# src/quant_platform/ml/unsupervised/hmm_pipeline.py
from __future__ import annotations

from typing import Dict, Optional, Any
import numpy as np
import pandas as pd

from quant_platform.data.feature_store.store import FeatureStore
from quant_platform.ml.unsupervised.hmm import RegimeHMM


def _polars_to_pandas(df):
    """
    Convert Polars → Pandas with a safe fallback when pyarrow is missing.

    Works for:
      - Polars DataFrame (primary path uses .to_pandas())
      - Polars DataFrame without pyarrow (fallback: to_dict → Pandas)
      - Pandas DataFrame (returned unchanged)
    """
    if hasattr(df, "to_pandas"):
        try:
            return df.to_pandas()
        except ModuleNotFoundError:
            try:
                return pd.DataFrame(df.to_dict(as_series=False))
            except Exception:
                pass
    return df


class HMMRegimePipeline:
    """
    Full-history HMM regime pipeline.

    Steps:
      - Pull features from FeatureStore
      - Convert Polars → Pandas if needed
      - Fit RegimeHMM on the chosen feature columns
      - Infer:
          * hard regime labels (regime_hmm)
          * smoothed state probabilities (prob_hmm_<state>)
      - Optionally write them back to FeatureStore.
    """

    def __init__(
        self,
        model: RegimeHMM,
        feature_store: FeatureStore,
        feature_name: str,
        feature_cols: Optional[list[str]] = None,
        regime_name: str = "hmm",
    ):
        self.model = model
        self.feature_store = feature_store
        self.feature_name = feature_name
        self.feature_cols = feature_cols
        self.regime_name = regime_name

        # Outputs
        self.labels_: Optional[np.ndarray] = None
        self.probs_: Optional[np.ndarray] = None

    @property
    def regime_col(self) -> str:
        return f"regime_{self.regime_name}"

    def run(self) -> Dict[str, Any]:
        """
        Fit HMM on full history and compute regimes and probabilities.
        """
        df = self.feature_store.get_features(self.feature_name)
        df = _polars_to_pandas(df)

        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"Expected Pandas DataFrame after conversion, got {type(df)}"
            )

        df = df.reset_index(drop=True)

        # Select feature columns: either explicit, or all non-date columns
        if self.feature_cols is not None:
            X = df[self.feature_cols]
        else:
            X = df.drop(columns=["date"], errors="ignore")

        # Fit & infer
        self.model.fit(X)
        labels = self.model.predict(X)
        probs = self.model.transform(X)

        self.labels_ = np.asarray(labels, dtype=int)
        self.probs_ = np.asarray(probs, dtype=float)

        return {
            "labels": self.labels_,
            "probs": self.probs_,
            "n_states": self.model.n_states,
        }

    def store_regime_features(self) -> None:
        """
        Store regime labels and probabilities in FeatureStore.

        Columns:
          - regime_<regime_name> (int)
          - prob_<regime_name>_<state> for each state index
        """
        if self.labels_ is None or self.probs_ is None:
            raise ValueError(
                "No labels_/probs_ found. Run the pipeline before storing regimes."
            )

        df_pl = self.feature_store.get_features(self.feature_name)

        try:
            import polars as pl
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("Polars is required to store regime features.") from exc

        if len(df_pl) != len(self.labels_):
            raise ValueError(
                f"Length mismatch between FeatureStore frame ({len(df_pl)}) "
                f"and labels_ ({len(self.labels_)})"
            )

        # regime labels
        regime_series = pl.Series(self.regime_col, self.labels_.tolist())
        df_pl = df_pl.with_columns(regime_series)

        # probabilities per state
        n_states = self.probs_.shape[1]
        for state in range(n_states):
            col_name = f"prob_{self.regime_name}_{state}"
            col_values = self.probs_[:, state].tolist()
            prob_series = pl.Series(col_name, col_values)
            df_pl = df_pl.with_columns(prob_series)

        self.feature_store.save_features(self.feature_name, df_pl)
