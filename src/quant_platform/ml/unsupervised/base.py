from __future__ import annotations
from typing import Protocol
import pandas as pd
import numpy as np


class UnsupervisedModel(Protocol):
    """
    Minimal interface for unsupervised models in the ML layer.

    Compatible with sklearn-like estimators and deep-learning wrappers.
    """

    def fit(self, X: pd.DataFrame) -> "UnsupervisedModel":
        ...

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return discrete labels (clusters or regimes).
        """
        ...

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return continuous embeddings or latent codes.
        """
        ...

    def score_samples(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return anomaly scores or log-likelihoods.
        """
        raise NotImplementedError("score_samples not implemented.")
