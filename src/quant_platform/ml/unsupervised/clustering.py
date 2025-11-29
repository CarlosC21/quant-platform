from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from quant_platform.ml.unsupervised.base import UnsupervisedModel


class KMeansClustering(UnsupervisedModel):
    """
    Simple wrapper around sklearn KMeans exposing the UnsupervisedModel protocol.
    """

    def __init__(self, n_clusters: int = 3, random_state: int = 42, **kwargs):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kwargs = kwargs
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state, **kwargs)

    def fit(self, X: pd.DataFrame) -> "KMeansClustering":
        # sklearn works fine with numpy arrays
        self.model.fit(X.values)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X.values)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return cluster distances (or embeddings if needed).
        For now: use transform() which returns distances to cluster centers.
        """
        return self.model.transform(X.values)

    def score_samples(self, X: pd.DataFrame) -> np.ndarray:
        """
        Negative distance to nearest cluster center = simple 'anomaly score'.
        """
        distances = self.model.transform(X.values)
        nearest = distances.min(axis=1)
        return -nearest
