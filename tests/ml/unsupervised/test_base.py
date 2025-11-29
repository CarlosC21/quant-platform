import pandas as pd
import numpy as np


class DummyModel:
    """Minimal dummy model implementing the protocol."""

    def fit(self, X: pd.DataFrame):
        self.mean_ = X.mean().iloc[0]
        return self

    def predict(self, X: pd.DataFrame):
        return (X.iloc[:, 0].values > self.mean_).astype(int)

    def transform(self, X: pd.DataFrame):
        return (X.iloc[:, 0].values - self.mean_).reshape(-1, 1)

    def score_samples(self, X: pd.DataFrame):
        return -(np.abs(X.iloc[:, 0].values - self.mean_))


def test_dummy_model_protocol():
    df = pd.DataFrame({"x": [1, 2, 10, 20]})
    model = DummyModel().fit(df)

    labels = model.predict(df)
    assert labels.shape == (4,)
    assert set(labels) <= {0, 1}

    emb = model.transform(df)
    assert emb.shape == (4, 1)

    scores = model.score_samples(df)
    assert scores.shape == (4,)
