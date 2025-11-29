import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from quant_platform.ml.walk_forward import WalkForwardRegression
from quant_platform.data.feature_store.store import FeatureStore


def test_walk_forward_basic():
    # synthetic feature store
    data = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=20, freq="D"),
            "feat1": np.arange(20),
            "feat2": np.arange(20, 40),
            "target": np.arange(1, 21),
        }
    )

    fs = FeatureStore()
    fs.save_features("synthetic_features", data)

    pipeline = WalkForwardRegression(
        model=LinearRegression(),
        feature_store=fs,
        feature_name="synthetic_features",
        target="target",
        val_window=5,
        embargo_days=1,
    )

    splits = list(pipeline.fit_predict_score())
    assert len(splits) > 0

    for split in splits:
        train_idx, val_idx, metrics, preds = split
        # ensure indices are arrays
        assert isinstance(train_idx, np.ndarray)
        assert isinstance(val_idx, np.ndarray)
        # ensure predictions array length matches validation set
        assert len(preds) == len(val_idx)
        # check metrics keys
        assert all(
            k in metrics
            for k in [
                "rmse",
                "directional_accuracy",
                "sharpe_ratio",
                "information_coefficient",
            ]
        )

    # scores list length matches number of splits
    assert len(pipeline.scores) == len(splits)
