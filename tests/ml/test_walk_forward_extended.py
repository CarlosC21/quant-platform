import pandas as pd
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from quant_platform.ml.walk_forward import WalkForwardRegression
from quant_platform.data.feature_store.store import FeatureStore


@pytest.fixture
def synthetic_feature_store():
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
    return fs


@pytest.mark.parametrize(
    "model",
    [
        LinearRegression(),
        Ridge(alpha=1.0),
        RandomForestRegressor(n_estimators=10, random_state=42),
    ],
)
def test_walk_forward_multi_model(synthetic_feature_store, model):
    pipeline = WalkForwardRegression(
        model=model,
        feature_store=synthetic_feature_store,
        feature_name="synthetic_features",
        target="target",
        val_window=5,
        embargo_days=1,
    )
    results = pipeline.fit_predict_score()
    assert len(results) > 0

    for train_idx, val_idx, metrics, preds in results:
        assert isinstance(train_idx, np.ndarray)
        assert isinstance(val_idx, np.ndarray)
        assert len(preds) == len(val_idx)
        assert all(
            k in metrics
            for k in [
                "rmse",
                "directional_accuracy",
                "sharpe_ratio",
                "information_coefficient",
            ]
        )

    # Check aggregated metrics
    agg = pipeline.aggregate_metrics()
    assert all(
        k in agg
        for k in [
            "rmse",
            "directional_accuracy",
            "sharpe_ratio",
            "information_coefficient",
        ]
    )

    # Check predictions_df
    df_preds = pipeline.predictions_df()
    assert "prediction" in df_preds.columns
    assert "target" in df_preds.columns
    assert len(df_preds) == sum(len(val_idx) for _, val_idx, _, _ in results)


def test_walk_forward_embargo(synthetic_feature_store):
    # Ensure embargo_days > 0 reduces overlap
    pipeline = WalkForwardRegression(
        model=LinearRegression(),
        feature_store=synthetic_feature_store,
        feature_name="synthetic_features",
        target="target",
        val_window=5,
        embargo_days=2,
    )
    results = pipeline.fit_predict_score()
    # Simple check: first val_idx should start after embargo_days from train end
    first_train_idx, first_val_idx, _, _ = results[0]
    assert first_val_idx[0] >= first_train_idx[-1] + 2
