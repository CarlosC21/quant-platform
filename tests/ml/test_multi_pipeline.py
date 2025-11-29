import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
import pytest
from quant_platform.ml.multi_pipeline import MultiModelWalkForwardPipeline
from quant_platform.data.feature_store.store import FeatureStore


@pytest.fixture
def synthetic_fs():
    data1 = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=20, freq="D"),
            "feat1": np.arange(20),
            "feat2": np.arange(20, 40),
            "target": np.arange(1, 21),
        }
    )
    data2 = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=20, freq="D"),
            "featA": np.arange(20, 40),
            "featB": np.arange(40, 60),
            "target": np.arange(1, 21),
        }
    )
    fs = FeatureStore()
    fs.save_features("features_1", data1)
    fs.save_features("features_2", data2)
    return fs


def test_multi_pipeline_run(synthetic_fs):
    models = {"linreg": LinearRegression(), "ridge": Ridge()}
    feature_sets = ["features_1", "features_2"]
    pipeline = MultiModelWalkForwardPipeline(
        models=models,
        feature_store=synthetic_fs,
        feature_sets=feature_sets,
        target="target",
        val_window=5,
        embargo_days=1,
    )
    results = pipeline.run()

    # check keys
    assert set(results.keys()) == set(feature_sets)
    for fs_name, model_out in results.items():
        assert set(model_out.keys()) == set(models.keys())
        for model_name, out in model_out.items():
            # check metrics exist
            assert all(
                k in out["metrics"]
                for k in [
                    "rmse",
                    "directional_accuracy",
                    "sharpe_ratio",
                    "information_coefficient",
                ]
            )
            # check predictions DataFrame
            assert "prediction" in out["predictions"].columns
            assert "target" in out["predictions"].columns


def test_best_model_per_feature_set(synthetic_fs):
    models = {"linreg": LinearRegression(), "ridge": Ridge()}
    feature_sets = ["features_1"]
    pipeline = MultiModelWalkForwardPipeline(
        models=models,
        feature_store=synthetic_fs,
        feature_sets=feature_sets,
        target="target",
        val_window=5,
        embargo_days=1,
    )
    pipeline.run()
    best = pipeline.best_model_per_feature_set()
    assert set(best.keys()) == set(feature_sets)
    assert best["features_1"] in models
