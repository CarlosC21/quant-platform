import pandas as pd
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression
from quant_platform.ml.pipeline import MLWalkForwardPipeline
from quant_platform.data.feature_store.store import FeatureStore


@pytest.fixture
def synthetic_fs():
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


def test_pipeline_run_and_report(synthetic_fs):
    pipeline = MLWalkForwardPipeline(
        model=LinearRegression(),
        feature_store=synthetic_fs,
        feature_name="synthetic_features",
        target="target",
        val_window=5,
        embargo_days=1,
    )
    output = pipeline.run()

    # check outputs exist
    assert "results" in output
    assert "metrics" in output
    assert "predictions" in output

    # check metrics keys
    assert all(
        k in output["metrics"]
        for k in [
            "rmse",
            "directional_accuracy",
            "sharpe_ratio",
            "information_coefficient",
        ]
    )

    # check predictions DataFrame
    df_preds = output["predictions"]
    assert "prediction" in df_preds.columns
    assert "target" in df_preds.columns
    assert not df_preds.empty

    # check report string
    report = pipeline.report()
    assert "Walk-Forward Regression Metrics" in report


def test_save_predictions(tmp_path, synthetic_fs):
    pipeline = MLWalkForwardPipeline(
        model=LinearRegression(),
        feature_store=synthetic_fs,
        feature_name="synthetic_features",
        target="target",
    )
    pipeline.run()
    path = tmp_path / "preds.csv"
    pipeline.save_predictions(str(path))
    # CSV file exists
    assert path.exists()
    df_loaded = pd.read_csv(path)
    assert "prediction" in df_loaded.columns
    assert "target" in df_loaded.columns
