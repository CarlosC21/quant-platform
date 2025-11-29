import pandas as pd
import polars as pl

from quant_platform.data.feature_store.store import FeatureStore
from quant_platform.ml.unsupervised.clustering import KMeansClustering
from quant_platform.ml.unsupervised.time_series_clustering import (
    TimeSeriesClusteringPipeline,
)


def test_time_series_clustering_pipeline_basic():
    # Build simple DataFrame (Polars for testing conversion path)
    df = pl.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=10, freq="D"),
            "feature1": [1, 2, 3, 4, 50, 60, 55, 53, 52, 51],
        }
    )

    fs = FeatureStore()
    fs.save_features("test_features", df)

    model = KMeansClustering(n_clusters=2)
    pipeline = TimeSeriesClusteringPipeline(
        model=model,
        feature_store=fs,
        feature_name="test_features",
        val_window=3,
        embargo_days=1,
    )

    result = pipeline.run()

    assert "results" in result
    assert "metrics" in result
    assert "labels" in result

    labels = result["labels"]
    assert len(labels) == 10
    assert set(labels) <= {0, 1, -1}  # -1 for train-only rows


def test_time_series_clustering_store_regime():
    df = pl.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=10, freq="D"),
            "feature1": [1, 2, 3, 4, 50, 60, 55, 53, 52, 51],
        }
    )

    fs = FeatureStore()
    fs.save_features("test_features_regime", df)

    model = KMeansClustering(n_clusters=2)
    pipeline = TimeSeriesClusteringPipeline(
        model=model,
        feature_store=fs,
        feature_name="test_features_regime",
        val_window=3,
        embargo_days=1,
        regime_name="kmeans",
    )

    result = pipeline.run()
    assert "labels" in result
    assert pipeline.labels_ is not None

    pipeline.store_regime_labels()

    df_after = fs.get_features("test_features_regime")
    assert "regime_kmeans" in df_after.columns
    assert df_after.height == 10

    # Ensure regime labels are integers and include only expected values
    regimes = df_after["regime_kmeans"].to_list()
    assert set(regimes) <= {0, 1, -1}
