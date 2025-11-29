# tests/ml/unsupervised/test_hmm_regime.py
import pytest
import numpy as np
import pandas as pd
import polars as pl

from quant_platform.data.feature_store.store import FeatureStore
from quant_platform.ml.unsupervised.hmm import RegimeHMM
from quant_platform.ml.unsupervised.hmm_pipeline import HMMRegimePipeline


@pytest.mark.skipif(
    pytest.importorskip("hmmlearn", reason="hmmlearn not installed") is None,
    reason="hmmlearn not available",
)
def test_regime_hmm_basic():
    # Two simple regimes: low and high values
    x = np.concatenate(
        [
            np.random.normal(loc=0.0, scale=0.5, size=50),
            np.random.normal(loc=5.0, scale=0.5, size=50),
        ]
    )
    df = pd.DataFrame({"x": x})

    model = RegimeHMM(n_states=2, n_iter=50, random_state=42)
    model.fit(df)

    labels = model.predict(df)
    probs = model.transform(df)
    logprob = model.score_samples(df)

    assert labels.shape == (100,)
    assert probs.shape == (100, 2)
    assert logprob.shape == (100,)
    # probabilities per row should sum to ~1
    row_sums = probs.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6)


@pytest.mark.skipif(
    pytest.importorskip("hmmlearn", reason="hmmlearn not installed") is None,
    reason="hmmlearn not available",
)
def test_hmm_regime_pipeline_store_features():
    # Simple synthetic feature set with a clear shift
    dates = pd.date_range("2020-01-01", periods=20, freq="D")
    feature = np.concatenate(
        [
            np.random.normal(loc=0.0, scale=0.5, size=10),
            np.random.normal(loc=3.0, scale=0.5, size=10),
        ]
    )
    df_pl = pl.DataFrame({"date": dates, "feature1": feature})

    fs = FeatureStore()
    fs.save_features("hmm_features", df_pl)

    model = RegimeHMM(n_states=2, n_iter=50, random_state=123)
    pipeline = HMMRegimePipeline(
        model=model,
        feature_store=fs,
        feature_name="hmm_features",
        feature_cols=["feature1"],
        regime_name="hmm",
    )

    result = pipeline.run()
    assert "labels" in result
    assert "probs" in result

    labels = result["labels"]
    probs = result["probs"]

    assert labels.shape == (20,)
    assert probs.shape[0] == 20
    assert probs.shape[1] == 2

    pipeline.store_regime_features()

    df_after = fs.get_features("hmm_features")
    # check columns exist
    assert "regime_hmm" in df_after.columns
    assert "prob_hmm_0" in df_after.columns
    assert "prob_hmm_1" in df_after.columns

    assert df_after.height == 20
