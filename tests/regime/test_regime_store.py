# tests/regime/test_regime_store.py

import numpy as np
import pandas as pd

from quant_platform.data.feature_store.store import FeatureStore
from quant_platform.data.regime_store.store import RegimeFeatureStore


def test_regime_store_in_memory():
    labels = np.array([0, 1, 0])
    probs = np.array([[0.8, 0.2], [0.1, 0.9], [0.7, 0.3]])

    store = RegimeFeatureStore()
    store.save_regime("test_feat", "hmm", labels, probs)

    df = store.load_regime("test_feat", "hmm", as_pandas=True)

    assert "regime_hmm" in df.columns
    assert "prob_hmm_0" in df.columns
    assert "prob_hmm_1" in df.columns
    assert len(df) == 3


def test_regime_store_with_feature_store():
    dates = pd.date_range("2020-01-01", periods=4)
    fs = FeatureStore()

    base = pd.DataFrame({"date": dates, "x": [1, 2, 3, 4]})

    fs.save_features("featA", base)

    labels = np.array([0, 1, 1, 0])
    probs = np.array(
        [
            [0.9, 0.1],
            [0.2, 0.8],
            [0.3, 0.7],
            [0.85, 0.15],
        ]
    )

    rstore = RegimeFeatureStore(feature_store=fs)
    rstore.save_regime("featA", "hmm", labels, probs)

    df = fs.get_features("featA")

    assert "regime_hmm" in df.columns
    assert "prob_hmm_0" in df.columns
    assert "prob_hmm_1" in df.columns
    assert df.height == 4

    # load_regime should filter only regime columns
    df_r = rstore.load_regime("featA", "hmm", as_pandas=True)
    assert "regime_hmm" in df_r.columns
    assert df_r.shape[0] == 4
