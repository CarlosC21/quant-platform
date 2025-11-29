import pandas as pd
from quant_platform.ml.cv import generate_time_series_splits


def test_generate_time_series_splits_basic():
    dates = pd.date_range("2020-01-01", periods=60, freq="D")
    ts = pd.Series(dates)
    splits = list(
        generate_time_series_splits(ts, n_splits=3, val_window=5, embargo_days=1)
    )
    assert len(splits) >= 1
    for train_idx, val_idx in splits:
        # ensure index order preserved
        assert max(train_idx) < min(val_idx)
