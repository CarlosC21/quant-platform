import polars as pl
from polars.testing import assert_frame_equal

from src.quant_platform.data.feature_store.store import FeatureStore


def test_feature_store_save_get():
    fs = FeatureStore()
    df = pl.DataFrame({"a": [1, 2, 3]})
    fs.save_features("test", df)

    df_out = fs.get_features("test")

    # Compare using polars.testing
    assert_frame_equal(df_out, df)

    assert "test" in fs.list_features()
