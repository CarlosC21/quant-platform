# tests/test_feature_store.py
import polars as pl
from polars.testing import assert_frame_equal
import pytest

from quant_platform.data.feature_store.store import FeatureStore


def test_feature_store_save_get():
    fs = FeatureStore()
    df = pl.DataFrame({"a": [1, 2, 3]})
    fs.save_features("test", df)

    df_out = fs.get_features("test")
    assert_frame_equal(df_out, df)
    assert "test" in fs.list_features()


# ----------------------------
# NEW TESTS FOR LAG & ROLLING
# ----------------------------
def test_add_lag_feature():
    fs = FeatureStore()
    df = pl.DataFrame({"returns": [0.1, 0.2, -0.1, 0.3, -0.2]})
    fs.save_features("lag_test", df)

    fs.add_lag_feature("lag_test", "returns", lag=1)
    df_out = fs.get_features("lag_test")

    # Check that new column exists
    assert "returns_lag1" in df_out.columns

    # Check values (shifted by 1)
    expected = [None, 0.1, 0.2, -0.1, 0.3]
    # Polars uses None for missing shift
    assert df_out["returns_lag1"].to_list() == expected


def test_add_rolling_feature_mean():
    fs = FeatureStore()
    df = pl.DataFrame({"returns": [1, 2, 3, 4, 5]})
    fs.save_features("roll_test", df)

    fs.add_rolling_feature("roll_test", "returns", window=3, func="mean")
    df_out = fs.get_features("roll_test")

    col_name = "returns_roll3_mean"
    assert col_name in df_out.columns

    # Polars rolling_mean fills initial NAs with None
    expected = [None, None, 2.0, 3.0, 4.0]
    assert df_out[col_name].to_list() == expected


def test_add_rolling_feature_std():
    fs = FeatureStore()
    df = pl.DataFrame({"returns": [1, 2, 3, 4, 5]})
    fs.save_features("roll_std_test", df)

    fs.add_rolling_feature("roll_std_test", "returns", window=2, func="std")
    df_out = fs.get_features("roll_std_test")

    col_name = "returns_roll2_std"
    assert col_name in df_out.columns

    # Polars rolling_std uses None for initial NA
    expected = [
        None,
        0.7071067811865476,
        0.7071067811865476,
        0.7071067811865476,
        0.7071067811865476,
    ]
    assert all(
        (a is None and b is None) or (abs(a - b) < 1e-12)
        for a, b in zip(df_out[col_name].to_list(), expected)
    )


def test_add_lag_feature_nonexistent_raises():
    fs = FeatureStore()
    df = pl.DataFrame({"returns": [1, 2, 3]})
    fs.save_features("x", df)

    with pytest.raises(ValueError):
        fs.add_lag_feature("x", "nonexistent", lag=1)


def test_add_rolling_feature_invalid_func_raises():
    fs = FeatureStore()
    df = pl.DataFrame({"returns": [1, 2, 3]})
    fs.save_features("y", df)

    with pytest.raises(ValueError):
        fs.add_rolling_feature("y", "returns", window=2, func="median")
