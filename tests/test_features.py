import pandas as pd
import pytest
from example_pipeline.data.features import add_lag_features, add_time_features, create_supervised


def make_sample_df():
    return pd.DataFrame({
        "bin_id": ["A", "A", "A", "A"],
        "timestamp": pd.date_range("2024-01-01", periods=4, freq="h"),
        "fill_level": [0.1, 0.2, 0.3, 0.4],
    })


def test_add_lag_features_columns_created():
    df = make_sample_df()
    df_lagged = add_lag_features(df, lags=[1, 2])
    assert "fill_lag_1" in df_lagged.columns
    assert "fill_lag_2" in df_lagged.columns


def test_add_lag_features_values():
    df = make_sample_df()
    df_lagged = add_lag_features(df, lags=[1])
    # First row should be NaN (no prior value)
    assert pd.isna(df_lagged["fill_lag_1"].iloc[0])
    # Second row should equal first fill level
    assert df_lagged["fill_lag_1"].iloc[1] == pytest.approx(0.1)


def test_add_time_features():
    df = make_sample_df()
    df_tf = add_time_features(df)
    assert "hour" in df_tf.columns
    assert "dayofweek" in df_tf.columns
    assert "month" in df_tf.columns
    assert "weekofyear" in df_tf.columns


def test_fill_level_clipping():
    from example_pipeline.data.preprocessors import preprocess_main
    df = pd.DataFrame({
        "bin_id": ["A", "A"],
        "timestamp": pd.date_range("2024-01-01", periods=2, freq="h"),
        "fill_level": [-0.5, 1.5],
    })
    df_clean = preprocess_main(df)
    assert df_clean["fill_level"].min() >= 0.0
    assert df_clean["fill_level"].max() <= 1.0
