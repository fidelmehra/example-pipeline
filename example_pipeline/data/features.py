import pandas as pd
from example_pipeline.utils.logging import get_logger
from example_pipeline.utils.config import CONFIG

logger = get_logger(__name__)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["weekofyear"] = df["timestamp"].dt.isocalendar().week.astype(int)
    df["month"] = df["timestamp"].dt.month
    return df


def add_lag_features(df: pd.DataFrame, lags: list = [1, 2, 3]) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["bin_id", "timestamp"])
    for lag in lags:
        df[f"fill_lag_{lag}"] = df.groupby("bin_id")["fill_level"].shift(lag)
    return df


def create_supervised(df: pd.DataFrame) -> pd.DataFrame:
    """Create supervised dataset: target is fill_level at t + horizon."""
    horizon_hours = CONFIG["training"]["target_horizon_hours"]
    df = df.copy()
    df = df.sort_values(["bin_id", "timestamp"])
    df["timestamp_target"] = df["timestamp"] + pd.to_timedelta(horizon_hours, unit="h")
    target = df[["bin_id", "timestamp_target", "fill_level"]].rename(
        columns={"timestamp_target": "timestamp", "fill_level": "target_fill"}
    )
    df_supervised = df.merge(target, on=["bin_id", "timestamp"], how="left")
    df_supervised = df_supervised.dropna(subset=["target_fill"])
    logger.info("Created supervised dataset with %d rows", len(df_supervised))
    return df_supervised
