import pandas as pd
from example_pipeline.utils.logging import get_logger

logger = get_logger(__name__)


def preprocess_main(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["bin_id", "timestamp"])
    df["fill_level"] = df["fill_level"].clip(lower=0, upper=1)
    df["fill_level"] = df.groupby("bin_id")["fill_level"].ffill()
    df = df.dropna(subset=["bin_id", "timestamp", "fill_level"])
    logger.info("Preprocessed main timeseries with %d rows", len(df))
    return df


def join_external(
    df_main: pd.DataFrame,
    df_weather: pd.DataFrame,
    df_calendar: pd.DataFrame,
) -> pd.DataFrame:
    df = df_main.copy()
    df_weather["timestamp"] = pd.to_datetime(df_weather["timestamp"])
    df_calendar["date"] = pd.to_datetime(df_calendar["date"])
    df = pd.merge_asof(
        df.sort_values("timestamp"),
        df_weather.sort_values("timestamp"),
        on="timestamp",
        direction="nearest",
    )
    df["date"] = df["timestamp"].dt.floor("D")
    df = df.merge(df_calendar, on="date", how="left")
    return df
