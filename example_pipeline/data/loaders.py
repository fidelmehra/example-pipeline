from pathlib import Path
import pandas as pd
from example_pipeline.utils.config import CONFIG


def load_main_timeseries() -> pd.DataFrame:
    path = Path(CONFIG["paths"]["raw_data_dir"]) / CONFIG["data"]["main_table"]
    return pd.read_parquet(path)


def load_weather() -> pd.DataFrame:
    path = Path(CONFIG["paths"]["raw_data_dir"]) / CONFIG["data"]["weather_table"]
    return pd.read_parquet(path)


def load_calendar() -> pd.DataFrame:
    path = Path(CONFIG["paths"]["raw_data_dir"]) / CONFIG["data"]["calendar_table"]
    return pd.read_parquet(path)
