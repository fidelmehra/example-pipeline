"""ETL script: load raw data, preprocess, build features, save supervised dataset."""
from pathlib import Path
import pandas as pd

from example_pipeline.utils.config import CONFIG
from example_pipeline.utils.logging import get_logger
from example_pipeline.data.loaders import load_main_timeseries, load_weather, load_calendar
from example_pipeline.data.preprocessors import preprocess_main, join_external
from example_pipeline.data.features import add_time_features, add_lag_features, create_supervised

logger = get_logger(__name__)


def main():
    df_main = load_main_timeseries()
    df_main = preprocess_main(df_main)
    df_weather = load_weather()
    df_calendar = load_calendar()
    df = join_external(df_main, df_weather, df_calendar)
    df = add_time_features(df)
    df = add_lag_features(df)
    df_supervised = create_supervised(df)
    processed_dir = Path(CONFIG["paths"]["processed_data_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / "supervised.parquet"
    df_supervised.to_parquet(out_path)
    logger.info("Saved supervised dataset to %s", out_path)


if __name__ == "__main__":
    main()
