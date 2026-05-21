from pathlib import Path
import joblib
import pandas as pd

from example_pipeline.utils.config import CONFIG
from example_pipeline.utils.logging import get_logger
from example_pipeline.data.loaders import load_main_timeseries, load_weather, load_calendar
from example_pipeline.data.preprocessors import preprocess_main, join_external
from example_pipeline.data.features import add_time_features, add_lag_features
from example_pipeline.models.train import FEATURE_COLUMNS

logger = get_logger(__name__)


def build_features_for_forecast() -> pd.DataFrame:
    df_main = load_main_timeseries()
    df_main = preprocess_main(df_main)
    df_weather = load_weather()
    df_calendar = load_calendar()
    df = join_external(df_main, df_weather, df_calendar)
    df = add_time_features(df)
    df = add_lag_features(df)
    df_latest = df.sort_values("timestamp").groupby("bin_id").tail(1)
    return df_latest


def run_batch_forecast() -> Path:
    df_latest = build_features_for_forecast()
    model_path = Path(CONFIG["paths"]["models_dir"]) / "bin_fill_model.joblib"
    model = joblib.load(model_path)
    X = df_latest[FEATURE_COLUMNS]
    df_latest = df_latest.copy()
    df_latest["predicted_fill"] = model.predict(X)
    forecasts_dir = Path(CONFIG["paths"]["forecasts_dir"])
    forecasts_dir.mkdir(parents=True, exist_ok=True)
    out_path = forecasts_dir / "forecasts.parquet"
    df_latest.to_parquet(out_path)
    logger.info("Saved forecasts to %s", out_path)
    return out_path
