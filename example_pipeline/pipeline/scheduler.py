from pathlib import Path
import pandas as pd
from example_pipeline.utils.config import CONFIG
from example_pipeline.utils.logging import get_logger

logger = get_logger(__name__)


def generate_schedule() -> Path:
    forecasts_dir = Path(CONFIG["paths"]["forecasts_dir"])
    forecast_path = forecasts_dir / "forecasts.parquet"
    df = pd.read_parquet(forecast_path)
    threshold = CONFIG["forecasting"]["threshold_fill_level"]
    df["collect_flag"] = df["predicted_fill"] >= threshold
    schedule = df[df["collect_flag"]].copy()
    horizon_hours = CONFIG["training"]["target_horizon_hours"]
    schedule["scheduled_date"] = schedule["timestamp"] + pd.to_timedelta(
        horizon_hours, unit="h"
    )
    out_path = forecasts_dir / "collection_schedule.parquet"
    schedule.to_parquet(out_path)
    logger.info("Generated schedule for %d bins", len(schedule))
    return out_path
