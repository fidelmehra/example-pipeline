from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from example_pipeline.utils.config import CONFIG
from example_pipeline.utils.logging import get_logger

logger = get_logger(__name__)

FEATURE_COLUMNS = [
    "fill_level",
    "fill_lag_1",
    "fill_lag_2",
    "fill_lag_3",
    "hour",
    "dayofweek",
    "weekofyear",
    "month",
]

TARGET_COLUMN = "target_fill"


def temporal_train_val_split(df: pd.DataFrame) -> tuple:
    train_start = CONFIG["training"]["train_start"]
    train_end = CONFIG["training"]["train_end"]
    val_start = CONFIG["training"]["val_start"]
    val_end = CONFIG["training"]["val_end"]
    train = df[(df["timestamp"] >= train_start) & (df["timestamp"] <= train_end)]
    val = df[(df["timestamp"] >= val_start) & (df["timestamp"] <= val_end)]
    return train, val


def train_model(df_supervised: pd.DataFrame) -> dict:
    train_df, val_df = temporal_train_val_split(df_supervised)
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]
    X_val = val_df[FEATURE_COLUMNS]
    y_val = val_df[TARGET_COLUMN]
    params = CONFIG["model"]["params"]
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    logger.info("Validation MAE: %.4f, RMSE: %.4f", mae, rmse)
    models_dir = Path(CONFIG["paths"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "bin_fill_model.joblib"
    joblib.dump(model, model_path)
    return {"model_path": str(model_path), "mae": mae, "rmse": rmse}
