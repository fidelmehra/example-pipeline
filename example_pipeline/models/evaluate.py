from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from example_pipeline.utils.config import CONFIG
from example_pipeline.utils.logging import get_logger
from example_pipeline.models.train import FEATURE_COLUMNS, TARGET_COLUMN, temporal_train_val_split

logger = get_logger(__name__)


def evaluate_model(df_supervised: pd.DataFrame) -> dict:
    _, val_df = temporal_train_val_split(df_supervised)
    model_path = Path(CONFIG["paths"]["models_dir"]) / "bin_fill_model.joblib"
    model = joblib.load(model_path)
    X_val = val_df[FEATURE_COLUMNS]
    y_val = val_df[TARGET_COLUMN]
    y_pred = model.predict(X_val)
    r2 = r2_score(y_val, y_pred)
    logger.info("Validation R^2: %.4f", r2)
    fig, ax = plt.subplots()
    ax.scatter(y_val, y_pred, alpha=0.3)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted Fill Level")
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True, parents=True)
    fig_path = output_dir / "actual_vs_predicted.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    return {"r2": r2, "plot_path": str(fig_path)}
