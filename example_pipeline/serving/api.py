from pathlib import Path
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from example_pipeline.utils.config import CONFIG
from example_pipeline.utils.logging import get_logger

logger = get_logger(__name__)
app = FastAPI(title="Example Pipeline API")


class BinState(BaseModel):
    bin_id: str
    fill_level: float
    fill_lag_1: Optional[float] = None
    fill_lag_2: Optional[float] = None
    fill_lag_3: Optional[float] = None
    hour: int
    dayofweek: int
    weekofyear: int
    month: int


model_path = Path(CONFIG["paths"]["models_dir"]) / "bin_fill_model.joblib"
model = joblib.load(model_path)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(state: BinState):
    x = [[
        state.fill_level,
        state.fill_lag_1,
        state.fill_lag_2,
        state.fill_lag_3,
        state.hour,
        state.dayofweek,
        state.weekofyear,
        state.month,
    ]]
    pred = float(model.predict(x)[0])
    threshold = CONFIG["forecasting"]["threshold_fill_level"]
    collect = pred >= threshold
    return {
        "bin_id": state.bin_id,
        "predicted_fill": pred,
        "collect": collect,
    }
