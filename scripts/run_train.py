"""Train and evaluate the bin fill-level model."""
from pathlib import Path
import pandas as pd

from example_pipeline.utils.config import CONFIG
from example_pipeline.models.train import train_model
from example_pipeline.models.evaluate import evaluate_model


def main():
    processed_dir = Path(CONFIG["paths"]["processed_data_dir"])
    df_supervised = pd.read_parquet(processed_dir / "supervised.parquet")
    train_info = train_model(df_supervised)
    eval_info = evaluate_model(df_supervised)
    print("Train:", train_info)
    print("Eval:", eval_info)


if __name__ == "__main__":
    main()
