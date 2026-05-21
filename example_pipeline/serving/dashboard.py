import pandas as pd
from pathlib import Path
import streamlit as st

from example_pipeline.utils.config import CONFIG


def load_forecasts() -> pd.DataFrame:
    path = Path(CONFIG["paths"]["forecasts_dir"]) / "forecasts.parquet"
    return pd.read_parquet(path)


def load_schedule() -> pd.DataFrame:
    path = Path(CONFIG["paths"]["forecasts_dir"]) / "collection_schedule.parquet"
    return pd.read_parquet(path)


def main():
    st.title("Example Pipeline - Smart Bin Forecasts")

    forecasts = load_forecasts()
    schedule = load_schedule()

    st.subheader("Forecasts overview")
    st.dataframe(forecasts[["bin_id", "timestamp", "predicted_fill"]])

    st.subheader("Collection schedule")
    st.dataframe(schedule[["bin_id", "scheduled_date", "predicted_fill"]])

    bin_ids = forecasts["bin_id"].unique()
    selected_bin = st.selectbox("Select bin", bin_ids)

    bin_data = forecasts[forecasts["bin_id"] == selected_bin].sort_values("timestamp")
    st.line_chart(bin_data.set_index("timestamp")[["predicted_fill"]])


if __name__ == "__main__":
    main()
