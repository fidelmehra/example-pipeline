"""Run batch forecast and generate collection schedule."""
from example_pipeline.pipeline.batch_forecast import run_batch_forecast
from example_pipeline.pipeline.scheduler import generate_schedule


def main():
    forecast_path = run_batch_forecast()
    schedule_path = generate_schedule()
    print("Forecasts saved to:", forecast_path)
    print("Schedule saved to:", schedule_path)


if __name__ == "__main__":
    main()
