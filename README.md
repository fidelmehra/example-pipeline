# example-pipeline

End-to-end ML pipeline for smart waste management forecasting and scheduling

## Overview

This project implements a production-ready machine learning pipeline for predicting waste generation volumes and optimising collection schedules across urban zones. It combines time series forecasting with route optimisation to reduce operational costs and improve sustainability metrics.

## Features

- **Data ingestion** – loads historical fill-level sensor readings and weather/calendar features
- **Feature engineering** – rolling statistics, Fourier seasonality terms, lag features
- **Forecasting models** – Prophet baseline, XGBoost, and LSTM ensemble
- **Route optimisation** – greedy and OR-Tools-based scheduler
- **REST API serving** – FastAPI endpoint for real-time predictions
- **Full test suite** – pytest-based unit and integration tests

## Project Structure

```
example-pipeline/
├── config/
│   └── config.yml            # Centralised configuration
├── example_pipeline/
│   ├── __init__.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── helpers.py        # Logging, timing, seed utilities
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loaders.py        # Data loading and validation
│   │   └── features.py       # Feature engineering
│   ├── models/
│   │   ├── __init__.py
│   │   ├── forecaster.py     # Prophet / XGBoost / LSTM models
│   │   └── ensemble.py       # Weighted ensemble and stacking
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── orchestrator.py   # End-to-end pipeline runner
│   └── serving/
│       ├── __init__.py
│       └── api.py            # FastAPI prediction endpoint
├── scripts/
│   ├── train.py              # CLI training script
│   └── predict.py            # CLI inference script
├── tests/
│   └── test_features.py      # Unit tests for feature engineering
├── .gitignore
├── README.md
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/fidelmehra/example-pipeline.git
cd example-pipeline
pip install -r requirements.txt
```

## Usage

### Train

```bash
python scripts/train.py --config config/config.yml
```

### Predict

```bash
python scripts/predict.py --config config/config.yml --input data/new_readings.csv
```

### Serve API

```bash
uvicorn example_pipeline.serving.api:app --reload
```

Then POST to `http://localhost:8000/predict` with JSON payload.

### Run Tests

```bash
pytest tests/ -v
```

## Configuration

All pipeline parameters live in `config/config.yml`:

| Section | Key parameters |
|---|---|
| `data` | `raw_path`, `processed_path`, `target_col`, `date_col` |
| `features` | `lag_periods`, `rolling_windows`, `fourier_order` |
| `model` | `type` (xgboost/prophet/lstm/ensemble), hyperparameters |
| `training` | `test_size`, `cv_folds`, `random_seed` |
| `serving` | `host`, `port`, `model_path` |

## Tech Stack

- **Python 3.10+**
- pandas, numpy, scikit-learn
- XGBoost, Prophet, PyTorch (LSTM)
- FastAPI + Uvicorn
- pytest

## License

MIT
