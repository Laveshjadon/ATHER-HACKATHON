# UP Heat-Risk Forecaster

Prototype for the **IIIT Lucknow Climate Intelligence Challenge 2026** (hosted by Aether,
the AI/ML Club of IIIT Lucknow) — 10-day-ahead Wet-Bulb Temperature (WBT) forecasting
across 125 districts of Uttar Pradesh & neighboring regions.

## Problem recap

WBT combines temperature and humidity into a single heat-stress signal:

| WBT | Risk |
|---|---|
| < 28°C | Safe |
| 28–32°C | Caution |
| 32–35°C | Danger |
| > 35°C | Fatal (unsurvivable beyond ~6h) |

Given 30 days of daily meteorological features per district, predict max WBT for each of
the next 10 days. Scored with weighted RMSE (weights decay linearly 1.0 → 0.1, Day 1 → Day 10).

## Approach

A **LightGBM** regressor predicts same-day WBT from NASA POWER features (temperature,
humidity, soil moisture, radiation, wind) plus engineered lag/rolling features
(1/3/7-day lags, 7-day rolling mean/std, day-over-day deltas), computed per district.

Because the meteorological drivers for the test period are themselves known NASA
reanalysis values (not forecasts), the model scores every day directly rather than
recursively forecasting 10 steps forward — this avoids compounding error. The 10-day-ahead
target vector for a given day is simply the model's same-day predictions on the following
10 days. `context.csv` seeds each district's first 30-day lag window before the test period
begins, exactly as the competition rules require.

This project also has an earlier deep-learning experiment (`experiments/weather_tcn_lstm.py`,
LSTM + LightGBM + CatBoost ensemble, and `experiments/competition_pipeline.py`) exploring the
same problem — the LightGBM path in `model/` was kept for the deployed prototype because it
trains fast on CPU and hits comparable accuracy.

## Architecture

```mermaid
flowchart LR
    A["Training data\n(train.csv, 1984-2014)"] --> B["Feature engineering\n(lags, rolling stats,\npsychrometric features)"]
    B --> C["LightGBM model\n(same-day WBT)"]
    C --> D["Trained model\n+ metrics"]

    E["Context + test data\n(2015-2025)"] --> F["Score every day\n+ shift 10-day targets"]
    D --> F
    F --> G["submission.csv"]
    F --> H["Streamlit dashboard"]

    D --> I["Presentation deck\n+ demo video"]
    H --> I
```

## Running it locally

```bash
pip install -r app/requirements.txt lightgbm scikit-learn joblib
py model/train_model.py     # trains model, ~3 min on CPU
py model/score_test.py      # writes results/submission.csv + app/data/*
streamlit run app/app.py    # opens the dashboard at http://localhost:8501
```

`data/train.csv` is not committed to this repo (262MB, over GitHub's 100MB file limit) —
download it from the competition's Kaggle dataset page and place it at `data/train.csv`
before running `model/train_model.py`.

## Team

Built by **Lavesh Jadon**, **Gaurav Jha**, and **Akash Bernwal** for the IIIT Lucknow
Climate Intelligence Challenge 2026 (Aether).
