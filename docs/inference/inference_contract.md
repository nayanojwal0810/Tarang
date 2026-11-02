# Inference contract

This document describes how TARANG creates predictions for one session or a batch of sessions.

## Prediction targets

TARANG predicts:

| Target | Meaning | Unit |
|---|---|---|
| `Duration` | Total time the vehicle stays connected to the charger. | Hours |
| `ConsumedkWh` | Energy delivered during the session. | kWh |

The inference package loads trained models. It does not train or update them while making predictions.

## Required model inputs

Every prediction uses exactly these 13 features in this order:

```text
weekday
weekend
holiday
month
StartCos
StartSin
StartHourNorm
TimeSinceLastStop
sessionsToday
LastDuration
LastConsumedkWh
CarKW
CarKWh
```

The feature definitions are in the [data contract](../data/data_contract.md).

## Single-session prediction

To predict a new session, provide:

- The arrival timestamp in UTC.
- The vehicle charger rating in kW.
- The vehicle battery capacity in kWh.
- The participant's completed session history.

The history must contain `StartTime`, `StopTime`, and `ConsumedkWh`. TARANG uses the previous completed session to build the history features. It rejects empty history, overlapping sessions, missing required values, and invalid inputs rather than filling them with assumed values.

Example:

```python
import pandas as pd
from tarang.inference import build_inference_features, predict_single

history = pd.DataFrame({
    "StartTime": [pd.Timestamp("2018-06-01 08:00:00", tz="UTC")],
    "StopTime": [pd.Timestamp("2018-06-01 12:00:00", tz="UTC")],
    "ConsumedkWh": [14.2],
})

features = build_inference_features(
    start_time="2018-06-01 18:00:00",
    car_kw=7.0,
    car_kwh=40.0,
    history=history,
)

duration_hours = predict_single(
    features=features,
    target="Duration",
    model_name="HistGradientBoostingRegressor",
)
print(duration_hours)
```

## Batch prediction

Batch inference accepts a CSV or Parquet file with the 13 features. It preserves the input rows and adds one prediction column named `predicted_Duration` or `predicted_ConsumedkWh`.

```bash
python scripts/run_inference.py \
    --input data/processed/electric_nation/features.parquet \
    --output duration_predictions.parquet \
    --target Duration
```

Use `python scripts/run_inference.py --help` to see all options.

## Models

TARANG supports these saved models:

- `LinearRegression`
- `QuantileRegressor`
- `HistGradientBoostingRegressor`
- `MedianValue`
- `MeanValue`
- `LastValue`

The default model is `HistGradientBoostingRegressor`. The model files must be present under `data/processed/electric_nation/modeling/`.

## Input checks and prediction values

TARANG checks the feature names, feature order, numeric values, timestamps, and session history before calling a model. All arrival and history timestamps use UTC. The global start-hour maximum must be available in `processing_summary.json`; inference does not estimate a replacement value from a prediction batch.

The inference API returns the estimator's raw numeric prediction. It does not clip or round the value. The decision support layer separately marks estimates as valid or invalid.
