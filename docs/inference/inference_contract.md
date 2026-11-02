# Inference contract

This document explains how TARANG creates predictions for one session or a batch of sessions.

## Prediction targets

TARANG predicts:

| Target | Meaning | Unit |
|---|---|---|
| Duration | Total time the vehicle stays connected to the charger. | Hours |
| ConsumedkWh | Energy delivered during the session. | kWh |

The inference package loads trained models. It does not train or update models while making predictions.

## Required model inputs

Each model receives exactly these 13 features in this order:

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

The [data contract](../data/data_contract.md) explains how we calculate these values.

## Single-session prediction

To predict a new session, provide:

- Arrival timestamp in UTC.
- Vehicle charger power in kW.
- Vehicle battery capacity in kWh.
- Completed charging history for the participant.

The history file must contain StartTime, StopTime, and ConsumedkWh. TARANG parses the timestamps as UTC and rejects sessions where StopTime comes before StartTime. It uses the latest session that started before the new arrival. That session must have ended before the new arrival and must have a finite ConsumedkWh value. TARANG does not fill missing values with assumptions.

Example:

    import pandas as pd
    from tarang.inference import build_inference_features, predict_single

    history = pd.DataFrame({
        "StartTime": [pd.Timestamp("2018-06-01 08:00:00", tz="UTC")],
        "StopTime": [pd.Timestamp("2018-06-01 12:00:00", tz="UTC")],
        "ConsumedkWh": [14.2],
    })

    features = build_inference_features(
        start_time="2018-06-01T18:00:00Z",
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

## Batch prediction

Batch inference accepts a CSV or Parquet file with the 13 feature columns. The command writes one prediction column named predicted_Duration or predicted_ConsumedkWh.

    python scripts/run_inference.py \
        --input data/processed/electric_nation/features.parquet \
        --output duration_predictions.parquet \
        --target Duration

Run python scripts/run_inference.py --help to see all options.

The Streamlit Batch Prediction page also requires StartTime so it can calculate the projected unplug timestamp.

## Available models

TARANG supports these model files:

- LinearRegression
- QuantileRegressor
- HistGradientBoostingRegressor
- MedianValue
- MeanValue
- LastValue

HistGradientBoostingRegressor is the default model for the application.

During training, TARANG stores model files under data/processed/electric_nation/modeling/. The application uses the matching models in deployment_artifacts/. The deployment bundle also contains processing_summary.json, which provides the maximum start hour needed to build the features.

## Input checks and output values

TARANG checks feature names, feature order, numeric values, timestamps, and session history before it calls a model. All arrival and history timestamps use UTC.

The feature builder needs the saved maximum start hour from processing_summary.json. It does not calculate a replacement from the prediction input.

The inference API returns the model's raw numeric prediction. It does not round or clip the value. The decision support layer separately marks an estimate as valid when it is finite and non-negative.
