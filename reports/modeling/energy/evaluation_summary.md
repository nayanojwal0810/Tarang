# Energy model evaluation

We evaluate how well TARANG predicts the energy delivered during a charging session. The target is measured in kilowatt-hours (kWh).

## Evaluation data

We use 59,520 sessions from 267 participants. We sort sessions by arrival time, train on the earlier 80% (47,616 sessions), and evaluate on the later 20% (11,904 sessions).

## Test results

| Model | MAE (kWh) | Median AE (kWh) | MAPE (%) | MdAPE (%) | sMAPE (%) |
|---|---:|---:|---:|---:|---:|
| LinearRegression | 4.44 | 2.56 | 299.9 | 32.5 | 46.4 |
| QuantileRegressor | 4.34 | 2.46 | 287.3 | 31.6 | 45.4 |
| HistGradientBoostingRegressor | 3.83 | 1.97 | 277.2 | 25.0 | 39.4 |
| MedianValue | 6.00 | 3.24 | 236.1 | 49.3 | 59.4 |
| MeanValue | 6.91 | 5.40 | 355.0 | 64.4 | 67.4 |
| LastValue | 5.10 | 2.73 | 329.8 | 37.2 | 54.4 |

HistGradientBoostingRegressor has the lowest MAE, median absolute error, MdAPE, and sMAPE in this comparison. Its MAE is 3.83 kWh and its median absolute error is 1.97 kWh. MedianValue has the lowest MAPE at 236.1%.

Across participants, the gradient boosting model has an average MAE of 4.84 kWh and an average median absolute error of 4.13 kWh.

MAE measures the average absolute difference between predictions and actual values. Median AE reports the middle absolute error. MAPE and MdAPE express percentage errors; sMAPE uses a symmetric percentage calculation. Percentage errors can become large when actual energy use is small, so read them together with the kWh-based errors.

## Same-day and overnight sessions

We train and test separate models for these two groups. We classify a session as same-day when its start and stop fall on the same UTC date. Otherwise, we classify it as overnight.

| Group | Sessions | Test sessions | MAE (kWh) | Median AE (kWh) |
|---|---:|---:|---:|---:|
| All sessions | 59,520 | 11,904 | 3.83 | 1.97 |
| Same-day | 21,436 | 4,288 | 2.88 | 1.95 |
| Overnight | 38,084 | 7,617 | 3.94 | 1.85 |

The subgroup results come from separate models. They are not a comparison of one model on two groups.

## Features linked to predictions

We use SHAP values to measure how much each feature contributes to the model's predictions. The highest global mean absolute values are:

| Feature | Mean absolute SHAP value |
|---|---:|
| CarKWh | 2.8210 |
| LastConsumedkWh | 2.0676 |
| TimeSinceLastStop | 1.4485 |
| StartHourNorm | 0.7093 |
| LastDuration | 0.5628 |

These values describe contributions to predictions. They do not show that a feature causes an energy outcome.

## Supporting results

- [Model performance table](model_performance.csv)
- [Same-day and overnight results](same_day_overnight_metrics.csv)
- [Global feature importance](shap_global_importance.csv)
- [Model performance figure](performance.png)
- [Prediction error distribution](best_model.png)
- [Feature importance figure](shap_values.png)
- [Same-day and overnight comparison](same_day_overnight_comparison.png)
