# Energy prediction results

This report summarizes how well TARANG predicts the energy used during a charging session. The target is measured in kilowatt-hours (kWh).

## Evaluation data

The evaluation uses 59,520 sessions from 267 participants. TARANG trains on the earlier 80% of sessions (47,616) and tests on the later 20% (11,904).

## Test results

The table shows errors across all held-out test sessions.

| Model | MAE (kWh) | Median AE (kWh) | MAPE (%) | Median APE (%) | sMAPE (%) |
|---|---:|---:|---:|---:|---:|
| LinearRegression | 4.44 | 2.56 | 299.9 | 32.5 | 46.4 |
| QuantileRegressor | 4.34 | 2.46 | 287.3 | 31.6 | 45.4 |
| HistGradientBoostingRegressor | 3.83 | 1.97 | 277.2 | 25.0 | 39.4 |
| MedianValue | 6.00 | 3.24 | 236.1 | 49.3 | 59.4 |
| MeanValue | 6.91 | 5.40 | 355.0 | 64.4 | 67.4 |
| LastValue | 5.10 | 2.73 | 329.8 | 37.2 | 54.4 |

HistGradientBoostingRegressor has the lowest MAE, median absolute error, median absolute percentage error, and sMAPE. MedianValue has the lowest MAPE at 236.1%.

The gradient boosting model has an average MAE of 4.84 kWh and an average median absolute error of 4.13 kWh across participants.

MAE is the average absolute error. Median AE is the middle absolute error. MAPE and median APE express error as a percentage, while sMAPE uses a symmetric percentage calculation. Percentage errors can become large when actual energy values are small, so review them together with the kWh-based errors.

## Same-day and overnight sessions

TARANG also evaluates separate models for sessions grouped by UTC calendar date.

| Group | Sessions | Test sessions | MAE (kWh) | Median AE (kWh) |
|---|---:|---:|---:|---:|
| All sessions | 59,520 | 11,904 | 3.83 | 1.97 |
| Same-day | 21,436 | 4,288 | 2.88 | 1.95 |
| Overnight | 38,084 | 7,617 | 3.94 | 1.85 |

These results come from separate same-day and overnight models. They are not a direct comparison of one model applied to both groups.

## Features linked to model predictions

The feature importance analysis reports the average absolute SHAP value. The largest values for the energy model are:

| Feature | Mean absolute SHAP value |
|---|---:|
| CarKWh | 2.8210 |
| LastConsumedkWh | 2.0676 |
| TimeSinceLastStop | 1.4485 |
| StartHourNorm | 0.7093 |
| LastDuration | 0.5628 |

These values describe how strongly each feature contributes to model predictions. They do not show that a feature causes an energy outcome.

## Additional evaluation

The project also reviews participant-level errors, model agreement, changes in error over time, seasonal patterns, and same-day and overnight performance. The supporting figures and CSV results are in this folder.

- [Model performance](performance.png)
- [Error distributions](best_model.png)
- [Feature importance](shap_values.png)
- [Same-day and overnight results](same_day_overnight_comparison.png)
