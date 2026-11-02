# Duration prediction results

This report summarizes how well TARANG predicts plugged-in connection time. The target is measured in hours and capped at 336 hours in the prepared dataset.

## Evaluation data

The evaluation uses 59,520 sessions from 267 participants. TARANG trains on the earlier 80% of sessions (47,616) and tests on the later 20% (11,904).

## Test results

The table shows errors across all held-out test sessions.

| Model | MAE (h) | Median AE (h) | MAPE (%) | Median APE (%) | sMAPE (%) |
|---|---:|---:|---:|---:|---:|
| LinearRegression | 6.45 | 4.05 | 279.4 | 34.5 | 61.2 |
| QuantileRegressor | 6.08 | 3.06 | 223.8 | 31.3 | 59.2 |
| HistGradientBoostingRegressor | 5.53 | 1.92 | 200.5 | 23.4 | 52.8 |
| MedianValue | 7.08 | 5.30 | 314.9 | 36.0 | 66.5 |
| MeanValue | 7.08 | 5.07 | 329.5 | 34.5 | 65.8 |
| LastValue | 8.93 | 5.02 | 294.4 | 57.1 | 78.3 |

HistGradientBoostingRegressor has the lowest error for each metric in this comparison. Its mean absolute error is 5.53 hours and its median absolute error is 1.92 hours.

Across participants, this model's average MAE is 6.20 hours and its average median absolute error is 3.28 hours.

MAE is the average absolute error. Median AE is the middle absolute error. MAPE and median APE express error as a percentage, while sMAPE uses a symmetric percentage calculation. Percentage errors can become large when actual session durations are small, so review them together with the hour-based errors.

## Same-day and overnight sessions

TARANG also evaluates separate models for sessions grouped by UTC calendar date.

| Group | Sessions | Test sessions | MAE (h) | Median AE (h) |
|---|---:|---:|---:|---:|
| All sessions | 59,520 | 11,904 | 5.53 | 1.92 |
| Same-day | 21,436 | 4,288 | 1.35 | 1.02 |
| Overnight | 38,084 | 7,617 | 4.05 | 1.13 |

These results come from separate same-day and overnight models. They are not a direct comparison of one model applied to both groups.

## Features linked to model predictions

The feature importance analysis reports the average absolute SHAP value. The largest values for the duration model are:

| Feature | Mean absolute SHAP value |
|---|---:|
| StartSin | 2.2364 |
| LastDuration | 1.3947 |
| CarKWh | 0.6599 |
| StartHourNorm | 0.6549 |
| StartCos | 0.6021 |

These values describe how strongly each feature contributes to model predictions. They do not show that a feature causes a charging outcome.

## Additional evaluation

The project also reviews participant-level errors, model agreement, changes in error over time, seasonal patterns, and predicted versus observed occupancy. The supporting figures and CSV results are in this folder.

- [Model performance](performance.png)
- [Error distributions](best_model.png)
- [Feature importance](shap_values.png)
- [Same-day and overnight results](same_day_overnight_comparison.png)
- [Participant occupancy examples](heatmap-best-0.png), [second example](heatmap-best-1.png), [third example](heatmap-best-2.png)
