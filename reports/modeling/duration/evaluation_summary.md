# Duration model evaluation

We evaluate how well TARANG predicts the time between plug-in and unplug. The model target is measured in hours and capped at 336 hours during data preparation.

## Evaluation data

We use 59,520 sessions from 267 participants. We sort sessions by arrival time, train on the earlier 80% (47,616 sessions), and evaluate on the later 20% (11,904 sessions).

## Test results

| Model | MAE (h) | Median AE (h) | MAPE (%) | MdAPE (%) | sMAPE (%) |
|---|---:|---:|---:|---:|---:|
| LinearRegression | 6.45 | 4.05 | 279.4 | 34.5 | 61.2 |
| QuantileRegressor | 6.08 | 3.06 | 223.8 | 31.3 | 59.2 |
| HistGradientBoostingRegressor | 5.53 | 1.92 | 200.5 | 23.4 | 52.8 |
| MedianValue | 7.08 | 5.30 | 314.9 | 36.0 | 66.5 |
| MeanValue | 7.08 | 5.07 | 329.5 | 34.5 | 65.8 |
| LastValue | 8.93 | 5.02 | 294.4 | 57.1 | 78.3 |

HistGradientBoostingRegressor has the lowest error across the five metrics in this comparison. Its mean absolute error is 5.53 hours. Its median absolute error is 1.92 hours.

Across participants, the model's average MAE is 6.20 hours and its average median absolute error is 3.28 hours.

MAE measures the average absolute difference between predictions and actual values. Median AE reports the middle absolute error. MAPE and MdAPE express percentage errors; sMAPE uses a symmetric percentage calculation. Percentage errors can become large for short sessions, so read them together with the hour-based errors.

## Same-day and overnight sessions

We train and test separate models for these two groups. We classify a session as same-day when its start and stop fall on the same UTC date. Otherwise, we classify it as overnight.

| Group | Sessions | Test sessions | MAE (h) | Median AE (h) |
|---|---:|---:|---:|---:|
| All sessions | 59,520 | 11,904 | 5.53 | 1.92 |
| Same-day | 21,436 | 4,288 | 1.35 | 1.02 |
| Overnight | 38,084 | 7,617 | 4.05 | 1.13 |

The subgroup results come from separate models. They are not a comparison of one model on two groups.

## Features linked to predictions

We use SHAP values to measure how much each feature contributes to the model's predictions. The highest global mean absolute values are:

| Feature | Mean absolute SHAP value |
|---|---:|
| StartSin | 2.2364 |
| LastDuration | 1.3947 |
| CarKWh | 0.6599 |
| StartHourNorm | 0.6549 |
| StartCos | 0.6021 |

These values describe contributions to predictions. They do not show that a feature causes a charging outcome.

## Supporting results

- [Model performance table](model_performance.csv)
- [Same-day and overnight results](same_day_overnight_metrics.csv)
- [Global feature importance](shap_global_importance.csv)
- [Model performance figure](performance.png)
- [Prediction error distribution](best_model.png)
- [Feature importance figure](shap_values.png)
- [Same-day and overnight comparison](same_day_overnight_comparison.png)
- [Participant occupancy examples](heatmap-best-0.png), [second example](heatmap-best-1.png), [third example](heatmap-best-2.png)
