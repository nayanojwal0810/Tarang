# Decision support summary

This report summarizes the duration and energy predictions and the historical charging flexibility results for the processed dataset.

## Session predictions

TARANG used HistGradientBoostingRegressor to predict both targets for 59,520 sessions.

| Target | Unit | Sessions | Valid predictions | Invalid predictions | Mean | Median | 90th percentile |
|---|---|---:|---:|---:|---:|---:|---:|
| Duration | Hours | 59,520 | 59,518 | 2 | 10.05 | 11.47 | 14.62 |
| Energy | kWh | 59,520 | 59,520 | 0 | 10.23 | 7.66 | 18.65 |

A valid prediction is finite and non-negative. TARANG keeps the original model output. It does not clip or replace invalid values.

There are 59,518 valid duration predictions, so the same number of projected unplug timestamps are available. TARANG calculates each timestamp by adding the predicted duration to the session arrival time. An invalid duration prediction produces `NaT`.

## Historical charging flexibility

The flexibility ratio compares active charging time with the total time a vehicle stayed connected. The analysis uses UK local dates to group same-day and overnight sessions.

| Session type | Total sessions | Valid ratios | Ratios in 0–1 histogram | Mean ratio | Median ratio |
|---|---:|---:|---:|---:|---:|
| Same-day | 21,700 | 15,947 | 15,947 | 0.689 | 0.780 |
| Overnight | 37,820 | 29,349 | 29,348 | 0.188 | 0.143 |

The summary statistics use all valid ratios. The histogram includes only ratios from 0 to 1. One valid overnight ratio is above 1, so the valid and histogram counts differ by one.

Overnight sessions have a lower median active-charging share than same-day sessions in these records. This describes historical timing only. It does not confirm that the remaining connection time or energy can be shifted.

## Scope

TARANG provides predictions and summary information for review. It does not:

- Choose a charging start time or create a vehicle schedule.
- Calculate a feasible amount of energy that can be delayed.
- Optimize tariffs, grid demand, or carbon use.
- Send commands to chargers or control charging power.

## Output files

- `prediction_summary.csv`: Prediction counts and distribution statistics.
- `flexibility_summary.csv`: Same-day and overnight ratio counts and summaries.
- The batch advisory dataset is stored under `data/processed/electric_nation/decision_support/`.
