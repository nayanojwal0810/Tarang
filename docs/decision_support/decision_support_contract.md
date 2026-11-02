# Decision support contract

This document describes the session information TARANG creates from duration and energy predictions.

## Purpose and limits

The decision support package combines two predictions for a charging session:

- Expected plugged-in duration, in hours.
- Expected energy use, in kWh.

It also calculates a projected unplug time and summarizes historical charging activity. These outputs support review and planning. TARANG does not schedule vehicles, control charging power, send commands to chargers, or optimize tariffs or grid dispatch.

## Single-session advisory

The `SessionAdvisory` result contains:

| Field | Meaning |
|---|---|
| `arrival_time_utc` | Session arrival time in UTC. |
| `predicted_duration_hours` | Raw duration estimate from the model. |
| `predicted_energy_kwh` | Raw energy estimate from the model. |
| `projected_unplug_time_utc` | Arrival time plus the duration estimate, when that estimate is valid. |
| `duration_prediction_valid` | True when the duration estimate is finite and non-negative. |
| `energy_prediction_valid` | True when the energy estimate is finite and non-negative. |

TARANG keeps the original estimates. It does not replace negative or non-finite values with a different prediction. If the duration estimate is invalid, the projected unplug time is `NaT`.

## Batch advisory

Batch input must contain `StartTime` and the 13 model features. TARANG retains the input rows, order, and index, then adds these columns:

| Output column | Meaning |
|---|---|
| `predicted_Duration` | Predicted plugged-in duration in hours. |
| `predicted_ConsumedkWh` | Predicted energy in kWh. |
| `projected_unplug_time_utc` | Projected unplug timestamp in UTC, or `NaT` when duration is invalid. |
| `duration_prediction_valid` | Numeric validity flag for the duration estimate. |
| `energy_prediction_valid` | Numeric validity flag for the energy estimate. |

Both predictions use the model selected when the decision support engine is created.

## Historical charging flexibility

TARANG calculates a descriptive ratio from session records:

`Flexibility ratio = (ChargingDuration in minutes / 60) / connection duration in hours`

The connection duration is calculated from `StopTime - StartTime`. The analysis classifies sessions by local UK dates in `Europe/London`:

- Same-day: the connection starts and ends on the same local date.
- Overnight: the connection ends on a later local date.

A ratio is included in the summary when `ChargingDuration` is present and the connection duration is greater than zero. TARANG does not clip the ratio. The mean and median use all valid ratios, including any value above 1.0.

The 25-bin histogram covers only the range from 0.0 to 1.0. Ratios outside this range are not included in the histogram, so the valid count can be one or more higher than the histogram count.

This ratio describes past charging activity. It does not show that the unused connection time or energy can be shifted, and it is not a measure of available power or guaranteed flexibility.

If the input is empty or lacks `ChargingDuration`, `StartTime`, or `StopTime`, the analysis reports that it is unavailable. It does not create replacement statistics or zero-filled results.

## Run the advisory workflow

The command reads a CSV or Parquet file and writes the batch advisories, prediction summary, flexibility summary when available, and a Markdown report.

```bash
python scripts/run_decision_support.py --help
```

The default model is `HistGradientBoostingRegressor`. The required model files and processed dataset must already exist.
