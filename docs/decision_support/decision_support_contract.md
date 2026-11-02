# Decision support contract

This document explains how TARANG combines duration and energy predictions into session advisories and historical charging summaries.

## Purpose and limits

For each session, TARANG provides:

- Predicted connection duration, in hours.
- Predicted energy consumption, in kWh.
- Projected unplug time.
- Validity flags for both predictions.

TARANG provides estimates for review. It does not schedule vehicles, control charging power, send commands to chargers, optimise tariffs, or dispatch energy.

## Single-session advisory

The SessionAdvisory result contains:

| Field | Meaning |
|---|---|
| arrival_time_utc | Session arrival time in UTC. |
| predicted_duration_hours | Raw duration estimate from the model. |
| predicted_energy_kwh | Raw energy estimate from the model. |
| projected_unplug_time_utc | Arrival time plus the duration estimate, when the duration estimate is valid. |
| duration_prediction_valid | True when the duration estimate is finite and non-negative. |
| energy_prediction_valid | True when the energy estimate is finite and non-negative. |

TARANG keeps the model's original estimates. It does not replace a negative or non-finite result with another value. When the duration estimate is invalid, the projected unplug time is unavailable (NaT).

## Batch advisory

Batch input must contain StartTime and the 13 model features. TARANG keeps the input rows, order, and index, then adds:

| Output column | Meaning |
|---|---|
| predicted_Duration | Predicted connection duration in hours. |
| predicted_ConsumedkWh | Predicted energy in kWh. |
| projected_unplug_time_utc | Projected unplug timestamp in UTC, or NaT when duration is invalid. |
| duration_prediction_valid | Validity flag for the duration estimate. |
| energy_prediction_valid | Validity flag for the energy estimate. |

Both predictions use the model selected when the decision support engine starts.

## Historical charging ratio

TARANG calculates the share of a session spent actively charging:

    Active charging ratio = (ChargingDuration in minutes / 60) / connection duration in hours

We calculate connection duration from StopTime minus StartTime. We classify sessions by local UK dates in Europe/London:

- Same-day: the session starts and ends on the same local date.
- Overnight: the session ends on a later local date.

A ratio is valid when ChargingDuration is present and connection duration is greater than zero. We do not clip valid ratios. The mean and median include every valid value, including a value above 1.

The 25-bin histogram covers only values from 0 to 1. Values outside this range do not appear in the histogram, so its count can be lower than the valid ratio count.

This ratio describes recorded charging activity. It does not show that unused connection time or energy can be shifted. It does not measure available power or guarantee charging flexibility.

If the input is empty or lacks ChargingDuration, StartTime, or StopTime, TARANG reports that the analysis is unavailable. It does not create replacement statistics or fill missing results with zero.

## Run the advisory workflow

The command reads a CSV or Parquet file and writes batch advisories, a prediction summary, a flexibility summary when the required fields are available, and a Markdown report.

    python scripts/run_decision_support.py --help

HistGradientBoostingRegressor is the default model. The input dataset and required model files must be available before you run the workflow.
