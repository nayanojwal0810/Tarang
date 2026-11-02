# Data contract

This document defines how TARANG reads charging records, prepares the session dataset, and creates the model inputs and targets.

## Input files

Place these workbooks in `data/raw/electric_nation/`:

| File | Sheet | Purpose |
|---|---|---|
| `CrowdCharge Transactions.xlsx` | `Transaction Data` | Charging sessions and transaction details. |
| `ChargerInstall.xlsx` | `20181126_ChargerInstall_DriveEl` | Participant, charger, and vehicle details. |

The transaction file must contain `TransactionID`, `ChargerID`, `ParticipantID`, `CarKW`, `CarKWh`, `StartTime`, `StopTime`, `ConsumedkWh`, and `Trial`.

The metadata file must contain `ParticipantID`, `Charger`, `DCSProvider`, `CarMake`, `CarModel`, `CarkW`, and `CarkWh`.

## Prediction targets

TARANG predicts two values at the time a vehicle plugs in:

| Target | Meaning | Unit |
|---|---|---|
| `Duration` | Total time between plug-in and unplug. | Hours |
| `ConsumedkWh` | Energy delivered during the charging session. | kWh |

TARANG calculates duration from `StopTime - StartTime`. It removes sessions shorter than five minutes and caps the duration target at 336 hours. It keeps the original `StopTime` for historical time calculations.

The data preparation process does not remove or cap energy readings based on their size. The audit report flags unusual values for review.

## Time handling

The source timestamps use UTC. TARANG parses them as UTC-aware timestamps and keeps their recorded clock values.

The model calendar and history inputs use UTC. The flexibility analysis classifies same-day and overnight sessions using local UK dates in `Europe/London`.

## Session preparation

TARANG applies these operations in order:

1. Exclude records where `Trial == 3`.
2. Calculate plugged-in duration and remove sessions shorter than five minutes.
3. Cap the duration target at 336 hours without changing `StopTime`.
4. Calculate the global maximum start hour after the first two filters. Include rows with a missing participant ID when calculating this value.
5. Sort each participant's sessions by `StartTime`.
6. Calculate the time from the previous session's raw `StopTime` to the current `StartTime`.
7. Remove sessions with a negative gap. Do not recalculate the gap for later rows.
8. Drop the first remaining session for each participant. Keep its duration and energy as history for the first retained session.
9. Keep participants with at least 10 retained sessions.
10. Keep participants whose retained session count is at least their observation span in days divided by seven.
11. Calculate the historical features and combine the participant records into one dataset sorted by `StartTime`.

The participant integer ID keeps the original group enumeration value. TARANG does not renumber IDs after filtering.

## Model inputs

Each model receives exactly these 13 inputs in this order:

| Feature | Meaning |
|---|---|
| `weekday` | 1 for Monday to Friday, except designated holidays; otherwise 0. |
| `weekend` | 1 for Saturday or Sunday, except designated holidays; otherwise 0. |
| `holiday` | 1 for one of the 13 listed holiday dates; otherwise 0. |
| `month` | UTC month number from 1 to 12. |
| `StartCos` | Cosine of 2 pi times the normalized start hour. |
| `StartSin` | Sine of the normalized start hour multiplied by (2\pi). |
| `StartHourNorm` | UTC start hour divided by the global maximum start hour. |
| `TimeSinceLastStop` | Hours from the previous session's raw stop time to the current start time. |
| `sessionsToday` | Count of earlier sessions for the participant on the same UTC date. |
| `LastDuration` | Duration of the immediately previous session, capped at 336 hours. |
| `LastConsumedkWh` | Energy from the immediately previous session. |
| `CarKW` | Vehicle onboard charger rating in kW. |
| `CarKWh` | Vehicle battery capacity in kWh. |

The 13 feature names and their order are defined in `tarang.config.FEATURE_NAMES`.

## Holiday dates

TARANG uses these 13 dates for 2017 and 2018:

- 2017-01-01, 2017-01-02, 2017-04-14, 2017-05-01, 2017-05-29, 2017-12-25, 2017-12-26
- 2018-01-01, 2018-03-30, 2018-05-07, 2018-05-28, 2018-12-25, 2018-12-26

On these dates, `weekday` and `weekend` are both zero and `holiday` is one.

## Use only information known at arrival

The current session's end time, duration, and energy are not known at plug-in time. TARANG must not use these values as model inputs.

Fields such as `StopTime`, `PluggedInTime`, `ConsumedkWh`, `ChargingDuration`, `EndCharge`, and `Hot_Unplug` are not current-session predictors. Historical features must use completed earlier sessions only.

## Processed dataset

The preparation script writes:

`data/processed/electric_nation/features.parquet`

The file contains 30 columns, including session details, targets, calendar fields, historical features, and a participant ID. The exact column names are defined in `tarang.config.CANONICAL_COLUMNS`.
