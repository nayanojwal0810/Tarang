# Data contract

This document explains which Electric Nation files TARANG reads, how we prepare charging sessions, and which values the models use.

## Source files

Place these workbooks in data/raw/electric_nation/:

| File | Sheet | Contents |
|---|---|---|
| CrowdCharge Transactions.xlsx | Transaction Data | Charging sessions and transaction details. |
| ChargerInstall.xlsx | 20181126_ChargerInstall_DriveEl | Participant, charger, and vehicle details. |

The transaction sheet must include TransactionID, ChargerID, ParticipantID, CarKW, CarKWh, StartTime, StopTime, ConsumedkWh, and Trial.

The vehicle sheet must include ParticipantID, Charger, DCSProvider, CarMake, CarModel, CarkW, and CarkWh.

## Prediction targets

TARANG predicts two values for a new session:

| Target | Meaning | Unit |
|---|---|---|
| Duration | Time from plug-in to unplug. | Hours |
| ConsumedkWh | Energy delivered during the session. | kWh |

We calculate Duration from StopTime minus StartTime. We remove sessions shorter than five minutes and cap the target at 336 hours. We keep the recorded StopTime unchanged.

We do not remove or cap energy readings because of their size. The audit report lists unusual readings for review.

## Time handling

The source timestamps use UTC. We parse them as UTC-aware timestamps and keep their recorded times.

The model's calendar and history features use UTC. The charging ratio analysis uses local UK dates in Europe/London to classify same-day and overnight sessions.

## Session preparation

We prepare the dataset in this order:

1. Exclude records where Trial equals 3.
2. Calculate connection duration and remove sessions shorter than five minutes.
3. Cap the Duration target at 336 hours without changing StopTime.
4. Calculate the global maximum start hour after these filters. Include rows with a missing ParticipantID in this calculation.
5. Sort each participant's records by StartTime.
6. Calculate the gap from the previous session's recorded StopTime to the current StartTime.
7. Remove sessions with a negative gap. We do not recalculate the gap for later rows.
8. Drop the first remaining session for each participant. We keep its duration and energy as history for the next retained session.
9. Keep participants with at least 10 retained sessions.
10. Apply the observation-density rule: retained sessions must be at least the observation span in days divided by seven.
11. Create the historical features and combine the retained sessions in StartTime order.

The prepared dataset contains 59,520 sessions from 267 participants.

## Model inputs

Each model receives these 13 inputs in this order:

| Feature | Meaning |
|---|---|
| weekday | 1 for Monday to Friday, except designated holidays; otherwise 0. |
| weekend | 1 for Saturday or Sunday, except designated holidays; otherwise 0. |
| holiday | 1 for one of the 13 listed holiday dates; otherwise 0. |
| month | UTC month number from 1 to 12. |
| StartCos | Cosine value for the normalized UTC start hour. |
| StartSin | Sine value for the normalized UTC start hour. |
| StartHourNorm | UTC start hour divided by the global maximum start hour. |
| TimeSinceLastStop | Hours from the previous session's recorded stop time to the current start time. |
| sessionsToday | Number of earlier sessions for the participant on the same UTC date. |
| LastDuration | Duration of the immediately previous session, capped at 336 hours. |
| LastConsumedkWh | Energy from the immediately previous session. |
| CarKW | Vehicle charger power rating in kW. |
| CarKWh | Vehicle battery capacity in kWh. |

The feature names and their order come from tarang.config.FEATURE_NAMES.

## Holiday dates

We mark these dates as holidays:

- 2017: January 1, January 2, April 14, May 1, May 29, December 25, December 26.
- 2018: January 1, March 30, May 7, May 28, December 25, December 26.

On these dates, weekday and weekend are 0 and holiday is 1.

## Information available at arrival

At plug-in time, the current session's end time, duration, and energy are not yet known. We do not use the current session's StopTime, Duration, ConsumedkWh, ChargingDuration, EndCharge, or Hot_Unplug as model inputs.

We calculate history features only from completed earlier sessions. This prevents information from the current session's outcome from entering its prediction.

## Prepared dataset

The preparation script writes data/processed/electric_nation/features.parquet. It contains 30 columns, including source session details, prediction targets, calendar features, history features, and participant identifiers. The full column list comes from tarang.config.CANONICAL_COLUMNS.
