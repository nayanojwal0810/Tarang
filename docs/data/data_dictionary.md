# Data dictionary

This document explains the main fields TARANG reads and the columns it creates for analysis and prediction.

## Source transaction fields

The source workbook is `CrowdCharge Transactions.xlsx`, sheet `Transaction Data`.

| Field | Meaning | Unit |
|---|---|---|
| `TransactionID` | Unique transaction record identifier. | ID |
| `ChargerID` | Identifier of the charging unit. | ID |
| `ParticipantID` | Anonymized participant identifier. | ID |
| `Trial` | Trial group value used to exclude Trial 3 records. | Category |
| `StartTime` | Time when the vehicle connects to the charger. | UTC timestamp |
| `StopTime` | Time when the vehicle disconnects. | UTC timestamp |
| `CarKW` | Vehicle onboard charger power rating. | kW |
| `CarKWh` | Vehicle battery capacity. | kWh |
| `ConsumedkWh` | Energy recorded for the session. | kWh |
| `PluggedInTime` | Total connection time recorded in the source. | Minutes |
| `ChargingDuration` | Active charging time recorded in the source. | Minutes |
| `Hot_Unplug` | Whether the vehicle disconnected while charging. | Category |

The audit also reads vehicle and charger details from `ChargerInstall.xlsx`. It uses `ParticipantID` to match records and checks charger rating, battery capacity, make, model, and provider information.

## Processed session fields

TARANG writes a 30-column dataset to `features.parquet`. The main fields are:

| Field | Meaning | Unit |
|---|---|---|
| `StartTime` | Current session arrival time. | UTC timestamp |
| `StopTime` | Recorded session end time. Kept unchanged. | UTC timestamp |
| `ParticipantID` | Participant identifier from the source. | ID |
| `Duration` | Plugged-in duration used as the first prediction target, capped at 336 hours. | Hours |
| `ConsumedkWh` | Session energy used as the second prediction target. | kWh |
| `CarKW` | Vehicle onboard charger rating. | kW |
| `CarKWh` | Vehicle battery capacity. | kWh |
| `PluggedInTime` | Source connection duration. | Minutes |
| `ChargingDuration` | Source active charging duration. | Minutes |
| `Hot_Unplug` | Source unplug status. | Category |
| `Mon` to `Sun` | Seven separate day-of-week indicators. | 0 or 1 |
| `weekday`, `weekend`, `holiday` | Calendar indicators used by the model. | 0 or 1 |
| `month` | UTC calendar month. | 1 to 12 |
| `StartHour` | Decimal hour of arrival in UTC. | Hours |
| `StartHourNorm` | Arrival hour divided by the global maximum start hour. | 0 to 1 |
| `StartCos`, `StartSin` | Cyclical encodings of arrival time. | Numeric |
| `TimeSinceLastStop` | Gap between the previous session stop and current arrival. | Hours |
| `sessionsToday` | Earlier sessions for the participant on the same UTC date. | Count |
| `LastDuration` | Previous session duration. | Hours |
| `LastConsumedkWh` | Previous session energy. | kWh |
| `id` | Integer participant group ID. | ID |

## Model inputs

The models use these 13 columns, in this order:

```text
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
```

The system does not use the current session's `StopTime`, `Duration`, `ConsumedkWh`, or `ChargingDuration` as model inputs. These values describe events that occur after arrival.

## Naming and units

- `Duration` means total time connected to the charger. It is not the same as `ChargingDuration`, which measures active charging time.
- `ConsumedkWh` is energy, measured in kilowatt-hours.
- `CarKW` is power, measured in kilowatts.
- `CarKWh` is battery capacity, measured in kilowatt-hours.
- Source timestamps and model calendar features use UTC. The flexibility report uses `Europe/London` local dates to classify same-day and overnight sessions.
