# Data dictionary

This document explains the main fields TARANG reads and the columns it creates for analysis and prediction.

## Source transaction fields

We read the CrowdCharge Transactions.xlsx workbook, from its Transaction Data sheet.

| Field | Meaning | Unit |
|---|---|---|
| TransactionID | Unique transaction record identifier. | ID |
| ChargerID | Charging unit identifier. | ID |
| ParticipantID | Anonymized participant identifier. | ID |
| Trial | Trial group used during data preparation. | Category |
| StartTime | Time when the vehicle connects to the charger. | UTC timestamp |
| StopTime | Time when the vehicle disconnects. | UTC timestamp |
| CarKW | Vehicle charger power rating. | kW |
| CarKWh | Vehicle battery capacity. | kWh |
| ConsumedkWh | Energy recorded for the session. | kWh |
| PluggedInTime | Connection duration recorded in the source. | Minutes |
| ChargingDuration | Active charging time recorded in the source. | Minutes |
| Hot_Unplug | Whether the vehicle disconnected while charging. | Category |

We also read ChargerInstall.xlsx. It provides participant, charger, provider, make, model, charger power, and battery capacity details. We use ParticipantID to match the vehicle details to charging records.

## Main processed fields

TARANG creates a 30-column file named features.parquet. The main columns are:

| Field | Meaning | Unit |
|---|---|---|
| StartTime | Current session arrival time. | UTC timestamp |
| StopTime | Recorded session end time. | UTC timestamp |
| ParticipantID | Participant identifier from the source. | ID |
| Duration | Plugged-in duration used as the first prediction target, capped at 336 hours. | Hours |
| ConsumedkWh | Session energy used as the second prediction target. | kWh |
| CarKW | Vehicle charger power rating. | kW |
| CarKWh | Vehicle battery capacity. | kWh |
| PluggedInTime | Source connection duration. | Minutes |
| ChargingDuration | Source active charging duration. | Minutes |
| Hot_Unplug | Source unplug status. | Category |
| Mon to Sun | Separate day-of-week indicators. | 0 or 1 |
| weekday, weekend, holiday | Calendar indicators used by the models. | 0 or 1 |
| month | UTC calendar month. | 1 to 12 |
| StartHour | Decimal hour of arrival in UTC. | Hours |
| StartHourNorm | Arrival hour divided by the global maximum start hour. | 0 to 1 |
| StartCos, StartSin | Cyclical encodings of arrival time. | Numeric |
| TimeSinceLastStop | Gap between the previous session stop and current arrival. | Hours |
| sessionsToday | Earlier sessions for the participant on the same UTC date. | Count |
| LastDuration | Duration of the previous session. | Hours |
| LastConsumedkWh | Energy from the previous session. | kWh |
| id | Integer participant group identifier. | ID |

## The 13 model inputs

The models use these columns in this order:

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

We do not use the current session's StopTime, Duration, ConsumedkWh, or ChargingDuration as model inputs. These values describe events that happen after arrival.

## Units and naming

- Duration measures the full time connected to the charger. ChargingDuration measures active charging time.
- ConsumedkWh measures energy in kilowatt-hours.
- CarKW measures power in kilowatts.
- CarKWh measures battery capacity in kilowatt-hours.
- Source timestamps and model calendar features use UTC.
- The charging ratio analysis uses Europe/London local dates to classify same-day and overnight sessions.
