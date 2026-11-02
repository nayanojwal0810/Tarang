# Dataset audit

We check the Electric Nation transaction and vehicle records before using them for charging analysis and prediction. The audit covers record quality, timestamps, participant details, vehicle specifications, and energy readings.

## Data checked

| Source file | Sheet | Records |
|---|---|---:|
| CrowdCharge Transactions.xlsx | Transaction Data | 71,264 |
| ChargerInstall.xlsx | 20181126_ChargerInstall_DriveEl | 673 |

The transaction data contains 300 non-empty participant IDs. The vehicle file contains 673 participant records.

## Main findings

| Check | Result |
|---|---:|
| Transaction records | 71,264 |
| Transaction participants matched to vehicle records | 300 of 300 |
| Records without ParticipantID | 3,899 |
| Duplicate transaction IDs | 0 |
| Duplicate complete rows | 0 |
| Stop time before start time | 0 |
| Overlapping sessions found by the audit | 160 |
| Missing energy readings | 1 |
| Energy readings above 100 kWh | 20 |
| Energy readings above recorded battery capacity | 1,788 |

All 300 participants with an ID in the transaction data match a record in the vehicle file.

The audit finds 3,899 transactions without a ParticipantID. Of these, 3,774 match known sessions by charger ID, start time, stop time, and energy. The remaining 125 records use a charger that does not appear in the vehicle file.

The audit also finds 160 overlapping sessions by comparing each session with the latest stop time of earlier sessions for the same participant. This is a diagnostic count. The preparation process uses a separate sequential gap check.

## How we prepare the dataset

We apply these main rules:

1. Exclude Trial 3 records.
2. Remove sessions shorter than five minutes.
3. Cap the duration target at 336 hours.
4. Remove sessions with a negative gap from the previous session.
5. Use the first available session as history and retain later sessions for modelling.
6. Keep participants with at least 10 retained sessions and enough observations for their observation period.

The prepared dataset contains 59,520 sessions from 267 participants. Each session has 13 model inputs and two prediction targets: duration and energy.

We keep energy readings without clipping them. The audit identifies unusual values for review, but it does not remove them from the prepared data.

## What the checks mean

The source records contain enough information to prepare the model inputs and targets. However, missing participant IDs, overlapping sessions, and unusually high energy readings should be considered when interpreting results.

## Supporting tables

- [Transaction quality](transaction_quality_summary.csv)
- [Participant reconciliation](participant_reconciliation.csv)
- [Vehicle metadata quality](metadata_quality_summary.csv)
- [Feature availability and leakage checks](feature_feasibility.csv)
