# Dataset audit report

## Purpose

We checked the two Electric Nation workbooks before using their records for charging analysis and prediction. The audit reviewed timestamps, transaction quality, participant details, vehicle data, energy readings, and the number of sessions available after filtering.

## Data checked

- `CrowdCharge Transactions.xlsx`, sheet `Transaction Data`
- `ChargerInstall.xlsx`, sheet `20181126_ChargerInstall_DriveEl`

The transaction file contains 71,264 records and 300 identified participants. The charger installation file contains 673 participant records.

## Main findings

| Check | Result |
|---|---:|
| Transaction records | 71,264 |
| Participants in transactions | 300 |
| Participants matched to installation records | 300 of 300 |
| Records without a participant ID | 3,899 |
| Duplicate transaction IDs | 0 |
| Duplicate complete rows | 0 |
| Records with an end time before the start time | 0 |
| Same-participant overlaps found by the audit | 160 |
| Missing energy readings | 1 |
| Energy readings above 100 kWh | 20 |
| Readings above the vehicle's recorded battery capacity | 1,788 |

All 300 participants found in the transaction data matched an installation record. The audit also found 3,899 sessions without a participant ID. Of these, 3,774 match known sessions on charger ID, start time, stop time, and energy; 125 are linked to a charger not listed in the installation file.

The audit found 160 overlapping sessions when it compared each session with the latest stop time of all earlier sessions for that participant. This is a diagnostic count. The preparation process applies its separate negative-gap filter in sequence.

## Prepared dataset

The preparation process excludes Trial 3 records, removes sessions shorter than five minutes, caps the duration target at 336 hours, and applies participant history and observation-density rules.

The current processed dataset contains:

- 59,520 sessions
- 267 participants
- 13 model input features
- Two prediction targets: duration and energy

The pipeline keeps energy readings without clipping them. The audit flags high readings for review but does not remove them from the processed data.

## Detailed audit tables

The CSV files in this folder contain the row counts and checks behind this summary:

- [Transaction quality](transaction_quality_summary.csv)
- [Participant reconciliation](participant_reconciliation.csv)
- [Vehicle metadata quality](metadata_quality_summary.csv)
- [Feature availability and leakage checks](feature_feasibility.csv)

## Conclusion

The transaction and vehicle records provide the fields needed to prepare the session dataset and build the two prediction targets. The audit findings above should be kept in mind when interpreting results, especially records without participant IDs and unusually high energy readings.
