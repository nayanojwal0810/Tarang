# TARANG — Initial Dataset Audit Report

**Project**: TARANG — EV Charging Intelligence System  
**Date**: November 2026  
**Status**: Completed and Validated  
**Branch**: `main`  
**Timezone Standard**: Coordinated Universal Time (UTC)  
**Datasets Audited**:
1. `data/raw/electric_nation/CrowdCharge Transactions.xlsx`
2. `data/raw/electric_nation/ChargerInstall.xlsx`

---

## 1. Executive Summary

A complete, evidence-based data audit was performed on the raw Electric Nation datasets downloaded locally in `data/raw/electric_nation/`. The primary objective was to establish whether the raw data supports the planned EV charging prediction (plugged-in duration and energy consumption) and behavioral analysis models without data leakage or fatal quality defects.

**Key Findings:**
- **Usability**: The dataset **supports** the planned system, provided specific preprocessing rules, duration capping, and outlier treatments are implemented.
- **Timestamp Standard**: All raw timestamps are verified and treated as **UTC-aware timestamps** without shifting their recorded clock values.
- **Transaction Coverage**: 71,264 transaction records spanning from 2017-03-01 19:55:00+00:00 to 2019-01-01 21:52:00+00:00 UTC.
- **Participant Coverage**: 300 distinct non-null participants in transactions and 673 participants in vehicle metadata.
- **Reconciliation**: 100.0% (300/300) of non-null transaction participants match metadata records (300 Crowd Charge). Row-level checks across all 67,365 matched records confirm 0 mismatches for `CarKW` and 0 mismatches for `CarKWh`.
- **Data Quality Invariants**: 0 negative timestamp intervals (`StopTime < StartTime`), 0 duplicate primary keys (`TransactionID`), and 0 full-row duplicates.
- **Unattributed Sessions**: 3,899 records (5.4712%) lack `ParticipantID`. Natural-session matching confirms 3,774 records match known participant sessions on the four session fields (`ChargerID, StartTime, StopTime, ConsumedkWh`), while 125 records belong to an unregistered charger (4441003917).
- **Participant Session Overlaps**: 160 overlapping sessions detected using the cumulative maximum stop time of all earlier sessions for that participant (uncovering 27 nested overlaps missed by sequential shift checks).
- **Energy Quality**: 20 records exceed the official Electric Nation 100 kWh implausibility threshold (12 exceed 150 kWh, reaching 37,487.88 kWh), and 1,788 records exceed their vehicle's rated battery capacity. 1 record lacks a meter reading.
- **Charging Duration Discrepancy**: 1 record (Transaction ID: 132725) exhibits `ChargingDuration` exceeding plugged-in connection duration. This is documented as an unexplained logging discrepancy requiring investigation.
- **Final Modeling Population**: Following the complete planned screening sequence (Trial 3 exclusion, duration threshold, cumulative overlap removal, initial session drop, $\ge 10$ retained sessions, and the observation-density filter), the final retained modeling population comprises **267 participants** and **59,494 sessions**.

---

## 2. Dataset Overview and Workbook Inventory

Both workbooks were inspected across all sheets. Original file sizes, available sheets, and dimensions are recorded below.

| Workbook | File Size (Bytes) | Sheet Name | Dimensions (Rows x Cols) | Content Description |
| :--- | :--- | :--- | :--- | :--- |
| `CrowdCharge Transactions.xlsx` | 11,141,809 | `Explanation` | 1 x 1 | Empty placeholder sheet |
| `CrowdCharge Transactions.xlsx` | 11,141,809 | `Transaction Data` | 71,265 x 28 | Primary EV charging transactions |
| `ChargerInstall.xlsx` | 58,976 | `Explanation` | 1 x 1 | Empty placeholder sheet |
| `ChargerInstall.xlsx` | 58,976 | `20181126_ChargerInstall_DriveEl` | 674 x 10 | Participant charger and vehicle metadata |

*Note: Excel row counts above include the header row.*

---

## 3. Timestamp Integrity and Timezone Handling

### 3.1 UTC Standardization

Raw timestamps in Electric Nation are supplied in UTC. The audit implements the following robust handling:
- Naive Excel timestamps are parsed and converted to UTC-aware timestamps without shifting recorded clock values.
- All derived temporal features (`weekday`, `weekend`, `holiday`, `month`, `StartCos`, `StartSin`, `StartHourNorm`, `sessionsToday`, and `TimeSinceLastStop`) are calculated strictly on this UTC time basis.
- Parsing success rate for non-missing raw timestamps is 100.0% (0 parse failures across 142528 non-missing timestamp cells; 0 missing cells).

| Timestamp Metric | Start Time (`StartTime`) | Stop Time (`StopTime`) |
| :--- | :--- | :--- |
| Column Name | `StartTime` | `StopTime` |
| Timezone | UTC (UTC-aware) | UTC (UTC-aware) |
| Earliest Value | `2017-03-01 19:55:00+00:00` | `2017-03-02 07:11:00+00:00` |
| Latest Value | `2018-12-31 23:11:00+00:00` | `2019-01-01 21:52:00+00:00` |
| Parse Failures | 0 | 0 |
| Missing Raw Values | 0 | 0 |
| Negative Intervals (`StopTime < StartTime`) | 0 (0.0%) | — |
| Zero Duration (`StopTime == StartTime`) | 92 (0.1291%) | — |

---

## 4. Unattributed-Session Investigation (`ParticipantID is Null`)

A rigorous natural-session matching investigation was conducted on the 3,899 unattributed records (5.4712% of transactions):

- **Chargers Represented**: 22 physical chargers appear in the unattributed records.
  - **21 chargers** exist in `ChargerInstall.xlsx` as valid participant installations.
  - **1 charger** (4441003917) does NOT exist in `ChargerInstall.xlsx`.
- **Natural Session Matching**: Using the matching key `(ChargerID, StartTime, StopTime, ConsumedkWh)` against the 67,365 known participant transactions:
  - **3,774 records** match known participant sessions on the four session fields. Other transaction fields may differ.
  - **125 records** belong to the unregistered charger (4441003917) and cannot be attributed to any trial participant.

---

## 5. Overlap Detection with Nested Overlap Resolution

The session overlap check was implemented using the cumulative maximum stop time of all earlier sessions for each participant:

$$\text{Overlap Condition: } \text{StartTime}_i < \max_{j < i} (\text{StopTime}_j)$$

- **Simple Shift(1) Overlaps**: 133 sessions detected when comparing only to the immediately preceding session.
- **Cumulative Max Earlier Overlaps**: **160 sessions** detected (0.2245% of transactions).
- **Nested Overlaps Identified**: Exactly 27 nested overlapping sessions were uncovered where a session started after the immediately prior short session ended but before an earlier long session concluded.
- Boundary sessions starting exactly when an earlier session ended (`StartTime == max(StopTime)`) are correctly treated as non-overlapping.

---

## 6. Row-Level Metadata Reconciliation

A row-level reconciliation was performed across all 67,365 matched transaction records against `ChargerInstall.xlsx`:

| Reconciliation Check | Count | Percentage | Status |
| :--- | :--- | :--- | :--- |
| Unique Participants in Transactions | 300 | 100.00% | Non-null participant IDs |
| Unique Participants in Metadata | 673 | 100.00% | Total trial participants across DCS providers |
| Matched Participants (Both Datasets) | 300 | 100.00% | 100.0% of transaction participants match |
| Unmatched Transaction Participants | 0 | 0.00% | 0 unknown participant IDs |
| Unmatched Metadata Participants | 373 | 55.42% | Metadata records without transactions |
| Row-Level `CarKW` Mismatches | 0 | 0.0000% | 100.00% agreement across all matched rows |
| Row-Level `CarKWh` Mismatches | 0 | 0.0000% | 100.00% agreement across all matched rows |
| Inconsistent Vehicle Specs per Participant | 0 | 0.00% | 0 participants with fluctuating specs |
| Row-Level `ChargerID` vs `Charger` Matches | 67,365 | 100.00% | Formats match when compared as strings |

---

## 7. Energy Quality and Battery Capacity Audit

Energy consumption readings (`ConsumedkWh`) were evaluated against physical thresholds:

| Metric | Record Count | % of Transactions | Assessment |
| :--- | :--- | :--- | :--- |
| Missing Energy | 1 | 0.0014% | 1 record lacks a meter reading |
| Non-Positive Energy (`<= 0`) | 0 | 0.0000% | No non-positive readings found (all recorded values are positive, min 0.01 kWh) |
| Energy > 100 kWh | 20 | 0.0281% | Exceeds official Electric Nation plausibility threshold |
| Energy > 150 kWh | 12 | 0.0168% | Severe outliers reaching 37,487.88 kWh |
| Energy > Registered `CarKWh` | 1,788 | 2.6542% | Records requiring review (requires investigation into telemetry or multi-cycle sessions) |

---

## 8. Planned Participant Screening and Observation-Density Filtering

The planned modeling population screening was executed in strict order:

| Step | Action | Filter / Rule Applied | Participants | Sessions |
| :---: | :--- | :--- | ---:| ---:|
| 1 | Baseline Population | Valid ParticipantID and valid timestamps | 300 | 67,365 |
| 2 | Exclude Trial 3 | `Trial != 3.0` | 300 | 61,242 |
| 3 | Duration Filter | `dur_min >= 5.0` | 299 | 60,733 |
| 4 | Chronological Sort | Sort by `[ParticipantID, StartTime, StopTime]` | 299 | 60,733 |
| 5 | Overlap Removal | Remove sessions where $\text{StartTime} < \max(\text{StopTime})$ | 299 | 60,575 |
| 6 | Drop First Session | Lag initialization (drop unlagged first session) | 298 | 60,276 |
| 7 | Minimum History Filter | Retain participants with $\ge 10$ retained sessions | 287 | 60,220 |
| 8 | Observation-Density Filter | Retain participants where $\text{retained\_sessions} \ge \text{observation\_days} / 7$ | **267** | **59,494** |

### Screening Transition Details
- **Step 5 (Overlap Removal)**: Yields 299 participants and 60,575 sessions.
- **Step 6 (Lag Initialization / Dropping First Session)**: Removes 299 sessions (one first session per participant after overlap removal). 1 participant had only a single session, leaving 0 retained sessions. This leaves **298 participants** with at least 1 retained session, totaling **60,276 retained sessions**.
- **Step 7 (Minimum History Filter)**: Of these 298 participants, exactly **11 participants** have fewer than 10 retained sessions (accounting for 56 sessions). The remaining **287 participants** (60,220 sessions) pass the minimum-history threshold.
- **Step 8 (Observation-Density Filter)**: Excludes 20 participants (726 sessions) whose retained session count is below $\text{observation\_days} / 7$, yielding the final population of **267 participants** and **59,494 sessions**.

### Observation-Density Results
- **Before Density Filtering**: 287 participants, 60,220 sessions.
- **Excluded by Density Rule**: 20 participants, 726 sessions.
- **After Density Filtering**: **267 participants**, **59,494 sessions**.
- **Final Population Density Metrics (267 vehicles)**:
  - Mean observation period: 374.88 days.
  - Mean sessions per week: 4.2 sessions/week.
  - Median sessions per week: 3.8 sessions/week.
  - Mean average interval between sessions: 2.44 days.

---

## 9. Planned Model Inputs and Anti-Leakage Boundaries

The planned model utilizes exactly **13 predictor inputs** and **2 prediction targets** evaluated at plug-in arrival ($t_0 = \text{StartTime}$):

### 9.1 Planned Model Features (13 Inputs)
1. `weekday`: UTC day of week (0..6 integer, Monday=0). Known at $t_0$.
2. `weekend`: UTC weekend binary flag (1 if Saturday or Sunday, else 0). Known at $t_0$.
3. `holiday`: UTC official UK bank holiday binary flag (1 if England & Wales holiday, else 0). Known at $t_0$.
4. `month`: UTC calendar month (1..12 integer). Known at $t_0$.
5. `StartCos`: $\cos(2\pi \cdot \text{hour}_{\text{float}} / 24.0)$ arrival cyclic component. Known at $t_0$.
6. `StartSin`: $\sin(2\pi \cdot \text{hour}_{\text{float}} / 24.0)$ arrival cyclic component. Known at $t_0$.
7. `StartHourNorm`: $\text{hour}_{\text{float}} / 24.0 \in [0.0, 1.0)$ normalized arrival hour. Known at $t_0$.
8. `TimeSinceLastStop`: Hours elapsed since previous session `StopTime`. Historical lag. Missing on initial session.
9. `sessionsToday`: Count of prior sessions initiated on same calendar date (UTC) before $t_0$. Historical lag.
10. `LastDuration`: Duration in hours of immediately prior completed session. Historical lag. Missing on initial session.
11. `LastConsumedkWh`: Energy consumed in kWh of immediately prior completed session. Historical lag. Missing on initial session.
12. `CarKW`: Vehicle onboard charger power rating in kW. Static vehicle hardware specification.
13. `CarKWh`: Usable battery capacity in kWh. Static vehicle hardware specification.

### 9.2 Prediction Targets
1. **Target 1: Plugged-In Duration (Hours)**: `(StopTime - StartTime) / 3600.0`, capped at 336.0 hours for the modeling target. Original `StopTime` is preserved for historical timing.
2. **Target 2: Energy Consumption (kWh)**: `ConsumedkWh`.

### 9.3 Strict Leakage Boundaries
The following fields occur post-arrival and are strictly excluded from predictors: `StopTime`, `AdjustedStopTime`, `PluggedInTime`, `ConsumedkWh`, `ActiveCharging_Start`, `EndCharge`, `ChargingDuration`, `t_inactive_start`, `t_inactive_end`, `Hot_Unplug`, `Max_Current_Drawn_for_T`, and trial curtailment indicators.

---

## 10. Final Dataset Feasibility Conclusion

The raw datasets **successfully pass the feasibility audit** for building the TARANG EV Charging Intelligence System, subject to the documented preprocessing pipeline:
- Complete, high-precision UTC-aware timestamps.
- 100.0% metadata concordance across all 300 active trial participants.
- The dual prediction targets (plugged-in duration capped at 336 hours and energy consumption) can be constructed reliably.
- All 13 planned features are feasible without target leakage.
- Data-cleaning rules must eliminate zero-duration sessions (92), extreme duration outliers, energy outliers (> 100 kWh, 20), overlapping records (160), and unattributed records (3,899).
- Participant screening yields a robust population of **267 participants** and **59,494 sessions**.
