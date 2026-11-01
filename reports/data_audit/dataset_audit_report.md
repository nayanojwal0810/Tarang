# TARANG — Initial Dataset Audit Report

**Project**: TARANG — EV Charging Intelligence System  
**Date**: November 2026  
**Status**: Completed  
**Branch**: `main`  
**Datasets Audited**:
1. `data/raw/electric_nation/CrowdCharge Transactions.xlsx`
2. `data/raw/electric_nation/ChargerInstall.xlsx`

---

## 1. Executive Summary

A complete, evidence-based data audit was performed on the raw Electric Nation datasets downloaded locally in `data/raw/electric_nation/`. The primary objective was to establish whether the raw data supports the planned EV charging prediction (plugged-in duration and energy consumption) and behavioral analysis models without data leakage or fatal quality defects.

**Key Findings:**
- **Usability**: The dataset **supports** the planned system, provided specific preprocessing rules and outlier treatments are implemented.
- **Transaction Coverage**: 71,264 transaction records spanning from March 1, 2017 to January 1, 2019 (22 months).
- **Participant Coverage**: 300 distinct non-null participants in transactions and 673 participants in vehicle metadata.
- **Reconciliation**: 100% of non-null transaction participants match metadata records (all belong to the Crowd Charge DCS provider). Metadata also contains 345 GreenFlux participants and 28 inactive Crowd Charge installations not present in this transaction workbook.
- **Data Quality Invariants**: Zero negative timestamp intervals (`StopTime < StartTime`), zero duplicate primary keys (`TransactionID`), and zero full-row duplicates.
- **Quality Anomalies**: 92 zero-duration sessions, 561 micro-sessions (< 5 minutes), 23 extreme durations (> 336 hours), 12 extreme energy outliers (> 150 kWh up to 37,487 kWh), 133 overlapping sessions for the same vehicle, and 3,899 unattributed sessions (of which 3,774 duplicate known charger sessions under the 200,000+ ID series).

---

## 2. Dataset Overview and Workbook Inventory

Both workbooks were inspected across all sheets. Original file sizes, available sheets, and dimensions are recorded below.

| Workbook | File Size (Bytes) | Sheet Name | Dimensions (Rows x Cols) | Content Description |
| :--- | :--- | :--- | :--- | :--- |
| `CrowdCharge Transactions.xlsx` | 11,141,809 (~10.63 MB) | `Explanation` | 1 x 1 | Empty placeholder sheet |
| `CrowdCharge Transactions.xlsx` | 11,141,809 (~10.63 MB) | `Transaction Data` | 71,265 x 28 | Primary EV charging transactions |
| `ChargerInstall.xlsx` | 58,976 (~57.59 KB) | `Explanation` | 1 x 1 | Empty placeholder sheet |
| `ChargerInstall.xlsx` | 58,976 (~57.59 KB) | `20181126_ChargerInstall_DriveEl` | 674 x 10 | Participant charger and vehicle metadata |

*Note: Row counts in the table above include the header row.*

---

## 3. Transaction Dataset Audit (`CrowdCharge Transactions.xlsx`)

### 3.1 Column Inventory, Data Types, and Missingness

The `Transaction Data` sheet contains 71,264 records across 28 columns.

| Column Name | Observed Data Type | Non-Null Count | Missing Count | Missing % | Description / Quality Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `TransactionID` | `int64` | 71,264 | 0 | 0.00% | Unique primary key (range 100000 to 204409) |
| `ChargerID` | `int64` | 71,264 | 0 | 0.00% | Charger unit identifier (301 unique chargers) |
| `ParticipantID` | `object` (`str`) | 67,365 | 3,899 | 5.47% | Participant identifier (300 unique IDs) |
| `CarKW` | `float64` | 67,365 | 3,899 | 5.47% | On-board charger rating (3.6 kW or 7.0 kW) |
| `CarKWh` | `float64` | 67,365 | 3,899 | 5.47% | Usable battery capacity (kWh) |
| `StartTime` | `datetime64[us]` | 71,264 | 0 | 0.00% | Plug-in connection timestamp |
| `StopTime` | `datetime64[us]` | 71,264 | 0 | 0.00% | Plug-out disconnection timestamp |
| `GroupID` | `float64` | 47,230 | 24,034 | 33.73% | Substation/feeder cluster identifier |
| `Trial` | `float64` | 47,230 | 24,034 | 33.73% | Managed trial phase (1.0, 2.0, 3.0) |
| `AdjustedStartTime` | `datetime64[us]` | 71,264 | 0 | 0.00% | Wall-clock adjusted start timestamp |
| `AdjustedStopTime` | `datetime64[us]` | 71,264 | 0 | 0.00% | Wall-clock adjusted stop timestamp |
| `PluggedInTime` | `int64` | 71,264 | 0 | 0.00% | Elapsed connection time in minutes |
| `ConsumedkWh` | `float64` | 71,263 | 1 | 0.0014% | Metered electrical energy delivered (kWh) |
| `Part_of_Managed_Group` | `object` (`str`) | 71,264 | 0 | 0.00% | Binary group flag ('Yes': 47,230, 'No': 24,034) |
| `Weekday_or_Weekend` | `object` (`str`) | 71,264 | 0 | 0.00% | Day category ('Weekday': 51,770, 'Weekend': 19,494) |
| `ActiveCharging_Start` | `float64` | 51,557 | 19,707 | 27.65% | Timestamp / offset of active charging start |
| `Max_Current_Drawn_for_T` | `float64` | 35,744 | 35,520 | 49.84% | Maximum observed current (A) |
| `EndCharge` | `float64` | 50,409 | 20,855 | 29.26% | Timestamp / offset of active charging end |
| `ChargingDuration` | `float64` | 50,403 | 20,861 | 29.27% | Net duration of active power transfer (minutes) |
| `t_inactive_start` | `float64` | 51,557 | 19,707 | 27.65% | Pre-charging idle connection time |
| `t_inactive_end` | `float64` | 50,409 | 20,855 | 29.26% | Post-charging idle connection time |
| `Used_a_Timer` | `float64` | 51,557 | 19,707 | 27.65% | Vehicle / charger timer flag |
| `Began_in_weekday_evening_peak` | `float64` | 51,557 | 19,707 | 27.65% | Peak hour start flag (17:00–21:00) |
| `Hot_Unplug` | `object` (`str`) | 51,425 | 19,839 | 27.84% | Unplug state ('No': 41,209, 'Yes': 9,080, etc.) |
| `T1_Managed` | `float64` | 28,107 | 43,157 | 60.56% | Managed indicator for Trial 1 |
| `T2_Managed` | `float64` | 12,998 | 58,266 | 81.76% | Managed indicator for Trial 2 |
| `Restricton T1` | `float64` | 2,074 | 69,190 | 97.09% | Demand curtailment active flag (Trial 1) |
| `Restriction T2` | `float64` | 475 | 70,789 | 99.33% | Demand curtailment active flag (Trial 2) |

### 3.2 Primary Key and Duplicate Verification

- **Total records**: 71,264
- **Unique `TransactionID` values**: 71,264 (0 duplicates, 100% unique primary keys)
- **Entire duplicate rows**: 0 (0.00%)

### 3.3 Timestamp Integrity and Temporal Range

- **`StartTime` Range**: Earliest = `2017-03-01 19:55:00`, Latest = `2018-12-31 23:11:00`
- **`StopTime` Range**: Earliest = `2017-03-02 07:11:00`, Latest = `2019-01-01 21:52:00`
- **Parsing Success**: 100.0% (0 unparsed or corrupted timestamps)
- **Negative Intervals (`StopTime < StartTime`)**: 0 (0.00%)
- **Zero Intervals (`StopTime == StartTime`)**: 92 records (0.129%)
- **Target Duration Relationship**: `PluggedInTime` directly equals `(StopTime - StartTime)` in minutes.

### 3.4 Target Variable Distributions

#### Target 1: Plugged-In Duration (`PluggedInTime` in minutes / hours)

$$\text{Duration Hours} = \frac{\text{StopTime} - \text{StartTime}}{3600}$$

| Percentile | Duration (Minutes) | Duration (Hours) | Quality Note |
| :--- | :--- | :--- | :--- |
| Min | 0.0 | 0.000 | 92 zero-duration glitches |
| 1% | 7.0 | 0.117 | Micro-sessions (< 5 min: 561 sessions, 0.787%) |
| 5% | 36.0 | 0.600 | Short top-up connections |
| 25% | 180.0 | 3.000 | Typical daytime session |
| **50% (Median)** | **682.0** | **11.367** | **Standard overnight plug-in** |
| 75% | 895.0 | 14.917 | Overnight connection extending to morning |
| 90% | 1,201.0 | 20.017 | Extended full-day connection |
| 99% | 3,862.0 | 64.367 | Weekend / holiday long connection |
| Max | 470,693.0 | 7,844.883 | Severe outlier (~326 days connected) |

#### Target 2: Energy Consumption (`ConsumedkWh`)

| Metric / Percentile | Value (kWh) | Assessment |
| :--- | :--- | :--- |
| Missing Count | 1 (0.0014%) | Single unmetered session |
| Non-null Count | 71,263 | Valid positive meter values |
| Negative / Zero Values | 0 (0.00%) | All values > 0 (minimum 0.01 kWh) |
| 25th Percentile | 4.36 kWh | Moderate partial top-up |
| **50th Percentile (Median)** | **7.06 kWh** | **Typical daily EV commute recharge** |
| 75th Percentile | 12.97 kWh | Larger battery recharge |
| 90th Percentile | 22.24 kWh | Substantial battery replenishment |
| 99th Percentile | 53.50 kWh | Full BEV recharge (Tesla, Leaf 40/62, etc.) |
| Extreme Outliers (> 150 kWh) | 12 records | Values up to 37,487.88 kWh (physically impossible) |

### 3.5 Trial Distribution

The trial structure reflects the Electric Nation smart charging trials:

| Trial Category | Record Count | Percentage | Smart Charging Intervention |
| :--- | :--- | :--- | :--- |
| Trial 1.0 | 28,107 | 39.44% | Managed charging algorithm Trial 1 |
| Unmanaged / Control (`NaN`) | 24,034 | 33.73% | Baseline unmanaged charging (`Part_of_Managed_Group == 'No'`) |
| Trial 2.0 | 13,000 | 18.24% | Managed charging algorithm Trial 2 |
| Trial 3.0 | 6,123 | 8.59% | Managed charging algorithm Trial 3 (time-of-use / tariff trial) |
| **Total** | **71,264** | **100.00%** | |

---

## 4. Vehicle Metadata Audit (`ChargerInstall.xlsx`)

The `20181126_ChargerInstall_DriveEl` sheet provides vehicle specifications and charger installation records for 673 participants.

### 4.1 Column Inventory and Missingness

| Column Name | Data Type | Non-Null | Missing | Missing % | Domain Meaning |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `ParticipantID` | `object` (`str`) | 673 | 0 | 0.00% | Participant primary key (673 unique) |
| `Charger` | `object` | 673 | 0 | 0.00% | Installed hardware charger ID |
| `ChargerInstallDate` | `datetime64[us]` | 673 | 0 | 0.00% | Date charger was installed |
| `DCSProvider` | `object` (`str`) | 673 | 0 | 0.00% | Trial management software provider |
| `CarInstallDate` | `datetime64[us]` | 569 | 104 | 15.45% | Date participant received trial vehicle |
| `CarMake` | `object` (`str`) | 673 | 0 | 0.00% | Vehicle manufacturer (18 unique) |
| `CarModel` | `object` (`str`) | 673 | 0 | 0.00% | Vehicle model (45 unique) |
| `PIVType` | `object` (`str`) | 673 | 0 | 0.00% | Powertrain category |
| `CarkW` | `float64` | 673 | 0 | 0.00% | Rated charger power (kW) |
| `CarkWh` | `float64` | 673 | 0 | 0.00% | Usable battery capacity (kWh) |

### 4.2 Key Metadata Distributions

1. **Provider Distribution**:
   - `Greenflux`: 345 (51.26%)
   - `Crowd Charge`: 328 (48.74%)
2. **Powertrain Breakdown (`PIVType`)**:
   - `Electric only (BEV)`: 315 (46.81%)
   - `Plug in Hybrid Electric Vehicle (PHEV)`: 271 (40.27%)
   - `Range extender (REX)`: 86 (12.78%)
   - `plug in Hybrid Electric Vehicle (PHEV)`: 1 (0.15%) *(Note: minor casing inconsistency with lowercase 'p')*
3. **Charger Power Rating (`CarkW`)**:
   - `7.0 kW` (32A single phase): 355 vehicles (52.75%)
   - `3.6 kW` (16A single phase): 318 vehicles (47.25%)
4. **Battery Capacity (`CarkWh`)**:
   - Min: 4.4 kWh (Toyota Prius Plug-in)
   - Median: 24.0 kWh (Nissan Leaf 24)
   - Mean: 29.71 kWh
   - Max: 100.0 kWh (Tesla Model S/X 100D)
5. **Top Vehicle Manufacturers (`CarMake`)**:
   - BMW (209), Tesla (119), Nissan (92), Volkswagen (75), Mitsubishi (50), Mercedes (42), Kia (25).

---

## 5. Cross-Dataset Consistency and Reconciliation

A strict participant-level reconciliation was conducted between transactions and metadata.

| Reconciliation Dimension | Count | Percentage | Audit Finding |
| :--- | :--- | :--- | :--- |
| Unique Non-Null Participants in Transactions | 300 | 100.00% | All non-null participant IDs in transactions |
| Unique Participants in Metadata | 673 | 100.00% | Total participants across both DCS providers |
| **Matched Participants (Present in Both)** | **300** | **100.00%** | **100% of transaction participants match metadata** |
| Transaction Participants Missing from Metadata | 0 | 0.00% | Zero unknown participant IDs in transactions |
| Metadata Participants Missing from Transactions | 373 | 55.42% | Composed of 345 GreenFlux + 28 Crowd Charge |
| Matched Participants Provider | 300 | 100.00% | All 300 matched participants are Crowd Charge |
| Unmatched Metadata — GreenFlux Provider | 345 | 92.49% | GreenFlux trials managed under separate external files |
| Unmatched Metadata — Crowd Charge Provider | 28 | 7.51% | Installations with no recorded sessions in this file |
| `CarKW` Specification Discrepancies | 0 | 0.00% | 100% agreement between transaction and metadata |
| `CarKWh` Battery Capacity Discrepancies | 0 | 0.00% | 100% agreement between transaction and metadata |

**Reconciliation Conclusion**:
The `CrowdCharge Transactions.xlsx` dataset represents exclusively the **Crowd Charge** operating arm of the Electric Nation project. It does not contain GreenFlux participants. For all 300 active participants, metadata coverage is 100% complete and perfectly concordant with transaction-level vehicle ratings.

---

## 6. Data-Quality Investigation

Each quality anomaly was investigated and categorized into standard data-quality severity classifications:

| Quality Check | Affected Records | % of Dataset | Severity Classification | Impact Assessment |
| :--- | :--- | :--- | :--- | :--- |
| **Zero-Duration Sessions (`StopTime == StartTime`)** | 92 | 0.1291% | Confirmed invalid records | Instantaneous plug/unplug events. Must be excluded from duration modeling. |
| **Micro-Sessions (< 5 minutes)** | 561 | 0.7872% | Potentially suspicious records | Includes 92 zero-duration records; remaining 469 records represent accidental plugs or immediate aborts without meaningful charging. |
| **Sessions Longer than 336 Hours (> 14 days)** | 23 | 0.0323% | Potentially suspicious records | Maximum duration reaches 7,844.88 hours (~326 days). Indicates unclosed communication sessions due to charger reboot or network loss. |
| **Inverted Timestamps (`StopTime < StartTime`)** | 0 | 0.0000% | Confirmed invalid records | Zero occurrences. Chronological consistency between arrival and departure is fully preserved. |
| **Duplicate Complete Rows** | 0 | 0.0000% | Confirmed invalid records | Zero duplicate rows detected across all 71,264 records. |
| **Duplicate `TransactionID` Values** | 0 | 0.0000% | Confirmed invalid records | All `TransactionID` values are strictly unique. |
| **Missing `ConsumedkWh` Target** | 1 | 0.0014% | Confirmed invalid records | Exactly 1 record (Transaction ID missing energy reading). Unusable for energy prediction. |
| **Extreme `ConsumedkWh` Outliers (> 150 kWh)** | 12 | 0.0168% | Potentially suspicious records | Unrealistic energy consumption up to 37,487.88 kWh. Requires clipping or exclusion. |
| **`ChargingDuration` Exceeds Plugged Duration** | 1 | 0.0014% | Confirmed invalid records | Session ID 132725 on 2018-03-25: active charging (478 min) > plugged duration (463 min). Caused by UK Daylight Saving Time 1-hour jump. |
| **Overlapping Sessions for Same Participant** | 133 | 0.1866% | Records requiring further investigation | 133 sessions start before the previous session of the same participant ended. Corrupts sequential lag calculations unless resolved. |
| **Missing Prior Session (Initial Vehicle Session)** | 300 | 0.4210% | Expected missing values | The first session of each vehicle has no prior history. Lag features (`LastDuration`, `LastConsumedkWh`, `TimeSinceLastStop`) naturally evaluate to `NaN`. |
| **Vehicles with Fewer than 10 Sessions** | 52 (across 11 vehicles) | 0.0730% | Records requiring further investigation | 11 participants have < 10 total sessions; insufficient history for reliable user-specific personalization. |
| **Vehicles with Low Observation Density (< 1 session/wk)** | 4,218 (across 28 vehicles) | 5.9188% | Records requiring further investigation | 28 participants exhibit sparse charging habits; inter-arrival intervals average > 7 days. |
| **Unattributed Sessions (Missing `ParticipantID`)** | 3,899 | 5.4712% | Records requiring further investigation | 3,899 records lack `ParticipantID`. 3,774 duplicate known charger transactions under the 200,000+ series IDs; 125 belong to unknown charger 4441003917. |

---

## 7. Feature Feasibility and Target Leakage Audit

### 7.1 Prediction Targets

The system has two distinct prediction targets evaluated at the time of vehicle arrival (`StartTime`):

1. **Target 1: Plugged-In Duration (Hours)**
   - Formula: `(StopTime - StartTime).total_seconds() / 3600.0`
   - **Crucial Distinction**: Must **NOT** be confused with active `ChargingDuration`. `PluggedInTime` measures total bay occupancy; `ChargingDuration` measures only the active current flow interval.
   - Status: **Feasible with Outlier Filtering** (exclude sessions < 5 min, > 336 hrs, and zero-duration records).
2. **Target 2: Energy Consumption (kWh)**
   - Formula: `ConsumedkWh`
   - Status: **Feasible with Outlier Filtering** (exclude 1 missing record and 12 extreme outliers > 150 kWh).

### 7.2 Planned Feature Feasibility

| Feature Name | Feature Type | Source / Calculation | Dependencies | Feasibility Status | Leakage Risk & Assessment |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Weekday` | Predictor | `StartTime.dt.dayofweek` | `StartTime` | Fully Feasible | **No Leakage**: Arrival timestamp is known at plug-in. |
| `Weekend` | Predictor | `StartTime.dt.dayofweek >= 5` | `StartTime` | Fully Feasible | **No Leakage**: Known at plug-in. |
| `Holiday` | Predictor | UK Bank Holidays (2017–2018) | `StartTime`, Holiday lookup | Fully Feasible | **No Leakage**: Calendar lookup known at plug-in. |
| `Month` | Predictor | `StartTime.dt.month` | `StartTime` | Fully Feasible | **No Leakage**: Known at plug-in. |
| `Start-Hour Cyclic` | Predictor | $\sin\left(\frac{2\pi h}{24}\right), \cos\left(\frac{2\pi h}{24}\right)$ | `StartTime` | Fully Feasible | **No Leakage**: Arrival hour known at plug-in. |
| `TimeSinceLastStop` | Historical Lag | `(StartTime - prev_StopTime) / 3600` | Chronological sort, `prev_StopTime` | Feasible with Conditions | **High Risk if unshifted**: Safe ONLY when strictly sorted per participant and referencing exclusively the immediately prior completed session. |
| `SessionsToday` | Historical Lag | Count of prior sessions on same date | Chronological sort, `StartTime` | Feasible with Conditions | **Leakage Risk if daily total used**: Must count ONLY sessions starting strictly before current `StartTime`. |
| `LastDuration` | Historical Lag | `prev_StopTime - prev_StartTime` | Chronological sort, prior timestamps | Feasible with Conditions | **Target Leakage Risk**: Must NEVER reference current session duration. Safe only from completed previous session. |
| `LastConsumedkWh` | Historical Lag | `prev_ConsumedkWh` | Chronological sort, prior energy | Feasible with Conditions | **Target Leakage Risk**: Safe only from prior completed session. Outlier clipping required for past corrupted readings. |
| `CarKW` | Static Metadata | `df_tx['CarKW']` / `df_meta['CarkW']` | Vehicle metadata | Fully Feasible | **No Leakage**: Vehicle charging capacity is known prior to connection. |
| `CarKWh` | Static Metadata | `df_tx['CarKWh']` / `df_meta['CarkWh']` | Vehicle metadata | Fully Feasible | **No Leakage**: Vehicle battery capacity is known prior to connection. |

### 7.3 Target Leakage Strict Boundaries

The following columns in the raw transaction dataset are generated **during or after** the charging session and **must NEVER be used as prediction inputs**:
- `StopTime` and `AdjustedStopTime` (defines the prediction target).
- `PluggedInTime` (equivalent to duration target).
- `ConsumedkWh` (defines the energy target).
- `ActiveCharging_Start` and `EndCharge` (occur during the session).
- `ChargingDuration` (net active charging time, unobserved at arrival).
- `t_inactive_start` and `t_inactive_end` (idle interval breakdowns).
- `Hot_Unplug` (disconnection event state).
- `Max_Current_Drawn_for_T` (maximum current drawn during the session).

---

## 8. Preprocessing Feasibility Analysis

A comparison between original data-preparation assumptions and actual raw data characteristics was conducted:

1. **UK Holiday Handling (2017 & 2018)**:
   - Feasible via an explicit reference set of official England & Wales bank holidays (8 holidays per year).
2. **Exclusion of Trial 3**:
   - `Trial == 3.0` contains exactly 6,123 sessions (8.59% of data).
   - Exclusion removes managed tariff-trial sessions while preserving 28,107 Trial 1 sessions, 13,000 Trial 2 sessions, and 24,034 unmanaged control sessions.
3. **Session Duration Thresholds (5 min min, 336 hrs max)**:
   - Removing sessions < 5 min purges 561 records (including 92 zero-duration records).
   - Removing sessions > 336 hours purges 23 unclosed sessions.
   - Combined duration filter impact: 584 records (0.82% of dataset).
4. **Chronological Participant Sorting**:
   - Raw records are indexed by `TransactionID`, which does not guarantee chronological order per participant. Sorting by `['ParticipantID', 'StartTime']` is strictly mandatory before generating lag features.
5. **Overlapping Sessions**:
   - 133 sessions overlap with the participant's prior session. These must either be resolved by adjusting overlapping boundaries or purged prior to lag feature calculation.
6. **Minimum Vehicle Session Count & Observation Density**:
   - 11 participants have < 10 sessions (52 records total). Excluding these participants ensures sufficient behavioral history for sequence modeling.

---

## 9. Final Dataset Feasibility Conclusion

The raw datasets **successfully pass the feasibility audit** for building the TARANG EV Charging Intelligence System, subject to the documented preprocessing pipeline:
- The data provides complete, high-precision timestamps, reliable meter readings for 99.99% of sessions, and 100% metadata concordance for all active trial participants.
- The dual prediction targets (plugged-in duration and energy consumption) can be reliably constructed from `StartTime`, `StopTime`, and `ConsumedkWh`.
- All planned temporal, cyclic, vehicle, and historical lag features are feasible without target leakage when calculated strictly using prior sessions.
- Data-cleaning rules must systematically eliminate the 92 zero-duration sessions, the 23 extreme long sessions, the 12 energy outliers, the 133 overlapping records, and the 3,899 unattributed/duplicate records before training.
