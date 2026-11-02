# TARANG — Data Dictionary

This document details the source columns, observed data types, units of measurement, and domain definitions across the TARANG project:
1. `CrowdCharge Transactions.xlsx` (`Transaction Data` sheet)
2. `ChargerInstall.xlsx` (`20181126_ChargerInstall_DriveEl` sheet)
3. Canonical Processed Dataset: `features.parquet`

Definitions reflect empirical observations and official trial documentation.

---

## 1. Raw Transaction Dataset (`Transaction Data`)

Primary transaction records representing individual EV connection and charging sessions.

| Column Name | Observed Type | Unit | Description | Observed Domain / Values |
| :--- | :--- | :--- | :--- | :--- |
| `TransactionID` | `int64` | Dimensionless | Unique sequential identifier assigned to each charging session. | Integer from 100000 to 204409. Globally unique primary key. |
| `ChargerID` | `int64` | Dimensionless | Hardware identifier for the smart EV charging station. | 301 distinct identifiers (e.g., 35000117, 568004118). |
| `ParticipantID` | `object` (`str`) | Dimensionless | Anonymized unique trial participant identifier. | Format `ENxxxx` (e.g., EN1001 to EN1954). 300 distinct IDs; 3,899 null values. |
| `CarKW` | `float64` | Kilowatts (kW) | Maximum charging power acceptance rate of the vehicle's onboard charger. | Discrete values: `3.6` (16A single phase) or `7.0` (32A single phase). 3,899 null values. |
| `CarKWh` | `float64` | Kilowatt-hours (kWh) | Usable battery capacity of the participant's vehicle. | Continuous values ranging from `6.2` to `100.0`. 3,899 null values. |
| `StartTime` | `datetime64[us]` | ISO 8601 UTC Timestamp | Timestamp when the vehicle was physically plugged into the charger (UTC). | `2017-03-01 19:55:00` to `2018-12-31 23:11:00`. Supplied in Coordinated Universal Time (UTC). |
| `StopTime` | `datetime64[us]` | ISO 8601 UTC Timestamp | Timestamp when the vehicle was unplugged or the connection was terminated (UTC). | `2017-03-02 07:11:00` to `2019-01-01 21:52:00`. Supplied in Coordinated Universal Time (UTC). |
| `GroupID` | `float64` | Dimensionless | Electrical substation feeder network group identifier for demand management. | Numeric cluster ID (e.g., 1.0, 2.0). 24,034 nulls for unmanaged participants. |
| `Trial` | `float64` | Dimensionless | Electric Nation smart charging trial phase indicator. | `1.0` (Trial 1: 28,107), `2.0` (Trial 2: 13,000), `3.0` (Trial 3: 6,123), or `NaN` (Control: 24,034). |
| `AdjustedStartTime` | `datetime64[us]` | ISO 8601 UTC Timestamp | Session start timestamp adjusted for network latency or clock alignment (UTC). | Identical to `StartTime` in this dataset. |
| `AdjustedStopTime` | `datetime64[us]` | ISO 8601 UTC Timestamp | Session stop timestamp adjusted for network latency or clock alignment (UTC). | Identical to `StopTime` in this dataset. |
| `PluggedInTime` | `int64` | Minutes | Total elapsed connection duration between `StartTime` and `StopTime`. | Integer ranging from `0` to `470693`. Median: 682 minutes. Matches `(StopTime - StartTime)` in minutes. |
| `ConsumedkWh` | `float64` | Kilowatt-hours (kWh) | Total electrical energy metered and supplied to the vehicle battery during the session. | Ranging from `0.01` to `37487.88`. Median: 7.06 kWh. Exactly 1 null value. 20 records > 100 kWh (official implausibility threshold); 12 records > 150 kWh. |
| `Part_of_Managed_Group` | `object` (`str`) | Categorical | Indicates whether the participant was assigned to an active smart management trial group. | `'Yes'` (47,230 sessions) or `'No'` (24,034 sessions). |
| `Weekday_or_Weekend` | `object` (`str`) | Categorical | Day-of-week classification based on `StartTime`. | `'Weekday'` (51,770 sessions) or `'Weekend'` (19,494 sessions). |
| `ActiveCharging_Start` | `float64` | Minutes / Offset | Time offset or elapsed minutes from plug-in until electrical power transfer commenced. | Non-negative numeric; 19,707 null values (27.65%). |
| `Max_Current_Drawn_for_T`| `float64` | Amperes (A) | Maximum electrical current drawn by the vehicle during the transaction. | Observed values from `0.0` to `32.0`. 35,520 null values (49.84%). |
| `EndCharge` | `float64` | Minutes / Offset | Time offset or elapsed minutes from plug-in until active battery charging ceased. | Non-negative numeric; 20,855 null values (29.26%). |
| `ChargingDuration` | `float64` | Minutes | Net active duration during which electrical current was flowing into the battery. | Ranging from `1.0` to `19570.0`. Median: 108 minutes. 20,861 null values (29.27%). 1 record (ID 132725) has ChargingDuration (478.0 min) > PluggedInTime (463.0 min), an unexplained logging discrepancy. |
| `t_inactive_start` | `float64` | Minutes | Idle duration between plug-in connection and active charging onset. | Non-negative numeric; 19,707 null values (27.65%). |
| `t_inactive_end` | `float64` | Minutes | Idle duration between charging completion and physical unplugging. | Non-negative numeric; 20,855 null values (29.26%). |
| `Used_a_Timer` | `float64` | Binary flag | Indicator of whether an in-vehicle or charger delay timer was utilized. | `0.0` (timer not detected) or `1.0` (timer used). 19,707 null values. |
| `Began_in_weekday_evening_peak` | `float64` | Binary flag | Indicator of whether the session started during peak weekday evening demand hours (17:00–21:00 UTC). | `0.0` or `1.0`. 19,707 null values (27.65%). |
| `Hot_Unplug` | `object` (`str`) | Categorical | Disconnection status indicating whether the vehicle was unplugged while active power was flowing. | `'No'` (41,209), `'Yes'` (9,080), `'May be departure timer'` (1,136), or `NaN` (19,839). |
| `T1_Managed` | `float64` | Binary flag | Indicator of smart demand management intervention in Trial 1. | `0.0` or `1.0`. Populated only for Trial 1 participants (43,157 nulls). |
| `T2_Managed` | `float64` | Binary flag | Indicator of smart demand management intervention in Trial 2. | `0.0` or `1.0`. Populated only for Trial 2 participants (58,266 nulls). |
| `Restricton T1` | `float64` | Binary flag | Indicator of active power curtailment during Trial 1. | `0.0` or `1.0`. 69,190 nulls (97.09%). |
| `Restriction T2` | `float64` | Binary flag | Indicator of active power curtailment during Trial 2. | `0.0` or `1.0`. 70,789 nulls (99.33%). |

---

## 2. Vehicle Metadata Dataset (`20181126_ChargerInstall_DriveEl`)

Hardware installation and vehicle registry records for participants in the Electric Nation trials.

| Column Name | Observed Type | Unit | Description | Observed Domain / Values |
| :--- | :--- | :--- | :--- | :--- |
| `ParticipantID` | `object` (`str`) | Dimensionless | Unique participant identifier linking metadata to transaction logs. | Format `ENxxxx` (e.g., EN1001 to EN1954). 673 unique values, 0 nulls. |
| `Charger` | `object` | Dimensionless | Serial number or hardware identifier of the installed charging unit. Matches transaction `ChargerID` when string-formatted. | 673 unique values (e.g., 568004118). 0 nulls. |
| `ChargerInstallDate`| `datetime64[us]` | ISO 8601 Date | Date when the domestic smart charger was installed at the participant's premises. | `2016-10-18` to `2018-04-13`. 0 nulls. |
| `DCSProvider` | `object` (`str`) | Categorical | Demand Control System (DCS) service provider operating the charger network. | `'Greenflux'` (345 installations) or `'Crowd Charge'` (328 installations). 0 nulls. |
| `CarInstallDate` | `datetime64[us]` | ISO 8601 Date | Date when the participant received delivery of the trial plug-in vehicle. | `2016-09-02` to `2018-05-30`. 104 null values (15.45%). |
| `CarMake` | `object` (`str`) | Categorical | Brand manufacturer of the plug-in vehicle. | 18 manufacturers (e.g., BMW, Tesla, Nissan, Volkswagen, Mitsubishi). 0 nulls. |
| `CarModel` | `object` (`str`) | Categorical | Commercial vehicle model designation. | 45 models (e.g., Model S, Leaf, i3 REX, Outlander PHEV, 330e, Golf GTE). 0 nulls. |
| `PIVType` | `object` (`str`) | Categorical | Plug-in Vehicle (PIV) powertrain technology architecture. | `'Electric only (BEV)'` (315), `'Plug in Hybrid Electric Vehicle (PHEV)'` (271), `'Range extender (REX)'` (86), or `'plug in Hybrid Electric Vehicle (PHEV)'` (1). 0 nulls. |
| `CarkW` | `float64` | Kilowatts (kW) | On-board charger maximum power rating. Matches transaction `CarKW` 100%. | `3.6` (318 vehicles) or `7.0` (355 vehicles). 0 nulls. |
| `CarkWh` | `float64` | Kilowatt-hours (kWh) | Usable battery pack capacity. Matches transaction `CarKWh` 100%. | Numeric values from `4.4` to `100.0`. Median: 24.0 kWh. 0 nulls. |

---

## 3. Canonical Processed Dataset (`features.parquet`)

The 30 columns present in `data/processed/electric_nation/features.parquet` (59,520 records across 267 participants), listed in exact canonical sequence:

| Column Name | Data Type | Category | Description | Domain / Observed Range |
| :--- | :--- | :--- | :--- | :--- |
| `StartTime` | `datetime64[ns, UTC]` | Source Field | Vehicle connection start timestamp (UTC). | `2017-03-01 21:09:00` to `2018-12-31 23:11:00` |
| `StopTime` | `datetime64[ns, UTC]` | Source Field | Vehicle disconnection stop timestamp (UTC, unmodified). | `2017-03-02 07:11:00` to `2019-01-01 21:52:00` |
| `ParticipantID` | `object` | Source Field | Unique trial participant identifier. | 267 distinct IDs (`ENxxxx`) |
| `ConsumedkWh` | `float64` | Target / Source | Delivered electrical energy during charging session. Retained without filtering missing energy or values > 100 kWh. | $0.01$ to $37,487.88$ kWh; median: 7.16 kWh (0 missing values in canonical dataset; single raw null excluded with Trial 3) |
| `CarKW` | `float64` | Feature / Source | Vehicle onboard charger power rating. | `3.6` or `7.0` kW |
| `CarKWh` | `float64` | Feature / Source | Vehicle usable battery capacity. | `6.2` to `100.0` kWh |
| `PluggedInTime` | `int64` | Source Field | Total connection duration in minutes. | Min: 5 minutes; max: 470,693 minutes |
| `ChargingDuration`| `float64` | Source Field | Net active charging duration in minutes. | 14,224 nulls (23.90%) |
| `Hot_Unplug` | `object` | Source Field | Disconnection event status. | 13,514 nulls (22.70%) |
| `Duration` | `float64` | Target | Elapsed plugged-in connection duration in hours, capped at 336.0h. | $0.0833$ to $336.0$ hours; median: 11.45 hours |
| `weekday` | `int64` | Feature | Binary flag: 1 for Monday–Friday, 0 for Saturday–Sunday; 0 on Bank Holidays. | $\in \{0, 1\}$ |
| `weekend` | `int64` | Feature | Binary flag: 1 for Saturday–Sunday, 0 for Monday–Friday; 0 on Bank Holidays. | $\in \{0, 1\}$ |
| `holiday` | `int64` | Feature | Binary flag for 13 designated UK Bank Holidays. On holidays, both `weekday` and `weekend` are 0. | $\in \{0, 1\}$ |
| `Mon` | `int64` | Day Flag | Binary indicator: Monday (1 if True, else 0). | $\in \{0, 1\}$ |
| `Tue` | `int64` | Day Flag | Binary indicator: Tuesday (1 if True, else 0). | $\in \{0, 1\}$ |
| `Wed` | `int64` | Day Flag | Binary indicator: Wednesday (1 if True, else 0). | $\in \{0, 1\}$ |
| `Thu` | `int64` | Day Flag | Binary indicator: Thursday (1 if True, else 0). | $\in \{0, 1\}$ |
| `Fri` | `int64` | Day Flag | Binary indicator: Friday (1 if True, else 0). | $\in \{0, 1\}$ |
| `Sat` | `int64` | Day Flag | Binary indicator: Saturday (1 if True, else 0). | $\in \{0, 1\}$ |
| `Sun` | `int64` | Day Flag | Binary indicator: Sunday (1 if True, else 0). | $\in \{0, 1\}$ |
| `month` | `int64` | Feature | Calendar month index (1 = January, 12 = December). | $1 \le \text{month} \le 12$ |
| `StartHour` | `float64` | Time Field | Continuous arrival hour in UTC: $\text{hour} + \text{min}/60 + \text{sec}/3600$. | $[0.0, 23.9833]$ |
| `StartHourNorm` | `float64` | Feature | Continuous arrival hour normalized by dataset maximum StartHour across filtered population. | $[0.0, 1.0]$ |
| `StartCos` | `float64` | Feature | Cyclical cosine: $\cos(2\pi \cdot \text{StartHourNorm})$. | $[-1.0, 1.0]$ |
| `StartSin` | `float64` | Feature | Cyclical sine: $\sin(2\pi \cdot \text{StartHourNorm})$. | $[-1.0, 1.0]$ |
| `TimeSinceLastStop` | `float64` | Feature | Elapsed hours between immediately preceding session stop and current start. | Min: 0.0, Max: 4,816.67 hours |
| `sessionsToday` | `int64` | Feature | Prior sessions initiated by participant on consecutive dates on or before StartTime. | $0 \le \text{sessionsToday} \le 7$ |
| `LastDuration` | `float64` | Feature | Elapsed duration in hours of immediately preceding session. First retained session populated from dropped preceding session (zero missingness). | $[0.0833, 336.0]$ hours |
| `LastConsumedkWh`| `float64` | Feature | Delivered energy in kWh of immediately preceding session. First retained session populated from dropped preceding session (0 missing values in canonical dataset; single raw null excluded with Trial 3; readings > 100 kWh retained without clipping). | $[0.01, 37,487.88]$ kWh |
| `id` | `int64` | Identifier | Integer participant group enumeration index from `enumerate(groupby("ParticipantID", sort=True))`. Non-contiguous (values 1 to 298 across 267 participants). Not used as predictor. | Integer values $1 \dots 298$ |

---

## 4. Authoritative Predictors (13 Features)

For supervised regression modeling, only these 13 features are used (evaluated at arrival time $t_0 = \text{StartTime}$):
1. `weekday`
2. `weekend`
3. `holiday`
4. `month`
5. `StartCos`
6. `StartSin`
7. `StartHourNorm`
8. `TimeSinceLastStop`
9. `sessionsToday`
10. `LastDuration`
11. `LastConsumedkWh`
12. `CarKW`
13. `CarKWh`
