# TARANG — Data Contract

**Specification Version**: 2.0.0  
**Domain**: Electric Vehicle Charging Intelligence System  
**System Scope**: Prediction of Plugged-In Duration and Energy Consumption at Plug-In Arrival ($t_0$)  
**Timezone Standard**: Coordinated Universal Time (UTC)

---

## 1. System Overview and Purpose

The TARANG system predicts two primary operational targets at the exact moment a vehicle arrives and is plugged into a charging bay:
1. **Target 1: Plugged-in Duration ($\hat{y}_{\text{dur}}$)** — Total elapsed time in hours the vehicle remains connected, capped at 336.0 hours for the modeling target.
2. **Target 2: Energy Consumption ($\hat{y}_{\text{energy}}$)** — Total electrical energy in kilowatt-hours (kWh) delivered to the vehicle battery during the session.

All predictions must be rendered at inference time $t_0 = \text{StartTime}$ using only information available prior to or at $t_0$.

---

## 2. Planned Model Features (13 Inputs Available at $t_0$)

All temporal and lag calculations are based on the **UTC** timezone standard.

| Feature Name | Expected Type | Unit | Domain Constraints | Availability at Prediction | Source / Derivation Logic |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `weekday` | `int64` | Day index (0..6) | $0 \le \text{weekday} \le 6$ (Monday = 0, Sunday = 6) | Known at $t_0$ | Derived from UTC `StartTime.dt.dayofweek`. Unambiguous day index. |
| `weekend` | `int64` | Binary flag (0/1) | $\in \{0, 1\}$ (1 if Saturday or Sunday, else 0) | Known at $t_0$ | Derived from UTC `StartTime.dt.dayofweek >= 5`. |
| `holiday` | `int64` | Binary flag (0/1) | $\in \{0, 1\}$ (1 if UK England & Wales Bank Holiday) | Known at $t_0$ | Lookup of UTC date against official 2017–2018 Bank Holiday calendar. |
| `month` | `int64` | Month index (1..12) | $1 \le \text{month} \le 12$ | Known at $t_0$ | Derived from UTC `StartTime.dt.month`. |
| `StartCos` | `float64` | Cyclical coordinate | $[-1.0, 1.0]$ | Known at $t_0$ | $\cos\left(\frac{2\pi \cdot (\text{hour} + \text{min}/60)}{24}\right)$ in UTC. |
| `StartSin` | `float64` | Cyclical coordinate | $[-1.0, 1.0]$ | Known at $t_0$ | $\sin\left(\frac{2\pi \cdot (\text{hour} + \text{min}/60)}{24}\right)$ in UTC. |
| `StartHourNorm` | `float64` | Normalized fraction | $[0.0, 1.0)$ | Known at $t_0$ | $(\text{hour} + \text{min}/60) / 24.0$ in UTC. |
| `TimeSinceLastStop` | `float64` | Hours | $\ge 0.0$ | Known at $t_0$ (from prior session) | Elapsed hours between previous session `StopTime` and current `StartTime` per participant. $NaN$ for first session. |
| `sessionsToday` | `int64` | Count ($\ge 0$) | Integer $\ge 0$ | Known at $t_0$ (from prior sessions) | Count of prior sessions initiated by the participant on the same UTC calendar date strictly before `StartTime`. |
| `LastDuration` | `float64` | Hours | $0.0833 \le \text{LastDuration} \le 336.0$ | Known at $t_0$ (from prior session) | Elapsed duration $(\text{StopTime} - \text{StartTime})/3600$ of immediately preceding completed session. $NaN$ for first session. |
| `LastConsumedkWh`| `float64` | Kilowatt-hours (kWh) | $0.01 \le \text{LastConsumedkWh} \le 100.0$ | Known at $t_0$ (from prior session) | Energy delivered in immediately preceding completed session. $NaN$ for first session. |
| `CarKW` | `float64` | Kilowatts (kW) | $\in \{3.6, 7.0\}$ | Known prior to arrival | On-board charger rating from metadata registry. |
| `CarKWh` | `float64` | Kilowatt-hours (kWh) | $4.4 \le \text{CarKWh} \le 100.0$ | Known prior to arrival | Usable battery capacity from metadata registry. |

---

## 3. Prediction Targets Specification

### 3.1 Target 1: Plugged-In Duration

- **Definition**: Total elapsed connection time between physical plug-in and physical disconnection.
- **Formula**:
  $$\text{Target}_{\text{duration}} = \min\left(\frac{\text{StopTime} - \text{StartTime}}{\text{Timedelta}(1, \text{'h'})}, 336.0\right)$$
- **Units**: Hours (`float64`).
- **Target Capping**: Sessions $> 336.0$ hours (14 days) are capped at 336.0 hours for the modeling target. The original `StopTime` timestamp is preserved unmodified for accurate historical lag calculations (`TimeSinceLastStop`).
- **Exclusion Threshold**: Sessions $< 5.0$ minutes (0.0833 hours) are excluded from the modeling population as micro-sessions.
- **Distinction**: Plugged-in duration represents infrastructure bay occupancy. It is **distinct** from active `ChargingDuration`.

### 3.2 Target 2: Energy Consumption

- **Definition**: Total cumulative electrical energy supplied by the charger to the vehicle battery during the session.
- **Source**: `ConsumedkWh`.
- **Units**: Kilowatt-hours (kWh, `float64`).
- **Plausibility Thresholds**:
  - Valid domain: $[0.01, 100.0]$ kWh.
  - Readings $> 100.0$ kWh are flagged as implausibly high per the official Electric Nation trial findings.
  - Readings $> 150.0$ kWh (reaching 37,487.88 kWh) represent gross outliers.
  - Missing energy (1 record): Excluded.

---

## 4. Timestamp Conventions and Temporal Integrity

1. **Timezone Standard**:
   - All timestamps are supplied and handled in **UTC** (Coordinated Universal Time).
   - Naive Excel timestamps are converted to UTC-aware timestamps (`datetime64[ns, UTC]`) without shifting their recorded clock values.
   - All temporal features (`weekday`, `month`, `StartHourNorm`, `sessionsToday`) must be computed on this uniform UTC time basis.
2. **Interval Consistency**:
   - Every valid record must strictly satisfy: $\text{StopTime} > \text{StartTime}$.
   - Zero duration sessions ($\text{StopTime} == \text{StartTime}$) represent instant hardware disconnect glitches and are invalid.
   - Negative durations ($\text{StopTime} < \text{StartTime}$) violate causality and are invalid.
3. **Chronological Sorting**:
   - All historical and sequential calculations must strictly sort records per participant:
     $$\text{Sort keys} = [\text{ParticipantID}, \text{StartTime}, \text{StopTime}]$$
   - Row index or `TransactionID` order must not be relied upon for chronological ordering.

---

## 5. Strict Anti-Leakage Boundaries

To guarantee valid offline evaluation and production inference, the following rules are mandatory:

1. **Prediction Horizon Boundary**:
   - Any feature whose value depends on events occurring after $t_0 = \text{StartTime}$ is **strictly prohibited**.
2. **Prohibited Features**:
   - `StopTime`, `AdjustedStopTime` (target leakage).
   - `PluggedInTime` (target leakage).
   - `ConsumedkWh` (target leakage).
   - `ChargingDuration` (unobserved at arrival).
   - `ActiveCharging_Start`, `EndCharge` (intra-session offsets).
   - `t_inactive_start`, `t_inactive_end` (post-session idle breakdown).
   - `Hot_Unplug` (disconnection event state).
   - `Max_Current_Drawn_for_T` (session summary metric).
   - `Restricton T1`, `Restriction T2` (real-time trial intervention indicators).
3. **Lag Feature Integrity**:
   - Historical lag features (`LastDuration`, `LastConsumedkWh`, `TimeSinceLastStop`) must reference **only** completed sessions where:
     $$t_{\text{stop, prev}} \le t_{0, \text{current}}$$
   - Overlapping sessions ($t_{\text{stop, prev}} > t_{0, \text{current}}$) must be detected using cumulative maximum stop times and purged or flagged before lag calculation.
   - Future sessions ($t > t_0$) must **never** be accessed during training feature calculation.

---

## 6. Planned Preprocessing and Participant Screening Sequence

The modeling dataset must be constructed by executing the following planned sequence:

| Step | Action | Filter / Processing Rule | Impact on Population |
| :--- | :--- | :--- | :--- |
| 1 | Baseline Population | Non-null `ParticipantID` transactions | 67,365 records (300 participants) |
| 2 | Trial Screening | Exclude Trial 3 (`Trial == 3.0`) | 6,123 records excluded $\rightarrow$ 61,242 records (300 participants) |
| 3 | Duration Screening | Exclude sessions $< 5.0$ minutes | 509 records excluded $\rightarrow$ 60,733 records (299 participants) |
| 4 | Chronological Sorting | Sort by `[ParticipantID, StartTime, StopTime]` | Enforces physical arrival order |
| 5 | Overlap Resolution | Exclude sessions where $\text{StartTime} < \max_{j < i}(\text{StopTime}_j)$ | 158 overlapping sessions excluded $\rightarrow$ 60,575 records (299 participants) |
| 6 | Lag Initialization | Drop first session per participant (no prior history) | 299 sessions dropped $\rightarrow$ 60,276 retained sessions |
| 7 | Minimum History Filter | Retain participants with $\ge 10$ sessions after step 6 | 12 participants excluded (56 sessions) $\rightarrow$ 287 participants (60,220 sessions retained) |
| 8 | Observation-Density Filter | Retain participants where $\text{retained\_sessions} \ge \text{observation\_days} / 7$ | 20 participants excluded (726 sessions) $\rightarrow$ **267 participants (59,494 sessions retained)** |
| 9 | Target Preparation | Cap target duration at 336.0 hours; preserve original StopTime | 23 sessions capped; 0 timestamps altered |
