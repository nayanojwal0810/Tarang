# TARANG — Data Contract

**Specification Version**: 1.0.0  
**Domain**: Electric Vehicle Charging Intelligence System  
**System Scope**: Prediction of Plugged-In Duration and Energy Consumption at Plug-In Arrival

---

## 1. System Overview and Purpose

The TARANG system predicts two primary operational targets at the exact moment a vehicle arrives and is plugged into a charging bay:
1. **Target 1: Plugged-in Duration ($\hat{y}_{\text{dur}}$)** — Total elapsed time in hours the vehicle remains connected.
2. **Target 2: Energy Consumption ($\hat{y}_{\text{energy}}$)** — Total electrical energy in kilowatt-hours (kWh) delivered to the vehicle battery during the session.

All predictions must be rendered at inference time $t_0 = \text{StartTime}$ using only information available prior to or at $t_0$.

---

## 2. Expected Inputs and Ingestion Schema

### 2.1 Primary Input Features (Available at Arrival $t_0$)

| Feature Name | Expected Type | Unit | Domain Constraints | Nullable | Source / Derivation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `ParticipantID` | `string` | Dimensionless | Format `^EN[0-9]{4}$` | No | Direct input / RFID token / app session |
| `StartTime` | `datetime64` | ISO 8601 | Valid datetime, `2017-01-01` to `2019-12-31` | No | Charger arrival sensor timestamp |
| `CarKW` | `float64` | Kilowatts (kW) | $\in \{3.6, 7.0\}$ | No | Vehicle metadata registry / charger hardware |
| `CarKWh` | `float64` | Kilowatt-hours (kWh) | $4.0 \le \text{CarKWh} \le 120.0$ | No | Vehicle metadata registry |
| `Weekday` | `int64` | Day index (0–6) | $0 \le \text{Weekday} \le 6$ (0 = Monday) | No | Derived from `StartTime.dt.dayofweek` |
| `Weekend` | `int64` | Binary flag | $\in \{0, 1\}$ (1 if Saturday or Sunday) | No | Derived from `StartTime.dt.dayofweek >= 5` |
| `Holiday` | `int64` | Binary flag | $\in \{0, 1\}$ (1 if UK bank holiday) | No | Lookup against official UK Bank Holiday calendar |
| `Month` | `int64` | Month index (1–12) | $1 \le \text{Month} \le 12$ | No | Derived from `StartTime.dt.month` |
| `StartHour_sin` | `float64` | Radians | $[-1.0, 1.0]$ | No | $\sin\left(\frac{2\pi \cdot (\text{hour} + \text{min}/60)}{24}\right)$ |
| `StartHour_cos` | `float64` | Radians | $[-1.0, 1.0]$ | No | $\cos\left(\frac{2\pi \cdot (\text{hour} + \text{min}/60)}{24}\right)$ |

### 2.2 Historical Lag Features (Derived from Prior Sessions)

| Feature Name | Expected Type | Unit | Domain Constraints | Nullable | Derivation Logic |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `TimeSinceLastStop` | `float64` | Hours | $\ge 0.0$ | Yes (Initial session) | Elapsed time between previous session `StopTime` and current `StartTime` for the same participant. |
| `SessionsToday` | `int64` | Count | $\ge 0$ | No | Number of prior sessions initiated by the participant on the same calendar day before `StartTime`. |
| `LastDuration` | `float64` | Hours | $0.0833 \le \text{LastDuration} \le 336.0$ | Yes (Initial session) | Elapsed duration $(\text{StopTime} - \text{StartTime})/3600$ of the immediately preceding session. |
| `LastConsumedkWh`| `float64` | kWh | $0.01 \le \text{LastConsumedkWh} \le 150.0$ | Yes (Initial session) | Energy consumed in the immediately preceding session. |

---

## 3. Prediction Targets Specification

### 3.1 Target 1: Plugged-In Duration

- **Definition**: Total elapsed connection time between physical plug-in and physical disconnection.
- **Formula**:
  $$\text{Target}_{\text{duration}} = \frac{\text{StopTime} - \text{StartTime}}{\text{Timedelta}(1, \text{'h'})}$$
- **Units**: Hours (`float64`).
- **Valid Modeling Range**: $[0.0833, 336.0]$ hours (5 minutes to 14 days).
- **Invalid Records**:
  - Zero-duration sessions ($\text{StopTime} = \text{StartTime}$): Strictly invalid (hardware glitch).
  - Micro-sessions ($< 5$ minutes): Excluded as non-charging aborted connections.
  - Ultra-long sessions ($> 336$ hours): Excluded as abandoned or unclosed sessions.
- **Critical Distinction**: Plugged-in duration reflects parking bay occupancy and charger unavailability. It is **distinct** from active `ChargingDuration` (when current was flowing).

### 3.2 Target 2: Energy Consumption

- **Definition**: Total cumulative electrical energy supplied by the charger to the vehicle battery during the session.
- **Source**: `ConsumedkWh`.
- **Units**: Kilowatt-hours (kWh, `float64`).
- **Valid Modeling Range**: $[0.01, 150.0]$ kWh.
- **Invalid Records**:
  - Missing energy readings: Strictly invalid.
  - Energy $\le 0.0$ kWh: None observed in raw data; strictly invalid if present.
  - Extreme energy $> 150.0$ kWh: Physically impossible for target passenger EV batteries (e.g. 37,487 kWh anomalies); must be excluded.

---

## 4. Timestamp Conventions and Temporal Integrity

1. **Timezone and Wall-Clock Alignment**:
   - All timestamps represent UK local wall-clock time (`datetime64[us]`).
   - Timestamps are naive and un-timezone-localized in raw source files.
   - **Daylight Saving Time (DST) Caution**: UK clocks shift +1 hour on the last Sunday of March (GMT to BST) and -1 hour on the last Sunday of October (BST to GMT). Unadjusted wall-clock subtractions will produce a 1-hour discrepancy during transition nights (as verified in session ID 132725 on 2018-03-25).
2. **Interval Consistency**:
   - Every valid record must strictly satisfy: $\text{StopTime} > \text{StartTime}$.
   - Negative durations ($\text{StopTime} < \text{StartTime}$) violate physical causality and are rejected.
3. **Chronological Participant Order**:
   - All participant histories must be sorted strictly in ascending order:
     $$\text{Sort keys} = [\text{ParticipantID}, \text{StartTime}]$$
   - Sequence indices must never be inferred from row indices or `TransactionID`.

---

## 5. Strict Anti-Leakage Boundaries

To maintain theoretical and practical model validity, features must satisfy the following boundary rules:

1. **Prediction Horizon Boundary**:
   - Any feature whose value depends on events occurring after $t_0 = \text{StartTime}$ is **strictly prohibited**.
2. **Prohibited Features**:
   - `StopTime`, `AdjustedStopTime` (target leakage).
   - `PluggedInTime` (target leakage).
   - `ConsumedkWh` (target leakage).
   - `ChargingDuration` (unobserved at arrival).
   - `ActiveCharging_Start`, `EndCharge` (session progression offsets).
   - `t_inactive_start`, `t_inactive_end` (post-session decomposition).
   - `Hot_Unplug` (disconnection event attribute).
   - `Max_Current_Drawn_for_T` (session summary metric).
   - `Restricton T1`, `Restriction T2` (real-time trial intervention indicators).
3. **Lag Feature Integrity**:
   - Historical lag features (`LastDuration`, `LastConsumedkWh`, `TimeSinceLastStop`) must reference **only** completed sessions where:
     $$t_{\text{stop, prev}} \le t_{0, \text{current}}$$
   - Overlapping sessions ($t_{\text{stop, prev}} > t_{0, \text{current}}$) must be flagged and omitted from clean lag generation.
   - Future sessions ($t > t_0$) must **never** be accessed during training feature calculation.

---

## 6. Preprocessing and Data Rejection Criteria

Records violating the following contract conditions must be quarantined or filtered before modeling:

| Filter ID | Filter Condition | Contract Action | Rationale |
| :--- | :--- | :--- | :--- |
| `FILT_01` | `StopTime <= StartTime` | Quarantine / Exclude | Zero or inverted duration; sensor communication failure. |
| `FILT_02` | `PluggedInDuration < 5.0 min` | Exclude | Micro-session; non-representative connection. |
| `FILT_03` | `PluggedInDuration > 336.0 hr` | Exclude | Outlier unclosed session (> 14 days). |
| `FILT_04` | `ConsumedkWh is Null` | Exclude | Missing target variable. |
| `FILT_05` | `ConsumedkWh > 150.0 kWh` | Exclude | Corrupted meter reading exceeding physical battery size. |
| `FILT_06` | `ParticipantID is Null` | Exclude / Deduplicate | 3,899 unattributed sessions (duplicate 200,000+ transmissions and unknown charger 4441003917). |
| `FILT_07` | `Trial == 3.0` | Optional Protocol Exclusion | Tariff/time-of-use trial protocol exclusion if evaluating baseline/demand-managed behavior. |
| `FILT_08` | `Participant Total Sessions < 10` | Exclude from Personalization | Insufficient observations for participant-level sequence training. |
