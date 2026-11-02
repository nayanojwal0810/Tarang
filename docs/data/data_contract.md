# TARANG — Data Contract

**Specification Version**: 3.0.0  
**Domain**: Electric Vehicle Charging Intelligence System  
**System Scope**: Dual-target prediction of bay connection duration and charging energy consumption at plug-in arrival ($t_0$)  
**Timezone Standard**: Coordinated Universal Time (UTC)

---

## 1. System Overview and Purpose

The TARANG system predicts two operational targets at the exact moment a vehicle arrives and is plugged into a charging bay:
1. **Target 1: Plugged-In Duration (`Duration`)** — Total elapsed connection time in hours the vehicle occupies the charging infrastructure, capped at 336.0 hours (14 days) for modeling.
2. **Target 2: Energy Consumption (`ConsumedkWh`)** — Total active electrical energy in kilowatt-hours (kWh) delivered to the vehicle battery during the session.

All predictions must be rendered at inference time $t_0 = \text{StartTime}$ using only information available prior to or at $t_0$.

---

## 2. Authoritative Predictors (13 Features Available at $t_0$)

All temporal and lag calculations are based on the **UTC** timezone standard.

| Feature Name | Expected Type | Unit | Domain Constraints | Availability at Prediction | Source / Derivation Logic |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `weekday` | `int64` | Day index (0..6) | $0 \le \text{weekday} \le 6$ (Monday = 0, Sunday = 6) | Known at $t_0$ | Derived from UTC `StartTime.dt.dayofweek`. Set to 0 on holidays. |
| `weekend` | `int64` | Binary flag (0/1) | $\in \{0, 1\}$ (1 if Saturday, Sunday, or Bank Holiday, else 0) | Known at $t_0$ | Derived from `(StartTime.dt.dayofweek >= 5) \| (holiday == 1)`. |
| `holiday` | `int64` | Binary flag (0/1) | $\in \{0, 1\}$ (1 if UK England & Wales Bank Holiday) | Known at $t_0$ | Lookup of UTC date against official 2017–2018 Bank Holiday calendar. |
| `month` | `int64` | Month index (1..12) | $1 \le \text{month} \le 12$ | Known at $t_0$ | Derived from UTC `StartTime.dt.month`. |
| `StartCos` | `float64` | Cyclical coordinate | $[-1.0, 1.0]$ | Known at $t_0$ | $\cos\left(\frac{2\pi \cdot (\text{hour} + \text{min}/60 + \text{sec}/3600)}{24}\right)$ in UTC. |
| `StartSin` | `float64` | Cyclical coordinate | $[-1.0, 1.0]$ | Known at $t_0$ | $\sin\left(\frac{2\pi \cdot (\text{hour} + \text{min}/60 + \text{sec}/3600)}{24}\right)$ in UTC. |
| `StartHourNorm` | `float64` | Normalized fraction | $[0.0, 1.0]$ | Known at $t_0$ | $(\text{hour} + \text{min}/60 + \text{sec}/3600) / \text{dataset\_max\_start\_hour}$ in UTC. |
| `TimeSinceLastStop` | `float64` | Hours | $\ge 0.0$ | Known at $t_0$ (from prior session) | Elapsed hours between previous session `StopTime` and current `StartTime` per participant. First session dropped. |
| `sessionsToday` | `int64` | Count ($\ge 0$) | Integer $\ge 0$ | Known at $t_0$ (from prior sessions) | Daily cumulative count of prior sessions initiated by the participant on the same UTC calendar date strictly before `StartTime`. |
| `LastDuration` | `float64` | Hours | $0.0833 \le \text{LastDuration} \le 336.0$ | Known at $t_0$ (from prior session) | Elapsed duration in hours of immediately preceding session. |
| `LastConsumedkWh`| `float64` | Kilowatt-hours (kWh) | $0.01 \le \text{LastConsumedkWh} \le 100.0$ | Known at $t_0$ (from prior session) | Energy delivered in immediately preceding session directly from sequential history. |
| `CarKW` | `float64` | Kilowatts (kW) | $\in \{3.6, 7.0\}$ | Known prior to arrival | On-board charger maximum power rating from trial record. |
| `CarKWh` | `float64` | Kilowatt-hours (kWh) | $4.4 \le \text{CarKWh} \le 100.0$ | Known prior to arrival | Usable battery capacity from trial record. |

---

## 3. Prediction Targets Specification

### 3.1 Target 1: Plugged-In Duration (`Duration`)

- **Definition**: Total elapsed connection time between physical plug-in and physical disconnection.
- **Formula**:
  $$\text{Duration} = \min\left(\frac{\text{StopTime} - \text{StartTime}}{\text{Timedelta}(1, \text{'h'})}, 336.0\right)$$
- **Units**: Hours (`float64`).
- **Target Capping**: Sessions $> 336.0$ hours (14 days) are capped at 336.0 hours for the modeling target. The original `StopTime` timestamp is preserved unmodified for accurate historical lag calculations (`TimeSinceLastStop`).
- **Exclusion Threshold**: Sessions $< 5.0$ minutes (0.0833 hours) are excluded from the modeling population as transient micro-sessions.
- **Distinction**: Plugged-in duration represents infrastructure bay occupancy. It is **distinct** from active `ChargingDuration`.

### 3.2 Target 2: Energy Consumption (`ConsumedkWh`)

- **Definition**: Total cumulative electrical energy supplied by the charger to the vehicle battery during the session.
- **Source**: `ConsumedkWh`.
- **Units**: Kilowatt-hours (kWh, `float64`).
- **Plausibility Thresholds**:
  - Valid range: $[0.01, 100.0]$ kWh.
  - Readings $> 100.0$ kWh are flagged as implausibly high per the official trial findings.
  - Missing energy: Excluded during screening.

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
     $$\text{Sort keys} = [\text{StartTime}]$$
   - Row index or arbitrary transaction IDs must not be relied upon for chronological ordering.

---

## 5. Strict Anti-Leakage Boundaries

To guarantee valid offline evaluation and production inference, the following rules are mandatory:

1. **Prediction Horizon Boundary**:
   - Any feature whose value depends on events occurring after $t_0 = \text{StartTime}$ is **strictly prohibited**.
2. **Prohibited Predictors**:
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
   - Overlapping sessions ($t_{\text{stop, prev}} > t_{0, \text{current}}$) must be detected and filtered before lag calculations.
   - Future sessions ($t > t_0$) must **never** be accessed during training feature calculation.

---

## 6. Preprocessing and Participant Screening Sequence

The canonical research dataset is constructed through the following deterministic pipeline:

| Step | Operation | Description | Impact on Population |
| :--- | :--- | :--- | :--- |
| 1 | Raw Ingestion | Read raw transaction records | 71,264 records (300 participants) |
| 2 | Trial Screening | Exclude Trial 3 records (`Trial != 3`) | 6,123 records excluded $\rightarrow$ 65,141 records (300 participants) |
| 3 | Field Selection | Retain 9 selected source fields | 9 source columns |
| 4 | Duration Calculation | Calculate duration in hours (`StopTime - StartTime`) | Elapsed connection hours |
| 5 | Micro-session Filter | Exclude sessions $< 5.0$ minutes | 511 records excluded $\rightarrow$ 64,630 records (299 participants) |
| 6 | Duration Capping | Cap `Duration` at 336.0 hours (preserve unmodified `StopTime`) | 23 sessions capped |
| 7 | Calendar Features | Derive `weekday`, `weekend`, `holiday`, `month`, and 7 day flags | Calendar columns added |
| 8 | Time Features | Continuous `StartHour`, dataset max normalization, cyclical `StartCos`/`StartSin` | Normalized & trigonometric columns added |
| 9 | Per-Participant Sorting | Sort each participant's records chronologically by `StartTime` | Arrival sequencing |
| 10 | Negative Gap Removal | Calculate `TimeSinceLastStop`; remove sessions with negative gap without recalculation | 132 sessions removed $\rightarrow$ 60,601 sessions (299 participants) |
| 11 | First Session Drop | Drop the first remaining session per participant | 299 sessions dropped; 1 participant has 0 remaining $\rightarrow$ 60,302 sessions (298 participants) |
| 12 | Minimum History Filter | Retain participants with $\ge 10$ retained sessions | 12 participants (56 sessions) excluded $\rightarrow$ 287 participants (60,246 sessions) |
| 13 | Observation-Density Filter | Retain participants with $\text{sessions} \ge \text{observation\_days} / 7.0$ | 20 participants (726 sessions) excluded $\rightarrow$ **267 participants (59,520 sessions)** |
| 14 | Integer ID Assignment | Assign integer participant identifier `id` ($0 \dots 266$) | Contiguous integer ID |
| 15 | Lag Calculation | Calculate `sessionsToday`, `LastDuration`, `LastConsumedkWh` | Lags derived from retained sessions |
| 16 | Concatenation & Sort | Global sort by `StartTime`, reset index | Final 59,520 records chronologically ordered |

---

## 7. Canonical Processed Dataset: `features.parquet`

The primary processed dataset is saved to:

`data/processed/electric_nation/features.parquet`

### Schema (30 Columns)

1. **Original Selected Source Fields (9)**:
   - `StartTime` (`datetime64[ns, UTC]`)
   - `StopTime` (`datetime64[ns, UTC]`)
   - `ParticipantID` (`object`)
   - `ConsumedkWh` (`float64`)
   - `CarKW` (`float64`)
   - `CarKWh` (`float64`)
   - `PluggedInTime` (`int64`)
   - `ChargingDuration` (`float64`)
   - `Hot_Unplug` (`object`)
2. **Operational Targets (2)**:
   - `Duration` (`float64`)
   - `ConsumedkWh` (`float64`) *(also listed in selected source fields)*
3. **Day-of-Week Indicators (7)**:
   - `Mon`, `Tue`, `Wed`, `Thu`, `Fri`, `Sat`, `Sun` (`int64`)
4. **Calendar Features (4)**:
   - `weekday` (`int64`)
   - `weekend` (`int64`)
   - `holiday` (`int64`)
   - `month` (`int64`)
5. **Start-Hour Features (4)**:
   - `StartHour` (`float64`)
   - `StartHourNorm` (`float64`)
   - `StartCos` (`float64`)
   - `StartSin` (`float64`)
6. **Historical Lag Features (4)**:
   - `TimeSinceLastStop` (`float64`)
   - `sessionsToday` (`int64`)
   - `LastDuration` (`float64`)
   - `LastConsumedkWh` (`float64`)
7. **Participant Identifier (1)**:
   - `id` (`int64`)
