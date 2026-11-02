# TARANG — Exploratory Data Analysis (EDA) Summary

**Project**: TARANG — EV Charging Intelligence System  
**Dataset**: Processed Electric Nation Modeling Dataset (`features.parquet`)  
**Scope**: 267 screened participants, 59,520 charging sessions  
**Generated Outputs**:
- `reports/eda/data_availability.png`: 4-panel participant availability distributions
- `reports/eda/duration_by_start_hour.png`: 2D histogram with marginal distributions (72h cutoff)
- `reports/eda/energy_by_start_hour.png`: 2D histogram with marginal distributions (75 kWh cutoff)
- `reports/eda/charging_flexibility.png`: Charging flexibility ratio (same-day vs overnight)

---

## 1. Participant Data Availability and Frequency

The screened cohort comprises **267 participants** monitored across the trial window.

| Availability Dimension | Mean | Median | Std Dev (pop) | Min | 25% | 75% | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Observation Span (Days) | 374.88 | 379.0 | 141.95 | 22 | 257.5 | 493.0 | 624 |
| Sessions per Participant | 222.92 | 186.0 | 161.54 | 14 | 111.5 | 296.5 | 1013 |
| Weekly Frequency (Sessions/Week) | 4.20 | 3.80 | 2.44 | 1.00 | 2.15 | 5.73 | 13.72 |
| Mean Plugged-In Duration (Hours) | 13.04 | 11.71 | 8.22 | 1.64 | 9.06 | 14.94 | 85.35 |

**Key Findings:**
- Average observation span per vehicle: 374.88 days (median: 379.0 days).
- Average session frequency: 4.20 sessions/week (population std: 2.44).
- Average session duration across participants: 13.04 hours (90th percentile: 18.66 hours).

---

## 2. Arrival Timing and Connection Duration Distributions

- **Modal Local Arrival Hour**: 18:00 Local UK Time (`Europe/London`)
- **Overall Raw Duration**:
  - Median: 11.45 hours
  - Mean: 12.14 hours
  - 75th Percentile: 14.92 hours
  - 95th Percentile: 25.77 hours
- **Distribution Output**: The joint distribution of connection durations across local UK arrival hours is visualized in `duration_by_start_hour.png` (binned in 0.5-hour increments up to the 72-hour cutoff).

---

## 3. Energy Consumption Profiles

- **Overall Energy Consumption**:
  - Median: 7.16 kWh
  - Mean: 10.44 kWh
  - 75th Percentile: 13.05 kWh
  - 95th Percentile: 29.48 kWh
- **Distribution Output**: The joint distribution of energy consumption across local UK arrival hours is visualized in `energy_by_start_hour.png` (binned in 0.5-hour arrival and 0.5-kWh energy increments up to the 75-kWh cutoff).

---

## 4. Smart-Charging Flexibility Ratio

The charging flexibility ratio is defined as:

$$\text{Flexibility Ratio} = \frac{\text{ChargingDuration} / 60}{(\text{StopTime} - \text{StartTime})_{\text{hours}}}$$

Evaluated using local UK dates (`Europe/London`):

| Session Category | Total Sessions | Evaluated in [0, 1] | Mean Ratio | Median Ratio |
| :--- | :---: | :---: | :---: | :---: |
| **Same-Day Sessions** | 21,700 | 15,947 | 0.689 | 0.780 |
| **Overnight Sessions** | 37,820 | 29,348 | 0.188 | 0.143 |

**Key Findings:**
- Same-day sessions exhibit a median flexibility ratio of **0.780** (mean: **0.689**).
- Overnight sessions exhibit a median flexibility ratio of **0.143** (mean: **0.188**).
- The full ratio distributions across 25 uniform bins in [0, 1] are rendered in `charging_flexibility.png`.
