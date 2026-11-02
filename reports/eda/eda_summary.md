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

The screened cohort comprises **267 participants** monitored across a multi-month trial window.

| Availability Dimension | Mean | Median | Std Dev | Min | 25% | 75% | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Observation Span (Days) | 374.88 | 379.0 | 142.21 | 22.0 | 257.5 | 493.0 | 624.0 |
| Sessions per Participant | 222.92 | 186.0 | 161.84 | 14.0 | 111.5 | 296.5 | 1013.0 |
| Weekly Frequency (Sessions/Week) | 4.2 | 3.8 | 2.45 | 1.0 | 2.16 | 5.73 | 13.72 |
| Mean Plugged-In Duration (Hours) | 13.04 | 11.71 | 8.23 | 1.64 | 9.06 | 14.94 | 85.35 |

**Key Insights:**
- Participants exhibit sustained charging engagement with a median observation period of 379.0 days.
- Weekly charging frequency averages 4.2 sessions per week, establishing regular behavioral cadence suitable for historical lag features.

---

## 2. Arrival Timing and Connection Duration Distributions

- **Peak Arrival Window**: Plug-in events peak in the late afternoon and early evening (16:00 – 19:00 UTC), reflecting typical domestic commuting patterns.
- **Overall Raw Duration**:
  - Median: 11.45 hours
  - Mean: 12.14 hours
  - 75th Percentile: 14.92 hours
  - 95th Percentile: 25.77 hours
- **Bivariate Pattern**: Evening arrivals (17:00–21:00) predominantly stay connected overnight (12–16 hours), creating a dense cluster visible in `duration_by_start_hour.png`. Daytime plug-ins (08:00–14:00) feature shorter connection periods (2–6 hours).

---

## 3. Energy Consumption Profiles

- **Overall Energy Consumption**:
  - Median: 7.16 kWh
  - Mean: 10.44 kWh
  - 75th Percentile: 13.05 kWh
  - 95th Percentile: 29.48 kWh
- **Bivariate Pattern**: Energy consumption varies substantially across all arrival hours, reflecting diverse state-of-charge upon connection. Evening connections exhibit both top-up sessions (5–15 kWh) and deep recharging events (25–60 kWh).

---

## 4. Smart-Charging Flexibility Ratio

The charging flexibility ratio is defined as:

$$\text{Flexibility Ratio} = \frac{\text{Charging Duration}}{\text{Plugged-In Duration}}$$

A ratio near 1.0 indicates that the vehicle charges for almost the entire time it is plugged in (low flexibility). A ratio near 0.0 indicates that the vehicle completes charging rapidly relative to its connection time, leaving substantial idle buffer time for load shifting and smart charging optimization.

| Session Category | Session Count | Mean Ratio | Median Ratio | Primary Characteristic |
| :--- | :---: | :---: | :---: | :--- |
| **Same-Day Sessions** | 15,947 | 0.689 | 0.780 | Moderate to low flexibility; vehicle unplugged shortly after charging completes |
| **Overnight Sessions** | 29,348 | 0.188 | 0.143 | High flexibility; vehicle remains connected for 10–14 hours with 2–4 hours of active charging |

**Key Findings:**
- Overnight sessions exhibit a median flexibility ratio of **0.143**, indicating that vehicles spend over **85.7%** of their connection window idle.
- This surplus connection time represents the primary operational buffer for managed charging, V1G peak shaving, and tariff-optimized scheduling.
