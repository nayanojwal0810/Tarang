# TARANG — Model Evaluation, Behavioral Analysis & Explainability
**Target:** Energy Consumption (target_energy_kwh)  
**Evaluation Split:** 20% Chronological Held-out Test Set  

---

## 1. Executive Summary

This report presents the model evaluation, participant-level behavioral analysis, and TreeSHAP explainability for **TARANG** predicting **Energy Consumption (target_energy_kwh)**.

- **Dataset Evaluated:** 59,520 canonical sessions across 267 unique participants (47,616 Train sessions / 11,904 Test sessions in chronological 80/20 split).
- **Model Performance Summary:** `HistGradientBoostingRegressor` (HGBR) achieves the lowest error on MAE, MdAE, MdAPE, SMAPE on the held-out test split (MAE: **3.83 kWh**, MdAE: **1.97 kWh**, SMAPE: **39.4%**); while `MedianValue` achieves the lowest aggregate MAPE (**236%**) and per-vehicle mean MAPE (**219%**).
- **Participant-Level Performance:** The reported per-vehicle mean MAE is **4.84 kWh** (MdAE: **4.13 kWh**, MAPE: **287%**), reflecting the distribution of individual participant charging patterns across the cohort.
- **Dominant Feature Drivers:** TreeSHAP analysis identifies `CarKWh`, `LastConsumedkWh`, `TimeSinceLastStop` as the primary global predictors driving model decisions.

---

## 2. Benchmark Model Performance

Evaluation results across all six benchmark models comparing aggregate session performance ('All data') against the average per-vehicle performance ('Per vehicle mean'):

| Model | All MAE | All MdAE | All MAPE (%) | All MdAPE (%) | All SMAPE (%) | Vehicle MAE | Vehicle MdAE | Vehicle MAPE (%) | Vehicle MdAPE (%) | Vehicle SMAPE (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LinearRegression** | 4.44 | 2.56 | 300 | 32.5 | 46.4 | 5.45 | 4.95 | 302 | 40.9 | 46.8 |
| **QuantileRegressor** | 4.34 | 2.46 | 287 | 31.6 | 45.4 | 5.43 | 4.86 | 291 | 38.4 | 46.4 |
| **HistGradientBoostingRegressor** | 3.83 | 1.97 | 277 | 25 | 39.4 | 4.84 | 4.13 | 287 | 32.4 | 40.7 |
| **MedianValue** | 6 | 3.24 | 236 | 49.3 | 59.4 | 7.78 | 7.29 | 219 | 50.6 | 64.3 |
| **MeanValue** | 6.91 | 5.4 | 355 | 64.4 | 67.4 | 8.02 | 7.52 | 320 | 69.3 | 66.4 |
| **LastValue** | 5.1 | 2.73 | 330 | 37.2 | 54.4 | 6.32 | 5.26 | 343 | 40.6 | 54.3 |

![Model Performance Comparison](performance.png)

---

## 3. Best Model Error Density & Distribution

Comparison of absolute error density between all sessions and the vehicle-level mean distribution for `HistGradientBoostingRegressor`:

![HGBR Error Distribution](best_model.png)

---

## 4. Behavioral & Vehicle-Level Analysis

Ordinary Least Squares (OLS) regression analyzing the dependence of participant-level Median Absolute Percentage Error (MdAPE) on user charging behaviors:

- **Variation of session duration (P90-P10 / Median)**: Slope $a = 1.209$, Intercept $b = 30.144$, $r^2 = 0.003$, $p = 0.4198$ (Not statistically significant)
- **Battery size (CarKWh)**: Slope $a = 0.370$, Intercept $b = 21.735$, $r^2 = 0.112$, $p < 0.0001$ (Statistically significant ($p < 0.05$))
- **Session start hour (StartHour)**: Slope $a = -1.609$, Intercept $b = 58.707$, $r^2 = 0.010$, $p = 0.1083$ (Not statistically significant)

![Performance Over Features](performance_over_features.png)

---

## 5. Model Agreement & Multi-Model Comparison

Visual comparison of participant-level Test MAE values between benchmark models (HGBR versus Linear Regression, HGBR versus Quantile Regressor, and Linear Regression versus Quantile Regressor). The main scatter plot displays agreement across all vehicles, while the inset axis zooms into the $[0, 20]$ error range:

![Model Agreement](model_agreement.png)

---

## 6. Temporal Stability & Seasonality Analysis

- **Temporal Error Shift Across Test Split:** The relative change in MAE from the first to second half of the test set is calculated per participant as $(\text{mean}(\text{first-half AE}) - \text{mean}(\text{second-half AE})) / \text{mean}(\text{all participant Test AE})$. Across evaluated vehicles, the mean relative change is **-0.054 \pm 0.353**. Under this formula, a negative value indicates that prediction error is higher in the second half of the participant's test sessions than in the first half.
- **Split Data Availability:**
  - **Train split:** 178.3 \pm 139.7 sessions over 298.8 \pm 138.2 days per vehicle.
  - **Test split:** 48.0 \pm 33.2 sessions over 70.5 \pm 17.3 days per vehicle.

![Error Over Time Histogram](ae_over_time_hist.png)

![Seasonality](seasonality.png)

---

## 7. TreeSHAP Explainability & Feature Importance

Global feature importance ranking evaluated via TreeSHAP on test sessions:

| Rank | Feature Name | Mean Absolute SHAP Value |
| :---: | :--- | :---: |
| 1 | `CarKWh` | 2.8210 |
| 2 | `LastConsumedkWh` | 2.0676 |
| 3 | `TimeSinceLastStop` | 1.4485 |
| 4 | `StartHourNorm` | 0.7093 |
| 5 | `LastDuration` | 0.5628 |
| 6 | `StartCos` | 0.3493 |
| 7 | `CarKW` | 0.2454 |
| 8 | `month` | 0.1853 |
| 9 | `StartSin` | 0.1727 |
| 10 | `weekday` | 0.0263 |
| 11 | `sessionsToday` | 0.0246 |
| 12 | `weekend` | 0.0069 |
| 13 | `holiday` | 0.0007 |

Data files: [shap_global_importance.csv](shap_global_importance.csv), [shap_vehicle_importance.csv](shap_vehicle_importance.csv)

![SHAP Values](shap_values.png)

---

## 8. Subgroup Analysis: Same-day vs. Overnight Charging

Independent evaluation comparing same-day charging sessions (where StopTime and StartTime fall on the same UTC calendar date, day difference == 0) against overnight charging sessions (where StopTime falls on a subsequent UTC calendar date, day difference > 0):

| Subgroup | Sessions | Train $N$ | Test $N$ | MAE | MdAE | MAPE (%) | MdAPE (%) | SMAPE (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **All Sessions** | 59,520 | 47,616 | 11,904 | 3.83 | 1.97 | 277 | 25 | 39.4 |
| **Same-day Sessions** | 21,436 | 17,148 | 4,288 | 2.88 | 1.95 | 384 | 35.1 | 48.7 |
| **Overnight Sessions** | 38,084 | 30,467 | 7,617 | 3.94 | 1.85 | 110 | 19.4 | 31.9 |

Data file: [same_day_overnight_metrics.csv](same_day_overnight_metrics.csv)

![Same-Day vs Overnight Comparison](same_day_overnight_comparison.png)

---

*Report generated automatically by TARANG Stage 3 Model Evaluation Suite.*