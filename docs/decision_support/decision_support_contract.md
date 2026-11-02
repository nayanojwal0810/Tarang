# TARANG — Stage 5: Decision Support Contract & Operational Specification

**Domain**: Electric Vehicle Charging Intelligence System  
**Stage**: Stage 5 — Decision Support  
**Module**: `src/tarang/decision_support/`  
**CLI Entry Point**: `scripts/run_decision_support.py`  
**Dependencies**: Stages 1–4 Accepted  

---

## 1. Operational Scope & Advisory Principles

TARANG Stage 5 establishes the decision support and flexibility advisory layer for EV charging infrastructure. It translates raw multi-target machine learning predictions (`Duration` and `ConsumedkWh`) and historical charging records into operational advisory metrics.

> [!IMPORTANT]
> **Advisory Scope Boundary**:
> TARANG operates strictly as an **advisory intelligence system** to support fleet managers, charge point operators (CPOs), distribution network operators (DNOs), and charging facility planners.
> 
> The system explicitly **does not**:
> 1. Execute direct charger hardware actuation, power modulation, or relay switching.
> 2. Implement automated charging dispatch, active vehicle throttling, or real-time grid curtailment.
> 3. Optimize dynamic tariffs, electricity spot-market pricing, or carbon-intensity dispatch.
> 4. Modify or clip raw regression predictions; estimator outputs are validated for non-negativity and finiteness via explicit validity flags while preserving original values. This numeric check validates model outputs against impossible negative or non-finite values but does not constitute a full physical feasibility assessment.

---

## 2. Session Advisory Schema & Data Structures

The decision support layer provides both single-session and high-throughput batch prediction APIs.

### 2.1 Single-Session Advisory: `SessionAdvisory`

Single-session predictions are encapsulated in an immutable, frozen dataclass:

```python
@dataclass(frozen=True)
class SessionAdvisory:
    arrival_time_utc: pd.Timestamp
    predicted_duration_hours: float
    predicted_energy_kwh: float
    projected_unplug_time_utc: pd.Timestamp
    duration_prediction_valid: bool
    energy_prediction_valid: bool
```

| Field Name | Type | Description |
| :--- | :--- | :--- |
| `arrival_time_utc` | `pd.Timestamp` (UTC) | Normalized session arrival timestamp in UTC basis. |
| `predicted_duration_hours` | `float` | Raw estimator prediction of plugged-in duration in continuous hours. |
| `predicted_energy_kwh` | `float` | Raw estimator prediction of delivered energy in kilowatt-hours. |
| `projected_unplug_time_utc` | `pd.Timestamp` (UTC) or `pd.NaT` | Calculated departure timestamp (`arrival_time_utc + predicted_duration_hours`). Evaluates to `pd.NaT` if duration prediction is flagged invalid. |
| `duration_prediction_valid` | `bool` | True if duration estimate is finite and non-negative (`>= 0.0`); False otherwise. |
| `energy_prediction_valid` | `bool` | True if energy estimate is finite and non-negative (`>= 0.0`); False otherwise. |

### 2.2 Batch Advisories DataFrame Schema

Batch processing via `predict_batch_advisories(features_df)` validates arrival timestamps before invoking models, rejects missing, `NaT`, or malformed timestamps, and preserves all existing input columns, metadata, and original index structures (including non-consecutive or duplicate indices), appending five standardized advisory columns:

| Appended Column Name | Data Type | Nullable | Validation & Computation Rule |
| :--- | :--- | :---: | :--- |
| `predicted_Duration` | `float64` | No | Raw model prediction from `Duration` estimator. |
| `predicted_ConsumedkWh` | `float64` | No | Raw model prediction from `ConsumedkWh` estimator. |
| `projected_unplug_time_utc` | `datetime64[ns, UTC]` | Yes (`NaT`) | Vectorized calculation: `StartTime_UTC + Timedelta(hours=predicted_Duration)` where valid; `pd.NaT` where duration is invalid. |
| `duration_prediction_valid` | `bool` | No | `np.isfinite(predicted_Duration) & (predicted_Duration >= 0.0)` |
| `energy_prediction_valid` | `bool` | No | `np.isfinite(predicted_ConsumedkWh) & (predicted_ConsumedkWh >= 0.0)` |

---

## 3. Numeric Validity & Timestamp Calculation Rules

### 3.1 Non-Negativity and Finiteness Verification

Estimator predictions represent physical quantities (time in hours, energy in kWh). Negative or infinite values can arise from unbounded linear regressors or extreme feature inputs. Rather than silently clipping or altering predictions:

$$\text{is\_valid}(y) = \begin{cases} \text{True} & \text{if } y \in \mathbb{R},\ y \ge 0 \\ \text{False} & \text{otherwise} \end{cases}$$

1. **Numeric Validity Check**: A finite, non-negative check verifies basic mathematical validity. It does not assert comprehensive physical feasibility or apply artificial upper bounds on energy or duration.
2. **Flag Retention**: `duration_prediction_valid` and `energy_prediction_valid` record model validity explicitly.
3. **Prediction Preservation**: The continuous values in `predicted_Duration` and `predicted_ConsumedkWh` remain untouched for transparent auditing and metric calculation.
4. **NaT Assignment**: When `duration_prediction_valid` is `False`, `projected_unplug_time_utc` is set to `pd.NaT` to protect downstream bay scheduling systems from computing erroneous or past departure times.

### 3.2 Timezone and Calendar Basis

1. **UTC Basis**: All arrival and unplug timestamps operate on UTC basis. Naive timestamps are interpreted as UTC; timezone-aware inputs are converted to UTC.
2. **Batch Timestamp Pre-Validation**: In batch processing, arrival timestamps in `StartTime` are validated before invoking any inference models. Missing timestamps, `NaT` values, or malformed strings raise an immediate `ValueError`.
3. **Local UK Date Roll-over**: For flexibility analysis and calendar day grouping, session timestamps are converted to `Europe/London` to correctly reflect local midnight transitions (accounting for GMT and BST daylight saving time).

---

## 4. Smart-Charging Flexibility Specification

Charging flexibility represents the descriptive ratio of active charging duration to total connection duration:

$$\text{Flexibility Ratio} = \frac{\text{ChargingDuration} / 60.0}{(\text{StopTime} - \text{StartTime})_{\text{hours}}}$$

### 4.1 Session Classification (Local UK Dates)

Using `Europe/London` local date evaluation:
- **Same-Day Sessions**: $\text{Date}(\text{StartTime}_{\text{local}}) == \text{Date}(\text{StopTime}_{\text{local}})$
- **Overnight Sessions**: $\text{Date}(\text{StartTime}_{\text{local}}) \ne \text{Date}(\text{StopTime}_{\text{local}})$

### 4.2 Ratio Computation Rules & Histogram Alignment

The decision support flexibility layer re-uses the exact established EDA calculation from `tarang.analysis.exploratory_data_analysis.calculate_charging_flexibility()` without modification. The following mathematical and structural rules govern its outputs:

1. **Validity Mask**: A session is valid for flexibility analysis if `ChargingDuration.notna()` and raw plugged-in duration `(StopTime - StartTime) > 0` in hours.
2. **No Clipping on Raw Ratios**: Raw valid flexibility ratios are not clipped or bounded artificially. Ratios exceeding 1.0 (e.g. from slight measurement or timestamp discrepancies in source data) are retained in the valid ratio arrays.
3. **Histogram Range [0.0, 1.0]**: The 25-bin histogram covers exactly the interval $[0.0, 1.0]$. Ratios falling outside this interval are not included in histogram bin counts.
4. **Histogram Percentage Normalization**: Histogram percentage distributions (`same_day_pct`, `overnight_pct`) are normalized over the count of in-range observations (`same_day_in_range`, `overnight_in_range`).
5. **Mean and Median Denominators**: Summary statistics (`mean_ratio`, `median_ratio`) are computed across **all valid ratios**, not only histogram-included ratios.
6. **Empirical Example (Canonical Dataset)**:
   - Same-day: 15,947 valid ratios; 15,947 included in `[0, 1]` histogram.
   - Overnight: 29,349 valid ratios; 29,348 included in `[0, 1]` histogram (1 session with ratio $> 1.0$).

### 4.3 Schema Fallback and Availability Check

If an input DataFrame lacks the required historical connection columns (`ChargingDuration`, `StartTime`, `StopTime`) or is empty:
- `available` is set to `False`.
- `reason` details why the analysis was unavailable (e.g., missing column names or empty DataFrame).
- `missing_columns` contains the list of missing column names when applicable, or `None`.
- Fabricated zero-valued statistics, counts, means, medians, or histogram distributions are **strictly not returned**.

---

## 5. Artifacts and Directory Layout

The decision support pipeline persists operational datasets, CSV statistical summaries, and publication-ready Markdown documentation:

```
Tarang/
|-- data/
|   |-- processed/
|       |-- electric_nation/
|           |-- decision_support/
|               |-- session_advisories.parquet   # Full batch advisories dataset (gitignored)
|
|-- reports/
|   |-- decision_support/
|       |-- prediction_summary.csv               # Distribution percentiles across valid predictions
|       |-- flexibility_summary.csv              # Same-day vs overnight flexibility metrics
|       |-- decision_support_summary.md          # Comprehensive Markdown summary report
```

### 5.1 Supported File Formats

Both the CLI (`scripts/run_decision_support.py`) and programmatic artifact saving (`save_decision_support_artifacts`) strictly support:
- `.parquet`
- `.csv`

File extensions are validated case-insensitively (`.parquet`, `.csv`, `.PARQUET`, `.CSV`). Any other format (e.g., `.txt`, `.json`, `.xlsx`) is rejected with an actionable `ValueError` or error log.

### 5.2 Prediction Summary Table (`prediction_summary.csv`)

| Column | Description |
| :--- | :--- |
| `target` | Prediction target (`Duration` or `ConsumedkWh`). |
| `unit` | Physical unit (`hours` or `kWh`). |
| `total_count` | Total sessions evaluated. |
| `valid_count` | Count of finite, non-negative predictions. |
| `invalid_count` | Count of flagged invalid predictions. |
| `mean` | Arithmetic mean of valid predictions. |
| `std` | Sample standard deviation of valid predictions ($ddof=1$). |
| `min` | Minimum valid prediction. |
| `25%` | 25th percentile (1st quartile) of valid predictions. |
| `50%` | 50th percentile (median) of valid predictions. |
| `75%` | 75th percentile (3rd quartile) of valid predictions. |
| `90%` | 90th percentile of valid predictions. |
| `max` | Maximum valid prediction. |

---

## 6. CLI and Python API Usage

### 6.1 CLI Execution

```bash
# Run batch decision support on canonical dataset with default HGBR models
python scripts/run_decision_support.py

# Custom input/output paths and specific model architecture
python scripts/run_decision_support.py \
    --input data/processed/electric_nation/features.parquet \
    --output data/processed/electric_nation/decision_support/session_advisories.parquet \
    --report-dir reports/decision_support \
    --model HistGradientBoostingRegressor
```

### 6.2 Python API

```python
import pandas as pd
from tarang.decision_support import DecisionSupportEngine, analyze_charging_flexibility

# Initialize engine with pre-trained models
engine = DecisionSupportEngine(model_name="HistGradientBoostingRegressor")

# 1. Single-session advisory with arrival parameters and participant history
history = pd.DataFrame({
    "StartTime": [pd.Timestamp("2018-05-15 08:30:00", tz="UTC")],
    "StopTime": [pd.Timestamp("2018-05-15 16:45:00", tz="UTC")],
    "ConsumedkWh": [22.4],
})

advisory = engine.predict_single_advisory(
    start_time="2018-05-15 18:00:00",
    car_kw=7.0,
    car_kwh=40.0,
    history=history,
)
print(f"Arrival (UTC): {advisory.arrival_time_utc}")
print(f"Predicted Duration: {advisory.predicted_duration_hours:.2f} h")
print(f"Projected Unplug (UTC): {advisory.projected_unplug_time_utc}")
print(f"Duration Numeric Validity: {advisory.duration_prediction_valid}")

# 2. Batch advisories on pre-engineered canonical features
features_df = pd.read_parquet("data/processed/electric_nation/features.parquet")
batch_df = engine.predict_batch_advisories(features_df)

# 3. Descriptive flexibility analysis
flex_stats = analyze_charging_flexibility(features_df)
print(f"Overnight Median Flexibility: {flex_stats['overnight_median_ratio']:.3f}")
```
