# TARANG — Inference Contract & Interface Specification

**Domain**: Electric Vehicle Charging Intelligence System  
**Stage**: Stage 4 — Inference  
**Targets**: Plugged-In Duration (`Duration`, hours) and Energy Consumption (`ConsumedkWh`, kWh)  
**Package**: `src/tarang/inference/`  
**CLI Tool**: `scripts/run_inference.py`

---

## 1. System Role & Scope Boundaries

The TARANG inference layer provides real-time single-session and high-throughput batch prediction engines for EV charging behavior. At vehicle arrival time ($t_0$), the system forecasts:

1. **Plugged-In Duration (`Duration`)**: Total bay occupancy duration in continuous hours.
2. **Energy Consumption (`ConsumedkWh`)**: Active energy delivered to the vehicle battery in kilowatt-hours.

### Scope Boundaries
- **Advisory Output**: Inference strictly returns raw continuous estimates directly from fitted estimators. It applies no artificial clipping, rounding, thresholding, or probability calibrations. Downstream interpretation, flexibility ratios ($\text{ChargingDuration} / \text{PluggedInDuration}$), and managed charging schedules are deferred to Stage 5 (Decision Support).
- **No In-Flight Training**: Inference exclusively loads pre-trained Stage 2 model artifacts. No fitting, refitting, or incremental learning occurs during inference execution.
- **Zero Imputation**: Missing or corrupted predictors are never silently backfilled or imputed.

---

## 2. Authoritative 13-Predictor Contract

All models receive exactly 13 predictors in the authoritative ordering defined by `tarang.config.FEATURE_NAMES`:

| Index | Predictor | Type | Valid Range / Format | Source / Description |
|---|---|---|---|---|
| 1 | `weekday` | `int` (binary) | `0` or `1` | 1 if arrival is Monday–Friday and not a UK bank holiday; 0 otherwise. |
| 2 | `weekend` | `int` (binary) | `0` or `1` | 1 if arrival is Saturday–Sunday and not a UK bank holiday; 0 otherwise. |
| 3 | `holiday` | `int` (binary) | `0` or `1` | 1 if arrival matches one of the 13 designated UK bank holidays; 0 otherwise. |
| 4 | `month` | `int` | `1` – `12` | Calendar month of arrival timestamp ($t_0$). |
| 5 | `StartCos` | `float` | $[-1.0, 1.0]$ | $\cos(2\pi \cdot \text{StartHourNorm})$. |
| 6 | `StartSin` | `float` | $[-1.0, 1.0]$ | $\sin(2\pi \cdot \text{StartHourNorm})$. |
| 7 | `StartHourNorm` | `float` | $[0.0, 1.0]$ | Arrival start hour continuous fraction divided by global `max_start_hour`. |
| 8 | `TimeSinceLastStop` | `float` | $\ge 0.0$ hours | Elapsed time from immediately preceding session's raw `StopTime` to new `StartTime`. |
| 9 | `sessionsToday` | `int` | $\ge 0$ | Cumulative count of prior sessions started by participant on same UTC date. |
| 10 | `LastDuration` | `float` | $[0.0, 336.0]$ hours | Duration of immediately preceding session, capped at 336.0 hours (14 days). |
| 11 | `LastConsumedkWh` | `float` | $\ge 0.0$ kWh | Metered energy delivered in immediately preceding session. |
| 12 | `CarKW` | `float` | $> 0.0$ kW | Vehicle onboard charger acceptance power rating. |
| 13 | `CarKWh` | `float` | $> 0.0$ kWh | Vehicle battery pack gross/usable capacity. |

### Normalization Anchor (`StartHourNorm`)
- The continuous start hour is computed as:
  $$\text{StartHour} = \text{hour} + \frac{\text{minute}}{60} + \frac{\text{second}}{3600}$$
- `StartHourNorm` normalizes this value against the global dataset-wide maximum `max_start_hour` determined in Stage 1 screening and persisted in `data/processed/electric_nation/processing_summary.json`.
- Imputation, recalculation from batch slices, or fallback to 24.0 is **prohibited**. If `processing_summary.json` is missing or invalid, inference raises an actionable error requiring Stage 1 execution.

### Holiday Override Rule
On UK bank holidays (the authoritative 13 trial holidays across 2017–2018), both `weekday` and `weekend` flags are set to `0`, and `holiday` is set to `1`.

---

## 3. Timestamp & History Integrity

1. **UTC Basis**: All arrival and historical timestamps are standardized to UTC. Naive datetime inputs are treated as UTC; timezone-aware inputs are converted to UTC without clock-hour modification.
2. **Participant History**: Single-session feature generation requires completed prior sessions for the participant containing:
   - `StartTime` (UTC)
   - `StopTime` (UTC)
   - `ConsumedkWh` (numeric float)
3. **Chronological Validity**:
   - Each historical session must have $\text{StopTime} \ge \text{StartTime}$.
   - All historical sessions must precede the new session ($\text{StartTime}_{\text{prev}} < \text{StartTime}_{\text{new}}$).
   - Negative gaps ($\text{StartTime}_{\text{new}} < \text{StopTime}_{\text{prev}}$) represent impossible session overlaps and are rejected.
4. **TimeSinceLastStop vs. LastDuration**:
   - `TimeSinceLastStop` is measured against the **unmodified raw `StopTime`** of the preceding session.
   - `LastDuration` applies the 336-hour cap (`min(raw_duration, 336.0)`). The cap does not alter the historical `StopTime` used for gap calculation.

---

## 4. Cold-Start and Missingness Policies

- **Cold-Start Participants**: New participants without completed prior sessions cannot supply the required lag states (`TimeSinceLastStop`, `LastDuration`, `LastConsumedkWh`). Rather than inventing default or imputed values, the inference engine explicitly rejects the request with a `ValueError("Cold-start condition: participant history is empty...")`.
- **Preceding Session Missing Energy**: If the immediately preceding session has missing/null `ConsumedkWh`, the lag feature cannot be resolved. In accordance with the Stage 2 model contract, the request is rejected without backward search or imputation.
- **Strict Numeric Verification**: All 13 predictor columns must be numeric floats/ints. Missing values (`NaN`, `None`), infinities (`inf`, `-inf`), and negative vehicle specs are strictly rejected before calling estimators.

---

## 5. Model Registry & Artifact Management

Model artifacts are loaded from Stage 2 directories:
- Duration: `data/processed/electric_nation/modeling/duration/models/{ModelName}.joblib`
- Energy: `data/processed/electric_nation/modeling/energy/models/{ModelName}.joblib`

### Supported Architectures
| Architecture | Model Name | Description | Default Role |
|---|---|---|---|
| `HistGradientBoostingRegressor` | `HistGradientBoostingRegressor` | Histogram-based gradient boosted decision trees with early stopping. | **Primary Default Model** (both targets) |
| `LinearRegression` | `LinearRegression` | Ordinary least squares regression. | Benchmark Comparator |
| `QuantileRegressor` | `QuantileRegressor` | L1-regularized median regression (quantile = 0.5, HiGHS solver). | Benchmark Comparator |
| `MedianValue` | `MedianValue` | Training target median baseline. | Baseline (Lowest Energy MAPE) |
| `MeanValue` | `MeanValue` | Training target arithmetic mean baseline. | Baseline Comparator |
| `LastValue` | `LastValue` | Target-specific lag baseline (`LastDuration` or `LastConsumedkWh`). | Benchmark Comparator |

### In-Memory Artifact Caching
To prevent redundant filesystem I/O across high-frequency predictions, `tarang.inference.loading.load_model` caches deserialized estimators in an in-memory registry keyed by `(target, model_name, models_dir)`. The cache can be purged at any time using `tarang.inference.clear_model_cache()`.

---

## 6. Public Python APIs

### Single-Session Feature Construction & Prediction
```python
from tarang.inference import build_inference_features, predict_single, InferenceEngine
import pandas as pd

# Historical sessions for participant
history_df = pd.DataFrame({
    "StartTime": [pd.Timestamp("2018-06-01 08:00:00", tz="UTC")],
    "StopTime": [pd.Timestamp("2018-06-01 12:00:00", tz="UTC")],
    "ConsumedkWh": [14.2],
})

# 1. Functional feature construction + single prediction
features_row = build_inference_features(
    start_time="2018-06-01 18:00:00",
    car_kw=7.0,
    car_kwh=40.0,
    history=history_df,
)

predicted_duration = predict_single(
    features=features_row,
    target="Duration",
    model_name="HistGradientBoostingRegressor",
)
# Returns continuous float hours (e.g. 7.42)

# 2. Stateful InferenceEngine class
engine = InferenceEngine(target="Duration", model_name="HistGradientBoostingRegressor")
predicted_duration = engine.build_and_predict(
    start_time="2018-06-01 18:00:00",
    car_kw=7.0,
    car_kwh=40.0,
    history=history_df,
)
```

### Batch Prediction
```python
from tarang.inference import predict_batch
import pandas as pd

# Load already-engineered features DataFrame (e.g. from features.parquet)
batch_df = pd.read_parquet("data/processed/electric_nation/features.parquet").head(100)

predictions_series = predict_batch(
    features_df=batch_df,
    target="ConsumedkWh",
    model_name="HistGradientBoostingRegressor",
)
# Returns pd.Series aligned with batch_df.index
```

---

## 7. Batch Inference CLI

Batch prediction can be executed directly from the terminal using `scripts/run_inference.py`.

```bash
# Predict Plugged-in Duration on a batch of sessions
python scripts/run_inference.py \
    --input data/processed/electric_nation/features.parquet \
    --output reports/inference/duration_predictions.parquet \
    --target Duration \
    --model HistGradientBoostingRegressor

# Predict Energy Consumption from CSV input to CSV output
python scripts/run_inference.py \
    --input test_sessions.csv \
    --output predictions_energy.csv \
    --target ConsumedkWh \
    --model HistGradientBoostingRegressor
```

### CLI Arguments
- `--input` / `-i` (required): Path to `.parquet` or `.csv` dataset containing the 13 canonical features. Extra metadata columns are preserved.
- `--output` / `-o` (required): Path to destination file (`.parquet` or `.csv`).
- `--target` / `-t` (required): Target name (`Duration` or `ConsumedkWh`, case-insensitive).
- `--model` / `-m` (optional): Model architecture (default: `HistGradientBoostingRegressor`).
- `--models-dir` (optional): Custom model artifacts root directory.

