# TARANG — EV Charging Intelligence System

TARANG is an Electric Vehicle (EV) Charging Intelligence System designed to predict charging session behaviors — specifically **plugged-in duration** and **energy consumption** — and model charging flexibility to support smart charging management, grid load integration, and bay occupancy planning as advisory decision support.

This repository implements the complete software foundation, data ingestion and auditing pipeline, deterministic participant screening, 13-predictor feature engineering, and exploratory data analysis using empirical records from the UK Electric Nation smart charging trial.

---

## Repository Structure

```
Tarang/
|-- .github/
|   |-- workflows/
|       |-- ci.yml                   # Automated continuous integration test suite
|
|-- data/
|   |-- raw/
|   |   |-- electric_nation/         # Source trial Excel workbooks (excluded from Git)
|   |-- processed/
|       |-- electric_nation/         # Canonical Parquet dataset and summary JSON (excluded from Git)
|
|-- docs/                            # In-depth architectural, data, and inference documentation
|   |-- architecture/project_structure.md
|   |-- data/data_contract.md
|   |-- data/data_dictionary.md
|   |-- inference/inference_contract.md
|
|-- reports/                         # Research findings, audit reports, and visual assets
|   |-- data_audit/                  # Feasibility and quality audit report and CSV summaries
|   |-- eda/                         # Exploratory data analysis report and distribution plots
|   |-- modeling/                    # Stage 3 evaluation summaries, SHAP plots, and metric tables
|       |-- duration/
|       |-- energy/
|
|-- scripts/                         # Canonical pipeline CLI entry points
|   |-- run_data_audit.py            # Stage 1: Data quality and reconciliation audit
|   |-- prepare_data.py              # Stage 1: Screening and 13-feature engineering
|   |-- run_eda.py                   # Stage 1: Descriptive EDA and flexibility analysis
|   |-- run_modeling.py              # Stage 2: Dual-target predictive modeling benchmark
|   |-- run_evaluation.py            # Stage 3: Model evaluation, TreeSHAP, and subgroup analysis
|   |-- run_inference.py             # Stage 4: Batch prediction on pre-engineered features
|
|-- src/
|   |-- tarang/
|       |-- __init__.py
|       |-- config.py                # Single source of truth for paths, schemas, and parameters
|       |-- data/                    # Ingestion, validation, auditing, and screening
|       |   |-- __init__.py
|       |   |-- loading.py           # Robust Excel/Parquet loading with validation
|       |   |-- validation.py        # Schema validation and timestamp verification
|       |   |-- quality_audit.py     # Quality auditing (overlaps, missingness, reconciliation)
|       |   |-- screening.py         # Exact research-grade participant screening pipeline
|       |   |-- preparation.py       # High-level pipeline orchestration and parquet persistence
|       |   |-- audit_reporting.py   # Dynamic Markdown and CSV audit report generator
|       |   |-- audit.py             # Re-export facade maintaining legacy compatibility
|       |-- features/                # Feature engineering modules
|       |   |-- __init__.py
|       |   |-- calendar_features.py # Exact 13 UK Bank Holidays, binary weekday/weekend, day indicators
|       |   |-- time_features.py     # StartHour, StartHourNorm, cyclical StartCos / StartSin
|       |   |-- historical_features.py # TimeSinceLastStop, sessionsToday, LastDuration, LastConsumedkWh
|       |   |-- feature_engineering.py # Authoritative 13 feature definitions and extraction
|       |-- modeling/                # Predictive modeling and benchmark regressors
|       |   |-- __init__.py
|       |   |-- models.py            # Benchmark regressors (LR, QR, HGB) and baselines (Mean, Median, LastValue)
|       |   |-- metrics.py           # Reference-aligned evaluation metrics and error distribution statistics
|       |   |-- experiment.py        # Chronological split, training, evaluation, and artifact persistence
|       |-- evaluation/              # Stage 3 evaluation, TreeSHAP, and behavioral analysis
|       |   |-- __init__.py
|       |   |-- loading.py           # MultiIndex performance, predictions, and model loader
|       |   |-- analysis.py          # Performance tables, user characteristics, model agreement, seasonality
|       |   |-- explainability.py    # TreeSHAP feature importance (global and vehicle distributions)
|       |   |-- subgroup_analysis.py # Same-day vs overnight session partitioning and modeling
|       |   |-- visualizations.py    # Publication-ready Matplotlib figures matching reference styling
|       |   |-- reporting.py         # Dynamic Markdown evaluation report generator
|       |-- inference/               # Single-session and batch inference pipeline
|       |   |-- __init__.py
|       |   |-- loading.py           # Model artifact and summary loader with in-memory caching
|       |   |-- validation.py        # Strict 13-predictor schema and session history validation
|       |   |-- feature_builder.py   # Single-session 13-predictor construction from history
|       |   |-- engine.py            # Fast prediction engine (predict_single, predict_batch)
|       |-- analysis/                # Statistical analysis and EDA
|       |   |-- __init__.py
|       |   |-- exploratory_data_analysis.py # Availability, flexibility, and bivariate distributions
|       |-- visualization/           # Visualization generators
|           |-- __init__.py
|           |-- eda_plots.py         # Multi-panel Matplotlib figures with marginal distributions
|
|-- tests/                           # Complete pytest test suite (121 unit & integration tests)
|   |-- unit/                        # Fast, isolated unit test suite
|   |-- integration/                 # End-to-end integration and pipeline tests
|
|-- .gitignore
|-- pyproject.toml                   # Modern PEP 517/621 package build configuration
|-- README.md                        # User-facing overview, quickstart, and inference guide
```

---

## Dataset Screening and Modeling Cohort

The TARANG data pipeline ingests raw smart charging records from the UK Electric Nation trial and executes a deterministic, multi-stage screening sequence:

| Screening Stage | Participants | Sessions | Impact Description |
| :--- | :---: | :---: | :--- |
| **Raw Ingestion** | 300 | 71,264 | Initial trial transactions |
| **Trial 3 Exclusion** | 300 | 65,141 | Exclude Trial 3 records (`Trial != 3`) |
| **Duration Filtering** | 299 | 64,630 | Remove micro-sessions ($< 5$ minutes) |
| **Negative-Gap Filtering** | 299 | 60,601 | Remove 132 sessions with `TimeSinceLastStop < 0` without gap recalculation |
| **First Session Removal** | 298 | 60,302 | Drop first session per participant (299 dropped; 1 user has 0 left) |
| **Minimum History Filter** | 287 | 60,246 | Exclude participants with $< 10$ retained sessions (12 excluded, 56 sessions) |
| **Observation-Density Filter** | **267** | **59,520** | Exclude participants where $\text{sessions} < \text{span} / 7.0$ (20 excluded, 726 sessions) |

The final eligible population consists of exactly **267 participants** and **59,520 sessions**. Missing energy values are retained in the research dataset without artificial exclusion.

### Authoritative 13 Features

All predictors are strictly evaluated at arrival time $t_0 = \text{StartTime}$ (UTC basis):
1. `weekday`: Binary indicator (1 for Monday–Friday, 0 for Saturday–Sunday; 0 on Bank Holidays)
2. `weekend`: Binary indicator (1 for Saturday–Sunday, 0 for Monday–Friday; 0 on Bank Holidays)
3. `holiday`: Binary indicator for exactly 13 designated UK holiday dates (2017–2018). On a holiday, both `weekday` and `weekend` are 0.
4. `month`: Calendar month index (1..12) of `StartTime`
5. `StartCos`: Cyclical cosine coordinate $\cos(2\pi \cdot \text{StartHourNorm})$
6. `StartSin`: Cyclical sine coordinate $\sin(2\pi \cdot \text{StartHourNorm})$
7. `StartHourNorm`: Continuous arrival hour normalized by dataset-wide maximum $\text{StartHour} / \max(\text{StartHour})$ across the filtered dataset (including null participant records)
8. `TimeSinceLastStop`: Elapsed hours since immediately preceding session stop time
9. `sessionsToday`: Daily cumulative count of prior sessions on the same calendar date for that participant
10. `LastDuration`: Elapsed duration in hours of immediately preceding session. For the first retained session, this is populated from the dropped preceding session (zero missingness).
11. `LastConsumedkWh`: Active energy in kWh of immediately preceding session. For the first retained session, this is populated from the dropped preceding session (missing only if preceding energy was missing).
12. `CarKW`: Vehicle onboard charger power rating (3.6 kW or 7.0 kW)
13. `CarKWh`: Vehicle battery capacity (6.2 kWh to 100.0 kWh)

### Canonical Processed Dataset

The deterministic screening and engineering pipeline generates the canonical modeling dataset:
- **Location**: `data/processed/electric_nation/features.parquet`
- **Dimensions**: Exactly **59,520 rows $\times$ 30 columns** across 267 screened participants.
- **Data Dictionary**: See [docs/data/data_dictionary.md](docs/data/data_dictionary.md) for full definitions, units, types, and value ranges for all 30 columns.
- **Data Contract**: See [docs/data/data_contract.md](docs/data/data_contract.md) for the authoritative screening sequence, feature availability, and anti-leakage rules.

---

## Project Stages & Current Progress

TARANG follows an established 7-stage architectural roadmap:

1. **Stage 1 — Foundation & Data Pipeline** (Completed): Ingestion, audit, deterministic screening, 13 features, and descriptive EDA.
2. **Stage 2 — Benchmark Predictive Modeling** (Completed): Dual-target benchmark models (LR, QR, HGBR) and baselines (Mean, Median, LastValue).
3. **Stage 3 — Model Evaluation, Behavioral Analysis & Explainability** (Completed): TreeSHAP explainability, behavioral regressions, model agreement, temporal stability, subgroup analysis, and publication reports.
4. **Stage 4 — Inference** (Completed): High-performance batch and single-session prediction engine, feature schema enforcement, lag state retrieval, and CLI tooling.
5. **Stage 5 — Decision Support** (Planned): Smart charging advisory analytics, flexibility modeling ($\text{ChargingDuration} / \text{PluggedInDuration}$), demand shifting potential, and bay occupancy planning.
6. **Stage 6 — Streamlit Application** (Planned): Interactive web dashboard for session inspection, predictive intelligence, flexibility visualization, and scenario planning.
7. **Stage 7 — Integration, Validation & Release** (Planned): End-to-end integration testing, system-wide performance benchmarking, release packaging, and user documentation.

---

## Getting Started

### Prerequisites

- Python 3.10+ (tested on Python 3.11, 3.12, and 3.13)
- Windows, macOS, or Linux

### Installation

Clone the repository and install the package:

```bash
git clone https://github.com/nayanojwal0810/Tarang.git
cd Tarang

# Standard package installation
pip install -e .

# Development and testing installation (includes pytest)
pip install -e ".[dev]"
```

### Raw Data Setup

Place the official Electric Nation trial workbooks in `data/raw/electric_nation/`:
- `CrowdCharge Transactions.xlsx` (Sheet: `Transaction Data`)
- `ChargerInstall.xlsx` (Sheet: `20181126_ChargerInstall_DriveEl`)

*Note: Raw trial workbooks and generated Parquet datasets are excluded from Git via `.gitignore`.*

---

## Execution Instructions

### 1. Run Data Quality Audit
Generates comprehensive audit metrics, CSV summaries, and dynamic Markdown report in `reports/data_audit/`:

```bash
python scripts/run_data_audit.py
```

### 2. Prepare Canonical Modeling Dataset
Executes deterministic screening and 13-feature engineering, producing `features.parquet` (59,520 sessions) and `processing_summary.json`:

```bash
python scripts/prepare_data.py
```

### 3. Run Exploratory Data Analysis (EDA)
Generates empirical summary statistics and visualization figures in `reports/eda/` using `Europe/London` local time for arrival hours and same-day vs. overnight classification:

```bash
python scripts/run_eda.py
```

Generated EDA artifacts:
- `reports/eda/eda_summary.md`: Summary report including participant availability and charging flexibility ratios.
- `reports/eda/data_availability.png`: 4-panel distribution of participant spans, sessions, weekly cadence, and durations (26 bins).
- `reports/eda/duration_by_start_hour.png`: 2D joint histogram of plugged-in duration vs. local arrival hour (0.5h bins, 72h cutoff, log scale).
- `reports/eda/energy_by_start_hour.png`: 2D joint histogram of energy consumption vs. local arrival hour (0.5h, 0.5 kWh bins, 75 kWh cutoff, log scale).
- `reports/eda/charging_flexibility.png`: Density distributions of flexibility ratios for same-day vs. overnight sessions.

### 4. Run Predictive Modeling Benchmark
Executes the established dual-target benchmark regression pipeline across both prediction targets (`Duration` and `ConsumedkWh`) using the 13 canonical predictors, chronological 80/20 train/test split, and the six benchmark models:

```bash
python scripts/run_modeling.py
```

Generated modeling artifacts stored in `data/processed/electric_nation/modeling/`:
- `duration/`:
  - `models/`: Trained serialized models (`LinearRegression.joblib`, `QuantileRegressor.joblib`, `HistGradientBoostingRegressor.joblib`, `MedianValue.joblib`, `MeanValue.joblib`, `LastValue.joblib`).
  - `performance.csv`: 3-level MultiIndex matrix containing aggregate (`ALL`) and per-participant metrics (`medae`, `mae`, `medape`, `mape`, `rmse`, `smape`, and `ae`/`ape` whisker statistics) across Train and Test splits.
  - `predictions.csv`: Canonical session records with split labels (`Train`/`Test`) and model predictions.
- `energy/`:
  - `models/`: Trained serialized models for energy consumption.
  - `performance.csv`: 3-level MultiIndex performance matrix for energy consumption.
  - `predictions.csv`: Canonical session records and energy predictions.

### 5. Run Model Evaluation, Behavioral Analysis & Explainability
Executes the complete Stage 3 evaluation suite for both prediction targets (`Duration` and `ConsumedkWh`):

```bash
# Evaluate both targets
python scripts/run_evaluation.py

# Or evaluate a specific target
python scripts/run_evaluation.py --target Duration
python scripts/run_evaluation.py --target ConsumedkWh
```

Generated evaluation artifacts under `reports/modeling/{duration,energy}/`:
- `evaluation_summary.md`: Publication-ready evaluation summary covering benchmark rankings, behavioral regressions, model agreement, temporal stability, TreeSHAP importance, and subgroup analysis.
- Visual figures (300 DPI): `performance.png`, `best_model.png`, `performance_over_features.png`, `model_agreement.png`, `ae_over_time_hist.png`, `seasonality.png`, `shap_values.png`, `same_day_overnight_comparison.png` (plus 6 charging occupancy heatmaps for duration).
- Quantitative CSV tables: `model_performance.csv`, `shap_global_importance.csv`, `shap_vehicle_importance.csv`, `same_day_overnight_metrics.csv`.

### 6. Run Inference (Batch & Single-Session)
TARANG supports both high-throughput batch prediction on engineered datasets and real-time single-session prediction from session arrival times and participant history.

#### Prediction Targets and Units
- **`Duration`**: Plugged-in bay occupancy duration in continuous hours.
- **`ConsumedkWh`**: Energy consumption delivered to the vehicle battery in kilowatt-hours (kWh).

#### Default and Configurable Models
- **Default Estimator**: `HistGradientBoostingRegressor` (leading performance across both targets).
- **Available Benchmark Models**: `LinearRegression`, `QuantileRegressor`, `HistGradientBoostingRegressor`, `MedianValue`, `MeanValue`, `LastValue`.
- Models are loaded from Stage 2 serialization artifacts in `data/processed/electric_nation/modeling/{duration,energy}/models/`.

#### Batch Inference CLI
Run batch predictions on an existing Parquet or CSV dataset containing the 13 canonical features:

```bash
# Predict Plugged-in Duration (hours) using default HGBR model
python scripts/run_inference.py \
    --input data/processed/electric_nation/features.parquet \
    --output reports/inference/duration_predictions.parquet \
    --target Duration

# Predict Energy Consumption (kWh) using LinearRegression from CSV to CSV
python scripts/run_inference.py \
    --input batch_features.csv \
    --output batch_predictions.csv \
    --target ConsumedkWh \
    --model LinearRegression
```

The CLI preserves all original columns, retains row ordering, and appends a `predicted_{target}` column.

#### Python Single-Session API & History Requirements
In real-time charging applications, single-session predictions are generated by combining arrival time ($t_0$), vehicle specifications, and the participant's prior charging history:

```python
import pandas as pd
from tarang.inference import build_inference_features, predict_single, InferenceEngine

# 1. Historical completed sessions for the participant
# Must contain at least StartTime, StopTime, and ConsumedkWh
history = pd.DataFrame({
    "StartTime": [pd.Timestamp("2018-05-15 08:30:00", tz="UTC")],
    "StopTime": [pd.Timestamp("2018-05-15 16:45:00", tz="UTC")],
    "ConsumedkWh": [22.4],
})

# 2. Build canonical 13-predictor row for new arrival
features_row = build_inference_features(
    start_time="2018-05-15 19:15:00",  # UTC arrival timestamp
    car_kw=7.0,                         # Charger power rating (kW)
    car_kwh=40.0,                       # Vehicle battery capacity (kWh)
    history=history,
)

# 3. Predict plugged-in duration (in continuous hours)
duration_hours = predict_single(
    features=features_row,
    target="Duration",
    model_name="HistGradientBoostingRegressor",
)
print(f"Predicted Duration: {duration_hours:.2f} hours")

# Alternatively, use the stateful InferenceEngine class:
engine = InferenceEngine(target="ConsumedkWh", model_name="HistGradientBoostingRegressor")
energy_kwh = engine.build_and_predict(
    start_time="2018-05-15 19:15:00",
    car_kw=7.0,
    car_kwh=40.0,
    history=history,
)
print(f"Predicted Energy: {energy_kwh:.2f} kWh")
```

#### Important Operational Requirements
- **UTC Timestamp Convention**: All timestamps follow UTC basis. Naive timestamps are interpreted as UTC without modification; timezone-aware timestamps are converted to UTC.
- **Participant History & Cold-Start Rejection**: At least one prior completed session is strictly required to compute historical lag features (`TimeSinceLastStop`, `LastDuration`, `LastConsumedkWh`). Cold-start sessions without history are explicitly rejected; imputation is disallowed.
- **Stage 1 Summary & Stage 2 Artifacts**: Start hour normalization (`StartHourNorm`) requires the dataset-wide `max_start_hour` stored in `data/processed/electric_nation/processing_summary.json` (fallback to 24.0 is disallowed). Prediction execution requires pre-trained Stage 2 `.joblib` model artifacts.
- **Zero Prediction Post-Processing**: The inference layer returns raw continuous estimates without clipping, rounding, or artificial thresholds; advisory interpretation and flexibility modeling are handled in Stage 5.

### 7. Run Test Suite
Run the full test suite including unit tests and integration tests:

```bash
python -m pytest tests/ -v
```

---

## Documentation

Detailed architectural, data contract, and inference specifications are maintained in `docs/`:

- [**Architecture & Project Structure**](docs/architecture/project_structure.md): System architecture, package responsibilities, subsystem data flows, and full roadmap.
- [**Data Contract & Screening Rules**](docs/data/data_contract.md): Exact 10-step screening pipeline, anti-leakage boundaries, and feature engineering specifications.
- [**Data Dictionary**](docs/data/data_dictionary.md): Raw transaction/metadata fields and canonical 30-column definitions, units, and ranges.
- [**Inference Contract & Specification**](docs/inference/inference_contract.md): Stage 4 single-session API, batch CLI, cold-start rules, and model artifact specifications.

---

## System Scope and Advisory Decision Support

TARANG provides predictive intelligence and charging behavioral insights to assist fleet managers, charge point operators, and smart grid planners. Predictions and flexibility metrics serve as **advisory decision support** for bay scheduling and demand management; TARANG does not execute automatic charger hardware dispatch or real-time grid switching control.

---

## References

1. Kreft, M., Brudermueller, T., Fleisch, E., & Staake, T. (2024). *Predictability of electric vehicle charging: Explaining extensive user behavior-specific heterogeneity*. Applied Energy, 370, 123544. https://doi.org/10.1016/j.apenergy.2024.123544
2. Reference Implementation: https://github.com/markus-kreft/ev-charging-prediction
