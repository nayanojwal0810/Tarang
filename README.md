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
|-- docs/
|   |-- architecture/
|   |   |-- project_structure.md     # Architectural design and module responsibilities
|   |-- data/
|       |-- data_contract.md         # Data contracts, anti-leakage rules, and screening specifications
|       |-- data_dictionary.md       # Raw and canonical column definitions, units, and ranges
|
|-- reports/
|   |-- data_audit/
|   |   |-- dataset_audit_report.md  # Comprehensive data quality and feasibility audit report
|   |   |-- transaction_quality_summary.csv
|   |   |-- metadata_quality_summary.csv
|   |   |-- participant_reconciliation.csv
|   |   |-- feature_feasibility.csv
|   |-- eda/
|       |-- eda_summary.md           # Descriptive EDA findings and flexibility analysis
|       |-- data_availability.png    # 4-panel participant availability distributions (26 bins)
|       |-- duration_by_start_hour.png # 2D joint histogram of duration vs arrival hour (0.5h bins, Europe/London)
|       |-- energy_by_start_hour.png   # 2D joint histogram of energy vs arrival hour (0.5h, 0.5 kWh bins, Europe/London)
|       |-- charging_flexibility.png   # Flexibility ratio distributions (same-day vs overnight)
|
|-- scripts/
|   |-- run_data_audit.py            # CLI entry point: run data quality audit
|   |-- prepare_data.py              # CLI entry point: data screening and feature engineering
|   |-- run_eda.py                   # CLI entry point: run exploratory data analysis and plot figures
|   |-- run_modeling.py              # CLI entry point: dual-target regression benchmark pipeline
|   |-- prepare_modeling_data.py     # Backward-compatible CLI shim for legacy workflows
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
|       |   |-- preprocessing.py     # Re-export facade maintaining legacy compatibility
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
|       |-- analysis/                # Statistical analysis and EDA
|       |   |-- __init__.py
|       |   |-- exploratory_data_analysis.py # Availability, flexibility, and bivariate distributions
|       |-- visualization/           # Visualization generators
|           |-- __init__.py
|           |-- eda_plots.py         # Multi-panel Matplotlib figures with marginal distributions
|
|-- tests/
|   |-- __init__.py
|   |-- unit/                        # Fast, isolated unit test suite
|   |   |-- test_audit.py
|   |   |-- test_loading.py
|   |   |-- test_screening.py
|   |   |-- test_features.py
|   |   |-- test_eda.py
|   |   |-- test_modeling.py
|   |-- integration/                 # End-to-end integration tests on real workbooks and datasets
|       |-- test_data_pipeline.py
|       |-- test_modeling_pipeline.py
|
|-- .gitignore
|-- pyproject.toml                   # Modern PEP 517/621 package build configuration
|-- pytest.ini                       # Test runner configuration
|-- requirements.txt                 # Lower-bounded dependencies
|-- README.md
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

The pipeline produces a single canonical Parquet dataset:
- **Location**: `data/processed/electric_nation/features.parquet`
- **Dimensions**: Exactly **59,520 rows $\times$ 30 columns**
- **Canonical Column Sequence**:
  1. `StartTime`
  2. `StopTime`
  3. `ParticipantID`
  4. `ConsumedkWh`
  5. `CarKW`
  6. `CarKWh`
  7. `PluggedInTime`
  8. `ChargingDuration`
  9. `Hot_Unplug`
  10. `Duration`
  11. `weekday`
  12. `weekend`
  13. `holiday`
  14. `Mon`
  15. `Tue`
  16. `Wed`
  17. `Thu`
  18. `Fri`
  19. `Sat`
  20. `Sun`
  21. `month`
  22. `StartHour`
  23. `StartHourNorm`
  24. `StartCos`
  25. `StartSin`
  26. `TimeSinceLastStop`
  27. `sessionsToday`
  28. `LastDuration`
  29. `LastConsumedkWh`
  30. `id` (integer group enumeration index from original participant grouping; non-contiguous)

---

## Getting Started

### Prerequisites

- Python 3.10+ (tested on Python 3.11, 3.12, and 3.13)
- Windows, macOS, or Linux

### Installation

Clone the repository and install the package in editable mode:

```bash
git clone https://github.com/nayanojwal0810/Tarang.git
cd Tarang
pip install -e .
```

*Note: Dependencies in `requirements.txt` specify minimum compatible versions (`>=`), not pinned versions.*

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

### 5. Run Test Suite
Run the full test suite including unit tests and integration tests:

```bash
python -m pytest tests/ -v
```

---

## System Scope and Advisory Decision Support

TARANG provides predictive intelligence and charging behavioral insights to assist fleet managers, charge point operators, and smart grid planners. Predictions and flexibility metrics serve as **advisory decision support** for bay scheduling and demand management; TARANG does not execute automatic charger hardware dispatch or real-time grid switching control.

---

## References

1. Kreft, M., Brudermueller, T., Fleisch, E., & Staake, T. (2024). *Predictability of electric vehicle charging: Explaining extensive user behavior-specific heterogeneity*. Applied Energy, 370, 123544. https://doi.org/10.1016/j.apenergy.2024.123544
2. Reference Implementation: https://github.com/markus-kreft/ev-charging-prediction
