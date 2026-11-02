# TARANG — EV Charging Intelligence System

TARANG is an Electric Vehicle (EV) Charging Intelligence System designed to predict charging session behaviors — specifically **plugged-in duration** and **energy consumption** — and model charging flexibility to support smart charging management, grid load integration, and bay occupancy planning.

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
|       |-- electric_nation/         # Canonical Parquet datasets and summary JSON (excluded from Git)
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
|       |-- data_availability.png    # 4-panel participant availability distributions
|       |-- duration_by_start_hour.png # 2D joint histogram of duration vs arrival hour
|       |-- energy_by_start_hour.png   # 2D joint histogram of energy vs arrival hour
|       |-- charging_flexibility.png   # Flexibility ratio distributions (same-day vs overnight)
|
|-- scripts/
|   |-- run_data_audit.py            # CLI entry point: run data quality audit
|   |-- prepare_data.py              # CLI entry point: data screening and feature engineering
|   |-- run_eda.py                   # CLI entry point: run exploratory data analysis and plot figures
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
|       |   |-- calendar_features.py # UK Bank Holidays, weekend/weekday override, day indicators
|       |   |-- time_features.py     # StartHour, StartHourNorm, cyclical StartCos / StartSin
|       |   |-- historical_features.py # TimeSinceLastStop, sessionsToday, LastDuration, LastConsumedkWh
|       |   |-- feature_engineering.py # Authoritative 13 feature definitions and extraction
|       |-- analysis/                # Statistical analysis and EDA
|       |   |-- __init__.py
|       |   |-- exploratory_data_analysis.py # Availability, flexibility, and bivariate distributions
|       |-- visualization/           # Visualization generators
|           |-- __init__.py
|           |-- eda_plots.py         # Multi-panel Matplotlib figures with marginal distributions
|
|-- tests/
|   |-- __init__.py
|   |-- unit/                        # Fast, isolated unit test suite (54 tests)
|   |   |-- test_audit.py
|   |   |-- test_loading.py
|   |   |-- test_screening.py
|   |   |-- test_features.py
|   |   |-- test_eda.py
|   |-- integration/                 # End-to-end integration tests on real workbooks (2 tests)
|       |-- test_data_pipeline.py
|
|-- .gitignore
|-- pyproject.toml                   # Modern PEP 517/621 package build configuration
|-- pytest.ini                       # Test runner configuration
|-- requirements.txt                 # Pinned dependencies
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
| **Negative-Gap Filtering** | 299 | 60,601 | Remove 132 sessions with `TimeSinceLastStop < 0` |
| **First Session Removal** | 298 | 60,302 | Drop first session per participant (299 dropped; 1 user has 0 left) |
| **Minimum History Filter** | 287 | 60,246 | Exclude participants with $< 10$ retained sessions (12 excluded, 56 sessions) |
| **Observation-Density Filter** | **267** | **59,520** | Exclude participants where $\text{sessions} < \text{span} / 7.0$ (20 excluded, 726 sessions) |

The final eligible population consists of exactly **267 participants** and **59,520 sessions**.

### Authoritative 13 Features

All predictors are strictly evaluated at arrival time $t_0 = \text{StartTime}$:
1. `weekday`: Day-of-week index (0 = Monday .. 6 = Sunday; 0 on Bank Holidays)
2. `weekend`: Binary flag (1 if Saturday, Sunday, or statutory UK Bank Holiday, else 0)
3. `holiday`: Binary flag for UK England & Wales statutory Bank Holidays (13 across 2017–2018)
4. `month`: Calendar month index (1..12)
5. `StartCos`: Cyclical cosine coordinate $\cos(2\pi \cdot \text{StartHour} / 24)$
6. `StartSin`: Cyclical sine coordinate $\sin(2\pi \cdot \text{StartHour} / 24)$
7. `StartHourNorm`: Continuous arrival hour normalized by dataset-wide maximum $\text{StartHour} / \max(\text{StartHour})$
8. `TimeSinceLastStop`: Elapsed hours since immediately preceding session stop time
9. `sessionsToday`: Daily cumulative count of prior sessions on the same calendar date
10. `LastDuration`: Elapsed duration in hours of immediately preceding session
11. `LastConsumedkWh`: Active energy consumption in kWh of immediately preceding session
12. `CarKW`: Vehicle onboard charger maximum power rating (3.6 kW or 7.0 kW)
13. `CarKWh`: Vehicle battery capacity (6.2 kWh to 100.0 kWh)

### Canonical Processed Dataset

The pipeline produces a single canonical Parquet dataset:
- **Location**: `data/processed/electric_nation/features.parquet`
- **Dimensions**: Exactly **59,520 rows $\times$ 30 columns**
- **Contents**: 9 selected raw source fields, `Duration` (capped at 336.0h), 7 day indicators (`Mon`..`Sun`), 4 calendar features, 4 start-hour features, 4 historical lag features, and integer participant `id`.

---

## Getting Started

### Prerequisites

- Python 3.10+ (tested on Python 3.13)
- Windows, macOS, or Linux

### Installation

Clone the repository and install the package in editable mode:

```bash
git clone https://github.com/nayanojwal0810/Tarang.git
cd Tarang
pip install -e .
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
Generates empirical summary statistics and publication-grade visualization figures in `reports/eda/`:

```bash
python scripts/run_eda.py
```

Generated EDA artifacts:
- `reports/eda/eda_summary.md`: Summary report including participant availability and charging flexibility ratios.
- `reports/eda/data_availability.png`: 4-panel distribution of participant spans, sessions, weekly cadence, and durations.
- `reports/eda/duration_by_start_hour.png`: 2D joint histogram of plugged-in duration vs. arrival hour (log scale).
- `reports/eda/energy_by_start_hour.png`: 2D joint histogram of energy consumption vs. arrival hour (log scale).
- `reports/eda/charging_flexibility.png`: Density distributions of flexibility ratios for same-day vs. overnight sessions.

### 4. Run Test Suite
Run the full 56-test test suite (54 unit tests and 2 integration tests):

```bash
python -m pytest tests/ -v
```

---

## Documentation

- [`docs/architecture/project_structure.md`](docs/architecture/project_structure.md): Modular architecture, responsibilities, and future stage roadmap.
- [`docs/data/data_contract.md`](docs/data/data_contract.md): Formal data schemas, anti-leakage constraints, and screening sequence.
- [`docs/data/data_dictionary.md`](docs/data/data_dictionary.md): Column definitions, types, units, and ranges across raw and processed datasets.
- [`reports/data_audit/dataset_audit_report.md`](reports/data_audit/dataset_audit_report.md): Full dataset quality audit and feasibility report.
- [`reports/eda/eda_summary.md`](reports/eda/eda_summary.md): Statistical findings and flexibility ratios.

---

## References

1. Kreft, M., Brudermueller, T., Fleisch, E., & Staake, T. (2024). Predictability of electric vehicle charging: Explaining extensive user behaviour with simple models. *Nature Communications*, 15(1), 1603. https://doi.org/10.1038/s41467-024-45899-7
2. Electric Nation Trial Dataset: National Grid Electricity Distribution & EA Technology. https://commercial.nationalgrid.co.uk/electric-nation-data
3. Reference Implementation: https://github.com/mkreft/predictability-of-ev-charging
