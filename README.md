# TARANG — EV Charging Intelligence System

TARANG is an Electric Vehicle (EV) Charging Intelligence System designed to predict charging session behaviors (plugged-in duration and energy consumption) and model charging profiles to support smart grid integration and demand management.

This repository contains the dataset audit, data contracts, quality assurance suite, and data processing foundation using empirical data from the UK Electric Nation smart charging trial.

---

## Repository Structure

```
tarang/
|-- data/
|   |-- raw/
|   |   |-- electric_nation/       # Raw trial Excel workbooks (excluded from Git)
|   |-- processed/                 # Prepared datasets (generated in downstream stages)
|
|-- docs/
|   |-- data/
|       |-- data_dictionary.md     # Source column descriptions, units, and data types
|       |-- data_contract.md       # Input schemas, targets, timestamps, and anti-leakage rules
|
|-- reports/
|   |-- data_audit/
|       |-- dataset_audit_report.md          # Comprehensive data audit report
|       |-- transaction_quality_summary.csv  # Record counts and percentages for quality checks
|       |-- metadata_quality_summary.csv     # Metadata quality and coverage summary
|       |-- participant_reconciliation.csv   # Cross-dataset participant reconciliation
|       |-- feature_feasibility.csv          # Feasibility and leakage evaluation per feature
|
|-- src/
|   |-- tarang/
|       |-- __init__.py
|       |-- data/
|           |-- __init__.py
|           |-- audit.py           # Reusable audit module and quality validation checks
|
|-- tests/
|   |-- __init__.py
|   |-- unit/
|       |-- __init__.py
|       |-- test_audit.py          # Unit tests covering core audit logic
|
|-- .gitignore
|-- README.md
|-- requirements.txt
```

---

## Dataset Audit Summary

The initial dataset audit evaluated two primary raw workbooks from the Electric Nation trial:
1. `CrowdCharge Transactions.xlsx` (11.14 MB, 71,264 records, 28 columns)
2. `ChargerInstall.xlsx` (58.98 KB, 673 records, 10 columns)

### Key Audit Findings

- **Record Completeness**: 71,264 transactions spanning March 2017 to January 2019 (22 months). 100% unique primary keys (`TransactionID`).
- **Participant Concordance**: 300 non-null participants in transactions. All 300 match the metadata registry with 100% agreement on vehicle specifications (`CarKW` and `CarKWh`).
- **Data Integrity**: 0 negative timestamp intervals (`StopTime < StartTime`), 0 duplicate complete rows.
- **Identified Anomalies**:
  - 92 zero-duration sessions (`StopTime == StartTime`).
  - 561 micro-sessions (< 5 minutes).
  - 23 long-duration outliers (> 336 hours, up to 7,844.88 hours).
  - 12 extreme energy consumption values (> 150 kWh, up to 37,487.88 kWh).
  - 1 session where active charging duration exceeded plugged-in duration due to the UK Daylight Saving Time transition on March 25, 2018.
  - 133 overlapping sessions for the same participant.
  - 3,899 unattributed sessions (3,774 duplicate transmissions under the 200,000+ ID series).

Detailed audit results and recommendations are documented in [`reports/data_audit/dataset_audit_report.md`](reports/data_audit/dataset_audit_report.md).

---

## Getting Started

### Prerequisites

- Python 3.10+ (tested on Python 3.13)
- CPU-only execution (no GPU required)

### Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/nayanojwal0810/Tarang.git
cd Tarang
pip install -r requirements.txt
```

### Running the Data Audit

To re-run the full data audit and generate all audit CSV deliverables:

```bash
python src/tarang/data/audit.py
```

### Running the Unit Test Suite

Execute the unit tests using `pytest`:

```bash
pytest tests/unit/test_audit.py -v
```

---

## Data Contracts and Guidelines

- Refer to [`docs/data/data_contract.md`](docs/data/data_contract.md) for input specifications, target definitions, timestamp conventions, and strict anti-leakage rules.
- Refer to [`docs/data/data_dictionary.md`](docs/data/data_dictionary.md) for comprehensive column descriptions, observed ranges, and units.
