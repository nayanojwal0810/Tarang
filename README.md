# TARANG — EV Charging Intelligence System

TARANG is an Electric Vehicle (EV) Charging Intelligence System designed to predict charging session behaviors (plugged-in duration and energy consumption) and model charging profiles to support smart grid integration and demand management.

This repository contains the dataset audit, formal data contracts, quality assurance suite, and data processing foundation using empirical data from the UK Electric Nation smart charging trial.

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
|       |-- dataset_audit_report.md          # Comprehensive data audit report (dynamically generated)
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
|           |-- audit.py           # Reusable audit module and report generator
|
|-- tests/
|   |-- __init__.py
|   |-- unit/
|       |-- __init__.py
|       |-- test_audit.py          # Comprehensive unit tests covering core audit logic
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

- **Timestamp Standard**: All raw timestamps are verified and treated as **UTC-aware timestamps** without shifting recorded clock values.
- **Record Completeness**: 71,264 transactions spanning March 2017 to January 2019 UTC (22 months). 100% unique primary keys (`TransactionID`).
- **Participant Concordance**: 300 non-null participants in transactions. All 300 match the metadata registry with 100% agreement on vehicle specifications (`CarKW` and `CarKWh`) across all 67,365 matched rows.
- **Data Integrity**: 0 negative timestamp intervals (`StopTime < StartTime`), 0 duplicate complete rows.
- **Unattributed Sessions**: 3,899 sessions lack `ParticipantID`. Natural-session matching on `(ChargerID, StartTime, StopTime, ConsumedkWh)` confirms 3,774 records match known participant sessions on the four session fields; other transaction fields may differ. An additional 125 records belong to an unregistered charger (`4441003917`).
- **Participant Overlaps**: 160 overlapping sessions detected using the cumulative maximum stop time of all earlier sessions for that participant (uncovering 27 nested overlaps missed by sequential shift checks).
- **Energy Quality**:
  - 1 record with missing energy.
  - 0 records with negative or zero energy.
  - 20 records exceed the official Electric Nation 100 kWh plausibility threshold.
  - 12 records exceed 150 kWh (reaching 37,487.88 kWh).
  - 1,788 records exceed their vehicle's rated battery capacity (documented as records requiring review; exceeding battery capacity does not automatically invalidate a transaction).
- **Planned Modeling Scope**:
  - Dual prediction targets: Plugged-in duration in hours (capped at 336.0 hours for the target, preserving original `StopTime`) and Energy consumption in kWh.
  - Exactly 13 planned features (`weekday`, `weekend`, `holiday`, `month`, `StartCos`, `StartSin`, `StartHourNorm`, `TimeSinceLastStop`, `sessionsToday`, `LastDuration`, `LastConsumedkWh`, `CarKW`, `CarKWh`).
  - Participant screening applies Trial 3 exclusion, duration filtering ($\ge 5$ min), cumulative overlap removal, dropping initial unlagged sessions, minimum history of 10 retained sessions (287 participants, 60,220 sessions), and observation-density filtering ($\text{retained\_sessions} \ge \text{observation\_days} / 7$), yielding a final eligible modeling population of **267 participants** and **59,494 sessions**.

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

### Raw Data Setup

The TARANG dataset audit relies on empirical smart charging records from the UK **Electric Nation** trial (funded by Western Power Distribution / National Grid Electricity Distribution and delivered by EA Technology and DriveElectric).

Before running the audit, both workbooks must be downloaded from the official repository and placed in the project raw directory:

- **Official Electric Nation dataset page**: https://commercial.nationalgrid.co.uk/electric-nation-data
- **Crowd Charge Transactions workbook**: https://commercial.nationalgrid.co.uk/downloads-view/81646
- **Charger Install workbook**: https://commercial.nationalgrid.co.uk/downloads/81655/chargerinstall.xlsx

**Target Directory**: `data/raw/electric_nation/`

**Required Workbooks & Sheets**:
1. `CrowdCharge Transactions.xlsx` — `Transaction Data` (71,264 rows $\times$ 28 columns)
2. `ChargerInstall.xlsx` — `20181126_ChargerInstall_DriveEl` (673 rows $\times$ 10 columns)

Both workbooks must be downloaded and placed in `data/raw/electric_nation/` before running the audit.

> **Note**: The raw Excel workbooks contain participant-level trial records and are excluded from Git version control via `.gitignore`. Do not commit raw workbooks to the repository.

### Running the Data Audit

To execute the complete audit, generate all CSV deliverables, and dynamically compile the Markdown report:

```bash
python src/tarang/data/audit.py
```

### Running the Unit Test Suite

Execute the unit test suite using `pytest`:

```bash
pytest tests/unit/test_audit.py -v
```

---

## Data Contracts and Guidelines

- Refer to [`docs/data/data_contract.md`](docs/data/data_contract.md) for input specifications, target definitions, timestamp conventions, and strict anti-leakage rules.
- Refer to [`docs/data/data_dictionary.md`](docs/data/data_dictionary.md) for comprehensive column descriptions, observed ranges, and units.
