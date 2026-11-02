# Project structure

TARANG is a Python application for preparing EV charging records, predicting session duration and energy use, and producing session advisories. The folders below show where we keep the main code, data, reports, and tests.

## Repository layout

    Tarang/
    ├── app.py
    ├── deployment_artifacts/
    │   ├── duration/
    │   ├── energy/
    │   ├── processing_summary.json
    │   └── manifest.json
    ├── data/
    │   ├── raw/electric_nation/
    │   └── processed/electric_nation/
    ├── docs/
    ├── reports/
    ├── scripts/
    ├── src/tarang/
    ├── tests/
    ├── pyproject.toml
    └── requirements.txt

The raw and processed data folders hold local files and are not tracked in Git. The deployment_artifacts folder contains the two trained models and the summary value that the application needs for inference.

## Main Python packages

| Package | Purpose |
|---|---|
| tarang.config | Defines file paths, feature names, targets, and shared settings. |
| tarang.data | Reads, checks, filters, and prepares charging records. |
| tarang.features | Builds the 13 model inputs. |
| tarang.analysis | Summarizes participant activity and charging patterns. |
| tarang.visualization | Creates data exploration figures. |
| tarang.modeling | Trains models and measures prediction errors. |
| tarang.evaluation | Compares models and examines prediction errors and feature contributions. |
| tarang.inference | Loads models and predicts duration or energy. |
| tarang.decision_support | Combines predictions with projected unplug times and charging summaries. |
| tarang.app | Runs the Overview, Single Prediction, and Batch Prediction pages. |

## Main scripts

Run scripts from the repository root.

| Script | Purpose |
|---|---|
| scripts/run_data_audit.py | Checks the source workbooks and writes audit tables. |
| scripts/prepare_data.py | Creates the prepared session dataset. |
| scripts/run_eda.py | Summarizes charging activity and creates figures. |
| scripts/run_modeling.py | Trains the duration and energy models. |
| scripts/run_evaluation.py | Evaluates models and writes result tables and figures. |
| scripts/run_inference.py | Runs single-target batch predictions. |
| scripts/run_decision_support.py | Creates duration and energy advisories and summaries. |
| scripts/verify_deployment.py | Checks the deployment bundle and runs sample inference. |

## Main data and output locations

- Prepared sessions: data/processed/electric_nation/features.parquet
- Training models and results: data/processed/electric_nation/modeling/
- Decision support outputs: data/processed/electric_nation/decision_support/
- Deployment models and metadata: deployment_artifacts/
- Reports and figures: reports/

## Run order

Prepare the data before training. Train the models before running inference or decision support.

    python scripts/run_data_audit.py
    python scripts/prepare_data.py
    python scripts/run_eda.py
    python scripts/run_modeling.py
    python scripts/run_evaluation.py

Use the help option to see the inputs and outputs for inference and decision support:

    python scripts/run_inference.py --help
    python scripts/run_decision_support.py --help

TARANG supports Python 3.11 and 3.12 in its CI workflow. Install the project with pip install -e ".[dev,app]" before running the full test suite.
