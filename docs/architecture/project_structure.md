# Project structure

TARANG is a Python system for EV charging session analysis and prediction. The code separates data preparation, feature creation, model training, evaluation, inference, and decision support.

## Main folders

```text
Tarang/
├── data/
│   ├── raw/electric_nation/          Source Excel workbooks
│   └── processed/electric_nation/    Prepared data and model files
├── docs/
│   ├── architecture/                 Project structure
│   ├── data/                         Data rules and column definitions
│   ├── inference/                    Prediction inputs and outputs
│   └── decision_support/             Advisory definitions
├── reports/
│   ├── data_audit/                   Data checks and CSV summaries
│   ├── eda/                          Data summaries and plots
│   ├── modeling/duration/             Duration model results
│   ├── modeling/energy/               Energy model results
│   └── decision_support/              Prediction and flexibility results
├── scripts/                           Command-line entry points
├── src/tarang/                        Python package
└── tests/                              Unit and integration tests
```

The raw and processed data folders contain files that are not stored in Git. The reports folder contains the project reports and selected result tables.

## Python package

| Package | Purpose |
|---|---|
| `tarang.config` | Defines paths, column names, targets, and shared settings. |
| `tarang.data` | Loads, checks, filters, and prepares session data. |
| `tarang.features` | Creates the 13 model inputs. |
| `tarang.analysis` | Calculates data availability and charging pattern summaries. |
| `tarang.visualization` | Creates data exploration plots. |
| `tarang.modeling` | Trains models and calculates prediction errors. |
| `tarang.evaluation` | Compares model results, studies participant differences, and calculates feature importance. |
| `tarang.inference` | Creates predictions for single sessions and batches. |
| `tarang.decision_support` | Combines duration and energy predictions with projected unplug times and flexibility summaries. |

## Command-line scripts

Run these scripts from the repository root:

| Script | What it does |
|---|---|
| `scripts/run_data_audit.py` | Checks the source workbooks and writes audit reports. |
| `scripts/prepare_data.py` | Builds the processed session dataset. |
| `scripts/run_eda.py` | Writes data summaries and plots. |
| `scripts/run_modeling.py` | Trains models for duration and energy. |
| `scripts/run_evaluation.py` | Creates model evaluation reports and figures. |
| `scripts/run_inference.py` | Predicts one target for a batch of sessions. |
| `scripts/run_decision_support.py` | Creates both predictions and session advisory outputs. |

## Data and output locations

The main processed dataset is:

`data/processed/electric_nation/features.parquet`

Model files and training results are stored under:

`data/processed/electric_nation/modeling/`

Decision support batch output is stored under:

`data/processed/electric_nation/decision_support/`

Reports and figures are stored under `reports/`.

## Run order

Prepare the data before training models. Train the models before running inference or decision support.

```bash
python scripts/run_data_audit.py
python scripts/prepare_data.py
python scripts/run_eda.py
python scripts/run_modeling.py
python scripts/run_evaluation.py
python scripts/run_inference.py --help
python scripts/run_decision_support.py --help
```

The project uses Python 3.10 or later. Install the development dependencies with `pip install -e ".[dev]"` and run tests with `pytest`.
