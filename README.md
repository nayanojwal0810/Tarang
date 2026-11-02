# TARANG — EV Charging Intelligence System

TARANG uses charging session history and vehicle details to estimate how long an electric vehicle will stay connected to a charger and how much energy it will use.

The system also reviews historical charging patterns and provides session-level advisory information. It does not control chargers or create charging schedules.

## What the system does

TARANG provides four main functions:

1. Prepare and check charging session data.
2. Explore charging times, connection duration, energy use, and participant activity.
3. Train and compare models for duration and energy predictions.
4. Generate single-session and batch predictions, projected unplug times, and historical flexibility summaries.

The two prediction targets are:

- `Duration`: plugged-in connection time in hours. The training target is capped at 336 hours.
- `ConsumedkWh`: energy delivered during the session in kilowatt-hours.

The models use 13 inputs that are available at or before plug-in time:

```python
[
    "weekday", "weekend", "holiday", "month",
    "StartCos", "StartSin", "StartHourNorm",
    "TimeSinceLastStop", "sessionsToday",
    "LastDuration", "LastConsumedkWh", "CarKW", "CarKWh"
]
```

## How the system works

The main workflow is:

1. Read the Electric Nation transaction and charger installation workbooks.
2. Check data quality and prepare the session dataset.
3. Review the data through summary tables and plots.
4. Train and compare six prediction models for both targets.
5. Review model errors, participant-level results, and feature importance.
6. Run predictions for one session or a batch of sessions.
7. Create session advisories and historical flexibility summaries.

## Project results

The current processed dataset contains 59,520 sessions from 267 participants. The reports directory contains the data audit, data exploration results, model evaluation reports, plots, and decision support summaries.

Start with these reports:

- [Dataset audit](reports/data_audit/dataset_audit_report.md)
- [Exploratory data analysis](reports/eda/eda_summary.md)
- [Duration model results](reports/modeling/duration/evaluation_summary.md)
- [Energy model results](reports/modeling/energy/evaluation_summary.md)
- [Decision support results](reports/decision_support/decision_support_summary.md)

## Set up the project

TARANG requires Python 3.10 or later.

Clone the repository and install the package:

```bash
git clone https://github.com/nayanojwal0810/Tarang.git
cd Tarang
pip install -e ".[dev]"
```

Place the two source workbooks in `data/raw/electric_nation/`:

- `CrowdCharge Transactions.xlsx`
- `ChargerInstall.xlsx`

The transaction workbook uses the `Transaction Data` sheet. The charger workbook uses the `20181126_ChargerInstall_DriveEl` sheet.

Raw workbooks, processed datasets, and trained model files stay outside Git.

## Run the workflow

Run the preparation and model commands in order. After the model files exist, use the last two commands to view the inference and decision support options.

```bash
python scripts/run_data_audit.py
python scripts/prepare_data.py
python scripts/run_eda.py
python scripts/run_modeling.py
python scripts/run_evaluation.py
python scripts/run_inference.py --help
python scripts/run_decision_support.py --help
```

The audit and preparation commands create the processed session dataset. The modeling command trains the models. The evaluation command creates model reports and figures. The inference and decision support commands use the saved dataset and trained models.

Run the test suite with:

```bash
pytest
```

## Predictions and advisory outputs

TARANG supports two prediction modes:

- Single session: provide an arrival time, vehicle specifications, and completed session history.
- Batch sessions: provide a CSV or Parquet file containing the 13 model inputs.

The inference command predicts one target at a time. The decision support command predicts both targets and adds:

- `predicted_Duration`
- `predicted_ConsumedkWh`
- `projected_unplug_time_utc`
- `duration_prediction_valid`
- `energy_prediction_valid`

The projected unplug time adds the predicted duration to the session arrival time. TARANG keeps the original model predictions and marks estimates as invalid when they are not finite and non-negative.

## What the results do not provide

A predicted connection duration is not a confirmed departure time. A historical flexibility ratio describes the share of a connection period spent charging; it does not prove that the remaining time or energy can be shifted.

TARANG does not schedule vehicles, optimize tariffs, dispatch energy, or send commands to charging hardware.

## Project documentation

- [Project structure](docs/architecture/project_structure.md)
- [Data contract](docs/data/data_contract.md)
- [Data dictionary](docs/data/data_dictionary.md)
- [Inference contract](docs/inference/inference_contract.md)
- [Decision support contract](docs/decision_support/decision_support_contract.md)
