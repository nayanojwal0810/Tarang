# TARANG — EV Charging Intelligence System

TARANG uses electric vehicle charging history and vehicle details to predict session connection duration and energy use. It also provides session-level advisory information based on those predictions and historical charging patterns.

TARANG is a Python analysis and prediction pipeline. It does not control chargers, create charging schedules, or dispatch energy.

## What it does

- Checks and prepares charging session data.
- Summarizes charging times, duration, energy use, and participant activity.
- Trains and compares models for duration and energy prediction.
- Produces predictions for single sessions and batches.
- Calculates projected unplug times and historical flexibility summaries.

The model predicts two targets:

| Target | Meaning | Unit |
|---|---|---|
| `Duration` | Time from plug-in to unplug. The training target is capped at 336 hours. | Hours |
| `ConsumedkWh` | Energy delivered during the session. | kWh |

Each model uses 13 inputs available at or before plug-in time:

```python
[
    "weekday", "weekend", "holiday", "month",
    "StartCos", "StartSin", "StartHourNorm",
    "TimeSinceLastStop", "sessionsToday",
    "LastDuration", "LastConsumedkWh", "CarKW", "CarKWh"
]
```

## Project design

```text
Electric Nation Data
        ↓
Data Audit & Preparation
        ↓
EDA & Feature Engineering (13 inputs)
        ↓
Model Training & Evaluation
        ↓
Saved Models
        ↓
Session Input + Charging History
        ↓
Inference: Duration & Energy Predictions
        ↓
Decision Support: Projected Unplug Time & Historical Flexibility
```

## Model comparison

The table shows results on the later 20% of the data (11,904 test sessions). Lower values indicate smaller errors.

| Model | Duration MAE (h) | Duration MdAPE (%) | Energy MAE (kWh) | Energy MdAPE (%) |
|---|---:|---:|---:|---:|
| LinearRegression | 6.45 | 34.5 | 4.44 | 32.5 |
| QuantileRegressor | 6.08 | 31.3 | 4.34 | 31.6 |
| HistGradientBoostingRegressor | 5.53 | 23.4 | 3.83 | 25.0 |
| MedianValue | 7.08 | 36.0 | 6.00 | 49.3 |
| MeanValue | 7.08 | 34.5 | 6.91 | 64.4 |
| LastValue | 8.93 | 57.1 | 5.10 | 37.2 |

MAE is mean absolute error. MdAPE is median absolute percentage error. HistGradientBoostingRegressor has the lowest MAE and MdAPE for both targets in this comparison. For energy, MedianValue has the lowest MAPE (236.1%); see the [energy evaluation report](reports/modeling/energy/evaluation_summary.md) for the full results and metric definitions.

The current processed dataset contains 59,520 sessions from 267 participants. The [reports folder](reports/) contains the data audit, exploration results, model evaluations, figures, and decision support summary.

## Technology

- **Language:** Python 3.10+
- **Data processing:** pandas, NumPy, PyArrow, openpyxl
- **Machine learning and statistics:** scikit-learn, SciPy, statsmodels, SHAP
- **Plots:** Matplotlib, Seaborn
- **Model files:** joblib
- **Tests:** pytest

## Set up

Clone the repository and install the package:

```bash
git clone https://github.com/nayanojwal0810/Tarang.git
cd Tarang
pip install -e ".[dev]"
```

Place the source workbooks in `data/raw/electric_nation/`:

- `CrowdCharge Transactions.xlsx` (sheet: `Transaction Data`)
- `ChargerInstall.xlsx` (sheet: `20181126_ChargerInstall_DriveEl`)

Raw workbooks, processed datasets, and trained model files are not stored in Git.

## Run the workflow

Run the preparation and modeling commands in order:

```bash
python scripts/run_data_audit.py
python scripts/prepare_data.py
python scripts/run_eda.py
python scripts/run_modeling.py
python scripts/run_evaluation.py
```

After training, view the inference and decision support command options:

```bash
python scripts/run_inference.py --help
python scripts/run_decision_support.py --help
```

Run the tests with:

```bash
pytest
```

## Predictions and advisory outputs

TARANG supports single-session prediction using an arrival time, vehicle details, and completed session history. It also supports batch prediction from a CSV or Parquet file containing the 13 model inputs.

Decision support adds these fields to batch results:

- `predicted_Duration`
- `predicted_ConsumedkWh`
- `projected_unplug_time_utc`
- `duration_prediction_valid`
- `energy_prediction_valid`

The projected unplug time adds the predicted duration to the arrival time. TARANG keeps the original predictions and marks estimates as invalid when they are not finite and non-negative.

A historical flexibility ratio describes the share of a connection period spent charging. It does not prove that the remaining time or energy can be shifted. TARANG does not schedule vehicles, optimize tariffs, or send commands to charging hardware.

## Documentation and reports

- [Project structure](docs/architecture/project_structure.md)
- [Data contract](docs/data/data_contract.md)
- [Data dictionary](docs/data/data_dictionary.md)
- [Inference contract](docs/inference/inference_contract.md)
- [Decision support contract](docs/decision_support/decision_support_contract.md)
- [Dataset audit](reports/data_audit/dataset_audit_report.md)
- [Exploratory data analysis](reports/eda/eda_summary.md)
- [Duration model results](reports/modeling/duration/evaluation_summary.md)
- [Energy model results](reports/modeling/energy/evaluation_summary.md)
- [Decision support results](reports/decision_support/decision_support_summary.md)

## References and data

- **Research paper:** [Kreft et al. (2024), *Predictability of electric vehicle charging: Explaining extensive user behavior-specific heterogeneity*, Applied Energy, 370, 123544](https://doi.org/10.1016/j.apenergy.2024.123544).
- **Dataset:** [Electric Nation data — National Grid Electricity Distribution](https://commercial.nationalgrid.co.uk/electric-nation-data). The project uses the CrowdCharge transaction and Charger Install data. See the source page for access and data information.

---

Author: A. Nayanojwal Sri Tej
