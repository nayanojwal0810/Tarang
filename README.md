# TARANG — EV Charging Intelligence System

TARANG uses electric vehicle charging history and vehicle details to predict how long a vehicle will stay connected to a charger and how much energy the session will use. It also provides projected unplug times and summaries of historical charging activity.

TARANG provides information for review. It does not control chargers, schedule charging sessions, or dispatch energy.

## What TARANG does

- Checks and prepares Electric Nation charging records.
- Explores charging times, connection duration, energy use, and participant activity.
- Trains and compares models for duration and energy prediction.
- Predicts one session from arrival details and completed charging history.
- Predicts multiple sessions from a CSV or Parquet file.
- Calculates projected unplug times and descriptive charging activity ratios.

## Prediction targets

| Target | Meaning | Unit |
|---|---|---|
| Duration | Time from plug-in to unplug. The training target is capped at 336 hours. | Hours |
| ConsumedkWh | Energy delivered during the charging session. | kWh |

Each model uses 13 inputs that are available at or before plug-in time:

    weekday, weekend, holiday, month,
    StartCos, StartSin, StartHourNorm,
    TimeSinceLastStop, sessionsToday,
    LastDuration, LastConsumedkWh, CarKW, CarKWh

## Project design

    Electric Nation charging data
                ↓
        Data audit and preparation
                ↓
        Exploration and feature creation
                ↓
        Model training and evaluation
                ↓
        Single-session or batch input
                ↓
        Duration and energy predictions
                ↓
        Projected unplug time and charging summaries
                ↓
        TARANG Streamlit application

## Model comparison

The table shows results on the later 20% of the data (11,904 test sessions). Lower error values indicate better predictions.

| Model | Duration MAE (h) | Duration MdAPE (%) | Energy MAE (kWh) | Energy MdAPE (%) |
|---|---:|---:|---:|---:|
| LinearRegression | 6.45 | 34.5 | 4.44 | 32.5 |
| QuantileRegressor | 6.08 | 31.3 | 4.34 | 31.6 |
| HistGradientBoostingRegressor | 5.53 | 23.4 | 3.83 | 25.0 |
| MedianValue | 7.08 | 36.0 | 6.00 | 49.3 |
| MeanValue | 7.08 | 34.5 | 6.91 | 64.4 |
| LastValue | 8.93 | 57.1 | 5.10 | 37.2 |

MAE is mean absolute error. MdAPE is median absolute percentage error. HistGradientBoostingRegressor has the lowest MAE and MdAPE for both targets in this comparison. For energy, MedianValue has the lowest MAPE (236.1%). The model reports explain the full results and metric definitions.

The prepared dataset contains 59,520 sessions from 267 participants.

## Technology

- Python 3.11 and 3.12
- pandas, NumPy, PyArrow
- scikit-learn, SciPy, SHAP, statsmodels
- Streamlit
- Matplotlib and Seaborn
- joblib and pytest

## Run TARANG locally

Use Python 3.11 or 3.12.

    git clone https://github.com/nayanojwal0810/Tarang.git
    cd Tarang
    pip install -e ".[dev,app]"
    streamlit run app.py

Open http://localhost:8501 in your browser.

To run the tests:

    python -m pytest

To prepare the data and generate the analysis reports, first place the two source workbooks in data/raw/electric_nation/:

- CrowdCharge Transactions.xlsx
- ChargerInstall.xlsx

Then run the workflow in order:

    python scripts/run_data_audit.py
    python scripts/prepare_data.py
    python scripts/run_eda.py
    python scripts/run_modeling.py
    python scripts/run_evaluation.py

Raw workbooks and processed datasets are not stored in Git. The application uses the model files and normalization metadata in deployment_artifacts/.

## Predictions and advisory outputs

Single prediction uses an arrival time, vehicle charger rating, battery capacity, and completed session history.

Batch prediction accepts a CSV or Parquet file with StartTime and the 13 model inputs. TARANG adds:

- predicted_Duration
- predicted_ConsumedkWh
- projected_unplug_time_utc
- duration_prediction_valid
- energy_prediction_valid

The projected unplug time adds the predicted duration to the arrival time. TARANG keeps the model's original estimates and marks values invalid when they are negative or non-finite.

TARANG also calculates a historical ratio of active charging time to total connection time. This ratio describes past sessions. It does not confirm that unused connection time or energy can be shifted.

## Reports

- [Dataset audit](reports/data_audit/dataset_audit_report.md) — source data checks and preparation results.
- [Charging data exploration](reports/eda/eda_summary.md) — participant activity, arrival times, connection duration, energy use, and historical charging ratios.
- [Duration model evaluation](reports/modeling/duration/evaluation_summary.md) — duration model comparison, subgroup results, and feature contributions.
- [Energy model evaluation](reports/modeling/energy/evaluation_summary.md) — energy model comparison, subgroup results, and feature contributions.
- [Decision support summary](reports/decision_support/decision_support_summary.md) — prediction output summaries and historical charging ratios.

The reports link to their supporting tables and figures.

## Technical documentation

- [Project structure](docs/architecture/project_structure.md)
- [Data contract](docs/data/data_contract.md)
- [Data dictionary](docs/data/data_dictionary.md)
- [Inference contract](docs/inference/inference_contract.md)
- [Decision support contract](docs/decision_support/decision_support_contract.md)

## Data source and limits

TARANG uses the Electric Nation trial data. See the [Electric Nation data page](https://commercial.nationalgrid.co.uk/electric-nation-data) for source information and access terms.

TARANG produces statistical estimates from historical charging records. It does not connect to live chargers or grid systems, control charging hardware, create charging schedules, optimise tariffs, or guarantee available flexibility.

## Application deployment

The public Streamlit deployment still needs owner activation and live verification. The application URL is not listed as active until both prediction workflows have been checked online.

---

Author: A. Nayanojwal Sri Tej
