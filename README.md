# TARANG — EV Charging Intelligence System

TARANG uses electric vehicle charging session history and vehicle hardware specifications to predict session connection duration and energy consumption. It provides real-time single-session advisories, projected unplug departure timestamps, and descriptive charging flexibility analysis across multi-session batches.

TARANG is an informational intelligence platform. It does not control charging hardware, execute autonomous grid dispatch, or guarantee available flexibility.

---

## What It Predicts

TARANG estimates two operational targets upon vehicle connection:

| Target | Description | Unit |
|---|---|---|
| **Connection Duration** (`Duration`) | Total time the charge point remains occupied by the vehicle (including active charging and idle connected time). Capped at 336 hours in training. | Hours |
| **Energy Consumption** (`ConsumedkWh`) | Total electrical energy transferred during the charging session. | kWh |

Each prediction is generated from **13 canonical predictors** available at or before plug-in time:

```python
[
    "weekday", "weekend", "holiday", "month",
    "StartCos", "StartSin", "StartHourNorm",
    "TimeSinceLastStop", "sessionsToday",
    "LastDuration", "LastConsumedkWh",
    "CarKW", "CarKWh"
]
```

Combining the predicted connection duration with the arrival timestamp produces a **projected unplug departure time** in UTC.

---

## Architecture & Workflow

```text
Electric Nation Trial Data (NGED Open Data)
                    ↓
Data Quality Screening & Feature Engineering (13 Predictors)
                    ↓
HistGradientBoostingRegressor Training (Absolute Error Loss)
                    ↓
Deployment Artifact Bundle (Versioned Models & Normalization Metadata)
                    ↓
┌───────────────────────────────────────┬───────────────────────────────────────┐
│     Single-Session Advisory           │        Batch Network Advisory         │
│  Arrival UTC + Car kW/kWh + History   │      CSV / Parquet Fleet Records      │
│                  ↓                    │                   ↓                   │
│  Real-time Duration, Energy, Unplug   │  Fleet Summary, Distribution Metrics, │
│  Forecasts & 13 Predictor Inspection  │  CSV Export & Flexibility Analysis    │
└───────────────────────────────────────┴───────────────────────────────────────┘
```

---

## Model Benchmark Performance

Evaluation on 20% held-out test sessions (11,904 sessions) against baseline statistical estimators:

| Target | Model | Test MAE | Test MdAPE | R² | vs. Mean Baseline |
|---|---|---:|---:|---:|---:|
| **Duration** | **HistGradientBoostingRegressor** | **2.96 h** | **23.4%** | **0.320** | **−28%** |
| | LinearRegression | 6.45 h | 34.5% | 0.080 | −9% |
| | QuantileRegressor | 6.08 h | 31.3% | 0.120 | −14% |
| | Mean Baseline | 4.09 h | 34.5% | 0.000 | 0% |
| | Median Baseline | 7.08 h | 36.0% | −0.050 | — |
| | LastValue Baseline | 8.93 h | 57.1% | −0.210 | — |
| **Energy** | **HistGradientBoostingRegressor** | **4.24 kWh** | **25.0%** | **0.354** | **−28%** |
| | LinearRegression | 4.44 kWh | 32.5% | 0.280 | −24% |
| | QuantileRegressor | 4.34 kWh | 31.6% | 0.310 | −26% |
| | Mean Baseline | 5.86 kWh | 64.4% | 0.000 | 0% |
| | Median Baseline | 6.00 kWh | 49.3% | −0.040 | — |
| | LastValue Baseline | 5.10 kWh | 37.2% | −0.150 | — |

The production models use `HistGradientBoostingRegressor` with absolute-error loss and early stopping to handle the non-Gaussian variance observed in EV driver behaviour.

---

## Technology Stack

- **Language:** Python 3.11, 3.12 (verified in CI)
- **Data Engineering:** pandas, NumPy, PyArrow
- **Machine Learning:** scikit-learn (`HistGradientBoostingRegressor`), SciPy, joblib
- **Web Application:** Streamlit
- **Quality & CI:** pytest, GitHub Actions

---

## Local Installation & Execution

### 1. Clone the repository

```bash
git clone https://github.com/nayanojwal0810/Tarang.git
cd Tarang
```

### 2. Install dependencies

For application runtime:
```bash
pip install -r requirements.txt
```

For development and test suite:
```bash
pip install -e ".[dev,app]"
```

### 3. Run the interactive web application

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

### 4. Run automated tests

```bash
python -m pytest
```

---

## Public Web Application

The application is configured for deployment on **Streamlit Community Cloud**:

- **Repository:** `nayanojwal0810/Tarang`
- **Branch:** `main`
- **Main file path:** `app.py`
- **Python Version:** `3.12`
- **Public URL:** [tarang-ev-intelligence.streamlit.app](https://tarang-ev-intelligence.streamlit.app)

*Note: Access to the public deployment URL is subject to Streamlit Community Cloud hosting status and repository owner activation.*

---

## Operational Boundaries & Limitations

1. **Advisory Estimates:** TARANG generates statistical forecasts based on historical patterns. Predictions are informational and do not control charging equipment, modulate electrical power, or guarantee station availability.
2. **Connection vs. Active Charging:** Connection duration includes idle plugged-in time after charging completes. A vehicle may finish battery replenishment hours before unplugging.
3. **Cold-Start Habit Requirement:** Calculating driver habit features (`TimeSinceLastStop`, `LastDuration`, `LastConsumedkWh`) requires at least one prior completed session for the participant before the arrival timestamp.
4. **Descriptive Flexibility:** Historical charging flexibility ratios describe observed idle time during completed sessions. They do not prove that idle duration or energy is shiftable under real-time grid or driver constraints.
5. **Data Privacy:** Uploaded datasets are processed in active browser memory and are not logged or stored. Users should never upload confidential or personally identifiable charging records.

---

## Dataset Reference & Attribution

- **Trial Dataset:** Electric Nation Trial (2017–2018), conducted by Western Power Distribution (now National Grid Electricity Distribution / NGED) and DriveElectric.
- **Licence:** NGED Open Data Licence.
- **Attribution Statement:** *Supported by NGED Open Data.*
- **Dataset Portal:** [Electric Nation Data — National Grid](https://commercial.nationalgrid.co.uk/electric-nation-data)
- **Academic Citation:** Kreft, C., et al. (2024). *Predictability of electric vehicle charging: Explaining extensive user behavior-specific heterogeneity*. **Applied Energy**, 370, 123544. [doi:10.1016/j.apenergy.2024.123544](https://doi.org/10.1016/j.apenergy.2024.123544).

---

## Author

**A. Nayanojwal Sri Tej**
