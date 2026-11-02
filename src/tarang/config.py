"""Central configuration module for TARANG — EV Charging Intelligence System.

This module provides path constants, column specifications, and authoritative configuration
settings across the data pipeline, feature engineering, and reporting suites.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Set
import pandas as pd

# ==============================================================================
# Filesystem Paths
# ==============================================================================

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# Data directories
DATA_DIR: Path = PROJECT_ROOT / "data"
DATA_RAW_DIR: Path = DATA_DIR / "raw" / "electric_nation"
DATA_INTERIM_DIR: Path = DATA_DIR / "interim"
DATA_PROCESSED_DIR: Path = DATA_DIR / "processed" / "electric_nation"

# Raw workbook files and sheets
RAW_TX_FILE: Path = DATA_RAW_DIR / "CrowdCharge Transactions.xlsx"
RAW_TX_SHEET: str = "Transaction Data"

RAW_META_FILE: Path = DATA_RAW_DIR / "ChargerInstall.xlsx"
RAW_META_SHEET: str = "20181126_ChargerInstall_DriveEl"

# Canonical processed dataset paths
PROCESSED_FEATURES_FILE: Path = DATA_PROCESSED_DIR / "features.parquet"
PROCESSING_SUMMARY_FILE: Path = DATA_PROCESSED_DIR / "processing_summary.json"

# Modeling directories and targets
MODELING_DIR: Path = DATA_PROCESSED_DIR / "modeling"
DURATION_MODELING_DIR: Path = MODELING_DIR / "duration"
ENERGY_MODELING_DIR: Path = MODELING_DIR / "energy"

# Evaluation directories
EVALUATION_DATA_DIR: Path = DATA_PROCESSED_DIR / "evaluation"
DURATION_EVALUATION_DATA_DIR: Path = EVALUATION_DATA_DIR / "duration"
ENERGY_EVALUATION_DATA_DIR: Path = EVALUATION_DATA_DIR / "energy"

TARGET_DURATION: str = "Duration"
TARGET_ENERGY: str = "ConsumedkWh"

LAG_FEATURE_DURATION: str = "LastDuration"
LAG_FEATURE_ENERGY: str = "LastConsumedkWh"

TARGET_LAG_MAP: dict[str, str] = {
    TARGET_DURATION: LAG_FEATURE_DURATION,
    TARGET_ENERGY: LAG_FEATURE_ENERGY,
}

MODEL_NAMES: List[str] = [
    "LinearRegression",
    "QuantileRegressor",
    "HistGradientBoostingRegressor",
    "MedianValue",
    "MeanValue",
    "LastValue",
]

TEST_SPLIT_SIZE: float = 0.2
SPLIT_RANDOM_STATE: int = 42
MODEL_RANDOM_STATE: int = 0

# Reports directories
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
AUDIT_REPORTS_DIR: Path = REPORTS_DIR / "data_audit"
EDA_REPORTS_DIR: Path = REPORTS_DIR / "eda"
EVALUATION_REPORTS_DIR: Path = REPORTS_DIR / "modeling"
DURATION_EVALUATION_REPORT_DIR: Path = EVALUATION_REPORTS_DIR / "duration"
ENERGY_EVALUATION_REPORT_DIR: Path = EVALUATION_REPORTS_DIR / "energy"

# ==============================================================================
# Schema Definitions
# ==============================================================================

# Required columns in raw transactions workbook
REQUIRED_TX_COLUMNS: List[str] = [
    "TransactionID",
    "ChargerID",
    "ParticipantID",
    "CarKW",
    "CarKWh",
    "StartTime",
    "StopTime",
    "ConsumedkWh",
    "Trial",
]

# Required columns in raw charger metadata workbook
REQUIRED_META_COLUMNS: List[str] = [
    "ParticipantID",
    "Charger",
    "DCSProvider",
    "CarMake",
    "CarModel",
    "CarkW",
    "CarkWh",
]

# Raw fields retained in the primary research pipeline
RAW_SELECTED_COLUMNS: List[str] = [
    "StartTime",
    "StopTime",
    "ParticipantID",
    "ConsumedkWh",
    "CarKW",
    "CarKWh",
    "PluggedInTime",
    "ChargingDuration",
    "Hot_Unplug",
]

# Authoritative 13 feature predictors
FEATURE_NAMES: List[str] = [
    "weekday",
    "weekend",
    "holiday",
    "month",
    "StartCos",
    "StartSin",
    "StartHourNorm",
    "TimeSinceLastStop",
    "sessionsToday",
    "LastDuration",
    "LastConsumedkWh",
    "CarKW",
    "CarKWh",
]

# Canonical 30 columns in features.parquet
CANONICAL_COLUMNS: List[str] = [
    "StartTime",
    "StopTime",
    "ParticipantID",
    "ConsumedkWh",
    "CarKW",
    "CarKWh",
    "PluggedInTime",
    "ChargingDuration",
    "Hot_Unplug",
    "Duration",
    "weekday",
    "weekend",
    "holiday",
    "Mon",
    "Tue",
    "Wed",
    "Thu",
    "Fri",
    "Sat",
    "Sun",
    "month",
    "StartHour",
    "StartHourNorm",
    "StartCos",
    "StartSin",
    "TimeSinceLastStop",
    "sessionsToday",
    "LastDuration",
    "LastConsumedkWh",
    "id",
]

# ==============================================================================
# Calendar and Reference Constants
# ==============================================================================

# Exact 13 UK Bank Holidays used in established benchmark (2017-2018)
UK_BANK_HOLIDAYS_REFERENCE: Set = {
    pd.Timestamp("2017-01-01").date(),  # New Year's Day 2017
    pd.Timestamp("2017-01-02").date(),  # New Year's Day observed 2017
    pd.Timestamp("2017-04-14").date(),  # Good Friday 2017
    pd.Timestamp("2017-05-01").date(),  # Early May Bank Holiday 2017
    pd.Timestamp("2017-05-29").date(),  # Spring Bank Holiday 2017
    pd.Timestamp("2017-12-25").date(),  # Christmas Day 2017
    pd.Timestamp("2017-12-26").date(),  # Boxing Day 2017
    pd.Timestamp("2018-01-01").date(),  # New Year's Day 2018
    pd.Timestamp("2018-03-30").date(),  # Good Friday 2018
    pd.Timestamp("2018-05-07").date(),  # Early May Bank Holiday 2018
    pd.Timestamp("2018-05-28").date(),  # Spring Bank Holiday 2018
    pd.Timestamp("2018-12-25").date(),  # Christmas Day 2018
    pd.Timestamp("2018-12-26").date(),  # Boxing Day 2018
}

# Screening parameters
MIN_DURATION_HOURS: float = 5.0 / 60.0  # 5 minutes
MAX_DURATION_HOURS: float = 336.0        # 14 days cap
MIN_RETAINED_SESSIONS: int = 10          # History threshold

