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

# Reports directories
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
AUDIT_REPORTS_DIR: Path = REPORTS_DIR / "data_audit"
EDA_REPORTS_DIR: Path = REPORTS_DIR / "eda"

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
    "Mon",
    "Tue",
    "Wed",
    "Thu",
    "Fri",
    "Sat",
    "Sun",
    "weekday",
    "weekend",
    "holiday",
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

# 13 UK Bank Holidays across the trial window (England and Wales)
UK_BANK_HOLIDAYS_REFERENCE: Set = {
    pd.Timestamp("2017-04-14").date(),  # Good Friday 2017
    pd.Timestamp("2017-04-17").date(),  # Easter Monday 2017
    pd.Timestamp("2017-05-01").date(),  # Early May bank holiday 2017
    pd.Timestamp("2017-05-29").date(),  # Spring bank holiday 2017
    pd.Timestamp("2017-08-28").date(),  # Summer bank holiday 2017
    pd.Timestamp("2017-12-25").date(),  # Christmas Day 2017
    pd.Timestamp("2017-12-26").date(),  # Boxing Day 2017
    pd.Timestamp("2018-01-01").date(),  # New Year's Day 2018
    pd.Timestamp("2018-03-30").date(),  # Good Friday 2018
    pd.Timestamp("2018-04-02").date(),  # Easter Monday 2018
    pd.Timestamp("2018-05-07").date(),  # Early May bank holiday 2018
    pd.Timestamp("2018-05-28").date(),  # Spring bank holiday 2018
    pd.Timestamp("2018-08-27").date(),  # Summer bank holiday 2018
}

# Screening parameters
MIN_DURATION_HOURS: float = 5.0 / 60.0  # 5 minutes
MAX_DURATION_HOURS: float = 336.0        # 14 days cap
MIN_RETAINED_SESSIONS: int = 10          # History threshold
