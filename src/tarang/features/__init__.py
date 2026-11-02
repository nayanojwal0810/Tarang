"""TARANG feature engineering package.

Exposes calendar, time-of-day, historical lag, and predictor extraction utilities.
"""

from tarang.config import FEATURE_NAMES, UK_BANK_HOLIDAYS_REFERENCE
from tarang.features.calendar_features import calculate_calendar_features
from tarang.features.feature_engineering import extract_features, get_feature_names
from tarang.features.historical_features import (
    calculate_last_consumed_kwh,
    calculate_last_duration,
    calculate_sessions_today,
    calculate_time_since_last_stop,
)
from tarang.features.time_features import (
    calculate_cyclical_time_features,
    calculate_start_hour,
)

__all__ = [
    "FEATURE_NAMES",
    "UK_BANK_HOLIDAYS_REFERENCE",
    "calculate_calendar_features",
    "calculate_start_hour",
    "calculate_cyclical_time_features",
    "calculate_time_since_last_stop",
    "calculate_sessions_today",
    "calculate_last_duration",
    "calculate_last_consumed_kwh",
    "get_feature_names",
    "extract_features",
]
