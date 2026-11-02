"""Feature construction engine for single-session TARANG inference.

Constructs the authoritative 13-predictor feature vector from a new session's arrival time,
vehicle specifications (CarKW, CarKWh), and historical completed sessions for that participant.
Enforces exact parity with Stage 1 feature engineering:
- UK bank holiday calendar overrides
- Global dataset-wide max_start_hour normalization
- Raw StopTime gap calculations (TimeSinceLastStop)
- 336-hour duration capping (LastDuration)
- Strict rejection of cold-start and missing lag states
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional
import numpy as np
import pandas as pd

from tarang.config import (
    FEATURE_NAMES,
    MAX_DURATION_HOURS,
    UK_BANK_HOLIDAYS_REFERENCE,
)
from tarang.features.calendar_features import calculate_calendar_features
from tarang.features.time_features import calculate_cyclical_time_features
from tarang.inference.loading import load_max_start_hour
from tarang.inference.validation import validate_session_inputs

logger = logging.getLogger("tarang.inference.feature_builder")


def build_inference_features(
    start_time: Any,
    car_kw: float,
    car_kwh: float,
    history: pd.DataFrame,
    max_start_hour: Optional[float] = None,
    summary_file: Optional[Path] = None,
) -> pd.DataFrame:
    """Construct canonical 13-predictor feature row for a single charging session.

    Args:
        start_time: Session arrival timestamp (naive timestamps treated as UTC).
        car_kw: Vehicle charger power in kW (strictly positive).
        car_kwh: Vehicle battery capacity in kWh (strictly positive).
        history: DataFrame containing participant's completed prior sessions with
            columns ['StartTime', 'StopTime', 'ConsumedkWh'].
        max_start_hour: Dataset-wide maximum StartHour from Stage 1. If None,
            dynamically loaded from Stage 1 processing_summary.json.
        summary_file: Optional path to processing_summary.json if max_start_hour
            is not provided.

    Returns:
        One-row DataFrame containing the 13 canonical predictors in FEATURE_NAMES order.

    Raises:
        ValueError: If inputs fail validation, history is empty (cold-start),
            prior session energy is missing, or max_start_hour is invalid.
        FileNotFoundError: If processing summary file is missing and max_start_hour
            is not explicitly provided.
    """
    # 1. Resolve and validate max_start_hour normalization denominator
    if max_start_hour is None:
        norm_max = load_max_start_hour(summary_file=summary_file)
    else:
        norm_max = float(max_start_hour)
        if not np.isfinite(norm_max) or norm_max <= 0.0:
            raise ValueError(
                f"Invalid max_start_hour {max_start_hour!r}: must be a positive finite float."
            )

    # 2. Validate session inputs and history integrity
    norm_start, kw, kwh, hist = validate_session_inputs(
        start_time=start_time,
        car_kw=car_kw,
        car_kwh=car_kwh,
        history=history,
    )

    # 3. Calendar and cyclical time features
    # Build single-row DataFrame for time and calendar feature extraction
    temp_df = pd.DataFrame({"StartTime": [norm_start]})
    temp_df = calculate_calendar_features(
        temp_df,
        time_col="StartTime",
        holidays=UK_BANK_HOLIDAYS_REFERENCE,
    )
    temp_df = calculate_cyclical_time_features(
        temp_df,
        start_hour_max=norm_max,
        time_col="StartTime",
    )

    weekday_val = int(temp_df["weekday"].iloc[0])
    weekend_val = int(temp_df["weekend"].iloc[0])
    holiday_val = int(temp_df["holiday"].iloc[0])
    month_val = int(temp_df["month"].iloc[0])
    start_cos_val = float(temp_df["StartCos"].iloc[0])
    start_sin_val = float(temp_df["StartSin"].iloc[0])
    start_hour_norm_val = float(temp_df["StartHourNorm"].iloc[0])

    # 4. Participant historical lag features
    # Immediately preceding session (chronologically last in sorted history)
    prev_session = hist.iloc[-1]
    prev_start = prev_session["StartTime"]
    prev_stop = prev_session["StopTime"]
    prev_consumed = float(prev_session["ConsumedkWh"])

    # TimeSinceLastStop: Hours elapsed from immediately preceding StopTime
    time_since_last_stop = (norm_start - prev_stop).total_seconds() / 3600.0

    # sessionsToday: Count of completed sessions in history on same UTC date
    arrival_date = norm_start.date()
    hist_dates = hist["StartTime"].dt.date
    sessions_today = int((hist_dates == arrival_date).sum())

    # LastDuration: Preceding session duration in hours capped at 336.0h
    raw_last_duration = (prev_stop - prev_start).total_seconds() / 3600.0
    last_duration = min(raw_last_duration, MAX_DURATION_HOURS)

    # LastConsumedkWh: Raw consumed energy of preceding session
    last_consumed_kwh = prev_consumed

    # 5. Assemble dictionary and return 1-row DataFrame in exact FEATURE_NAMES order
    feature_dict = {
        "weekday": weekday_val,
        "weekend": weekend_val,
        "holiday": holiday_val,
        "month": month_val,
        "StartCos": start_cos_val,
        "StartSin": start_sin_val,
        "StartHourNorm": start_hour_norm_val,
        "TimeSinceLastStop": time_since_last_stop,
        "sessionsToday": sessions_today,
        "LastDuration": last_duration,
        "LastConsumedkWh": last_consumed_kwh,
        "CarKW": kw,
        "CarKWh": kwh,
    }

    feature_df = pd.DataFrame([feature_dict], columns=FEATURE_NAMES)
    return feature_df

