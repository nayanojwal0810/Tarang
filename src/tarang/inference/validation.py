"""Input validation and feature schema verification for TARANG inference.

Ensures strict adherence to the canonical 13-predictor schema, verifies numeric integrity,
prohibits missing or non-finite values, and enforces cold-start and chronological constraints
on participant history.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from tarang.config import FEATURE_NAMES


def validate_inference_features(
    features: Union[pd.DataFrame, pd.Series, Dict[str, Any]],
) -> pd.DataFrame:
    """Validate and align feature input against the authoritative 13-predictor contract.

    Args:
        features: Input feature row(s) as a DataFrame, Series, or dictionary.

    Returns:
        DataFrame containing strictly the 13 canonical features in FEATURE_NAMES order,
        preserving the original index.

    Raises:
        ValueError: If input is empty, missing required features, contains non-numeric
            data, or contains NaN, null, or infinite values.
    """
    if features is None:
        raise ValueError("Inference features input cannot be None.")

    if isinstance(features, dict):
        df = pd.DataFrame([features])
    elif isinstance(features, pd.Series):
        df = features.to_frame().T
    elif isinstance(features, pd.DataFrame):
        df = features
    else:
        raise ValueError(
            f"Unsupported features type: {type(features).__name__}. "
            "Expected DataFrame, Series, or dict."
        )

    if df.empty:
        raise ValueError("Inference features DataFrame is empty.")

    # Validate presence of all 13 canonical features
    missing_features = [f for f in FEATURE_NAMES if f not in df.columns]
    if missing_features:
        raise ValueError(
            f"Missing required predictor column(s): {missing_features}. "
            f"Inference requires all 13 canonical predictors: {FEATURE_NAMES}."
        )

    # Extract strictly the 13 features in canonical order
    extracted = df[FEATURE_NAMES].copy()

    # Validate numeric types
    for col in FEATURE_NAMES:
        if not pd.api.types.is_numeric_dtype(extracted[col]):
            raise ValueError(
                f"Predictor '{col}' contains non-numeric data type '{extracted[col].dtype}'. "
                "All 13 predictors must be numeric."
            )

    # Validate absence of null / NaN values (imputation disallowed)
    null_counts = extracted.isna().sum()
    if (null_counts > 0).any():
        bad_cols = null_counts[null_counts > 0].to_dict()
        raise ValueError(
            f"Predictor column(s) contain missing/null values: {bad_cols}. "
            "Imputation is strictly disallowed."
        )

    # Validate absence of infinite values
    arr = extracted.to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        bad_cols_inf = [col for col in FEATURE_NAMES if not np.all(np.isfinite(extracted[col]))]
        raise ValueError(
            f"Predictor column(s) contain non-finite numbers (inf or -inf): {bad_cols_inf}."
        )

    return extracted


def normalize_utc_timestamp(ts: Any) -> pd.Timestamp:
    """Normalize input timestamp to UTC-aware pandas Timestamp.

    Naive timestamps are treated as UTC; timezone-aware timestamps are converted to UTC.

    Args:
        ts: Input timestamp (string, datetime, Timestamp).

    Returns:
        UTC-aware pandas Timestamp.

    Raises:
        ValueError: If timestamp cannot be parsed or is NaT.
    """
    try:
        dt = pd.to_datetime(ts)
    except Exception as exc:
        raise ValueError(f"Invalid timestamp value: {ts!r}. Parse error: {exc}") from exc

    if pd.isna(dt):
        raise ValueError(f"Timestamp value is missing or NaT: {ts!r}.")

    if dt.tzinfo is None:
        return dt.tz_localize("UTC")
    return dt.tz_convert("UTC")


def validate_session_inputs(
    start_time: Any,
    car_kw: float,
    car_kwh: float,
    history: Optional[pd.DataFrame],
) -> Tuple[pd.Timestamp, float, float, pd.DataFrame]:
    """Validate single-session arrival inputs and participant history.

    Args:
        start_time: Session arrival timestamp.
        car_kw: Vehicle charging rate in kW.
        car_kwh: Vehicle battery capacity in kWh.
        history: DataFrame containing participant's completed historical sessions.

    Returns:
        Tuple of (normalized_start_time, validated_car_kw, validated_car_kwh, validated_history).

    Raises:
        ValueError: If specifications are non-positive/non-finite, timestamps are invalid,
            history is empty (cold-start), history lacks required columns, previous
            sessions contain negative gaps, or the immediately prior session lacks energy.
    """
    # 1. Validate start_time
    norm_start = normalize_utc_timestamp(start_time)

    # 2. Validate vehicle specifications
    try:
        kw = float(car_kw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid CarKW value {car_kw!r}: must be a numeric float.") from exc

    if not np.isfinite(kw) or kw <= 0.0:
        raise ValueError(
            f"Invalid CarKW value {car_kw!r}: must be a strictly positive finite number."
        )

    try:
        kwh = float(car_kwh)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid CarKWh value {car_kwh!r}: must be a numeric float.") from exc

    if not np.isfinite(kwh) or kwh <= 0.0:
        raise ValueError(
            f"Invalid CarKWh value {car_kwh!r}: must be a strictly positive finite number."
        )

    # 3. Validate participant history (cold-start condition)
    if history is None or (isinstance(history, pd.DataFrame) and history.empty):
        raise ValueError(
            "Cold-start condition: participant history is empty. "
            "Model requires at least one prior completed session to compute lag features "
            "(TimeSinceLastStop, LastDuration, LastConsumedkWh)."
        )

    if not isinstance(history, pd.DataFrame):
        raise ValueError(
            f"Participant history must be a pandas DataFrame, got {type(history).__name__}."
        )

    required_history_cols = ["StartTime", "StopTime", "ConsumedkWh"]
    missing_cols = [c for c in required_history_cols if c not in history.columns]
    if missing_cols:
        raise ValueError(
            f"Participant history is missing required column(s): {missing_cols}. "
            f"Required columns: {required_history_cols}."
        )

    # Do not mutate input DataFrame
    hist = history.copy()

    # Parse and normalize timestamps in history
    try:
        raw_start_dt = pd.to_datetime(hist["StartTime"])
        if raw_start_dt.dt.tz is None:
            hist["StartTime"] = raw_start_dt.dt.tz_localize("UTC")
        else:
            hist["StartTime"] = raw_start_dt.dt.tz_convert("UTC")

        raw_stop_dt = pd.to_datetime(hist["StopTime"])
        if raw_stop_dt.dt.tz is None:
            hist["StopTime"] = raw_stop_dt.dt.tz_localize("UTC")
        else:
            hist["StopTime"] = raw_stop_dt.dt.tz_convert("UTC")
    except Exception as exc:
        raise ValueError(
            f"Failed to parse timestamps in participant history as UTC: {exc}"
        ) from exc

    if hist["StartTime"].isna().any() or hist["StopTime"].isna().any():
        raise ValueError(
            "Participant history contains unparseable or missing StartTime or StopTime values."
        )

    # Validate individual session duration in history
    raw_duration = (hist["StopTime"] - hist["StartTime"]).dt.total_seconds()
    if (raw_duration < 0.0).any():
        raise ValueError(
            "Participant history contains invalid session(s) where StopTime precedes StartTime."
        )

    # Sort chronologically by StartTime
    hist = hist.sort_values("StartTime", ascending=True).reset_index(drop=True)

    # Immediately preceding session
    prev_session = hist.iloc[-1]
    prev_start = prev_session["StartTime"]
    prev_stop = prev_session["StopTime"]

    # Reject future history records
    if prev_start >= norm_start:
        raise ValueError(
            f"Invalid history: preceding session StartTime ({prev_start}) "
            f"must be strictly before new session StartTime ({norm_start}). "
            "Future history records are rejected."
        )

    # Reject negative time gaps (overlapping sessions)
    gap_hours = (norm_start - prev_stop).total_seconds() / 3600.0
    if gap_hours < 0.0:
        raise ValueError(
            f"Negative time gap ({gap_hours:.4f} hours): new session StartTime ({norm_start}) "
            f"precedes immediately preceding session StopTime ({prev_stop}). "
            "Overlapping sessions are rejected."
        )

    # Validate immediately preceding session has valid ConsumedkWh
    prev_kwh = prev_session["ConsumedkWh"]
    if pd.isna(prev_kwh) or not np.isfinite(float(prev_kwh)):
        raise ValueError(
            "Immediately preceding session has missing or non-finite ConsumedkWh; "
            "imputation is disallowed by the model contract."
        )

    return norm_start, kw, kwh, hist

