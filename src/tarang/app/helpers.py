"""Data validation, model caching, and helper utilities for TARANG Streamlit Application."""

from __future__ import annotations

from datetime import datetime, timezone
import io
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import streamlit as st

from tarang.config import (
    FEATURE_NAMES,
    PROCESSED_FEATURES_FILE,
)
from tarang.decision_support.engine import DecisionSupportEngine

REQUIRED_HISTORY_COLUMNS: List[str] = ["StartTime", "StopTime", "ConsumedkWh"]


@st.cache_resource(show_spinner="Loading TARANG predictive models...")
def get_decision_support_engine(
    model_name: str = "HistGradientBoostingRegressor",
    models_dir: Optional[Union[str, Path]] = None,
    summary_file: Optional[Union[str, Path]] = None,
) -> DecisionSupportEngine:
    """Load and cache the pre-trained DecisionSupportEngine instance across sessions.

    Honors explicit arguments or TARANG_MODELS_DIR and TARANG_SUMMARY_FILE environment
    variables for clean test isolation and deployment environments.
    """
    env_models = os.environ.get("TARANG_MODELS_DIR")
    env_summary = os.environ.get("TARANG_SUMMARY_FILE")

    effective_models_dir = Path(models_dir) if models_dir else (Path(env_models) if env_models else None)
    effective_summary_file = Path(summary_file) if summary_file else (Path(env_summary) if env_summary else None)

    return DecisionSupportEngine(
        model_name=model_name,
        models_dir=effective_models_dir,
        summary_file=effective_summary_file,
    )


def validate_history_dataframe(
    df: Optional[pd.DataFrame],
) -> Tuple[bool, Optional[str], Optional[List[str]]]:
    """Validate user-uploaded completed session history for single-session inference.

    Handles numeric and string participant identifiers consistently and explicitly
    rejects ambiguous histories with missing or partially missing identifiers.

    Args:
        df: Uploaded session history DataFrame.

    Returns:
        Tuple of (is_valid, error_reason, list_of_participant_ids_if_multiple).
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return False, "Uploaded history file is empty.", None

    missing = [col for col in REQUIRED_HISTORY_COLUMNS if col not in df.columns]
    if missing:
        return (
            False,
            f"Missing required history column(s): {missing}. Required columns: {REQUIRED_HISTORY_COLUMNS}.",
            None,
        )

    # Check for ParticipantID column
    participants: Optional[List[str]] = None
    if "ParticipantID" in df.columns:
        p_series = df["ParticipantID"]
        # Check if all rows are null
        if p_series.isna().all():
            return False, "ParticipantID column is present but contains only missing/null values.", None

        # Check for mixed nulls (some null rows alongside identified records)
        if p_series.isna().any():
            return (
                False,
                "Uploaded history contains rows with missing/null participant identifiers alongside identified records. "
                "Every record must have a valid identifier.",
                None,
            )

        # Convert identifiers (numeric or string) to clean strings
        raw_unique = p_series.dropna().unique()
        unique_p = sorted(list(set(str(x).strip() for x in raw_unique if str(x).strip())))

        if len(unique_p) == 0:
            return False, "ParticipantID column contains no valid non-empty identifiers.", None
        elif len(unique_p) > 1:
            participants = unique_p

    return True, None, participants


def validate_batch_dataframe(
    df: Optional[pd.DataFrame],
) -> Tuple[bool, Optional[str], List[str]]:
    """Validate uploaded batch inference dataset schema.

    Must contain 'StartTime' and all 13 canonical model predictors.

    Args:
        df: Uploaded batch DataFrame.

    Returns:
        Tuple of (is_valid, error_reason, list_of_missing_columns).
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return False, "Batch dataset is empty or could not be parsed.", []

    required = ["StartTime"] + FEATURE_NAMES
    missing = [c for c in required if c not in df.columns]
    if missing:
        return (
            False,
            f"Dataset is missing required column(s): {missing}.",
            missing,
        )

    # Check for nulls in the 13 predictors
    null_counts = df[FEATURE_NAMES].isna().sum()
    cols_with_nulls = null_counts[null_counts > 0].to_dict()
    if cols_with_nulls:
        return (
            False,
            f"Input features contain missing/null values: {cols_with_nulls}. Predictors must be fully populated.",
            list(cols_with_nulls.keys()),
        )

    return True, None, []


def load_file_into_dataframe(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> pd.DataFrame:
    """Safely parse uploaded CSV or Parquet file into a pandas DataFrame."""
    filename = uploaded_file.name.lower()
    if filename.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif filename.endswith(".parquet"):
        return pd.read_parquet(uploaded_file)
    else:
        raise ValueError(f"Unsupported file format '{uploaded_file.name}'. Please upload a .csv or .parquet file.")


def get_sample_history_csv(
    reference_time: Optional[Union[str, pd.Timestamp, datetime]] = None,
) -> str:
    """Generate sample historical completed sessions CSV string for testing.

    Places prior completed sessions relative to reference_time (or current UTC time)
    to ensure realistic interval gaps without extreme TimeSinceLastStop values.
    """
    if reference_time is not None:
        try:
            ref_ts = pd.to_datetime(reference_time, utc=True)
            if pd.isna(ref_ts):
                ref_ts = pd.Timestamp.now(timezone.utc)
        except Exception:
            ref_ts = pd.Timestamp.now(timezone.utc)
    else:
        ref_ts = pd.Timestamp.now(timezone.utc)

    sample_df = pd.DataFrame(
        [
            {
                "ParticipantID": "WP1-042",
                "StartTime": (ref_ts - pd.Timedelta(hours=66)).strftime("%Y-%m-%d %H:%M:%S+00:00"),
                "StopTime": (ref_ts - pd.Timedelta(hours=56)).strftime("%Y-%m-%d %H:%M:%S+00:00"),
                "ConsumedkWh": 14.8,
            },
            {
                "ParticipantID": "WP1-042",
                "StartTime": (ref_ts - pd.Timedelta(hours=40)).strftime("%Y-%m-%d %H:%M:%S+00:00"),
                "StopTime": (ref_ts - pd.Timedelta(hours=32)).strftime("%Y-%m-%d %H:%M:%S+00:00"),
                "ConsumedkWh": 18.2,
            },
            {
                "ParticipantID": "WP1-042",
                "StartTime": (ref_ts - pd.Timedelta(hours=16)).strftime("%Y-%m-%d %H:%M:%S+00:00"),
                "StopTime": (ref_ts - pd.Timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S+00:00"),
                "ConsumedkWh": 22.5,
            },
        ]
    )
    return sample_df.to_csv(index=False)


def get_sample_batch_dataframe(n_rows: int = 50) -> Tuple[pd.DataFrame, bool]:
    """Retrieve sample batch evaluation records.

    Returns:
        Tuple of (DataFrame, is_synthetic: bool).
        is_synthetic is False if records are sampled from Electric Nation canonical dataset,
        and True if generated via synthetic fallback.
    """
    if PROCESSED_FEATURES_FILE.is_file():
        try:
            full_df = pd.read_parquet(PROCESSED_FEATURES_FILE)
            sample = full_df.head(n_rows).copy()
            return sample, False
        except Exception:
            pass

    # Synthetic fallback for clean environments without local private data
    np.random.seed(42)
    start_times = pd.date_range("2018-05-01 08:00:00", periods=n_rows, freq="6h", tz="UTC")
    data = {
        "StartTime": start_times,
        "weekday": start_times.weekday,
        "weekend": (start_times.weekday >= 5).astype(int),
        "holiday": 0,
        "month": start_times.month,
        "StartCos": np.cos(2 * np.pi * start_times.hour / 24.0),
        "StartSin": np.sin(2 * np.pi * start_times.hour / 24.0),
        "StartHourNorm": start_times.hour / 24.0,
        "TimeSinceLastStop": np.random.uniform(4.0, 36.0, size=n_rows),
        "sessionsToday": np.random.choice([0, 1, 2], size=n_rows, p=[0.7, 0.25, 0.05]),
        "LastDuration": np.random.uniform(2.0, 14.0, size=n_rows),
        "LastConsumedkWh": np.random.uniform(4.0, 32.0, size=n_rows),
        "CarKW": np.random.choice([3.6, 7.0, 11.0, 22.0], size=n_rows),
        "CarKWh": np.random.choice([24.0, 30.0, 40.0, 75.0, 100.0], size=n_rows),
        # Historical flexibility columns
        "StopTime": start_times + pd.to_timedelta(np.random.uniform(3.0, 14.0, size=n_rows), unit="h"),
        "ChargingDuration": np.random.uniform(60.0, 300.0, size=n_rows),
    }
    return pd.DataFrame(data), True


def convert_dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Encode DataFrame to UTF-8 CSV bytes for Streamlit file download button."""
    return df.to_csv(index=False).encode("utf-8")

