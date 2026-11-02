"""Unit tests for TARANG Streamlit application helpers and schema validators."""

from __future__ import annotations

import io
import numpy as np
import pandas as pd
import pytest

from tarang.app.helpers import (
    convert_dataframe_to_csv_bytes,
    get_sample_batch_dataframe,
    get_sample_history_csv,
    validate_batch_dataframe,
    validate_history_dataframe,
)
from tarang.config import FEATURE_NAMES


def test_validate_history_dataframe_empty_and_none() -> None:
    """Verify history validator rejects empty and None inputs."""
    ok, err, participants = validate_history_dataframe(None)
    assert not ok
    assert "empty" in str(err)
    assert participants is None

    ok_empty, err_empty, participants_empty = validate_history_dataframe(pd.DataFrame())
    assert not ok_empty
    assert "empty" in str(err_empty)
    assert participants_empty is None


def test_validate_history_dataframe_missing_columns() -> None:
    """Verify history validator rejects DataFrames missing required columns."""
    df_missing = pd.DataFrame({"StartTime": ["2018-05-15 10:00:00"], "ConsumedkWh": [15.0]})
    ok, err, participants = validate_history_dataframe(df_missing)
    assert not ok
    assert "StopTime" in str(err)
    assert participants is None


def test_validate_history_dataframe_single_participant() -> None:
    """Verify history validator succeeds for valid single participant records."""
    df = pd.DataFrame(
        [
            {
                "ParticipantID": "WP1-001",
                "StartTime": "2018-05-10 18:00:00",
                "StopTime": "2018-05-11 06:00:00",
                "ConsumedkWh": 15.0,
            }
        ]
    )
    ok, err, participants = validate_history_dataframe(df)
    assert ok
    assert err is None
    assert participants is None  # Single participant -> no multiple participants prompt needed


def test_validate_history_dataframe_multiple_participants() -> None:
    """Verify history validator detects multiple participants and returns sorted list."""
    df = pd.DataFrame(
        [
            {"ParticipantID": "WP1-002", "StartTime": "2018-05-10 18:00:00", "StopTime": "2018-05-11 06:00:00", "ConsumedkWh": 15.0},
            {"ParticipantID": "WP1-001", "StartTime": "2018-05-12 18:00:00", "StopTime": "2018-05-13 06:00:00", "ConsumedkWh": 20.0},
        ]
    )
    ok, err, participants = validate_history_dataframe(df)
    assert ok
    assert err is None
    assert participants == ["WP1-001", "WP1-002"]


def test_validate_batch_dataframe_empty_and_none() -> None:
    """Verify batch validator rejects empty and None inputs."""
    ok, err, missing = validate_batch_dataframe(None)
    assert not ok
    assert "empty" in str(err)

    ok_empty, err_empty, missing_empty = validate_batch_dataframe(pd.DataFrame())
    assert not ok_empty
    assert "empty" in str(err_empty)


def test_validate_batch_dataframe_missing_starttime_or_features() -> None:
    """Verify batch validator rejects datasets missing StartTime or any of 13 features."""
    # Missing StartTime
    df_no_start = pd.DataFrame({col: [1.0] for col in FEATURE_NAMES})
    ok_no_start, err_no_start, missing_no_start = validate_batch_dataframe(df_no_start)
    assert not ok_no_start
    assert "StartTime" in missing_no_start

    # Missing one canonical feature (e.g. CarKW)
    cols_without_carkw = [c for c in FEATURE_NAMES if c != "CarKW"]
    df_missing_feat = pd.DataFrame({col: [1.0] for col in cols_without_carkw})
    df_missing_feat["StartTime"] = "2018-05-15 10:00:00"
    ok_feat, err_feat, missing_feat = validate_batch_dataframe(df_missing_feat)
    assert not ok_feat
    assert "CarKW" in missing_feat


def test_validate_batch_dataframe_null_detection() -> None:
    """Verify batch validator detects missing/null values in predictors."""
    row = {col: 1.0 for col in FEATURE_NAMES}
    row["StartTime"] = "2018-05-15 10:00:00"
    row["LastDuration"] = np.nan
    df_null = pd.DataFrame([row])

    ok, err, missing = validate_batch_dataframe(df_null)
    assert not ok
    assert "LastDuration" in str(err)
    assert "LastDuration" in missing


def test_validate_batch_dataframe_success() -> None:
    """Verify batch validator succeeds on fully populated canonical schema."""
    row = {col: 1.0 for col in FEATURE_NAMES}
    row["StartTime"] = "2018-05-15 10:00:00"
    df_valid = pd.DataFrame([row])

    ok, err, missing = validate_batch_dataframe(df_valid)
    assert ok
    assert err is None
    assert len(missing) == 0


def test_sample_generators_and_converters() -> None:
    """Verify sample history and batch generators produce valid structures."""
    sample_hist = get_sample_history_csv()
    df_hist = pd.read_csv(io.StringIO(sample_hist))
    assert len(df_hist) >= 3
    assert set(["StartTime", "StopTime", "ConsumedkWh"]).issubset(df_hist.columns)

    sample_batch = get_sample_batch_dataframe(n_rows=15)
    assert len(sample_batch) == 15
    assert "StartTime" in sample_batch.columns
    assert set(FEATURE_NAMES).issubset(sample_batch.columns)

    csv_bytes = convert_dataframe_to_csv_bytes(sample_batch)
    assert isinstance(csv_bytes, bytes)
    assert len(csv_bytes) > 0
