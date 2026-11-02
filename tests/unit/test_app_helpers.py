"""Unit tests for TARANG Streamlit application helpers and schema validators."""

from __future__ import annotations

import io
import numpy as np
import pandas as pd
import pytest

from tarang.app.batch_prediction import _compute_dataset_id
from tarang.app.helpers import (
    convert_dataframe_to_csv_bytes,
    get_sample_batch_dataframe,
    get_sample_history_csv,
    normalize_participant_id,
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


def test_validate_history_dataframe_numeric_participants() -> None:
    """Verify history validator handles numeric participant IDs consistently as strings."""
    df = pd.DataFrame(
        [
            {"ParticipantID": 102, "StartTime": "2018-05-10 18:00:00", "StopTime": "2018-05-11 06:00:00", "ConsumedkWh": 15.0},
            {"ParticipantID": 101, "StartTime": "2018-05-12 18:00:00", "StopTime": "2018-05-13 06:00:00", "ConsumedkWh": 20.0},
        ]
    )
    ok, err, participants = validate_history_dataframe(df)
    assert ok
    assert err is None
    assert participants == ["101", "102"]


def test_validate_history_dataframe_null_participants_rejection() -> None:
    """Verify history validator explicitly rejects all-null and mixed-null participant IDs."""
    # Case 1: All null participant IDs
    df_all_null = pd.DataFrame(
        [
            {"ParticipantID": np.nan, "StartTime": "2018-05-10 18:00:00", "StopTime": "2018-05-11 06:00:00", "ConsumedkWh": 15.0},
            {"ParticipantID": None, "StartTime": "2018-05-12 18:00:00", "StopTime": "2018-05-13 06:00:00", "ConsumedkWh": 20.0},
        ]
    )
    ok_all, err_all, p_all = validate_history_dataframe(df_all_null)
    assert not ok_all
    assert "missing/null" in str(err_all)
    assert p_all is None

    # Case 2: Mixed null and non-null participant IDs
    df_mixed = pd.DataFrame(
        [
            {"ParticipantID": "WP1-001", "StartTime": "2018-05-10 18:00:00", "StopTime": "2018-05-11 06:00:00", "ConsumedkWh": 15.0},
            {"ParticipantID": np.nan, "StartTime": "2018-05-12 18:00:00", "StopTime": "2018-05-13 06:00:00", "ConsumedkWh": 20.0},
        ]
    )
    ok_mix, err_mix, p_mix = validate_history_dataframe(df_mixed)
    assert not ok_mix
    assert "missing/null participant identifiers" in str(err_mix)
    assert p_mix is None


def test_normalize_participant_id_cases() -> None:
    """Verify normalize_participant_id handles numeric, whitespace, and null cases correctly."""
    # Numeric IDs
    assert normalize_participant_id(101) == "101"
    assert normalize_participant_id(101.0) == "101"

    # Leading and trailing whitespace
    assert normalize_participant_id("  WP1-001  ") == "WP1-001"
    assert normalize_participant_id(" 102 ") == "102"

    # Whitespace-only, empty, and null/nan identifiers
    assert normalize_participant_id("   ") is None
    assert normalize_participant_id("") is None
    assert normalize_participant_id(None) is None
    assert normalize_participant_id(np.nan) is None
    assert normalize_participant_id("nan") is None
    assert normalize_participant_id("None") is None


def test_validate_history_dataframe_whitespace_and_invalid_identifiers() -> None:
    """Verify history validator rejects whitespace-only IDs and normalizes whitespace."""
    # Case 1: Whitespace-only ID mixed with valid ID -> rejected
    df_whitespace_mixed = pd.DataFrame(
        [
            {"ParticipantID": "WP1-001", "StartTime": "2018-05-10 18:00:00", "StopTime": "2018-05-11 06:00:00", "ConsumedkWh": 15.0},
            {"ParticipantID": "   ", "StartTime": "2018-05-12 18:00:00", "StopTime": "2018-05-13 06:00:00", "ConsumedkWh": 20.0},
        ]
    )
    ok_ws, err_ws, p_ws = validate_history_dataframe(df_whitespace_mixed)
    assert not ok_ws
    assert "missing/null participant identifiers" in str(err_ws)
    assert p_ws is None

    # Case 2: Empty string ID mixed with valid ID -> rejected
    df_empty_mixed = pd.DataFrame(
        [
            {"ParticipantID": "WP1-001", "StartTime": "2018-05-10 18:00:00", "StopTime": "2018-05-11 06:00:00", "ConsumedkWh": 15.0},
            {"ParticipantID": "", "StartTime": "2018-05-12 18:00:00", "StopTime": "2018-05-13 06:00:00", "ConsumedkWh": 20.0},
        ]
    )
    ok_em, err_em, p_em = validate_history_dataframe(df_empty_mixed)
    assert not ok_em
    assert "missing/null participant identifiers" in str(err_em)
    assert p_em is None

    # Case 3: Valid IDs with leading and trailing whitespace -> normalized cleanly
    df_ws_valid = pd.DataFrame(
        [
            {"ParticipantID": "  WP1-001  ", "StartTime": "2018-05-10 18:00:00", "StopTime": "2018-05-11 06:00:00", "ConsumedkWh": 15.0},
            {"ParticipantID": " WP1-002 ", "StartTime": "2018-05-12 18:00:00", "StopTime": "2018-05-13 06:00:00", "ConsumedkWh": 20.0},
        ]
    )
    ok_clean, err_clean, p_clean = validate_history_dataframe(df_ws_valid)
    assert ok_clean
    assert err_clean is None
    assert p_clean == ["WP1-001", "WP1-002"]


def test_history_filtering_normalized_participant() -> None:
    """Verify history filtering matches normalized participant and excludes others."""
    raw_df = pd.DataFrame(
        [
            {"ParticipantID": " 101 ", "StartTime": "2026-11-01 10:00:00", "StopTime": "2026-11-01 18:00:00", "ConsumedkWh": 15.0},
            {"ParticipantID": 102, "StartTime": "2026-11-02 10:00:00", "StopTime": "2026-11-02 18:00:00", "ConsumedkWh": 20.0},
            {"ParticipantID": "101", "StartTime": "2026-11-03 10:00:00", "StopTime": "2026-11-03 18:00:00", "ConsumedkWh": 25.0},
        ]
    )
    # Apply normalization as single_prediction does
    raw_df["ParticipantID"] = raw_df["ParticipantID"].apply(normalize_participant_id)
    filtered = raw_df[raw_df["ParticipantID"].notna() & (raw_df["ParticipantID"] == "101")].copy()

    assert len(filtered) == 2
    assert list(filtered["ParticipantID"]) == ["101", "101"]
    assert "102" not in list(filtered["ParticipantID"])


def test_compute_dataset_id_fingerprint() -> None:
    """Verify _compute_dataset_id changes when middle row, index, or dtype changes."""
    df1, _ = get_sample_batch_dataframe(n_rows=25)
    id1 = _compute_dataset_id(df1)

    # 1. Modify middle row only (keep shape, columns, first and last rows identical)
    df2 = df1.copy()
    mid_idx = 12
    df2.loc[mid_idx, "CarKW"] = df2.loc[mid_idx, "CarKW"] + 10.0
    assert df1.shape == df2.shape
    assert list(df1.columns) == list(df2.columns)
    assert df1.iloc[0].equals(df2.iloc[0])
    assert df1.iloc[-1].equals(df2.iloc[-1])
    id2 = _compute_dataset_id(df2)
    assert id1 != id2

    # 2. Modify index
    df3 = df1.copy()
    df3.index = range(100, 125)
    id3 = _compute_dataset_id(df3)
    assert id1 != id3

    # 3. Modify column dtype
    df4 = df1.copy()
    df4["holiday"] = df4["holiday"].astype(float)
    id4 = _compute_dataset_id(df4)
    assert id1 != id4


def test_sample_history_relative_timestamps() -> None:
    """Verify demonstration history timestamps are aligned relative to reference arrival time."""
    ref_time = "2026-11-03 18:00:00+00:00"
    sample_csv = get_sample_history_csv(reference_time=ref_time)
    df_hist = pd.read_csv(io.StringIO(sample_csv))

    assert len(df_hist) == 3
    # Most recent session stop time
    last_stop = pd.to_datetime(df_hist["StopTime"].iloc[-1], utc=True)
    arrival = pd.to_datetime(ref_time, utc=True)
    time_since_stop = (arrival - last_stop).total_seconds() / 3600.0

    # Must be realistic interval (8 hours), not thousands of hours
    assert time_since_stop == 8.0


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

    sample_batch, is_synth = get_sample_batch_dataframe(n_rows=15)
    assert len(sample_batch) == 15
    assert isinstance(is_synth, bool)
    assert "StartTime" in sample_batch.columns
    assert set(FEATURE_NAMES).issubset(sample_batch.columns)

    csv_bytes = convert_dataframe_to_csv_bytes(sample_batch)
    assert isinstance(csv_bytes, bytes)
    assert len(csv_bytes) > 0

