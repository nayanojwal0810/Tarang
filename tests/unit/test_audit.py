"""Unit tests for the TARANG data audit functions.

Uses small synthetic DataFrames to thoroughly verify core audit logic,
including timestamp handling, nested overlaps, row-level metadata reconciliation,
energy thresholds, participant history screening, and anti-leakage boundaries.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tarang.data.audit import (
    audit_energy_quality,
    audit_feature_feasibility,
    audit_metadata_row_level,
    calculate_missingness,
    detect_duplicates,
    detect_session_overlaps,
    investigate_unattributed_sessions,
    parse_utc_timestamps,
    screen_participant_history,
)


def test_parse_utc_timestamps_valid_and_failures() -> None:
    """Test UTC parsing, handling of malformed and missing timestamps, and interval checks."""
    df = pd.DataFrame({
        "StartTime": ["2017-04-01 10:00:00", "2017-04-02 12:00:00", "not-a-date", None, "2017-04-05 08:00:00"],
        "StopTime": ["2017-04-01 11:00:00", "2017-04-02 12:00:00", "2017-04-03 14:00:00", "2017-04-04 15:00:00", "2017-04-05 07:00:00"],
    })
    start_utc, stop_utc, metrics = parse_utc_timestamps(df, "StartTime", "StopTime")

    assert str(start_utc.dt.tz) == "UTC"
    assert str(stop_utc.dt.tz) == "UTC"
    assert metrics["start_parse_failures"] == 1  # 'not-a-date'
    assert metrics["stop_parse_failures"] == 0
    assert metrics["missing_start"] == 2  # 'not-a-date' coerced + None
    assert metrics["missing_stop"] == 0
    assert metrics["zero_intervals"] == 1  # 2017-04-02 12:00 == 12:00
    assert metrics["negative_intervals"] == 1  # 2017-04-05 07:00 < 08:00
    assert metrics["negative_intervals_pct"] == 20.0  # 1 out of 5
    assert metrics["zero_intervals_pct"] == 20.0  # 1 out of 5


def test_detect_duplicates() -> None:
    """Test duplicate detection on full rows and specific column subsets with dynamic counts."""
    df = pd.DataFrame({
        "TransactionID": [101, 102, 102, 103],
        "Value": ["a", "b", "b", "c"],
    })
    assert detect_duplicates(df) == 1
    assert detect_duplicates(df, subset=["TransactionID"]) == 1

    df_unique = pd.DataFrame({
        "TransactionID": [101, 102, 103],
        "Value": ["a", "b", "c"],
    })
    assert detect_duplicates(df_unique) == 0


def test_calculate_missingness_dynamic() -> None:
    """Test calculation of missing values count and non-zero dynamic percentages."""
    df = pd.DataFrame({
        "col_a": [1.0, 2.0, None, 4.0],
        "col_b": [None, None, None, None],
        "col_c": ["x", "y", "z", "w"],
    })
    res = calculate_missingness(df)
    assert len(res) == 3

    row_a = res[res["column_name"] == "col_a"].iloc[0]
    assert row_a["missing_count"] == 1
    assert row_a["missing_percentage"] == 25.0

    row_b = res[res["column_name"] == "col_b"].iloc[0]
    assert row_b["missing_count"] == 4
    assert row_b["missing_percentage"] == 100.0


def test_nested_session_overlaps() -> None:
    """Test detection of nested session overlaps using cumulative maximum stop times.

    Session 1: 10:00 to 20:00 (Long session)
    Session 2: 11:00 to 12:00 (Direct overlap with Session 1)
    Session 3: 13:00 to 14:00 (Starts after Session 2 ends, but before Session 1 ends -> nested overlap)
    Session 4: 20:00 to 22:00 (Starts exactly when Session 1 ends -> non-overlapping)
    """
    df = pd.DataFrame({
        "ParticipantID": ["EN001", "EN001", "EN001", "EN001", "EN002"],
        "StartTime": [
            "2017-05-01 10:00:00",
            "2017-05-01 11:00:00",
            "2017-05-01 13:00:00",
            "2017-05-01 20:00:00",
            "2017-05-01 09:00:00",
        ],
        "StopTime": [
            "2017-05-01 20:00:00",
            "2017-05-01 12:00:00",
            "2017-05-01 14:00:00",
            "2017-05-01 22:00:00",
            "2017-05-01 10:00:00",
        ],
    })
    overlap_mask, overlap_count = detect_session_overlaps(df, "ParticipantID", "StartTime", "StopTime")

    # Session 2 and Session 3 for EN001 must both be flagged as overlapping
    assert overlap_count == 2
    assert not overlap_mask.iloc[0]  # Session 1: initial session
    assert overlap_mask.iloc[1]      # Session 2: overlap with Session 1
    assert overlap_mask.iloc[2]      # Session 3: nested overlap with Session 1
    assert not overlap_mask.iloc[3]  # Session 4: exact boundary match (non-overlapping)
    assert not overlap_mask.iloc[4]  # EN002: independent participant


def test_metadata_row_level_mismatches_and_consistency() -> None:
    """Test row-level comparison of vehicle specifications and participant consistency."""
    df_tx = pd.DataFrame({
        "ParticipantID": ["EN001", "EN001", "EN002", "EN003"],
        "ChargerID": [3501, 3501, 3502, 3503],
        "CarKW": [7.0, 7.0, 3.6, 7.0],
        "CarKWh": [33.0, 33.0, 24.0, 75.0],
    })
    df_meta = pd.DataFrame({
        "ParticipantID": ["EN001", "EN002", "EN003"],
        "Charger": ["3501", "3502", "9999"],  # EN003 has mismatched charger
        "CarkW": [7.0, 7.0, 7.0],             # EN002 has mismatched CarkW (7.0 vs 3.6)
        "CarkWh": [33.0, 24.0, 85.0],         # EN003 has mismatched CarkWh (85.0 vs 75.0)
    })
    metrics = audit_metadata_row_level(df_tx, df_meta)

    assert metrics["matched_participants"] == 3
    assert metrics["total_matched_transaction_rows"] == 4
    assert metrics["row_level_carkw_mismatches"] == 1       # EN002 session
    assert metrics["row_level_carkw_mismatch_pct"] == 25.0
    assert metrics["row_level_carkwh_mismatches"] == 1      # EN003 session
    assert metrics["row_level_carkwh_mismatch_pct"] == 25.0
    assert metrics["row_level_charger_mismatches"] == 1     # EN003 charger
    assert metrics["participants_with_inconsistent_carkw"] == 0


def test_participant_inconsistent_vehicle_specs() -> None:
    """Test detection when a participant has fluctuating CarKW or CarKWh across transactions."""
    df_tx = pd.DataFrame({
        "ParticipantID": ["EN001", "EN001"],
        "ChargerID": [3501, 3501],
        "CarKW": [7.0, 3.6],  # Inconsistent CarKW within EN001!
        "CarKWh": [33.0, 33.0],
    })
    df_meta = pd.DataFrame({
        "ParticipantID": ["EN001"],
        "Charger": ["3501"],
        "CarkW": [7.0],
        "CarkWh": [33.0],
    })
    metrics = audit_metadata_row_level(df_tx, df_meta)
    assert metrics["participants_with_inconsistent_carkw"] == 1


def test_unattributed_sessions_and_duplicate_matching() -> None:
    """Test natural session matching rule on unattributed records."""
    df_tx = pd.DataFrame({
        "TransactionID": [101, 102, 201, 202],
        "ChargerID": [3501, 3502, 3501, 9999],
        "ParticipantID": ["EN001", "EN002", None, None],
        "StartTime": ["2017-05-01 10:00:00", "2017-05-02 11:00:00", "2017-05-01 10:00:00", "2017-05-03 14:00:00"],
        "StopTime": ["2017-05-01 12:00:00", "2017-05-02 13:00:00", "2017-05-01 12:00:00", "2017-05-03 16:00:00"],
        "ConsumedkWh": [10.5, 8.0, 10.5, 12.0],
    })
    df_meta = pd.DataFrame({
        "ParticipantID": ["EN001", "EN002"],
        "Charger": ["3501", "3502"],  # 9999 is unregistered
    })
    res = investigate_unattributed_sessions(df_tx, df_meta)

    assert res["total_unattributed_records"] == 2
    assert res["chargers_involved"] == 2
    assert res["known_chargers_count"] == 1
    assert res["unknown_chargers_count"] == 1
    assert res["unknown_charger_ids"] == ["9999"]
    assert res["unknown_charger_records"] == 1
    assert res["exact_duplicate_session_records"] == 1  # Tx 201 matches Tx 101 on natural session key
    assert res["excess_duplicate_rows"] == 1
    assert res["unattributed_non_duplicate_records"] == 1  # Tx 202


def test_energy_quality_thresholds() -> None:
    """Test energy quality checks against 100 kWh, 150 kWh, and CarKWh capacity."""
    df_tx = pd.DataFrame({
        "TransactionID": [1, 2, 3, 4, 5],
        "CarKWh": [33.0, 33.0, 75.0, 24.0, None],
        "ConsumedkWh": [15.0, 38.0, 110.0, 160.0, None],
    })
    metrics = audit_energy_quality(df_tx)

    assert metrics["total_records"] == 5
    assert metrics["missing_energy"] == 1
    assert metrics["missing_energy_pct"] == 20.0
    assert metrics["non_positive_energy"] == 0
    assert metrics["energy_gt_100kwh"] == 2  # 110.0 and 160.0
    assert metrics["energy_gt_100kwh_pct"] == 40.0
    assert metrics["energy_gt_150kwh"] == 1  # 160.0
    assert metrics["energy_gt_150kwh_pct"] == 20.0
    # ConsumedkWh > CarKWh: 38.0 > 33.0, 110.0 > 75.0, 160.0 > 24.0 (3 records out of 4 valid pairs)
    assert metrics["energy_exceeds_carkwh"] == 3
    assert metrics["energy_exceeds_carkwh_pct"] == 75.0


def test_screen_participant_history_retained_and_density() -> None:
    """Test participant screening: Trial 3 exclusion, duration threshold, overlap, dropping first session, and retained >= 10."""
    # Build 14 valid sessions for EN001, 5 valid sessions for EN002, and 1 Trial 3 session for EN001
    dates_en001 = pd.date_range("2017-05-01 10:00:00", periods=15, freq="2D")
    df_en001 = pd.DataFrame({
        "TransactionID": range(101, 116),
        "ParticipantID": "EN001",
        "StartTime": dates_en001,
        "StopTime": dates_en001 + pd.Timedelta(hours=4),
        "Trial": [1.0] * 14 + [3.0],  # 1 Trial 3 session
    })
    dates_en002 = pd.date_range("2017-06-01 10:00:00", periods=5, freq="3D")
    df_en002 = pd.DataFrame({
        "TransactionID": range(201, 206),
        "ParticipantID": "EN002",
        "StartTime": dates_en002,
        "StopTime": dates_en002 + pd.Timedelta(hours=2),
        "Trial": [1.0] * 5,
    })
    df_combined = pd.concat([df_en001, df_en002], ignore_index=True)

    metrics = screen_participant_history(df_combined)

    assert metrics["initial_valid_records"] == 20
    assert metrics["initial_participants"] == 2
    assert metrics["trial_3_excluded"] == 1  # 1 Trial 3 excluded
    assert metrics["remaining_after_t3"] == 19
    # Dropped first session: 2 (1 per participant)
    # Remaining: EN001 has 13 retained, EN002 has 4 retained
    assert metrics["retained_sessions_total"] == 17  # 13 + 4
    assert metrics["participants_lt_10_retained"] == 1  # EN002 has 4 (< 10)
    assert metrics["sessions_in_lt_10_retained"] == 4
    assert metrics["eligible_participants_gte_10"] == 1  # EN001 has 13 (>= 10)
    assert metrics["eligible_sessions_total"] == 13
    assert metrics["density_mean_span_days"] > 0.0


def test_feature_feasibility_completeness_and_leakage() -> None:
    """Test that all 13 planned features and 2 prediction targets are audited with strict leakage rules."""
    df_tx = pd.DataFrame()
    df_meta = pd.DataFrame()
    feasibility = audit_feature_feasibility(df_tx, df_meta)

    assert len(feasibility) == 15  # 13 features + 2 targets

    expected_13_features = {
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
    }
    actual_features = set(feasibility[feasibility["type"] == "Predictor Feature"]["feature_name"])
    assert actual_features == expected_13_features

    targets = set(feasibility[feasibility["type"].str.contains("Prediction Target")]["feature_name"])
    assert "Plugged-in Duration (Hours)" in targets
    assert "Energy Consumption (kWh)" in targets

    # Verify that all time bases are UTC
    utc_rows = feasibility[feasibility["time_basis"].str.contains("UTC")]
    assert len(utc_rows) >= 11  # All temporal features and targets reference UTC

    # Verify target leakage boundaries
    for _, row in feasibility[feasibility["type"].str.contains("Prediction Target")].iterrows():
        assert "TARGET LEAKAGE" in row["leakage_risk"]
