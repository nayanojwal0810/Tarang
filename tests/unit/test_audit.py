"""Unit tests for the TARANG data audit functions.

Uses small synthetic DataFrames to thoroughly verify core audit logic,
including timestamp handling, nested overlaps, row-level metadata reconciliation,
energy thresholds, participant history screening, observation-density filtering,
anti-leakage boundaries, schema validation, and dynamic report rendering.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from tarang.data.audit import (
    REQUIRED_META_COLUMNS,
    REQUIRED_TX_COLUMNS,
    audit_energy_quality,
    audit_feature_feasibility,
    audit_metadata_row_level,
    calculate_missingness,
    detect_duplicates,
    detect_session_overlaps,
    generate_markdown_report,
    generate_metadata_quality_summary,
    generate_participant_reconciliation,
    generate_transaction_quality_summary,
    investigate_unattributed_sessions,
    load_dataset,
    parse_utc_timestamps,
    screen_participant_history,
    validate_workbook_schema,
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


def test_unattributed_sessions_and_natural_session_matching() -> None:
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
    assert res["matched_4field_session_records"] == 1  # Tx 201 matches Tx 101 on natural 4-field session key
    assert res["unmatched_session_records"] == 1       # Tx 202


def test_natural_key_matches_differing_transaction_fields() -> None:
    """Test natural session matching where other transaction fields (Trial, GroupID, telemetry) differ."""
    df_tx = pd.DataFrame({
        "TransactionID": [101, 102, 201, 202],
        "ChargerID": [3501, 3502, 3501, 9999],
        "ParticipantID": ["EN001", "EN002", None, None],
        "StartTime": ["2017-05-01 10:00:00", "2017-05-02 11:00:00", "2017-05-01 10:00:00", "2017-05-03 14:00:00"],
        "StopTime": ["2017-05-01 12:00:00", "2017-05-02 13:00:00", "2017-05-01 12:00:00", "2017-05-03 16:00:00"],
        "ConsumedkWh": [10.5, 8.0, 10.5, 12.0],
        # Differing transaction fields:
        "Trial": [1.0, 1.0, 2.0, 1.0],
        "GroupID": [5.0, 5.0, 10.0, None],
        "Max_Current_Drawn_for_T": [16.0, 32.0, 32.0, 16.0],
    })
    df_meta = pd.DataFrame({
        "ParticipantID": ["EN001", "EN002"],
        "Charger": ["3501", "3502"],  # 9999 is unregistered
    })
    res = investigate_unattributed_sessions(df_tx, df_meta)

    assert res["total_unattributed_records"] == 2
    assert res["matched_4field_session_records"] == 1
    assert res["unknown_charger_records"] == 1
    assert res["unknown_charger_ids"] == ["9999"]


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
    assert metrics["max_energy"] == 160.0


def test_density_rule_participant_fails_minimum_density() -> None:
    """A participant who passes the minimum session count (>=10) but fails the density rule.

    Observation period of retained sessions: 77 days (earliest retained StartTime to latest retained StopTime).
    Required sessions: observation_days / 7 = 77 / 7 = 11.0 sessions.
    Retained sessions: 10 sessions.
    Since 10 < 11.0, the participant is excluded by the density filter.
    """
    base = pd.Timestamp("2017-01-01 10:00:00")
    retained_start = pd.Timestamp("2017-01-02 10:00:00")
    # 10 retained sessions spaced across 77 days
    retained_starts = [retained_start + pd.Timedelta(days=i * 8) for i in range(10)]
    all_starts = [base] + retained_starts
    all_stops = [base + pd.Timedelta(hours=2)] + [t + pd.Timedelta(hours=2) for t in retained_starts[:-1]] + [retained_start + pd.Timedelta(days=77, hours=2)]
    df = pd.DataFrame({
        "TransactionID": range(1, 12),
        "ParticipantID": "EN_SPARSE",
        "StartTime": all_starts,
        "StopTime": all_stops,
        "Trial": 1.0,
    })
    metrics = screen_participant_history(df)
    assert metrics["participants_before_density"] == 1
    assert metrics["sessions_before_density"] == 10
    assert metrics["participants_excluded_density"] == 1
    assert metrics["sessions_excluded_density"] == 10
    assert metrics["final_eligible_participants"] == 0
    assert metrics["final_eligible_sessions"] == 0


def test_density_rule_participant_passes_density() -> None:
    """A participant who passes the observation-density rule.

    Observation period of retained sessions: 70 days (earliest retained StartTime to latest retained StopTime).
    Required sessions: observation_days / 7 = 70 / 7 = 10.0 sessions.
    Retained sessions: 15 sessions.
    Since 15 >= 10.0, the participant is retained.
    """
    base = pd.Timestamp("2017-01-01 10:00:00")
    retained_base = pd.Timestamp("2017-01-02 10:00:00")
    times = [base] + [retained_base + pd.Timedelta(days=i * 5) for i in range(15)]
    stops = [base + pd.Timedelta(hours=2)] + [t + pd.Timedelta(hours=2) for t in times[1:-1]] + [retained_base + pd.Timedelta(days=70, hours=2)]
    df = pd.DataFrame({
        "TransactionID": range(1, 17),
        "ParticipantID": "EN_ACTIVE",
        "StartTime": times,
        "StopTime": stops,
        "Trial": 1.0,
    })
    metrics = screen_participant_history(df)
    assert metrics["participants_before_density"] == 1
    assert metrics["sessions_before_density"] == 15
    assert metrics["participants_excluded_density"] == 0
    assert metrics["sessions_excluded_density"] == 0
    assert metrics["final_eligible_participants"] == 1
    assert metrics["final_eligible_sessions"] == 15


def test_density_rule_participant_boundary_exact_threshold() -> None:
    """A participant exactly on the density threshold (retained_count == observation_days / 7).

    The screening rule is strictly 'retained_session_count < observation_days / 7'.
    When retained_session_count == observation_days / 7, the participant is NOT excluded.

    Observation period of retained sessions: 70 days (earliest retained StartTime to latest retained StopTime).
    Required sessions: observation_days / 7 = 70 / 7 = 10.0 sessions.
    Retained sessions: exactly 10 sessions.
    Since 10 is not less than 10.0, the participant is retained.
    """
    base = pd.Timestamp("2017-01-01 10:00:00")
    retained_start = pd.Timestamp("2017-01-02 10:00:00")
    retained_dates = [retained_start + pd.Timedelta(days=i * 7) for i in range(10)]
    all_starts = [base] + retained_dates
    all_stops = [base + pd.Timedelta(hours=1)] + [t + pd.Timedelta(hours=1) for t in retained_dates[:-1]] + [retained_start + pd.Timedelta(days=70, hours=1)]
    df = pd.DataFrame({
        "TransactionID": range(1, 12),
        "ParticipantID": "EN_BOUNDARY",
        "StartTime": all_starts,
        "StopTime": all_stops,
        "Trial": 1.0,
    })
    metrics = screen_participant_history(df)
    assert metrics["participants_before_density"] == 1
    assert metrics["sessions_before_density"] == 10
    assert metrics["participants_excluded_density"] == 0
    assert metrics["final_eligible_participants"] == 1
    assert metrics["final_eligible_sessions"] == 10


def test_final_screening_counts_on_real_workbooks() -> None:
    """Validate expected screening counts on the actual Electric Nation transactions workbook:
    - Step 5 (After overlap removal): 299 participants, 60,575 sessions
    - Step 6 (After dropping first sessions): 298 participants, 60,276 sessions (299 dropped)
    - Step 7 (Minimum history filter): 11 participants excluded (56 sessions), leaving 287 participants, 60,220 sessions
    - Step 8 (Observation density filter): 20 participants excluded (726 sessions), leaving 267 participants, 59,494 sessions
    """
    tx_path = Path("data/raw/electric_nation/CrowdCharge Transactions.xlsx")
    if not tx_path.is_file():
        pytest.skip("Raw transactions workbook not available in local workspace")

    df_tx = load_dataset(tx_path, sheet_name="Transaction Data")
    metrics = screen_participant_history(df_tx)

    # Step 5: Overlaps
    assert metrics["participants_after_overlaps"] == 299
    assert metrics["remaining_after_overlaps"] == 60575

    # Step 6: Drop first session
    assert metrics["dropped_first_sessions"] == 299
    assert metrics["participants_after_dropped_first"] == 298
    assert metrics["participants_zero_retained_after_dropped_first"] == 1
    assert metrics["retained_sessions_total"] == 60276

    # Step 7: Minimum history (>=10 retained sessions)
    assert metrics["participants_lt_10_retained"] == 11
    assert metrics["sessions_in_lt_10_retained"] == 56
    assert metrics["participants_before_density"] == 287
    assert metrics["sessions_before_density"] == 60220

    # Step 8: Density filter
    assert metrics["participants_excluded_density"] == 20
    assert metrics["sessions_excluded_density"] == 726
    assert metrics["final_eligible_participants"] == 267
    assert metrics["final_eligible_sessions"] == 59494


def test_audit_safe_with_malformed_and_missing_timestamps() -> None:
    """Verify that the audit handles malformed and missing timestamps safely without crashing,
    and invalid timestamp records do not enter duration or screening calculations.
    """
    df = pd.DataFrame({
        "TransactionID": [1, 2, 3, 4],
        "ParticipantID": ["EN001", "EN001", "EN001", "EN001"],
        "StartTime": ["2017-05-01 10:00:00", "corrupt-date", None, "2017-05-01 14:00:00"],
        "StopTime": ["2017-05-01 12:00:00", "2017-05-01 13:00:00", "2017-05-01 15:00:00", "broken-stop"],
        "Trial": [1.0, 1.0, 1.0, 1.0],
    })
    # Must parse without exception
    start_utc, stop_utc, ts_metrics = parse_utc_timestamps(df, "StartTime", "StopTime")
    assert ts_metrics["start_parse_failures"] == 1
    assert ts_metrics["stop_parse_failures"] == 1
    assert ts_metrics["missing_start"] == 2
    assert ts_metrics["missing_stop"] == 1

    # detect_session_overlaps must complete safely
    overlap_mask, overlap_count = detect_session_overlaps(df, "ParticipantID", "StartTime", "StopTime")
    assert overlap_count == 0

    # screen_participant_history must complete safely without raising an exception
    screening = screen_participant_history(df)
    assert screening["initial_valid_records"] == 1  # Only row 1 has both valid timestamps


def test_schema_validation_clear_errors() -> None:
    """Test schema validation raises descriptive ValueError detailing missing columns and workbook name."""
    valid_tx = pd.DataFrame(columns=REQUIRED_TX_COLUMNS)
    validate_workbook_schema(valid_tx, REQUIRED_TX_COLUMNS, "Transactions")

    # Incomplete transactions schema missing TransactionID and ConsumedkWh
    incomplete_tx = pd.DataFrame(columns=["ChargerID", "ParticipantID", "StartTime"])
    with pytest.raises(ValueError) as excinfo:
        validate_workbook_schema(incomplete_tx, REQUIRED_TX_COLUMNS, "Transactions Workbook")

    err_msg = str(excinfo.value)
    assert "Transactions Workbook" in err_msg
    assert "Missing required column(s)" in err_msg
    assert "TransactionID" in err_msg
    assert "ConsumedkWh" in err_msg

    # Test metadata schema validation missing CarkW and CarkWh
    incomplete_meta = pd.DataFrame(columns=["ParticipantID", "Charger"])
    with pytest.raises(ValueError) as excinfo_meta:
        validate_workbook_schema(incomplete_meta, REQUIRED_META_COLUMNS, "ChargerInstall.xlsx")
    assert "ChargerInstall.xlsx" in str(excinfo_meta.value)
    assert "CarkW" in str(excinfo_meta.value)
    assert "CarkWh" in str(excinfo_meta.value)


def test_report_values_change_dynamically_with_metric_changes() -> None:
    """Test that generate_markdown_report produces dynamically updated text matching input metrics."""
    base_tx_meta = {"size_bytes": 10000000, "sheet_dimensions": {"Transaction Data": {"max_row": 70001, "max_column": 28}, "Explanation": {"max_row": 1, "max_column": 1}}}
    base_meta_meta = {"size_bytes": 50000, "sheet_dimensions": {"20181126_ChargerInstall_DriveEl": {"max_row": 601, "max_column": 10}, "Explanation": {"max_row": 1, "max_column": 1}}}
    ts_metrics = {
        "start_col": "StartTime", "stop_col": "StopTime", "timezone": "UTC",
        "earliest_start": "2017-03-01 10:00:00+00:00", "latest_start": "2018-12-31 10:00:00+00:00",
        "earliest_stop": "2017-03-01 12:00:00+00:00", "latest_stop": "2019-01-01 10:00:00+00:00",
        "start_parse_failures": 0, "stop_parse_failures": 0, "missing_start": 0, "missing_stop": 0,
        "negative_intervals": 0, "zero_intervals": 92, "negative_intervals_pct": 0.0, "zero_intervals_pct": 0.13,
    }
    unattr_metrics = {
        "total_unattributed_records": 3899, "unattributed_percentage": 5.47,
        "chargers_involved": 22, "known_chargers_count": 21, "unknown_chargers_count": 1,
        "unknown_charger_ids": ["4441003917"], "unknown_charger_records": 125,
        "matched_4field_session_records": 3774, "unmatched_session_records": 125,
    }
    row_meta_metrics = {
        "unique_tx_participants": 300, "unique_meta_participants": 673, "matched_participants": 300,
        "tx_only_participants": 0, "meta_only_participants": 373, "total_matched_transaction_rows": 67365,
        "row_level_carkw_mismatches": 0, "row_level_carkw_mismatch_pct": 0.0,
        "row_level_carkwh_mismatches": 0, "row_level_carkwh_mismatch_pct": 0.0,
        "row_level_charger_matches": 67365, "row_level_charger_mismatches": 0, "row_level_charger_mismatch_pct": 0.0,
        "participants_with_inconsistent_carkw": 0, "participants_with_inconsistent_carkwh": 0,
    }
    energy_metrics = {
        "total_records": 71264, "missing_energy": 1, "missing_energy_pct": 0.0014,
        "non_positive_energy": 0, "non_positive_energy_pct": 0.0,
        "energy_gt_100kwh": 20, "energy_gt_100kwh_pct": 0.028,
        "energy_gt_150kwh": 12, "energy_gt_150kwh_pct": 0.0168,
        "energy_exceeds_carkwh": 1788, "energy_exceeds_carkwh_pct": 2.65,
        "max_energy": 37487.88,
    }
    ch_discrepancy = {"count": 1, "ids": [132725]}

    screening_a = {
        "initial_valid_records": 67365, "initial_participants": 300, "trial_3_excluded": 6123,
        "remaining_after_t3": 61242, "participants_after_t3": 300, "short_sessions_excluded": 509,
        "remaining_after_short": 60733, "participants_after_short": 299, "screened_overlaps": 158,
        "remaining_after_overlaps": 60575, "participants_after_overlaps": 299, "dropped_first_sessions": 299,
        "retained_sessions_total": 60276, "participants_after_dropped_first": 298,
        "participants_zero_retained_after_dropped_first": 1,
        "participants_lt_10_retained": 11, "sessions_in_lt_10_retained": 56,
        "participants_before_density": 287, "sessions_before_density": 60220,
        "participants_excluded_density": 20, "sessions_excluded_density": 726,
        "final_eligible_participants": 267, "final_eligible_sessions": 59494,
        "density_mean_span_days": 374.88, "density_mean_sessions_per_week": 4.20,
        "density_median_sessions_per_week": 3.80, "density_mean_avg_interval_days": 2.44,
    }

    report_a = generate_markdown_report(
        base_tx_meta, base_meta_meta, ts_metrics, unattr_metrics, row_meta_metrics,
        energy_metrics, overlap_count=160, simple_overlap_count=133,
        ch_discrepancy_info=ch_discrepancy, screening_metrics=screening_a, total_tx=71264,
    )
    assert "267 participants" in report_a
    assert "59,494 sessions" in report_a
    assert "160 sessions" in report_a
    assert "1 participant(s) had only a single session, leaving 0 retained sessions" in report_a

    # Metric set B with altered participant count and overlaps
    screening_b = dict(screening_a)
    screening_b["final_eligible_participants"] = 240
    screening_b["final_eligible_sessions"] = 51234
    screening_b["participants_zero_retained_after_dropped_first"] = 5
    report_b = generate_markdown_report(
        base_tx_meta, base_meta_meta, ts_metrics, unattr_metrics, row_meta_metrics,
        energy_metrics, overlap_count=95, simple_overlap_count=80,
        ch_discrepancy_info=ch_discrepancy, screening_metrics=screening_b, total_tx=71264,
    )
    assert "240 participants" in report_b
    assert "51,234 sessions" in report_b
    assert "95 sessions" in report_b
    assert "5 participant(s) had only a single session, leaving 0 retained sessions" in report_b
    assert report_a != report_b


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


def test_screen_participant_history_zero_retained_count() -> None:
    """Verify that participants_zero_retained_after_dropped_first dynamically counts participants
    who have no remaining sessions after dropping their earliest session.
    """
    base = pd.Timestamp("2017-05-01 10:00:00")
    df = pd.DataFrame({
        "TransactionID": [1, 2, 3, 4],
        "ParticipantID": ["P1", "P2", "P2", "P3"],
        "StartTime": [base, base, base + pd.Timedelta(hours=5), base],
        "StopTime": [base + pd.Timedelta(hours=2), base + pd.Timedelta(hours=2), base + pd.Timedelta(hours=7), base + pd.Timedelta(hours=2)],
        "Trial": [1.0, 1.0, 1.0, 1.0],
    })
    metrics = screen_participant_history(df)
    assert metrics["participants_after_overlaps"] == 3
    assert metrics["participants_after_dropped_first"] == 1  # Only P2 remains
    assert metrics["participants_zero_retained_after_dropped_first"] == 2  # P1 and P3 have 0 retained


def test_timestamp_parse_success_when_missing_or_malformed() -> None:
    """Verify parsing success percentage over non-missing raw timestamps,
    and verify display of 'N/A' when no non-missing timestamps exist.
    """
    df = pd.DataFrame({
        "TransactionID": [1, 2, 3, 4],
        "StartTime": ["2017-05-01 10:00:00", "2017-05-02 10:00:00", "corrupt-ts", None],
        "StopTime": ["2017-05-01 12:00:00", "2017-05-02 12:00:00", "2017-05-03 12:00:00", "2017-05-04 12:00:00"],
    })
    _, _, ts_metrics = parse_utc_timestamps(df, "StartTime", "StopTime")
    dummy_meta = {"size_bytes": 100, "sheet_dimensions": {}}
    dummy_unattr = {"total_unattributed_records": 0, "unattributed_percentage": 0.0, "chargers_involved": 0, "known_chargers_count": 0, "unknown_chargers_count": 0, "unknown_charger_ids": [], "unknown_charger_records": 0, "matched_4field_session_records": 0}
    dummy_row_meta = {"unique_tx_participants": 1, "unique_meta_participants": 1, "matched_participants": 1, "tx_only_participants": 0, "meta_only_participants": 0, "total_matched_transaction_rows": 4, "row_level_carkw_mismatches": 0, "row_level_carkw_mismatch_pct": 0.0, "row_level_carkwh_mismatches": 0, "row_level_carkwh_mismatch_pct": 0.0, "row_level_charger_matches": 4, "row_level_charger_mismatches": 0, "row_level_charger_mismatch_pct": 0.0, "participants_with_inconsistent_carkw": 0, "participants_with_inconsistent_carkwh": 0, "participants_with_inconsistent_specs": 0}
    dummy_energy = {"missing_energy": 0, "missing_energy_pct": 0.0, "non_positive_energy": 0, "non_positive_energy_pct": 0.0, "energy_gt_100kwh": 0, "energy_gt_100kwh_pct": 0.0, "energy_gt_150kwh": 0, "energy_gt_150kwh_pct": 0.0, "energy_exceeds_carkwh": 0, "energy_exceeds_carkwh_pct": 0.0, "max_energy": 10.0}
    dummy_screening = {"initial_participants": 1, "initial_valid_records": 4, "participants_after_t3": 1, "remaining_after_t3": 4, "participants_after_short": 1, "remaining_after_short": 4, "participants_after_overlaps": 1, "remaining_after_overlaps": 4, "dropped_first_sessions": 1, "participants_after_dropped_first": 1, "participants_zero_retained_after_dropped_first": 0, "retained_sessions_total": 3, "participants_lt_10_retained": 0, "sessions_in_lt_10_retained": 0, "participants_before_density": 1, "sessions_before_density": 3, "participants_excluded_density": 0, "sessions_excluded_density": 0, "final_eligible_participants": 1, "final_eligible_sessions": 3, "density_mean_span_days": 10.0, "density_mean_sessions_per_week": 2.0, "density_median_sessions_per_week": 2.0, "density_mean_avg_interval_days": 3.0}

    report = generate_markdown_report(
        dummy_meta, dummy_meta, ts_metrics, dummy_unattr, dummy_row_meta,
        dummy_energy, overlap_count=0, simple_overlap_count=0,
        ch_discrepancy_info={"count": 0, "ids": []}, screening_metrics=dummy_screening, total_tx=4,
    )
    assert "85.7%" in report

    # Test display of 'N/A' when 0 non-missing timestamps exist
    all_missing_df = pd.DataFrame({"StartTime": [None, None], "StopTime": [None, None]})
    _, _, all_missing_ts = parse_utc_timestamps(all_missing_df, "StartTime", "StopTime")
    report_na = generate_markdown_report(
        dummy_meta, dummy_meta, all_missing_ts, dummy_unattr, dummy_row_meta,
        dummy_energy, overlap_count=0, simple_overlap_count=0,
        ch_discrepancy_info={"count": 0, "ids": []}, screening_metrics=dummy_screening, total_tx=2,
    )
    assert "N/A" in report_na


def test_dynamic_energy_quality_report_wording() -> None:
    """Verify that report text adapts dynamically to positive, non-positive, and missing energy counts."""
    dummy_meta = {"size_bytes": 100, "sheet_dimensions": {}}
    dummy_ts = {"start_col": "StartTime", "stop_col": "StopTime", "start_parse_failures": 0, "stop_parse_failures": 0, "missing_start": 0, "missing_stop": 0, "negative_intervals": 0, "negative_intervals_pct": 0.0, "zero_intervals": 0, "zero_intervals_pct": 0.0, "earliest_start": "2017-01-01", "latest_start": "2017-01-02", "earliest_stop": "2017-01-01", "latest_stop": "2017-01-02"}
    dummy_unattr = {"total_unattributed_records": 0, "unattributed_percentage": 0.0, "chargers_involved": 0, "known_chargers_count": 0, "unknown_chargers_count": 0, "unknown_charger_ids": [], "unknown_charger_records": 0, "matched_4field_session_records": 0}
    dummy_row_meta = {"unique_tx_participants": 1, "unique_meta_participants": 1, "matched_participants": 1, "tx_only_participants": 0, "meta_only_participants": 0, "total_matched_transaction_rows": 1, "row_level_carkw_mismatches": 0, "row_level_carkw_mismatch_pct": 0.0, "row_level_carkwh_mismatches": 0, "row_level_carkwh_mismatch_pct": 0.0, "row_level_charger_matches": 1, "row_level_charger_mismatches": 0, "row_level_charger_mismatch_pct": 0.0, "participants_with_inconsistent_carkw": 0, "participants_with_inconsistent_carkwh": 0, "participants_with_inconsistent_specs": 0}
    dummy_screening = {"initial_participants": 1, "initial_valid_records": 1, "participants_after_t3": 1, "remaining_after_t3": 1, "participants_after_short": 1, "remaining_after_short": 1, "participants_after_overlaps": 1, "remaining_after_overlaps": 1, "dropped_first_sessions": 1, "participants_after_dropped_first": 1, "participants_zero_retained_after_dropped_first": 0, "retained_sessions_total": 1, "participants_lt_10_retained": 0, "sessions_in_lt_10_retained": 0, "participants_before_density": 1, "sessions_before_density": 1, "participants_excluded_density": 0, "sessions_excluded_density": 0, "final_eligible_participants": 1, "final_eligible_sessions": 1, "density_mean_span_days": 10.0, "density_mean_sessions_per_week": 2.0, "density_median_sessions_per_week": 2.0, "density_mean_avg_interval_days": 3.0}

    # Case A: non_positive == 0, min_positive == 0.05
    energy_a = {"missing_energy": 0, "missing_energy_pct": 0.0, "non_positive_energy": 0, "non_positive_energy_pct": 0.0, "energy_gt_100kwh": 0, "energy_gt_100kwh_pct": 0.0, "energy_gt_150kwh": 0, "energy_gt_150kwh_pct": 0.0, "energy_exceeds_carkwh": 0, "energy_exceeds_carkwh_pct": 0.0, "max_energy": 50.0, "min_positive_energy": 0.05}
    rep_a = generate_markdown_report(dummy_meta, dummy_meta, dummy_ts, dummy_unattr, dummy_row_meta, energy_a, 0, 0, {"count": 0, "ids": []}, dummy_screening, total_tx=1)
    assert "No non-positive readings found (all recorded values are positive, min 0.05 kWh)" in rep_a
    assert "No missing energy readings found (all records contain metered energy)" in rep_a

    # Case B: non_positive == 4, missing_energy == 2
    energy_b = dict(energy_a)
    energy_b["non_positive_energy"] = 4
    energy_b["missing_energy"] = 2
    rep_b = generate_markdown_report(dummy_meta, dummy_meta, dummy_ts, dummy_unattr, dummy_row_meta, energy_b, 0, 0, {"count": 0, "ids": []}, dummy_screening, total_tx=1)
    assert "4 non-positive energy reading(s) detected (<= 0 kWh)" in rep_b
    assert "Confirmed invalid (2 records lack meter reading)" in rep_b

    # Case C: missing_energy == 1
    energy_c = dict(energy_a)
    energy_c["missing_energy"] = 1
    rep_c = generate_markdown_report(dummy_meta, dummy_meta, dummy_ts, dummy_unattr, dummy_row_meta, energy_c, 0, 0, {"count": 0, "ids": []}, dummy_screening, total_tx=1)
    assert "Confirmed invalid (1 record lacks meter reading)" in rep_c


def test_combined_vehicle_specification_inconsistency() -> None:
    """Verify that participants_with_inconsistent_specs correctly unions participants
    with fluctuating CarKW and CarKWh, without double-counting participants with both.
    """
    df_tx = pd.DataFrame({
        "TransactionID": [1, 2, 3, 4, 5, 6, 7, 8],
        "ParticipantID": ["P1", "P1", "P2", "P2", "P3", "P3", "P4", "P4"],
        "CarKW": [3.6, 7.0, 7.0, 7.0, 3.6, 7.0, 7.0, 7.0],  # P1 and P3 change
        "CarKWh": [33.0, 33.0, 40.0, 60.0, 30.0, 40.0, 50.0, 50.0],  # P2 and P3 change
        "ChargerID": ["C1"] * 8,
    })
    df_meta = pd.DataFrame({
        "ParticipantID": ["P1", "P2", "P3", "P4"],
        "CarkW": [7.0, 7.0, 7.0, 7.0],
        "CarkWh": [33.0, 60.0, 40.0, 50.0],
        "Charger": ["C1"] * 4,
        "DCSProvider": ["Crowd Charge"] * 4,
    })
    metrics = audit_metadata_row_level(df_tx, df_meta)
    assert metrics["participants_with_inconsistent_carkw"] == 2  # P1, P3
    assert metrics["participants_with_inconsistent_carkwh"] == 2  # P2, P3
    assert metrics["participants_with_inconsistent_specs"] == 3  # P1, P2, P3 (P3 counted once)

    reconcile_df = generate_participant_reconciliation(
        metrics, df_meta, {"matched_4field_session_records": 0, "unknown_charger_records": 0}, total_tx=8
    )
    row = reconcile_df[reconcile_df["metric"] == "Participants with Inconsistent Specifications Across Sessions"].iloc[0]
    assert row["count"] == 3
    assert row["percentage"] == 75.0  # 3 out of 4 matched participants


def test_empty_dataframe_quality_summary_percentages() -> None:
    """Verify that generate_transaction_quality_summary and generate_metadata_quality_summary
    return 0.0% for total records when given an empty DataFrame.
    """
    empty_tx = pd.DataFrame(columns=["TransactionID", "ParticipantID", "StartTime", "StopTime", "ConsumedkWh", "CarKW", "CarKWh"])
    tx_summary = generate_transaction_quality_summary(
        empty_tx,
        ts_metrics={"negative_intervals": 0, "zero_intervals": 0, "negative_intervals_pct": 0.0, "zero_intervals_pct": 0.0},
        unattributed_metrics={"total_unattributed_records": 0, "unattributed_percentage": 0.0, "matched_4field_session_records": 0, "unknown_charger_records": 0},
        energy_metrics={"missing_energy": 0, "missing_energy_pct": 0.0, "non_positive_energy": 0, "non_positive_energy_pct": 0.0, "energy_gt_100kwh": 0, "energy_gt_100kwh_pct": 0.0, "energy_gt_150kwh": 0, "energy_gt_150kwh_pct": 0.0, "energy_exceeds_carkwh": 0, "energy_exceeds_carkwh_pct": 0.0, "max_energy": 0.0},
        overlap_count=0,
        ch_discrepancy_info={"count": 0, "ids": []},
    )
    assert tx_summary.iloc[0]["record_count"] == 0
    assert tx_summary.iloc[0]["percentage"] == 0.0

    empty_meta = pd.DataFrame(columns=["ParticipantID", "Charger", "DCSProvider", "CarKW", "CarKWh"])
    meta_summary = generate_metadata_quality_summary(empty_meta)
    assert meta_summary.iloc[0]["record_count"] == 0
    assert meta_summary.iloc[0]["percentage"] == 0.0
