"""Unit tests for the TARANG preprocessing and modeling dataset pipeline.

Tests all 16 required test cases using fast, deterministic, self-contained
synthetic DataFrames without reading real Excel files during unit testing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from tarang.data.audit import screen_participant_history, screen_participant_sessions
from tarang.data.preprocessing import (
    MODEL_METADATA_COLUMNS,
    MODEL_READY_COLUMNS,
    PROHIBITED_FEATURE_COLUMNS,
    TARGET_COLUMNS,
    WARM_START_FEATURES,
    build_screened_and_model_ready_datasets,
    engineer_features,
    prepare_targets,
    save_processed_datasets,
    validate_input_datasets,
)


def make_valid_meta_and_tx(
    n_sessions: int = 15,
    participant_id: str = "EN001",
    start_date: str = "2017-05-01 10:00:00",
    spacing_hours: float = 24.0,
    session_duration_hours: float = 2.0,
    car_kw: float = 7.0,
    car_kwh: float = 30.0,
    energy_kwh: float = 12.0,
    trial: float = 1.0,
    start_tx_id: int = 1001,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Helper to create matching synthetic transaction and metadata DataFrames."""
    base_time = pd.Timestamp(start_date)
    start_times = [base_time + pd.Timedelta(hours=i * spacing_hours) for i in range(n_sessions)]
    stop_times = [t + pd.Timedelta(hours=session_duration_hours) for t in start_times]

    df_tx = pd.DataFrame({
        "TransactionID": list(range(start_tx_id, start_tx_id + n_sessions)),
        "ParticipantID": [participant_id] * n_sessions,
        "StartTime": start_times,
        "StopTime": stop_times,
        "ConsumedkWh": [energy_kwh] * n_sessions,
        "CarKW": [car_kw] * n_sessions,
        "CarKWh": [car_kwh] * n_sessions,
        "Trial": [trial] * n_sessions,
        "ChargerID": [101] * n_sessions,
    })

    df_meta = pd.DataFrame({
        "ParticipantID": [participant_id],
        "Charger": ["101"],
        "DCSProvider": ["Provider A"],
        "CarMake": ["Nissan"],
        "CarModel": ["Leaf"],
        "CarkW": [car_kw],
        "CarkWh": [car_kwh],
    })

    return df_tx, df_meta


def test_01_shared_screening_returns_both_and_preserves_metrics() -> None:
    """1. Shared screening returns screened_sessions and final eligible rows while preserving all audit metrics."""
    # Participant with 12 sessions spaced over 20 days (density: 12 sessions in 20 days passes)
    df_tx, _ = make_valid_meta_and_tx(n_sessions=12, spacing_hours=36.0)

    # Call shared screening function
    df_screened, df_final, screening_metrics = screen_participant_sessions(df_tx)
    # Call original audit function
    audit_metrics = screen_participant_history(df_tx)

    # Screening metrics must be bit-exact identical to audit metrics
    assert screening_metrics == audit_metrics

    # Screened sessions contains all 12 sessions (post-overlap, no first session drop)
    assert len(df_screened) == 12
    # Final eligible contains 11 sessions (first session dropped, 11 >= 10 retained, density passes)
    assert len(df_final) == 11
    assert screening_metrics["final_eligible_sessions"] == 11
    assert screening_metrics["final_eligible_participants"] == 1


def test_02_nested_overlap_removal_retains_earlier_session() -> None:
    """2. Nested overlap removal correctly retains earlier session and discards nested session at boundaries."""
    # Session 1: 10:00 to 18:00 (Long session)
    # Session 2: 11:00 to 12:00 (Nested inside Session 1) -> dropped
    # Session 3: 13:00 to 15:00 (Nested inside Session 1) -> dropped
    # Session 4: 18:00 to 20:00 (Starts exactly when Session 1 stops -> retained)
    df_tx = pd.DataFrame({
        "TransactionID": [1, 2, 3, 4],
        "ParticipantID": ["EN001"] * 4,
        "StartTime": [
            "2017-05-01 10:00:00",
            "2017-05-01 11:00:00",
            "2017-05-01 13:00:00",
            "2017-05-01 18:00:00",
        ],
        "StopTime": [
            "2017-05-01 18:00:00",
            "2017-05-01 12:00:00",
            "2017-05-01 15:00:00",
            "2017-05-01 20:00:00",
        ],
        "ConsumedkWh": [15.0, 5.0, 6.0, 8.0],
        "CarKW": [7.0] * 4,
        "CarKWh": [30.0] * 4,
        "Trial": [1.0] * 4,
        "ChargerID": [101] * 4,
    })

    df_screened, _, metrics = screen_participant_sessions(df_tx)
    assert metrics["screened_overlaps"] == 2
    assert len(df_screened) == 2
    assert list(df_screened["TransactionID"]) == [1, 4]


def test_03_first_sessions_in_screened_but_dropped_from_model_ready() -> None:
    """3. First sessions are present in intermediate screened_sessions but dropped from warm_start model_ready."""
    df_tx, df_meta = make_valid_meta_and_tx(n_sessions=15, spacing_hours=24.0)

    df_screened, df_model_ready, summary = build_screened_and_model_ready_datasets(df_tx, df_meta)

    # First session (TransactionID 1001) should be in screened but not model-ready
    assert 1001 in df_screened["TransactionID"].values
    assert 1001 not in df_model_ready["TransactionID"].values

    assert len(df_screened) == 15
    assert len(df_model_ready) == 14
    assert summary["first_sessions_excluded_warm_start"] == 1


def test_04_minimum_history_and_density_rule_sequence() -> None:
    """4. Minimum history and observation density rules are applied in the correct sequence."""
    # Participant A: 8 retained sessions (9 total). Excluded at minimum history stage (< 10).
    tx_a, _ = make_valid_meta_and_tx(n_sessions=9, participant_id="EN_FEW", spacing_hours=24.0)

    # Participant B: 10 retained sessions (11 total) across 80 days.
    # Passes minimum history (10 >= 10), but fails density filter (10 < 80/7 = 11.4).
    base = pd.Timestamp("2017-01-01 10:00:00")
    times_b = [base] + [base + pd.Timedelta(days=1 + i * 8) for i in range(10)]
    stops_b = [t + pd.Timedelta(hours=2) for t in times_b[:-1]] + [times_b[0] + pd.Timedelta(days=80)]
    tx_b = pd.DataFrame({
        "TransactionID": range(2001, 2012),
        "ParticipantID": ["EN_SPARSE"] * 11,
        "StartTime": times_b,
        "StopTime": stops_b,
        "ConsumedkWh": [10.0] * 11,
        "CarKW": [7.0] * 11,
        "CarKWh": [30.0] * 11,
        "Trial": [1.0] * 11,
        "ChargerID": [102] * 11,
    })

    # Participant C: 12 retained sessions across 20 days. Passes both.
    tx_c, _ = make_valid_meta_and_tx(n_sessions=13, participant_id="EN_QUALIFIED", spacing_hours=24.0, start_tx_id=3001)

    df_tx = pd.concat([tx_a, tx_b, tx_c], ignore_index=True)
    df_meta = pd.DataFrame({
        "ParticipantID": ["EN_FEW", "EN_SPARSE", "EN_QUALIFIED"],
        "Charger": ["101", "102", "103"],
        "DCSProvider": ["Provider A", "Provider B", "Provider C"],
        "CarMake": ["Nissan", "BMW", "Tesla"],
        "CarModel": ["Leaf", "i3", "Model S"],
        "CarkW": [7.0, 7.0, 7.0],
        "CarkWh": [30.0, 30.0, 30.0],
    })

    df_screened, df_model_ready, summary = build_screened_and_model_ready_datasets(df_tx, df_meta)

    # Screening metrics must show exact stages
    assert summary["participants_after_first_session_drop"] == 3
    assert summary["participants_passing_minimum_history"] == 2  # EN_SPARSE and EN_QUALIFIED
    assert summary["participants_passing_observation_density"] == 1  # EN_QUALIFIED only
    assert summary["final_model_ready_participants"] == 1

    assert set(df_model_ready["ParticipantID"].unique()) == {"EN_QUALIFIED"}
    assert len(df_model_ready) == 12


def test_05_duration_capped_at_336h_original_stoptime_unchanged() -> None:
    """5. Duration is capped at 336.0 hours for target_duration_hours, while original StopTime remains uncapped."""
    start = pd.Timestamp("2017-01-01 00:00:00", tz="UTC")
    stop = pd.Timestamp("2017-01-17 16:00:00", tz="UTC")  # 400.0 hours
    df = pd.DataFrame({
        "TransactionID": [1],
        "ParticipantID": ["EN001"],
        "StartTime": [start],
        "StopTime": [stop],
        "ConsumedkWh": [50.0],
    })

    res = prepare_targets(df)
    assert res.iloc[0]["duration_hours_raw"] == 400.0
    assert res.iloc[0]["target_duration_hours"] == 336.0
    assert res.iloc[0]["StopTime"] == stop  # Original StopTime uncapped and untouched


def test_06_time_since_last_stop_uses_uncapped_stoptime() -> None:
    """6. TimeSinceLastStop accurately calculates hours from previous session's original uncapped StopTime."""
    # Session 1: 400 hours duration (capped to 336 for target, but original stop at Jan 17 16:00)
    # Session 2: Starts Jan 18 16:00 (24 hours after Session 1 uncapped StopTime)
    s1_start = pd.Timestamp("2017-01-01 00:00:00", tz="UTC")
    s1_stop = pd.Timestamp("2017-01-17 16:00:00", tz="UTC")
    s2_start = pd.Timestamp("2017-01-18 16:00:00", tz="UTC")
    s2_stop = pd.Timestamp("2017-01-18 20:00:00", tz="UTC")

    df = pd.DataFrame({
        "TransactionID": [1, 2],
        "ParticipantID": ["EN001", "EN001"],
        "StartTime": [s1_start, s2_start],
        "StopTime": [s1_stop, s2_stop],
        "ConsumedkWh": [25.0, 20.0],
        "CarKW": [7.0, 7.0],
        "CarKWh": [30.0, 30.0],
    })

    df_targets = prepare_targets(df)
    df_features = engineer_features(df_targets)

    # Session 1 has no previous stop -> NaN
    assert np.isnan(df_features.iloc[0]["TimeSinceLastStop"])
    # Session 2 should measure exactly 24.0 hours from uncapped StopTime
    assert df_features.iloc[1]["TimeSinceLastStop"] == 24.0


def test_07_last_duration_uses_previous_capped_duration() -> None:
    """7. LastDuration uses previous session's capped target duration."""
    # Session 1: 400 hours raw duration -> capped to 336.0
    # Session 2: LastDuration should be 336.0
    s1_start = pd.Timestamp("2017-01-01 00:00:00", tz="UTC")
    s1_stop = pd.Timestamp("2017-01-17 16:00:00", tz="UTC")
    s2_start = pd.Timestamp("2017-01-18 16:00:00", tz="UTC")
    s2_stop = pd.Timestamp("2017-01-18 18:00:00", tz="UTC")

    df = pd.DataFrame({
        "TransactionID": [1, 2],
        "ParticipantID": ["EN001", "EN001"],
        "StartTime": [s1_start, s2_start],
        "StopTime": [s1_stop, s2_stop],
        "ConsumedkWh": [25.0, 20.0],
        "CarKW": [7.0, 7.0],
        "CarKWh": [30.0, 30.0],
    })

    df_targets = prepare_targets(df)
    df_features = engineer_features(df_targets)

    assert np.isnan(df_features.iloc[0]["LastDuration"])
    assert df_features.iloc[1]["LastDuration"] == 336.0


def test_08_sessions_today_counts_strictly_prior_same_day() -> None:
    """8. sessionsToday accurately counts only strictly prior same-day sessions for the participant."""
    times = [
        pd.Timestamp("2017-05-01 08:00:00", tz="UTC"),
        pd.Timestamp("2017-05-01 12:00:00", tz="UTC"),
        pd.Timestamp("2017-05-01 18:00:00", tz="UTC"),
        pd.Timestamp("2017-05-02 09:00:00", tz="UTC"),
    ]
    stops = [t + pd.Timedelta(hours=1) for t in times]

    df = pd.DataFrame({
        "TransactionID": [1, 2, 3, 4],
        "ParticipantID": ["EN001"] * 4,
        "StartTime": times,
        "StopTime": stops,
        "ConsumedkWh": [5.0] * 4,
        "CarKW": [7.0] * 4,
        "CarKWh": [30.0] * 4,
    })

    df_targets = prepare_targets(df)
    df_features = engineer_features(df_targets)

    # sessionsToday should be 0, 1, 2 for May 1st, and reset to 0 on May 2nd
    assert list(df_features["sessionsToday"]) == [0, 1, 2, 0]


def test_09_calendar_and_cyclical_features_contract() -> None:
    """9. Calendar and daily cyclical features strictly follow data contract formulas."""
    # 1. Monday 2017-05-01 00:00:00 UTC (UK Early May Bank Holiday)
    # 2. Saturday 2017-06-10 12:00:00 UTC
    # 3. Wednesday 2017-08-02 06:00:00 UTC
    times = [
        pd.Timestamp("2017-05-01 00:00:00", tz="UTC"),
        pd.Timestamp("2017-06-10 12:00:00", tz="UTC"),
        pd.Timestamp("2017-08-02 06:00:00", tz="UTC"),
    ]
    stops = [t + pd.Timedelta(hours=2) for t in times]

    df = pd.DataFrame({
        "TransactionID": [1, 2, 3],
        "ParticipantID": ["EN001"] * 3,
        "StartTime": times,
        "StopTime": stops,
        "ConsumedkWh": [10.0] * 3,
        "CarKW": [7.0] * 3,
        "CarKWh": [30.0] * 3,
    })

    df_targets = prepare_targets(df)
    df_feat = engineer_features(df_targets)

    # Session 1: Monday, bank holiday, midnight
    s1 = df_feat.iloc[0]
    assert s1["weekday"] == 0
    assert s1["weekend"] == 0
    assert s1["holiday"] == 1
    assert s1["month"] == 5
    assert s1["StartHourNorm"] == pytest.approx(0.0)
    assert s1["StartCos"] == pytest.approx(1.0)
    assert s1["StartSin"] == pytest.approx(0.0)

    # Session 2: Saturday, noon
    s2 = df_feat.iloc[1]
    assert s2["weekday"] == 5
    assert s2["weekend"] == 1
    assert s2["holiday"] == 0
    assert s2["month"] == 6
    assert s2["StartHourNorm"] == pytest.approx(0.5)
    assert s2["StartCos"] == pytest.approx(-1.0)
    assert s2["StartSin"] == pytest.approx(0.0, abs=1e-12)

    # Session 3: Wednesday, 6 AM
    s3 = df_feat.iloc[2]
    assert s3["weekday"] == 2
    assert s3["weekend"] == 0
    assert s3["holiday"] == 0
    assert s3["month"] == 8
    assert s3["StartHourNorm"] == pytest.approx(0.25)
    assert s3["StartCos"] == pytest.approx(0.0, abs=1e-12)
    assert s3["StartSin"] == pytest.approx(1.0)


def test_10_energy_gt_100kwh_available_in_lag_feature() -> None:
    """10. Sessions with energy > 100 kWh remain available in lag feature LastConsumedkWh."""
    times = [
        pd.Timestamp("2017-05-01 10:00:00", tz="UTC"),
        pd.Timestamp("2017-05-02 10:00:00", tz="UTC"),
    ]
    stops = [t + pd.Timedelta(hours=4) for t in times]

    df = pd.DataFrame({
        "TransactionID": [1, 2],
        "ParticipantID": ["EN001", "EN001"],
        "StartTime": times,
        "StopTime": stops,
        "ConsumedkWh": [125.5, 15.0],
        "CarKW": [7.0, 7.0],
        "CarKWh": [30.0, 30.0],
    })

    df_targets = prepare_targets(df)
    df_feat = engineer_features(df_targets)

    # Session 2 should have LastConsumedkWh == 125.5
    assert df_feat.iloc[1]["LastConsumedkWh"] == 125.5


def test_11_missing_or_non_positive_energy_yields_nan_without_skipping() -> None:
    """11. Missing/non-positive previous energy yields NaN without skipping backwards to older sessions."""
    times = [
        pd.Timestamp("2017-05-01 10:00:00", tz="UTC"),
        pd.Timestamp("2017-05-02 10:00:00", tz="UTC"),
        pd.Timestamp("2017-05-03 10:00:00", tz="UTC"),
    ]
    stops = [t + pd.Timedelta(hours=2) for t in times]

    # Session 1: valid 20.0 kWh
    # Session 2: invalid 0.0 kWh (or missing)
    # Session 3: valid 15.0 kWh
    df = pd.DataFrame({
        "TransactionID": [1, 2, 3],
        "ParticipantID": ["EN001", "EN001", "EN001"],
        "StartTime": times,
        "StopTime": stops,
        "ConsumedkWh": [20.0, 0.0, 15.0],
        "CarKW": [7.0, 7.0, 7.0],
        "CarKWh": [30.0, 30.0, 30.0],
    })

    df_targets = prepare_targets(df)
    df_feat = engineer_features(df_targets)

    assert np.isnan(df_feat.iloc[0]["LastConsumedkWh"])
    assert df_feat.iloc[1]["LastConsumedkWh"] == 20.0
    # Session 3 must be NaN because Session 2 was non-positive; it must NOT skip backwards to 20.0!
    assert np.isnan(df_feat.iloc[2]["LastConsumedkWh"])


def test_12_leakage_future_modifications_do_not_alter_earlier_features() -> None:
    """12. Leakage test: modifying or corrupting future session data does not alter any earlier feature."""
    df_tx1, _ = make_valid_meta_and_tx(n_sessions=6, spacing_hours=24.0)

    # Modify future sessions (sessions 4 and 5) in df_tx2
    df_tx2 = df_tx1.copy()
    df_tx2.loc[4, "ConsumedkWh"] = 999.0
    df_tx2.loc[4, "StopTime"] = df_tx2.loc[4, "StartTime"] + pd.Timedelta(hours=50)
    df_tx2.loc[5, "ConsumedkWh"] = 0.001

    df_feat1 = engineer_features(prepare_targets(df_tx1))
    df_feat2 = engineer_features(prepare_targets(df_tx2))

    # Compare features for sessions 0..3 (the past sessions)
    for col in WARM_START_FEATURES:
        pd.testing.assert_series_equal(
            df_feat1.loc[0:3, col],
            df_feat2.loc[0:3, col],
            check_names=False,
            check_exact=True,
        )


def test_13_model_ready_allowlist_and_no_prohibited_columns() -> None:
    """13. Model-ready dataset output strictly adheres to explicit allowlist and contains no prohibited columns."""
    df_tx, df_meta = make_valid_meta_and_tx(n_sessions=15, spacing_hours=24.0)

    _, df_model_ready, _ = build_screened_and_model_ready_datasets(df_tx, df_meta)

    # Verify exact column list and order
    assert list(df_model_ready.columns) == MODEL_READY_COLUMNS
    assert len(df_model_ready.columns) == 18

    # Ensure none of the prohibited feature columns exist in WARM_START_FEATURES
    assert len(set(WARM_START_FEATURES).intersection(PROHIBITED_FEATURE_COLUMNS)) == 0

    # Ensure prohibited source columns are not in df_model_ready (except allowable metadata & targets)
    prohibited_disallowed = PROHIBITED_FEATURE_COLUMNS - set(MODEL_METADATA_COLUMNS) - set(TARGET_COLUMNS)
    for col in prohibited_disallowed:
        assert col not in df_model_ready.columns


def test_14_energy_gt_100kwh_in_baseline_target_flagged() -> None:
    """14. Energy > 100 kWh records remain included in baseline target dataset and flagged in intermediate data."""
    df_tx, df_meta = make_valid_meta_and_tx(n_sessions=15, spacing_hours=24.0, energy_kwh=120.0)

    df_screened, df_model_ready, summary = build_screened_and_model_ready_datasets(df_tx, df_meta)

    # In screened sessions, flag should be True
    assert "energy_gt_100kwh" in df_screened.columns
    assert df_screened["energy_gt_100kwh"].all()

    # In model ready, 120.0 kWh remains untouched and included
    assert (df_model_ready["target_energy_kwh"] == 120.0).all()
    assert summary["model_ready_energy_gt_100kwh"] == len(df_model_ready)


def test_15_parquet_io_roundtrip_schema_and_types(tmp_path: Path) -> None:
    """15. Parquet outputs can be saved and read back with identical row counts, expected schema, and types."""
    df_tx, df_meta = make_valid_meta_and_tx(n_sessions=15, spacing_hours=24.0)

    df_screened, df_model_ready, summary = build_screened_and_model_ready_datasets(df_tx, df_meta)
    file_map = save_processed_datasets(df_screened, df_model_ready, summary, tmp_path)

    assert file_map["screened_sessions"].is_file()
    assert file_map["model_ready_sessions"].is_file()
    assert file_map["processing_summary"].is_file()

    # Roundtrip read
    df_screened_in = pd.read_parquet(file_map["screened_sessions"])
    df_model_in = pd.read_parquet(file_map["model_ready_sessions"])

    assert len(df_screened_in) == len(df_screened)
    assert len(df_model_in) == len(df_model_ready)
    assert list(df_model_in.columns) == MODEL_READY_COLUMNS

    # Read summary JSON
    with open(file_map["processing_summary"], "r", encoding="utf-8") as f:
        summary_in = json.load(f)
    assert summary_in["final_model_ready_rows"] == len(df_model_ready)


def test_16_pipeline_determinism() -> None:
    """16. Repeated pipeline runs on identical inputs yield bit-exact, deterministic outputs."""
    df_tx, df_meta = make_valid_meta_and_tx(n_sessions=15, spacing_hours=24.0)

    s1, m1, summ1 = build_screened_and_model_ready_datasets(df_tx, df_meta)
    s2, m2, summ2 = build_screened_and_model_ready_datasets(df_tx, df_meta)

    pd.testing.assert_frame_equal(s1, s2, check_exact=True)
    pd.testing.assert_frame_equal(m1, m2, check_exact=True)
    assert summ1 == summ2


def test_input_validation_catches_invalid_datasets() -> None:
    """Extra test: validate_input_datasets correctly raises ValueError on corrupted data."""
    # Duplicate TransactionID
    df_tx, df_meta = make_valid_meta_and_tx(n_sessions=5)
    df_tx_dup = df_tx.copy()
    df_tx_dup.loc[1, "TransactionID"] = df_tx_dup.loc[0, "TransactionID"]
    with pytest.raises(ValueError, match="duplicate values in transactions"):
        validate_input_datasets(df_tx_dup, df_meta)

    # Missing participant in metadata
    df_meta_missing = df_meta[df_meta["ParticipantID"] != "EN001"]
    with pytest.raises(ValueError, match="not found in metadata"):
        validate_input_datasets(df_tx, df_meta_missing)
