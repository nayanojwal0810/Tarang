"""Unit tests for participant screening and preparation pipeline in TARANG.

Covers:
2. Trial 3 exclusion.
3. Sessions below five minutes removed.
4. Duration capping at 336 hours.
5. Original StopTime preserved.
6. Immediate-previous-session gap calculation.
7. Correct negative-gap removal sequence.
8. First session dropped.
9. Minimum-history condition.
10. Observation-density condition.
11. Participant ID assignment.
12. No additional energy exclusions.
31. Deterministic output.
34. Empty input handling.
35. Clear invalid-input errors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tarang.config import CANONICAL_COLUMNS
from tarang.data.screening import screen_and_prepare_dataset, screen_participant_history


def make_synthetic_tx(
    n_sessions: int = 15,
    participant_id: str = "EN001",
    start_date: str = "2017-05-01 10:00:00",
    spacing_hours: float = 24.0,
    session_duration_hours: float = 2.0,
    car_kw: float = 7.0,
    car_kwh: float = 30.0,
    consumed_kwh: float = 12.0,
    trial: float = 1.0,
) -> pd.DataFrame:
    """Construct synthetic transactions DataFrame with standard columns."""
    base_time = pd.Timestamp(start_date)
    start_times = [base_time + pd.Timedelta(hours=i * spacing_hours) for i in range(n_sessions)]
    stop_times = [t + pd.Timedelta(hours=session_duration_hours) for t in start_times]

    return pd.DataFrame({
        "StartTime": start_times,
        "StopTime": stop_times,
        "ParticipantID": [participant_id] * n_sessions,
        "ConsumedkWh": [consumed_kwh] * n_sessions,
        "CarKW": [car_kw] * n_sessions,
        "CarKWh": [car_kwh] * n_sessions,
        "PluggedInTime": [session_duration_hours * 60.0] * n_sessions,
        "ChargingDuration": [session_duration_hours * 50.0] * n_sessions,
        "Hot_Unplug": [False] * n_sessions,
        "Trial": [trial] * n_sessions,
    })


def test_02_trial_3_sessions_are_excluded() -> None:
    """Trial 3 sessions (Trial == 3.0) must be excluded from the dataset."""
    df_t1 = make_synthetic_tx(n_sessions=15, trial=1.0)
    df_t3 = make_synthetic_tx(n_sessions=5, participant_id="EN002", trial=3.0)
    df_comb = pd.concat([df_t1, df_t3], ignore_index=True)

    final_df, metrics = screen_and_prepare_dataset(df_comb)
    assert metrics["trial_3_excluded"] == 5
    assert "EN002" not in final_df["ParticipantID"].values


def test_03_sessions_under_five_minutes_are_removed() -> None:
    """Sessions with duration < 5 minutes (5/60 hours) must be discarded."""
    df = make_synthetic_tx(n_sessions=15, spacing_hours=24.0, session_duration_hours=2.0)
    # Inject a 3-minute session
    short_start = pd.Timestamp("2017-06-01 10:00:00")
    short_session = pd.DataFrame({
        "StartTime": [short_start],
        "StopTime": [short_start + pd.Timedelta(minutes=3)],
        "ParticipantID": ["EN001"],
        "ConsumedkWh": [0.5],
        "CarKW": [7.0],
        "CarKWh": [30.0],
        "PluggedInTime": [3.0],
        "ChargingDuration": [2.0],
        "Hot_Unplug": [False],
        "Trial": [1.0],
    })
    df_with_short = pd.concat([df, short_session], ignore_index=True)

    final_df, metrics = screen_and_prepare_dataset(df_with_short)
    assert metrics["short_sessions_excluded"] == 1
    # Duration must be >= 5 minutes (5/60 hours)
    assert (final_df["Duration"] >= (5.0 / 60.0)).all()


def test_04_05_duration_capped_at_336_and_original_stoptime_preserved() -> None:
    """Duration values above 336.0 hours are capped at 336.0, while original StopTime is preserved."""
    df = make_synthetic_tx(n_sessions=15, spacing_hours=24.0, session_duration_hours=2.0)
    # Replace the last session with a 400-hour duration
    long_start = df.loc[14, "StartTime"]
    long_stop = long_start + pd.Timedelta(hours=400.0)
    df.loc[14, "StopTime"] = long_stop

    final_df, _ = screen_and_prepare_dataset(df)
    capped_row = final_df[final_df["StartTime"] == long_start.tz_localize("UTC")]
    assert len(capped_row) == 1
    # Duration is capped at 336.0
    assert capped_row["Duration"].iloc[0] == 336.0
    # Original StopTime is preserved (remains long_stop)
    assert capped_row["StopTime"].iloc[0] == long_stop.tz_localize("UTC")


def test_06_07_immediate_gap_and_negative_gap_removal_sequence() -> None:
    """Negative gap sessions are removed without recalculating subsequent gaps.

    Scenario:
    Session 1: Stop at 12:00
    Session 2: Start at 11:00 (Gap = -1h, negative) -> Discarded
    Session 3: Start at 13:00 (Original Gap vs Session 2 Stop [14:00] was -1h)
    Under reference implementation, Session 3 gap is calculated against Session 2 BEFORE Session 2 is removed,
    so Session 3 also has negative gap and is removed!
    """
    s1_start = pd.Timestamp("2017-05-01 10:00:00")
    s1_stop = pd.Timestamp("2017-05-01 12:00:00")
    s2_start = pd.Timestamp("2017-05-01 11:00:00")  # starts before s1 ends -> negative gap vs s1
    s2_stop = pd.Timestamp("2017-05-01 14:00:00")
    s3_start = pd.Timestamp("2017-05-01 13:00:00")  # starts before s2 ends -> negative gap vs s2

    # Provide 15 regular sessions afterwards
    subsequent = [s3_start + pd.Timedelta(days=i * 2) for i in range(1, 16)]
    sub_stops = [t + pd.Timedelta(hours=2) for t in subsequent]

    starts = [s1_start, s2_start, s3_start] + subsequent
    stops = [s1_stop, s2_stop, s3_start + pd.Timedelta(hours=2)] + sub_stops

    df = pd.DataFrame({
        "StartTime": starts,
        "StopTime": stops,
        "ParticipantID": ["EN001"] * len(starts),
        "ConsumedkWh": [10.0] * len(starts),
        "CarKW": [7.0] * len(starts),
        "CarKWh": [30.0] * len(starts),
        "PluggedInTime": [120.0] * len(starts),
        "ChargingDuration": [100.0] * len(starts),
        "Hot_Unplug": [False] * len(starts),
        "Trial": [1.0] * len(starts),
    })

    final_df, metrics = screen_and_prepare_dataset(df)
    assert metrics["negative_gaps_removed"] == 2  # s2 and s3 both had negative gaps


def test_08_first_remaining_session_is_dropped() -> None:
    """The first remaining session per participant must be dropped."""
    df = make_synthetic_tx(n_sessions=15, spacing_hours=24.0, session_duration_hours=2.0)
    final_df, metrics = screen_and_prepare_dataset(df)
    assert metrics["dropped_first_sessions"] == 1
    # Original first session was 2017-05-01 10:00:00; retained starts with second session
    assert len(final_df) == 14
    assert final_df["StartTime"].min() == pd.Timestamp("2017-05-02 10:00:00", tz="UTC")


def test_09_minimum_history_condition() -> None:
    """Participants with fewer than 10 retained sessions are excluded."""
    # 10 initial sessions -> 1 dropped first session -> 9 retained sessions (< 10) -> Excluded!
    df = make_synthetic_tx(n_sessions=10, spacing_hours=24.0, session_duration_hours=2.0)
    final_df, metrics = screen_and_prepare_dataset(df)
    assert metrics["participants_lt_10_retained"] == 1
    assert len(final_df) == 0


def test_10_observation_density_condition() -> None:
    """Participants with retained_sessions < observation_span_days / 7 are excluded."""
    base = pd.Timestamp("2017-01-01 10:00:00")
    # 1 dropped first session + 10 retained sessions across 77 days
    retained_starts = [base + pd.Timedelta(days=1 + i * 8) for i in range(10)]
    all_starts = [base] + retained_starts
    all_stops = [base + pd.Timedelta(hours=2)] + [t + pd.Timedelta(hours=2) for t in retained_starts[:-1]] + [retained_starts[0] + pd.Timedelta(days=77, hours=2)]

    df = pd.DataFrame({
        "StartTime": all_starts,
        "StopTime": all_stops,
        "ParticipantID": ["EN_SPARSE"] * 11,
        "ConsumedkWh": [10.0] * 11,
        "CarKW": [7.0] * 11,
        "CarKWh": [30.0] * 11,
        "PluggedInTime": [120.0] * 11,
        "ChargingDuration": [100.0] * 11,
        "Hot_Unplug": [False] * 11,
        "Trial": [1.0] * 11,
    })

    final_df, metrics = screen_and_prepare_dataset(df)
    assert metrics["participants_before_density"] == 1
    assert metrics["sessions_before_density"] == 10
    assert metrics["participants_excluded_density"] == 1
    assert metrics["sessions_excluded_density"] == 10
    assert len(final_df) == 0


def test_11_participant_integer_id_assignment() -> None:
    """Each retained participant is assigned a sequential integer id starting from 0."""
    df1 = make_synthetic_tx(n_sessions=15, participant_id="EN001", spacing_hours=24.0)
    df2 = make_synthetic_tx(n_sessions=15, participant_id="EN002", spacing_hours=24.0)
    df_comb = pd.concat([df1, df2], ignore_index=True)

    final_df, metrics = screen_and_prepare_dataset(df_comb)
    assert metrics["final_eligible_participants"] == 2
    assert "id" in final_df.columns
    p_ids = sorted(final_df["id"].unique())
    assert p_ids == [0, 1]


def test_12_no_additional_energy_exclusions() -> None:
    """Extreme energy readings (> 100 kWh or > CarKWh) are NOT dropped by research screening."""
    df = make_synthetic_tx(n_sessions=15, spacing_hours=24.0, consumed_kwh=120.0, car_kwh=30.0)
    final_df, metrics = screen_and_prepare_dataset(df)
    # The 120 kWh sessions are preserved
    assert len(final_df) == 14
    assert (final_df["ConsumedkWh"] == 120.0).all()


def test_31_deterministic_screening_output() -> None:
    """Running screen_and_prepare_dataset twice on identical input yields bit-exact identical DataFrames."""
    df = make_synthetic_tx(n_sessions=15, spacing_hours=24.0)
    res1, metrics1 = screen_and_prepare_dataset(df)
    res2, metrics2 = screen_and_prepare_dataset(df)

    pd.testing.assert_frame_equal(res1, res2)
    assert metrics1 == metrics2


def test_34_empty_input_handling() -> None:
    """Empty input DataFrame returns empty DataFrame with CANONICAL_COLUMNS and zeroed metrics without crashing."""
    df_empty = pd.DataFrame()
    final_df, metrics = screen_and_prepare_dataset(df_empty)
    assert final_df.empty
    assert list(final_df.columns) == CANONICAL_COLUMNS
    assert metrics["final_eligible_participants"] == 0
    assert metrics["final_eligible_sessions"] == 0
