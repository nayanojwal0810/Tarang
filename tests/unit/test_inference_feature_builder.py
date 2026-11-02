"""Unit tests for single-session inference feature builder.

Validates that feature construction adheres strictly to:
- Exact 13 canonical predictor schema and ordering
- Weekday/weekend logic with holiday overrides
- Accurate month and cyclical time trigonometric calculations
- Max start hour normalization from summary configuration without fallback
- Immediate predecessor lag state calculation and 336-hour duration capping
- Same-day cumulative session count tracking (sessionsToday)
- Cold-start rejection and chronological integrity enforcement
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from tarang.config import FEATURE_NAMES, UK_BANK_HOLIDAYS_REFERENCE
from tarang.inference.feature_builder import build_inference_features
from tarang.inference.loading import load_max_start_hour
from tarang.inference.validation import normalize_utc_timestamp, validate_session_inputs


@pytest.fixture
def valid_history() -> pd.DataFrame:
    """Provide minimal valid single-participant history."""
    return pd.DataFrame(
        {
            "StartTime": [
                pd.Timestamp("2018-05-01 08:00:00", tz="UTC"),
                pd.Timestamp("2018-05-01 14:00:00", tz="UTC"),
            ],
            "StopTime": [
                pd.Timestamp("2018-05-01 11:30:00", tz="UTC"),
                pd.Timestamp("2018-05-01 16:00:00", tz="UTC"),
            ],
            "ConsumedkWh": [12.5, 8.0],
        }
    )


def test_feature_names_and_order(valid_history: pd.DataFrame) -> None:
    """Verify that build_inference_features returns exactly 13 canonical features in order."""
    start_time = "2018-05-01 18:00:00"
    df_feat = build_inference_features(
        start_time=start_time,
        car_kw=7.0,
        car_kwh=30.0,
        history=valid_history,
        max_start_hour=24.0,
    )
    assert isinstance(df_feat, pd.DataFrame)
    assert len(df_feat) == 1
    assert list(df_feat.columns) == FEATURE_NAMES


def test_calendar_weekday_and_weekend(valid_history: pd.DataFrame) -> None:
    """Verify weekday and weekend classification for regular days."""
    # Wednesday 2018-05-02 (not a holiday)
    df_wed = build_inference_features(
        start_time="2018-05-02 10:00:00",
        car_kw=7.0,
        car_kwh=40.0,
        history=valid_history,
        max_start_hour=24.0,
    )
    assert df_wed["weekday"].iloc[0] == 1
    assert df_wed["weekend"].iloc[0] == 0
    assert df_wed["holiday"].iloc[0] == 0
    assert df_wed["month"].iloc[0] == 5

    # Saturday 2018-05-05 (not a holiday)
    df_sat = build_inference_features(
        start_time="2018-05-05 10:00:00",
        car_kw=7.0,
        car_kwh=40.0,
        history=valid_history,
        max_start_hour=24.0,
    )
    assert df_sat["weekday"].iloc[0] == 0
    assert df_sat["weekend"].iloc[0] == 1
    assert df_sat["holiday"].iloc[0] == 0


def test_holiday_override(valid_history: pd.DataFrame) -> None:
    """Verify that on UK bank holidays, both weekday and weekend are 0, and holiday is 1."""
    # 2018-05-07 is Early May Bank Holiday (Monday)
    assert pd.Timestamp("2018-05-07").date() in UK_BANK_HOLIDAYS_REFERENCE

    df_hol = build_inference_features(
        start_time="2018-05-07 10:00:00",
        car_kw=7.0,
        car_kwh=40.0,
        history=valid_history,
        max_start_hour=24.0,
    )
    assert df_hol["holiday"].iloc[0] == 1
    assert df_hol["weekday"].iloc[0] == 0
    assert df_hol["weekend"].iloc[0] == 0


def test_cyclical_time_features(valid_history: pd.DataFrame) -> None:
    """Verify StartHourNorm, StartCos, and StartSin exact mathematical formulas."""
    # Start at 06:00:00 -> 6.0 hours
    # If max_start_hour = 24.0 -> StartHourNorm = 0.25
    # StartCos = cos(2 * pi * 0.25) = cos(pi / 2) = 0.0
    # StartSin = sin(2 * pi * 0.25) = sin(pi / 2) = 1.0
    df_feat = build_inference_features(
        start_time="2018-05-02 06:00:00",
        car_kw=7.0,
        car_kwh=40.0,
        history=valid_history,
        max_start_hour=24.0,
    )
    assert pytest.approx(df_feat["StartHourNorm"].iloc[0], abs=1e-6) == 0.25
    assert pytest.approx(df_feat["StartCos"].iloc[0], abs=1e-6) == 0.0
    assert pytest.approx(df_feat["StartSin"].iloc[0], abs=1e-6) == 1.0


def test_time_since_last_stop_and_sessions_today(valid_history: pd.DataFrame) -> None:
    """Verify TimeSinceLastStop gap and sessionsToday cumulative count."""
    # Immediately preceding session stopped at 2018-05-01 16:00:00 UTC
    # New session arrives at 2018-05-01 18:30:00 UTC -> gap = 2.5 hours
    # On 2018-05-01, history contains 2 prior sessions -> sessionsToday = 2
    df_feat = build_inference_features(
        start_time="2018-05-01 18:30:00",
        car_kw=3.7,
        car_kwh=24.0,
        history=valid_history,
        max_start_hour=24.0,
    )
    assert pytest.approx(df_feat["TimeSinceLastStop"].iloc[0], abs=1e-6) == 2.5
    assert df_feat["sessionsToday"].iloc[0] == 2
    assert pytest.approx(df_feat["LastDuration"].iloc[0], abs=1e-6) == 2.0  # 16:00 - 14:00
    assert pytest.approx(df_feat["LastConsumedkWh"].iloc[0], abs=1e-6) == 8.0
    assert df_feat["CarKW"].iloc[0] == 3.7
    assert df_feat["CarKWh"].iloc[0] == 24.0


def test_sessions_today_resets_on_new_date(valid_history: pd.DataFrame) -> None:
    """Verify that sessionsToday is 0 when the new session is on a date with no prior sessions."""
    # New session arrives on next day 2018-05-02
    df_feat = build_inference_features(
        start_time="2018-05-02 08:00:00",
        car_kw=7.0,
        car_kwh=30.0,
        history=valid_history,
        max_start_hour=24.0,
    )
    assert df_feat["sessionsToday"].iloc[0] == 0


def test_last_duration_capping() -> None:
    """Verify that LastDuration is capped at 336.0 hours but TimeSinceLastStop uses raw StopTime."""
    # Prior session lasted 400 hours: from 2018-01-01 00:00 to 2018-01-17 16:00
    # StopTime is 2018-01-17 16:00:00
    # New session starts 2018-01-17 20:00:00 -> gap is 4.0 hours
    long_history = pd.DataFrame(
        {
            "StartTime": [pd.Timestamp("2018-01-01 00:00:00", tz="UTC")],
            "StopTime": [pd.Timestamp("2018-01-17 16:00:00", tz="UTC")],
            "ConsumedkWh": [50.0],
        }
    )
    df_feat = build_inference_features(
        start_time="2018-01-17 20:00:00",
        car_kw=7.0,
        car_kwh=60.0,
        history=long_history,
        max_start_hour=24.0,
    )
    # LastDuration must be capped at 336.0 hours
    assert df_feat["LastDuration"].iloc[0] == 336.0
    # TimeSinceLastStop must be exactly 4.0 hours (raw StopTime)
    assert pytest.approx(df_feat["TimeSinceLastStop"].iloc[0], abs=1e-6) == 4.0


def test_cold_start_rejection() -> None:
    """Verify that empty or None history raises ValueError under cold-start condition."""
    empty_history = pd.DataFrame(columns=["StartTime", "StopTime", "ConsumedkWh"])
    with pytest.raises(ValueError, match="Cold-start condition"):
        build_inference_features(
            start_time="2018-05-01 12:00:00",
            car_kw=7.0,
            car_kwh=30.0,
            history=empty_history,
            max_start_hour=24.0,
        )

    with pytest.raises(ValueError, match="Cold-start condition"):
        build_inference_features(
            start_time="2018-05-01 12:00:00",
            car_kw=7.0,
            car_kwh=30.0,
            history=None,  # type: ignore[arg-type]
            max_start_hour=24.0,
        )


def test_missing_preceding_energy_rejection() -> None:
    """Verify that missing ConsumedkWh in preceding session is rejected without imputation."""
    history_missing_kwh = pd.DataFrame(
        {
            "StartTime": ["2018-05-01 08:00:00"],
            "StopTime": ["2018-05-01 10:00:00"],
            "ConsumedkWh": [np.nan],
        }
    )
    with pytest.raises(ValueError, match="missing or non-finite ConsumedkWh"):
        build_inference_features(
            start_time="2018-05-01 12:00:00",
            car_kw=7.0,
            car_kwh=30.0,
            history=history_missing_kwh,
            max_start_hour=24.0,
        )


def test_future_history_and_negative_gap_rejection(valid_history: pd.DataFrame) -> None:
    """Verify rejection of history records that occur after new session or overlap."""
    # Preceding session StopTime is 16:00:00
    # New session arrival at 15:00:00 -> negative gap (-1 hour)
    with pytest.raises(ValueError, match="Negative time gap"):
        build_inference_features(
            start_time="2018-05-01 15:00:00",
            car_kw=7.0,
            car_kwh=30.0,
            history=valid_history,
            max_start_hour=24.0,
        )

    # Preceding session StartTime is 14:00:00
    # New session arrival at 13:00:00 -> future record
    with pytest.raises(ValueError, match="must be strictly before new session StartTime"):
        build_inference_features(
            start_time="2018-05-01 13:00:00",
            car_kw=7.0,
            car_kwh=30.0,
            history=valid_history,
            max_start_hour=24.0,
        )


def test_vehicle_specs_validation(valid_history: pd.DataFrame) -> None:
    """Verify rejection of non-positive or non-finite CarKW and CarKWh."""
    with pytest.raises(ValueError, match="Invalid CarKW"):
        build_inference_features(
            start_time="2018-05-01 18:00:00",
            car_kw=-7.0,
            car_kwh=30.0,
            history=valid_history,
            max_start_hour=24.0,
        )

    with pytest.raises(ValueError, match="Invalid CarKW"):
        build_inference_features(
            start_time="2018-05-01 18:00:00",
            car_kw=0.0,
            car_kwh=30.0,
            history=valid_history,
            max_start_hour=24.0,
        )

    with pytest.raises(ValueError, match="Invalid CarKWh"):
        build_inference_features(
            start_time="2018-05-01 18:00:00",
            car_kw=7.0,
            car_kwh=-30.0,
            history=valid_history,
            max_start_hour=24.0,
        )


def test_history_dataframe_immutability(valid_history: pd.DataFrame) -> None:
    """Ensure the input history DataFrame is not modified by feature construction."""
    history_copy = valid_history.copy(deep=True)
    build_inference_features(
        start_time="2018-05-01 18:00:00",
        car_kw=7.0,
        car_kwh=30.0,
        history=valid_history,
        max_start_hour=24.0,
    )
    pd.testing.assert_frame_equal(valid_history, history_copy)


def test_max_start_hour_loading(tmp_path: Path, valid_history: pd.DataFrame) -> None:
    """Verify loading max_start_hour from custom temporary processing_summary.json."""
    summary_file = tmp_path / "processing_summary.json"
    summary_file.write_text(json.dumps({"max_start_hour": 23.5}), encoding="utf-8")

    loaded_max = load_max_start_hour(summary_file)
    assert loaded_max == 23.5

    df_feat = build_inference_features(
        start_time="2018-05-01 18:00:00",  # 18.0 hours
        car_kw=7.0,
        car_kwh=30.0,
        history=valid_history,
        summary_file=summary_file,
    )
    assert pytest.approx(df_feat["StartHourNorm"].iloc[0], abs=1e-6) == 18.0 / 23.5


def test_missing_summary_file_rejection(tmp_path: Path, valid_history: pd.DataFrame) -> None:
    """Verify FileNotFoundError when processing summary does not exist and max_start_hour is None."""
    missing_file = tmp_path / "non_existent_summary.json"
    with pytest.raises(FileNotFoundError, match="Processing summary file not found"):
        build_inference_features(
            start_time="2018-05-01 18:00:00",
            car_kw=7.0,
            car_kwh=30.0,
            history=valid_history,
            summary_file=missing_file,
        )
