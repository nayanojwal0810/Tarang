"""Unit tests for feature engineering modules in TARANG.

Covers:
13. Exact 13-feature list and ordering.
14. Correct binary weekday/weekend values.
15. Exact 13 holiday dates.
16. Holiday override (weekday and weekend both set to 0 on holidays).
17. Seven day indicators.
18. StartHour includes seconds.
19. Correct dataset-wide maximum normalization.
20. Correct cosine and sine values.
21. Correct sessionsToday.
22. Correct previous Duration.
23. Correct previous energy.
24. Missing previous energy not skipped backward.
25. No mutation of source input.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tarang.config import FEATURE_NAMES, UK_BANK_HOLIDAYS_REFERENCE
from tarang.features.calendar_features import calculate_calendar_features
from tarang.features.feature_engineering import extract_features, get_feature_names
from tarang.features.historical_features import (
    calculate_last_consumed_kwh,
    calculate_last_duration,
    calculate_sessions_today,
    calculate_time_since_last_stop,
)
from tarang.features.time_features import (
    calculate_cyclical_time_features,
    calculate_start_hour,
)


def test_13_exact_13_features_and_authoritative_ordering() -> None:
    """The authoritative FEATURE_NAMES must match exactly 13 items in mandatory sequence."""
    expected = [
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
    ]
    assert get_feature_names() == expected
    assert FEATURE_NAMES == expected


def test_14_17_binary_weekday_weekend_and_seven_day_indicators() -> None:
    """Correct binary flags for weekday (Mon-Fri) and weekend (Sat-Sun) and Mon..Sun indicators."""
    # 2017-05-01 is a bank holiday, so test with non-holiday dates:
    # 2017-05-02 is Tuesday (weekday=1, weekend=0, Tue=1, Mon=0)
    # 2017-05-06 is Saturday (weekday=0, weekend=1, Sat=1)
    df = pd.DataFrame({
        "StartTime": [pd.Timestamp("2017-05-02 10:00:00"), pd.Timestamp("2017-05-06 14:00:00")]
    })
    out = calculate_calendar_features(df)

    # Tuesday:
    assert out.loc[0, "weekday"] == 1
    assert out.loc[0, "weekend"] == 0
    assert out.loc[0, "Tue"] == 1
    assert out.loc[0, "Mon"] == 0
    assert out.loc[0, "holiday"] == 0
    assert out.loc[0, "month"] == 5

    # Saturday:
    assert out.loc[1, "weekday"] == 0
    assert out.loc[1, "weekend"] == 1
    assert out.loc[1, "Sat"] == 1
    assert out.loc[1, "Sun"] == 0
    assert out.loc[1, "holiday"] == 0


def test_15_16_exact_13_holidays_and_override() -> None:
    """Exact 13 holidays are recognized, setting holiday=1 and overriding BOTH weekday and weekend to 0.

    Also tests dates that must NOT be treated as holidays (e.g. Easter Monday and August bank holidays).
    """
    expected_holidays = {
        pd.Timestamp("2017-01-01").date(),
        pd.Timestamp("2017-01-02").date(),
        pd.Timestamp("2017-04-14").date(),
        pd.Timestamp("2017-05-01").date(),
        pd.Timestamp("2017-05-29").date(),
        pd.Timestamp("2017-12-25").date(),
        pd.Timestamp("2017-12-26").date(),
        pd.Timestamp("2018-01-01").date(),
        pd.Timestamp("2018-03-30").date(),
        pd.Timestamp("2018-05-07").date(),
        pd.Timestamp("2018-05-28").date(),
        pd.Timestamp("2018-12-25").date(),
        pd.Timestamp("2018-12-26").date(),
    }
    assert UK_BANK_HOLIDAYS_REFERENCE == expected_holidays

    # Dates that must NOT be treated as holidays in the established benchmark
    non_holidays = [
        pd.Timestamp("2017-04-17 12:00:00"),  # Easter Monday 2017
        pd.Timestamp("2017-08-28 12:00:00"),  # Summer Bank Holiday 2017
        pd.Timestamp("2018-04-02 12:00:00"),  # Easter Monday 2018
        pd.Timestamp("2018-08-27 12:00:00"),  # Summer Bank Holiday 2018
    ]
    df_non = pd.DataFrame({"StartTime": non_holidays})
    out_non = calculate_calendar_features(df_non)
    assert (out_non["holiday"] == 0).all()
    # All are Mondays, so weekday must be 1, weekend must be 0
    assert (out_non["weekday"] == 1).all()
    assert (out_non["weekend"] == 0).all()

    # All 13 designated holidays
    all_holidays = [pd.Timestamp(f"{d} 10:00:00") for d in sorted(expected_holidays)]
    df_hol = pd.DataFrame({"StartTime": all_holidays})
    out_hol = calculate_calendar_features(df_hol)

    assert (out_hol["holiday"] == 1).all()
    # On holidays, both weekday and weekend must be overridden to 0
    assert (out_hol["weekday"] == 0).all()
    assert (out_hol["weekend"] == 0).all()


def test_18_starthour_includes_seconds() -> None:
    """StartHour formula: hour + minute/60.0 + second/3600.0 must include seconds precision."""
    # 14:30:45 -> 14 + 30/60 + 45/3600 = 14 + 0.5 + 0.0125 = 14.5125
    df = pd.DataFrame({"StartTime": [pd.Timestamp("2017-05-01 14:30:45")]})
    s = calculate_start_hour(df)
    assert pytest.approx(s.iloc[0], rel=1e-6) == 14.5125


def test_19_20_dataset_wide_maximum_normalization_and_cyclical_trig() -> None:
    """StartHourNorm = StartHour / max(StartHour) and cosine/sine calculations."""
    df = pd.DataFrame({
        "StartTime": [
            pd.Timestamp("2017-05-01 00:00:00"),
            pd.Timestamp("2017-05-01 12:00:00"),
            pd.Timestamp("2017-05-01 23:59:00"),
        ]
    })
    max_hour = 23.983333333333334
    out = calculate_cyclical_time_features(df, start_hour_max=max_hour)

    # Check normalization
    assert pytest.approx(out.loc[0, "StartHourNorm"], abs=1e-5) == 0.0
    assert pytest.approx(out.loc[0, "StartCos"], abs=1e-5) == 1.0
    assert pytest.approx(out.loc[0, "StartSin"], abs=1e-5) == 0.0

    # Halfway
    norm_half = 12.0 / max_hour
    assert pytest.approx(out.loc[1, "StartHourNorm"], abs=1e-5) == norm_half
    assert pytest.approx(out.loc[1, "StartCos"], abs=1e-5) == np.cos(2.0 * np.pi * norm_half)
    assert pytest.approx(out.loc[1, "StartSin"], abs=1e-5) == np.sin(2.0 * np.pi * norm_half)


def test_21_sessionstoday_running_daily_count() -> None:
    """sessionsToday counts sessions started on the same calendar date for that participant."""
    df = pd.DataFrame({
        "StartTime": [
            pd.Timestamp("2017-05-01 08:00:00"),
            pd.Timestamp("2017-05-01 14:00:00"),
            pd.Timestamp("2017-05-01 20:00:00"),
            pd.Timestamp("2017-05-02 09:00:00"),
        ]
    })
    counts = calculate_sessions_today(df)
    assert counts.tolist() == [0, 1, 2, 0]


def test_22_23_last_duration_and_last_consumed_kwh() -> None:
    """LastDuration and LastConsumedkWh extract the shift(1) values directly."""
    df = pd.DataFrame({
        "Duration": [2.5, 4.0, 1.2],
        "ConsumedkWh": [8.0, 15.5, 3.2],
    })
    last_dur = calculate_last_duration(df)
    last_energy = calculate_last_consumed_kwh(df)

    assert pd.isna(last_dur.iloc[0])
    assert last_dur.iloc[1] == 2.5
    assert last_dur.iloc[2] == 4.0

    assert pd.isna(last_energy.iloc[0])
    assert last_energy.iloc[1] == 8.0
    assert last_energy.iloc[2] == 15.5


def test_24_missing_previous_energy_not_skipped_backward() -> None:
    """When previous session had NaN energy, LastConsumedkWh must be NaN (not searching backwards)."""
    df = pd.DataFrame({
        "ConsumedkWh": [10.0, np.nan, 20.0],
    })
    last_energy = calculate_last_consumed_kwh(df)
    # Row 0: NaN (first session)
    assert pd.isna(last_energy.iloc[0])
    # Row 1: 10.0 (from row 0)
    assert last_energy.iloc[1] == 10.0
    # Row 2: NaN (from row 1, NOT skipping backward to row 0's 10.0!)
    assert pd.isna(last_energy.iloc[2])


def test_25_no_mutation_of_source_input() -> None:
    """Feature engineering functions must not mutate input DataFrames in place."""
    df = pd.DataFrame({"StartTime": [pd.Timestamp("2017-05-02 10:00:00")]})
    cols_before = list(df.columns)
    _ = calculate_calendar_features(df)
    assert list(df.columns) == cols_before
    _ = calculate_cyclical_time_features(df)
    assert list(df.columns) == cols_before


def test_extract_features_enforces_exact_13_columns() -> None:
    """extract_features selects exactly the 13 predictors and errors if any are missing."""
    df = pd.DataFrame({col: [1.0] for col in FEATURE_NAMES})
    df["extra_col"] = [99.0]

    extracted = extract_features(df)
    assert list(extracted.columns) == FEATURE_NAMES

    # Missing column raises ValueError
    df_missing = df.drop(columns=["StartCos"])
    with pytest.raises(ValueError) as exc_info:
        extract_features(df_missing)
    assert "StartCos" in str(exc_info.value)

