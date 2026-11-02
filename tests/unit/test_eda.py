"""Unit tests for Exploratory Data Analysis (EDA) functions in TARANG.

Covers:
26. Correct participant availability metrics.
27. Correct raw duration calculations.
28. Correct local same-day and overnight classification.
29. Correct charging ratio.
30. Correct bin boundaries and display ranges.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tarang.analysis.exploratory_data_analysis import (
    calculate_charging_flexibility,
    calculate_data_availability,
    generate_eda_markdown_summary,
)


def test_26_27_participant_availability_and_raw_duration() -> None:
    """Participant availability metrics: span, session count, weekly frequency, and raw mean duration."""
    # Participant with 10 sessions spread over 70 days.
    # Sessions are 400 hours long each, but capped Duration column is 336.0.
    # The calculation must use raw StopTime - StartTime (400.0), NOT capped Duration!
    base = pd.Timestamp("2017-05-01 10:00:00", tz="UTC")
    starts = [base + pd.Timedelta(days=i * 7) for i in range(10)]
    stops = [t + pd.Timedelta(hours=400.0) for t in starts]

    df = pd.DataFrame({
        "ParticipantID": ["EN001"] * 10,
        "StartTime": starts,
        "StopTime": stops,
        "Duration": [336.0] * 10,  # Capped at 336.0
        "ConsumedkWh": [25.0] * 10,
    })

    summary_df, agg_stats = calculate_data_availability(df)
    assert len(summary_df) == 1
    row = summary_df.iloc[0]

    # Span in days: (latest StopTime - earliest StartTime).days = (2017-05-01 + 63 days + 400h - 2017-05-01).days = 79 days
    expected_span = (stops[-1] - starts[0]).days
    assert row["span_days"] == expected_span
    assert row["session_count"] == 10
    # weekly frequency = 10 / (expected_span / 7.0)
    assert pytest.approx(row["weekly_frequency"], rel=1e-2) == (10.0 / (expected_span / 7.0))
    # Mean duration must be 400.0 hours (raw), NOT 336.0!
    assert pytest.approx(row["mean_duration_hours"], rel=1e-3) == 400.0


def test_28_local_sameday_vs_overnight_classification() -> None:
    """Sessions are classified into same-day vs overnight based on Europe/London local dates.

    Example in BST (UTC+1, e.g. July):
    - Session A: Start 2017-07-01 22:30 UTC -> London is 2017-07-01 23:30 BST
                 Stop  2017-07-01 23:45 UTC -> London is 2017-07-02 00:45 BST
      In UTC, start and stop dates are both 2017-07-01 (same-day in UTC).
      In London BST, stop date is 2017-07-02 (overnight in London!).
      Must correctly classify as overnight!
    """
    start_utc = pd.Timestamp("2017-07-01 22:30:00", tz="UTC")
    stop_utc = pd.Timestamp("2017-07-01 23:45:00", tz="UTC")
    # Duration: 1.25 hours (75 minutes). ChargingDuration: 60 minutes.
    df = pd.DataFrame({
        "StartTime": [start_utc],
        "StopTime": [stop_utc],
        "ChargingDuration": [60.0],
    })

    flex_stats = calculate_charging_flexibility(df, n_bins=25)
    # Must be classified as overnight because in London it crosses midnight!
    assert flex_stats["overnight_count"] == 1
    assert flex_stats["same_day_count"] == 0


def test_29_30_charging_flexibility_ratio_and_bin_boundaries() -> None:
    """Charging ratio = (ChargingDuration / 60) / raw_duration_hours, binned into 25 bins in [0, 1]."""
    # Session 1: 4 hours plugged in, 2 hours charging -> ratio = 2.0 / 4.0 = 0.50
    # Session 2: 10 hours plugged in, 1 hour charging -> ratio = 1.0 / 10.0 = 0.10
    start1 = pd.Timestamp("2017-05-02 10:00:00", tz="UTC")
    stop1 = pd.Timestamp("2017-05-02 14:00:00", tz="UTC")
    start2 = pd.Timestamp("2017-05-03 10:00:00", tz="UTC")
    stop2 = pd.Timestamp("2017-05-03 20:00:00", tz="UTC")

    df = pd.DataFrame({
        "StartTime": [start1, start2],
        "StopTime": [stop1, stop2],
        "ChargingDuration": [120.0, 60.0],  # minutes
    })

    flex_stats = calculate_charging_flexibility(df, n_bins=25)
    # Exactly 26 bin edges for 25 bins over [0, 1]
    assert len(flex_stats["bin_edges"]) == 26
    assert flex_stats["bin_edges"][0] == 0.0
    assert flex_stats["bin_edges"][-1] == 1.0

    # Both sessions are same-day (10:00 to 14:00 and 10:00 to 20:00)
    assert flex_stats["same_day_count"] == 2
    # Ratios are 0.50 and 0.10
    # Median is (0.50 + 0.10) / 2 = 0.30
    assert pytest.approx(flex_stats["same_day_median_ratio"], abs=1e-3) == 0.30


def test_eda_markdown_summary_generation() -> None:
    """generate_eda_markdown_summary produces markdown containing key figures."""
    df = pd.DataFrame({
        "ParticipantID": ["EN001"] * 10,
        "StartTime": [pd.Timestamp("2017-05-01 10:00:00", tz="UTC") + pd.Timedelta(days=i) for i in range(10)],
        "StopTime": [pd.Timestamp("2017-05-01 14:00:00", tz="UTC") + pd.Timedelta(days=i) for i in range(10)],
        "ConsumedkWh": [15.0] * 10,
        "ChargingDuration": [120.0] * 10,
        "StartHour": [10.0] * 10,
    })
    _, avail_stats = calculate_data_availability(df)
    flex_stats = calculate_charging_flexibility(df)
    md = generate_eda_markdown_summary(avail_stats, flex_stats, df)

    assert "# TARANG — Exploratory Data Analysis (EDA) Summary" in md
    assert "Participant Data Availability" in md
    assert "Smart-Charging Flexibility Ratio" in md
