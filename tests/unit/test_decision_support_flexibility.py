"""Unit tests for TARANG Stage 5 smart-charging flexibility analysis.

Validates:
- Availability detection and graceful fallback on empty, None, or missing schema inputs.
- Accurate calendar date classification in Europe/London (same-day vs overnight).
- Exact calculation of charging-to-connection duration ratios:
  (ChargingDuration / 60.0) / (StopTime - StartTime in hours).
- Binning into 25 uniform intervals across [0.0, 1.0] and percentage PMF normalization.
- Parity with EDA flexibility methodology.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tarang.analysis.exploratory_data_analysis import calculate_charging_flexibility
from tarang.decision_support.flexibility import (
    REQUIRED_FLEXIBILITY_COLUMNS,
    analyze_charging_flexibility,
)


def test_flexibility_empty_and_none() -> None:
    """Test flexibility analysis graceful fallback on empty or None inputs."""
    res_none = analyze_charging_flexibility(None)  # type: ignore
    assert res_none["available"] is False
    assert "empty or None" in res_none["reason"]
    assert res_none["same_day_total"] == 0
    assert len(res_none["bin_edges"]) == 26

    res_empty = analyze_charging_flexibility(pd.DataFrame())
    assert res_empty["available"] is False
    assert "empty or None" in res_empty["reason"]
    assert res_empty["same_day_total"] == 0


def test_flexibility_missing_columns() -> None:
    """Test flexibility analysis graceful fallback when required historical columns are missing."""
    df_missing = pd.DataFrame({"StartTime": ["2018-05-15 10:00:00"]})
    res = analyze_charging_flexibility(df_missing)

    assert res["available"] is False
    assert "Missing required column(s)" in res["reason"]
    assert "ChargingDuration" in res["reason"]
    assert "StopTime" in res["reason"]


def test_flexibility_date_classification_europe_london() -> None:
    """Test same-day vs overnight classification adhering to Europe/London local dates."""
    # Session 1: Same day in Europe/London (BST: UTC+1)
    # 2018-06-15 10:00:00 UTC (11:00 BST) -> 2018-06-15 14:00:00 UTC (15:00 BST)
    # Plugged-in: 4h. Charging: 120m (2h). Ratio: 2.0 / 4.0 = 0.5.
    s1 = {
        "StartTime": "2018-06-15 10:00:00+00:00",
        "StopTime": "2018-06-15 14:00:00+00:00",
        "ChargingDuration": 120.0,
    }

    # Session 2: Overnight in Europe/London
    # 2018-06-15 20:00:00 UTC (21:00 BST) -> 2018-06-16 06:00:00 UTC (07:00 BST)
    # Plugged-in: 10h. Charging: 120m (2h). Ratio: 2.0 / 10.0 = 0.2.
    s2 = {
        "StartTime": "2018-06-15 20:00:00+00:00",
        "StopTime": "2018-06-16 06:00:00+00:00",
        "ChargingDuration": 120.0,
    }

    # Session 3: Midnight rollover edge case in BST (UTC 23:30 is next day 00:30 BST)
    # 2018-06-15 22:30:00 UTC (23:30 BST June 15) -> 2018-06-15 23:30:00 UTC (00:30 BST June 16)
    # Plugged-in: 1h. Charging: 30m (0.5h). Overnight in London!
    s3 = {
        "StartTime": "2018-06-15 22:30:00+00:00",
        "StopTime": "2018-06-15 23:30:00+00:00",
        "ChargingDuration": 30.0,
    }

    df = pd.DataFrame([s1, s2, s3])
    res = analyze_charging_flexibility(df, n_bins=25)

    assert res["available"] is True
    assert res["reason"] is None
    assert res["same_day_total"] == 1
    assert res["overnight_total"] == 2
    assert res["same_day_in_range"] == 1
    assert res["overnight_in_range"] == 2

    # Check ratios
    assert pytest.approx(res["same_day_mean_ratio"], rel=1e-3) == 0.5
    assert pytest.approx(res["same_day_median_ratio"], rel=1e-3) == 0.5

    # Overnight ratios: 0.2 and 0.5 -> mean = 0.35
    assert pytest.approx(res["overnight_mean_ratio"], rel=1e-3) == 0.35
    assert pytest.approx(res["overnight_median_ratio"], rel=1e-3) == 0.35


def test_flexibility_parity_with_eda() -> None:
    """Verify exact numerical parity between decision support flexibility wrapper and EDA function."""
    df = pd.DataFrame(
        [
            {
                "StartTime": "2017-03-01 18:00:00+00:00",
                "StopTime": "2017-03-02 07:00:00+00:00",
                "ChargingDuration": 180.0,
            },
            {
                "StartTime": "2017-03-02 12:00:00+00:00",
                "StopTime": "2017-03-02 16:00:00+00:00",
                "ChargingDuration": 60.0,
            },
        ]
    )

    eda_res = calculate_charging_flexibility(df, n_bins=25)
    ds_res = analyze_charging_flexibility(df, n_bins=25)

    assert ds_res["available"] is True
    assert ds_res["bin_edges"] == eda_res["bin_edges"]
    assert ds_res["same_day_total"] == eda_res["same_day_total"]
    assert ds_res["overnight_total"] == eda_res["overnight_total"]
    assert ds_res["same_day_median_ratio"] == eda_res["same_day_median_ratio"]
    assert ds_res["overnight_median_ratio"] == eda_res["overnight_median_ratio"]
    assert ds_res["same_day_pct"] == eda_res["same_day_pct"]
    assert ds_res["overnight_pct"] == eda_res["overnight_pct"]
