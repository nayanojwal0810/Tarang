"""Unit tests for the TARANG data audit functions.

Uses small synthetic DataFrames to verify core audit logic without
relying on the full production datasets.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tarang.data.audit import (
    audit_data_quality,
    audit_feature_feasibility,
    calculate_missingness,
    detect_duplicates,
    reconcile_participants,
    validate_timestamps,
)


@pytest.fixture
def sample_transaction_df() -> pd.DataFrame:
    """Create a synthetic transaction DataFrame."""
    return pd.DataFrame({
        "TransactionID": [101, 102, 103, 104, 105],
        "ChargerID": [3501, 3501, 3502, 3503, 3503],
        "ParticipantID": ["EN001", "EN001", "EN002", "EN003", None],
        "CarKW": [7.0, 7.0, 3.6, 7.0, None],
        "CarKWh": [33.0, 33.0, 24.0, 75.0, None],
        "StartTime": [
            "2017-04-01 10:00:00",
            "2017-04-01 18:00:00",
            "2017-04-02 08:30:00",
            "2017-04-03 14:00:00",
            "2017-04-04 12:00:00",
        ],
        "StopTime": [
            "2017-04-01 12:00:00",
            "2017-04-02 06:00:00",
            "2017-04-02 08:30:00",  # Zero duration
            "2017-04-03 14:02:00",  # 2 min duration (< 5 min)
            "2017-04-04 15:00:00",
        ],
        "ConsumedkWh": [12.5, 22.0, 0.5, 0.1, None],  # 1 missing
        "ChargingDuration": [90.0, 180.0, 0.0, 2.0, 45.0],
        "Trial": [1.0, 1.0, 2.0, 3.0, None],
    })


@pytest.fixture
def sample_metadata_df() -> pd.DataFrame:
    """Create a synthetic metadata DataFrame."""
    return pd.DataFrame({
        "ParticipantID": ["EN001", "EN002", "EN004"],
        "Charger": [3501, 3502, 3504],
        "DCSProvider": ["Crowd Charge", "Crowd Charge", "Greenflux"],
        "CarMake": ["BMW", "Nissan", "Tesla"],
        "CarModel": ["i3", "Leaf", "Model S"],
        "PIVType": ["Electric only (BEV)", "Electric only (BEV)", "Electric only (BEV)"],
        "CarkW": [7.0, 3.6, 7.0],
        "CarkWh": [33.0, 24.0, 75.0],
    })


def test_detect_duplicates() -> None:
    """Test duplicate detection on full rows and specific column subsets."""
    df = pd.DataFrame({
        "TransactionID": [1, 2, 2, 3],
        "Value": ["a", "b", "b", "c"],
    })
    assert detect_duplicates(df) == 1
    assert detect_duplicates(df, subset=["TransactionID"]) == 1

    df_unique = pd.DataFrame({
        "TransactionID": [1, 2, 3],
        "Value": ["a", "b", "c"],
    })
    assert detect_duplicates(df_unique) == 0


def test_calculate_missingness() -> None:
    """Test calculation of missing values count and percentages."""
    df = pd.DataFrame({
        "col_a": [1, 2, None, 4],
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

    row_c = res[res["column_name"] == "col_c"].iloc[0]
    assert row_c["missing_count"] == 0
    assert row_c["missing_percentage"] == 0.0


def test_validate_timestamps() -> None:
    """Test timestamp parsing, negative intervals, and zero intervals."""
    df = pd.DataFrame({
        "StartTime": ["2017-01-01 10:00:00", "2017-01-02 12:00:00", "2017-01-03 14:00:00"],
        "StopTime": ["2017-01-01 11:00:00", "2017-01-02 12:00:00", "2017-01-03 13:00:00"],
    })
    val = validate_timestamps(df, "StartTime", "StopTime")

    assert val["start_parse_failures"] == 0
    assert val["stop_parse_failures"] == 0
    assert val["zero_intervals"] == 1
    assert val["negative_intervals"] == 1
    assert "2017-01-01 10:00:00" in val["earliest_start"]
    assert "2017-01-03 14:00:00" in val["latest_start"]


def test_reconcile_participants(sample_transaction_df: pd.DataFrame, sample_metadata_df: pd.DataFrame) -> None:
    """Test participant reconciliation between transactions and metadata."""
    reconciliation = reconcile_participants(sample_transaction_df, sample_metadata_df)
    assert isinstance(reconciliation, pd.DataFrame)
    assert "metric" in reconciliation.columns
    assert "count" in reconciliation.columns

    # Transactions has non-null EN001, EN002, EN003 (3 participants)
    # Metadata has EN001, EN002, EN004 (3 participants)
    # Matched should be EN001, EN002 (2 participants)
    matched_row = reconciliation[reconciliation["metric"] == "Matched Participants (Present in Both)"].iloc[0]
    assert matched_row["count"] == 2

    tx_only_row = reconciliation[reconciliation["metric"] == "Transaction Participants Missing from Metadata"].iloc[0]
    assert tx_only_row["count"] == 1  # EN003

    meta_only_row = reconciliation[reconciliation["metric"] == "Metadata Participants Missing from Transactions"].iloc[0]
    assert meta_only_row["count"] == 1  # EN004


def test_feature_feasibility(sample_transaction_df: pd.DataFrame, sample_metadata_df: pd.DataFrame) -> None:
    """Test feature feasibility audit table generation."""
    feasibility = audit_feature_feasibility(sample_transaction_df, sample_metadata_df)
    assert isinstance(feasibility, pd.DataFrame)
    assert len(feasibility) == 13

    feature_names = set(feasibility["feature_name"])
    expected_features = {
        "Weekday",
        "Weekend",
        "Holiday",
        "Month",
        "Start-Hour Cyclic (sin/cos)",
        "TimeSinceLastStop",
        "SessionsToday",
        "LastDuration",
        "LastConsumedkWh",
        "CarKW",
        "CarKWh",
        "Plugged-in Duration (Hours)",
        "Energy Consumption (kWh)",
    }
    assert expected_features == feature_names

    # Check that targets are identified
    targets = feasibility[feasibility["type"].str.contains("Prediction Target")]
    assert len(targets) == 2


def test_audit_data_quality(sample_transaction_df: pd.DataFrame) -> None:
    """Test data quality investigation check classifications."""
    quality = audit_data_quality(sample_transaction_df)
    assert isinstance(quality, pd.DataFrame)
    assert "check_name" in quality.columns
    assert "severity_classification" in quality.columns

    # Check severity classes exist
    severities = set(quality["severity_classification"])
    expected_severities = {
        "Confirmed invalid records",
        "Potentially suspicious records",
        "Expected missing values",
        "Records requiring further investigation",
    }
    assert expected_severities.issubset(severities)
