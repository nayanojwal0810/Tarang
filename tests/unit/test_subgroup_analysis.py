"""Unit tests for TARANG Stage 3 same-day vs overnight subgroup analysis."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from tarang.config import FEATURE_NAMES, TARGET_DURATION
from tarang.evaluation.subgroup_analysis import (
    partition_sessions_by_timing,
    run_subgroup_analysis,
    train_and_evaluate_subgroup_hgbr,
)


@pytest.fixture
def synthetic_timing_sessions():
    """Create synthetic sessions with known same-day and overnight timing."""
    start_same = pd.Timestamp("2017-06-01 08:00:00")
    stop_same = pd.Timestamp("2017-06-01 16:00:00")  # Same day: delta days == 0

    start_overnight = pd.Timestamp("2017-06-01 20:00:00")
    stop_overnight = pd.Timestamp("2017-06-02 07:00:00")  # Overnight: delta days == 1

    records = []
    # 30 same-day sessions
    for i in range(30):
        t_start = start_same + pd.Timedelta(days=i)
        t_stop = stop_same + pd.Timedelta(days=i)
        rec = {
            "StartTime": t_start,
            "StopTime": t_stop,
            "ParticipantID": f"P{i % 3 + 1:02d}",
            "Duration": 8.0,
            "ConsumedkWh": 28.0,
        }
        for feat in FEATURE_NAMES:
            rec[feat] = 1.0
        records.append(rec)

    # 20 overnight sessions
    for i in range(20):
        t_start = start_overnight + pd.Timedelta(days=i)
        t_stop = stop_overnight + pd.Timedelta(days=i)
        rec = {
            "StartTime": t_start,
            "StopTime": t_stop,
            "ParticipantID": f"P{i % 3 + 1:02d}",
            "Duration": 11.0,
            "ConsumedkWh": 38.5,
        }
        for feat in FEATURE_NAMES:
            rec[feat] = 2.0
        records.append(rec)

    df = pd.DataFrame(records).sort_values("StartTime").reset_index(drop=True)
    return df


def test_partition_sessions_by_timing(synthetic_timing_sessions):
    """Test same-day vs overnight session partitioning."""
    df_same, df_over = partition_sessions_by_timing(synthetic_timing_sessions)

    assert len(df_same) == 30
    assert len(df_over) == 20
    assert len(df_same) + len(df_over) == len(synthetic_timing_sessions)

    # Verify no date leakage
    assert ((df_same["StopTime"].dt.normalize() - df_same["StartTime"].dt.normalize()).dt.days == 0).all()
    assert ((df_over["StopTime"].dt.normalize() - df_over["StartTime"].dt.normalize()).dt.days > 0).all()


def test_train_and_evaluate_subgroup_hgbr(synthetic_timing_sessions, tmp_path):
    """Test independent training and metric evaluation for a subgroup."""
    df_same, _ = partition_sessions_by_timing(synthetic_timing_sessions)

    metrics, model, preds = train_and_evaluate_subgroup_hgbr(
        sessions_subgroup=df_same,
        target=TARGET_DURATION,
        subgroup_name="same_day",
        models_dir=tmp_path,
    )

    assert "mae" in metrics
    assert "medae" in metrics
    assert "mape" in metrics
    assert metrics["n_sessions"] == 30
    assert metrics["n_train"] == 24  # 80%
    assert metrics["n_test"] == 6    # 20%

    # Saved model
    saved_model = tmp_path / "HistGradientBoostingRegressor_same_day.joblib"
    assert saved_model.exists()


def test_run_subgroup_analysis(synthetic_timing_sessions, tmp_path):
    """Test full subgroup analysis workflow."""
    df_metrics = run_subgroup_analysis(
        sessions=synthetic_timing_sessions,
        target=TARGET_DURATION,
        output_dir=tmp_path,
        evaluation_data_dir=tmp_path,
    )

    assert isinstance(df_metrics, pd.DataFrame)
    assert len(df_metrics) == 3
    expected_subgroups = ["All Sessions", "Same-day Sessions", "Overnight Sessions"]
    assert list(df_metrics["subgroup"]) == expected_subgroups

    csv_path = tmp_path / "same_day_overnight_metrics.csv"
    assert csv_path.exists()
