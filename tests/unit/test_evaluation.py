"""Unit tests for TARANG Stage 3 evaluation and analytical metrics."""

from __future__ import annotations

import datetime as dt
import numpy as np
import pandas as pd
import pytest

from tarang.config import TARGET_DURATION, TARGET_ENERGY
from tarang.evaluation.analysis import (
    compute_data_in_splits,
    compute_error_over_time,
    compute_model_agreement,
    compute_participant_summary_stats,
    compute_seasonality,
    compute_vehicle_characteristics,
    format_performance_markdown_table,
    format_sci,
    generate_performance_table,
)
from tarang.evaluation.loading import normalize_target_name


@pytest.fixture
def sample_performance_data():
    """Create synthetic MultiIndex performance dataset."""
    models = ["HistGradientBoostingRegressor", "LinearRegression", "QuantileRegressor"]
    metrics = ["mae", "medae", "mape", "medape", "smape"]
    cols = []
    for m in models:
        for split in ["Train", "Test"]:
            for score in metrics:
                cols.append((m, split, score))

    col_index = pd.MultiIndex.from_tuples(cols, names=["model", "dataset", "score"])

    # Aggregate row 'ALL'
    all_data = {
        col: 5.0 if "mae" in col[2] or "medae" in col[2] else 25.0
        for col in cols
    }
    perf_all = pd.Series(all_data, name="ALL")

    # Participant rows
    pids = [f"P{i:03d}" for i in range(1, 11)]
    part_matrix = []
    for i in range(10):
        row = [5.0 + i * 0.1 if "mae" in col[2] or "medae" in col[2] else 20.0 + i for col in cols]
        part_matrix.append(row)

    perf_part = pd.DataFrame(part_matrix, index=pids, columns=col_index)
    perf_part.index.name = "ParticipantID"

    return perf_all, perf_part


@pytest.fixture
def sample_predictions_df():
    """Create synthetic predictions DataFrame with datetime columns."""
    n_rows = 100
    dates = pd.date_range("2017-01-01", periods=n_rows, freq="12h")
    pids = [f"P{(i % 5) + 1:03d}" for i in range(n_rows)]

    actual_duration = np.random.uniform(2.0, 14.0, size=n_rows)
    pred_hgbr = actual_duration + np.random.normal(0, 1.0, size=n_rows)
    pred_linear = actual_duration + np.random.normal(0, 1.5, size=n_rows)

    df = pd.DataFrame(
        {
            "StartTime": dates,
            "StopTime": dates + pd.to_timedelta(actual_duration, unit="h"),
            "ParticipantID": pids,
            "Duration": actual_duration,
            "ConsumedkWh": actual_duration * 3.5,
            "CarKWh": 40.0,
            "StartHourNorm": 0.5,
            "dataset": ["Train"] * 70 + ["Test"] * 30,
            "HistGradientBoostingRegressor": pred_hgbr,
            "LinearRegression": pred_linear,
        }
    )
    return df


def test_normalize_target_name():
    """Test target name normalization."""
    assert normalize_target_name("Duration") == TARGET_DURATION
    assert normalize_target_name("duration") == TARGET_DURATION
    assert normalize_target_name("ConsumedkWh") == TARGET_ENERGY
    assert normalize_target_name("energy") == TARGET_ENERGY
    assert normalize_target_name("consumption") == TARGET_ENERGY

    with pytest.raises(ValueError):
        normalize_target_name("unsupported_target")


def test_format_sci():
    """Test 3-sig-fig scientific number formatter."""
    assert format_sci(5.5261) == "5.53"
    assert format_sci(201.2) == "201"
    assert format_sci(0.0423) == "0.0423"
    assert format_sci(0.0) == "0"
    assert format_sci(np.nan) == "N/A"


def test_generate_performance_table(sample_performance_data):
    """Test benchmark performance table structure and content."""
    perf_all, perf_part = sample_performance_data
    table = generate_performance_table(perf_part, perf_all, dataset="Test")

    assert isinstance(table, pd.DataFrame)
    assert ("All data", "MAE") in table.columns
    assert ("Per vehicle mean", "MdAPE") in table.columns
    assert "HistGradientBoostingRegressor" in table.index

    md_str = format_performance_markdown_table(table)
    assert isinstance(md_str, str)
    assert "| Model |" in md_str
    assert "HistGradientBoostingRegressor" in md_str


def test_compute_participant_summary_stats(sample_performance_data):
    """Test participant metric distribution quantiles."""
    _, perf_part = sample_performance_data
    stats = compute_participant_summary_stats(perf_part, dataset="Test")

    assert "mae" in stats.index
    assert "mean" in stats.columns
    assert "median" in stats.columns
    assert stats.loc["mae", "mean"] > 0


def test_compute_vehicle_characteristics(sample_performance_data, sample_predictions_df):
    """Test user charging characteristics calculation and OLS regression."""
    _, perf_part = sample_performance_data
    reg_results = compute_vehicle_characteristics(
        perf_part, sample_predictions_df, model_name="HistGradientBoostingRegressor", dataset="Test"
    )

    assert "duration_variability" in reg_results
    assert "battery_size" in reg_results
    assert "start_hour" in reg_results

    for key in ["duration_variability", "battery_size", "start_hour"]:
        assert "slope" in reg_results[key]
        assert "p_value" in reg_results[key]
        assert "x" in reg_results[key]


def test_compute_model_agreement(sample_performance_data):
    """Test pairwise model agreement calculations."""
    _, perf_part = sample_performance_data
    agreement = compute_model_agreement(perf_part, dataset="Test", score="mae")

    assert len(agreement) > 0
    first_pair = list(agreement.keys())[0]
    assert "pearson_r" in agreement[first_pair]
    assert "mean_abs_diff" in agreement[first_pair]


def test_compute_error_over_time(sample_predictions_df):
    """Test temporal error shift across test split."""
    eot = compute_error_over_time(
        sample_predictions_df, target=TARGET_DURATION, model_name="HistGradientBoostingRegressor", dataset="Test"
    )

    assert "relative_changes" in eot
    assert "mean_relative_change" in eot
    assert "std_relative_change" in eot
    assert isinstance(eot["relative_changes"], np.ndarray)


def test_compute_seasonality(sample_predictions_df):
    """Test monthly seasonality calculation."""
    season = compute_seasonality(
        sample_predictions_df, target=TARGET_DURATION, model_name="HistGradientBoostingRegressor"
    )

    assert "Train" in season
    assert "Test" in season
    assert len(season["Train"]["dates"]) > 0
    assert len(season["Train"]["means"]) > 0


def test_compute_data_in_splits(sample_predictions_df):
    """Test session count and timespan availability in splits."""
    availability = compute_data_in_splits(sample_predictions_df)

    assert "Train" in availability
    assert "Test" in availability
    assert availability["Train"]["session_count_mean"] > 0
    assert availability["Test"]["timespan_days_mean"] >= 0
