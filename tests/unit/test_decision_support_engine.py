"""Unit tests for TARANG Stage 5 Decision Support Engine.

Validates:
- Single-session advisory generation from arrival parameters and pre-built features.
- SessionAdvisory dataclass immutability.
- Numeric validity checks for non-negativity and finiteness.
- Projected unplug timestamp calculations in UTC.
- Assignment of pd.NaT when duration predictions are negative or non-finite.
- Batch advisories schema, column retention, and index alignment (including duplicate indices).
- Input error handling (missing StartTime, empty DataFrame, invalid types).
- Strict arrival timestamp validation (NaT, null, malformed strings rejected before model invocation).
- Support for naive UTC timestamps and timezone-aware timestamps converted to UTC.
- Regression testing for target leakage invariance.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import pytest

from tarang.config import FEATURE_NAMES, TARGET_DURATION, TARGET_ENERGY
from tarang.decision_support.engine import (
    DecisionSupportEngine,
    SessionAdvisory,
    calculate_projected_unplug_time,
    generate_batch_advisories,
    generate_session_advisory,
    is_prediction_valid,
)


class MockEstimator:
    """Mock regression estimator returning deterministic predictions."""

    def __init__(self, return_val: float) -> None:
        self.return_val = return_val
        self.call_count = 0

    def predict(self, X: Any) -> np.ndarray:
        self.call_count += 1
        n_samples = len(X)
        return np.full(n_samples, self.return_val, dtype=np.float64)


@pytest.fixture
def mock_models() -> dict[str, Any]:
    """Provide mock models returning standard positive values."""
    return {
        "duration": MockEstimator(return_val=8.5),
        "energy": MockEstimator(return_val=22.0),
    }


@pytest.fixture
def sample_feature_df() -> pd.DataFrame:
    """Provide a minimal valid feature DataFrame with StartTime and 13 predictors."""
    row = {
        "StartTime": "2018-05-15 18:00:00",
        "weekday": 1,
        "weekend": 0,
        "holiday": 0,
        "month": 5,
        "StartCos": 0.5,
        "StartSin": 0.866,
        "StartHourNorm": 0.75,
        "TimeSinceLastStop": 10.0,
        "sessionsToday": 0,
        "LastDuration": 6.0,
        "LastConsumedkWh": 15.0,
        "CarKW": 7.0,
        "CarKWh": 40.0,
        "extra_metadata_col": "preserve_me",
    }
    return pd.DataFrame([row])


def test_is_prediction_valid() -> None:
    """Test non-negativity and finiteness validation."""
    assert is_prediction_valid(0.0) is True
    assert is_prediction_valid(12.5) is True
    assert is_prediction_valid("5.2") is True
    assert is_prediction_valid(-0.01) is False
    assert is_prediction_valid(-10.0) is False
    assert is_prediction_valid(np.nan) is False
    assert is_prediction_valid(float("nan")) is False
    assert is_prediction_valid(float("inf")) is False
    assert is_prediction_valid(float("-inf")) is False
    assert is_prediction_valid(None) is False
    assert is_prediction_valid("invalid_string") is False


def test_calculate_projected_unplug_time() -> None:
    """Test projected unplug timestamp calculation and NaT on invalid duration."""
    start = pd.Timestamp("2018-05-15 18:00:00", tz="UTC")

    # Valid duration
    unplug = calculate_projected_unplug_time(start, 4.5, is_valid=True)
    assert unplug == pd.Timestamp("2018-05-15 22:30:00", tz="UTC")

    # Invalid duration -> pd.NaT
    unplug_invalid = calculate_projected_unplug_time(start, -2.0, is_valid=False)
    assert pd.isna(unplug_invalid)


def test_session_advisory_immutability(mock_models: dict[str, Any], sample_feature_df: pd.DataFrame) -> None:
    """Verify that SessionAdvisory dataclass is frozen and cannot be mutated."""
    engine = DecisionSupportEngine(
        duration_model=mock_models["duration"],
        energy_model=mock_models["energy"],
    )
    advisory = engine.predict_single_advisory(features=sample_feature_df)

    assert advisory.predicted_duration_hours == 8.5
    assert advisory.predicted_energy_kwh == 22.0
    assert advisory.duration_prediction_valid is True
    assert advisory.energy_prediction_valid is True
    assert advisory.projected_unplug_time_utc == pd.Timestamp("2018-05-16 02:30:00", tz="UTC")

    with pytest.raises(FrozenInstanceError):
        advisory.predicted_duration_hours = 10.0  # type: ignore


def test_predict_single_advisory_invalid_duration(sample_feature_df: pd.DataFrame) -> None:
    """Verify that an estimator predicting negative duration flags invalidity and yields NaT."""
    negative_dur_model = MockEstimator(return_val=-3.0)
    valid_energy_model = MockEstimator(return_val=15.0)

    engine = DecisionSupportEngine(
        duration_model=negative_dur_model,
        energy_model=valid_energy_model,
    )
    advisory = engine.predict_single_advisory(features=sample_feature_df)

    # Raw negative prediction is preserved without clipping
    assert advisory.predicted_duration_hours == -3.0
    assert advisory.duration_prediction_valid is False
    assert advisory.energy_prediction_valid is True
    assert pd.isna(advisory.projected_unplug_time_utc)


def test_predict_single_advisory_from_history(mock_models: dict[str, Any]) -> None:
    """Test generating single advisory directly from arrival params and participant history."""
    engine = DecisionSupportEngine(
        duration_model=mock_models["duration"],
        energy_model=mock_models["energy"],
        max_start_hour=23.95,
    )

    history = pd.DataFrame(
        [
            {
                "StartTime": "2018-05-14 10:00:00",
                "StopTime": "2018-05-14 16:00:00",
                "ConsumedkWh": 18.0,
            }
        ]
    )

    advisory = engine.predict_single_advisory(
        start_time="2018-05-15 18:00:00",
        car_kw=7.0,
        car_kwh=40.0,
        history=history,
    )

    assert isinstance(advisory, SessionAdvisory)
    assert advisory.arrival_time_utc == pd.Timestamp("2018-05-15 18:00:00", tz="UTC")
    assert advisory.predicted_duration_hours == 8.5
    assert advisory.predicted_energy_kwh == 22.0
    assert advisory.duration_prediction_valid is True
    assert advisory.energy_prediction_valid is True
    assert advisory.projected_unplug_time_utc == pd.Timestamp("2018-05-16 02:30:00", tz="UTC")


def test_predict_batch_advisories_columns_and_index(mock_models: dict[str, Any]) -> None:
    """Test batch advisory generation preserves input columns, handles duplicate index, and assigns outputs."""
    engine = DecisionSupportEngine(
        duration_model=mock_models["duration"],
        energy_model=mock_models["energy"],
    )

    # DataFrame with duplicate indices and non-consecutive labels
    rows = [
        {
            "StartTime": "2018-05-15 10:00:00",
            "weekday": 1, "weekend": 0, "holiday": 0, "month": 5,
            "StartCos": 0.5, "StartSin": 0.866, "StartHourNorm": 0.4,
            "TimeSinceLastStop": 8.0, "sessionsToday": 0, "LastDuration": 4.0,
            "LastConsumedkWh": 10.0, "CarKW": 7.0, "CarKWh": 40.0,
            "metadata_id": "session_A",
        },
        {
            "StartTime": "2018-05-15 18:00:00",
            "weekday": 1, "weekend": 0, "holiday": 0, "month": 5,
            "StartCos": -0.5, "StartSin": 0.866, "StartHourNorm": 0.75,
            "TimeSinceLastStop": 12.0, "sessionsToday": 1, "LastDuration": 2.0,
            "LastConsumedkWh": 8.0, "CarKW": 7.0, "CarKWh": 40.0,
            "metadata_id": "session_B",
        },
    ]
    df = pd.DataFrame(rows, index=[42, 42])

    result = engine.predict_batch_advisories(df)

    # Check index preservation
    assert list(result.index) == [42, 42]

    # Check input column preservation
    assert "metadata_id" in result.columns
    assert list(result["metadata_id"]) == ["session_A", "session_B"]

    # Check output columns
    expected_new_cols = [
        "predicted_Duration",
        "predicted_ConsumedkWh",
        "projected_unplug_time_utc",
        "duration_prediction_valid",
        "energy_prediction_valid",
    ]
    for col in expected_new_cols:
        assert col in result.columns

    assert (result["predicted_Duration"] == 8.5).all()
    assert (result["predicted_ConsumedkWh"] == 22.0).all()
    assert (result["duration_prediction_valid"] == True).all()
    assert (result["energy_prediction_valid"] == True).all()

    expected_unplug_0 = pd.Timestamp("2018-05-15 18:30:00", tz="UTC")
    expected_unplug_1 = pd.Timestamp("2018-05-16 02:30:00", tz="UTC")
    assert result["projected_unplug_time_utc"].iloc[0] == expected_unplug_0
    assert result["projected_unplug_time_utc"].iloc[1] == expected_unplug_1


def test_predict_batch_advisories_with_invalid_rows() -> None:
    """Test batch advisories correctly handle mixed valid/invalid model outputs."""
    class MixedDurationModel:
        def predict(self, X: Any) -> np.ndarray:
            return np.array([5.0, -1.0, np.nan])

    class MixedEnergyModel:
        def predict(self, X: Any) -> np.ndarray:
            return np.array([12.0, 15.0, -4.0])

    engine = DecisionSupportEngine(
        duration_model=MixedDurationModel(),
        energy_model=MixedEnergyModel(),
    )

    rows = [
        {
            "StartTime": f"2018-05-15 1{i}:00:00",
            "weekday": 1, "weekend": 0, "holiday": 0, "month": 5,
            "StartCos": 0.5, "StartSin": 0.866, "StartHourNorm": 0.4,
            "TimeSinceLastStop": 8.0, "sessionsToday": 0, "LastDuration": 4.0,
            "LastConsumedkWh": 10.0, "CarKW": 7.0, "CarKWh": 40.0,
        }
        for i in range(3)
    ]
    df = pd.DataFrame(rows)

    result = engine.predict_batch_advisories(df)

    assert result["duration_prediction_valid"].tolist() == [True, False, False]
    assert result["energy_prediction_valid"].tolist() == [True, True, False]

    # Row 0 valid -> timestamp populated
    assert result["projected_unplug_time_utc"].iloc[0] == pd.Timestamp("2018-05-15 15:00:00", tz="UTC")
    # Rows 1 and 2 invalid duration -> NaT
    assert pd.isna(result["projected_unplug_time_utc"].iloc[1])
    assert pd.isna(result["projected_unplug_time_utc"].iloc[2])


def test_batch_advisories_error_handling(mock_models: dict[str, Any]) -> None:
    """Test error handling when StartTime is missing or DataFrame is empty."""
    engine = DecisionSupportEngine(
        duration_model=mock_models["duration"],
        energy_model=mock_models["energy"],
    )

    # Empty DataFrame
    with pytest.raises(ValueError, match="empty"):
        engine.predict_batch_advisories(pd.DataFrame())

    # Missing StartTime
    df_no_start = pd.DataFrame([{col: 1.0 for col in FEATURE_NAMES}])
    with pytest.raises(ValueError, match="StartTime"):
        engine.predict_batch_advisories(df_no_start)


def test_batch_timestamp_validation_rejects_invalid_before_model_invocation() -> None:
    """Verify that NaT, null, and malformed timestamps are rejected BEFORE calling models."""
    mock_dur = MockEstimator(return_val=5.0)
    mock_ene = MockEstimator(return_val=10.0)
    engine = DecisionSupportEngine(
        duration_model=mock_dur,
        energy_model=mock_ene,
    )

    base_row = {col: 1.0 for col in FEATURE_NAMES}

    # Case 1: StartTime = pd.NaT
    df_nat = pd.DataFrame([{**base_row, "StartTime": pd.NaT}])
    with pytest.raises(ValueError, match="missing or null/NaT"):
        engine.predict_batch_advisories(df_nat)
    assert mock_dur.call_count == 0
    assert mock_ene.call_count == 0

    # Case 2: StartTime = None
    df_none = pd.DataFrame([{**base_row, "StartTime": None}])
    with pytest.raises(ValueError, match="missing or null/NaT"):
        engine.predict_batch_advisories(df_none)
    assert mock_dur.call_count == 0
    assert mock_ene.call_count == 0

    # Case 3: StartTime = malformed string
    df_malformed = pd.DataFrame([{**base_row, "StartTime": "not-a-timestamp-12345"}])
    with pytest.raises(ValueError, match="Failed to parse 'StartTime'"):
        engine.predict_batch_advisories(df_malformed)
    assert mock_dur.call_count == 0
    assert mock_ene.call_count == 0


def test_batch_timestamp_valid_formats(mock_models: dict[str, Any]) -> None:
    """Test valid naive and timezone-aware timestamps are parsed into UTC."""
    engine = DecisionSupportEngine(
        duration_model=mock_models["duration"],
        energy_model=mock_models["energy"],
    )

    base_row = {col: 1.0 for col in FEATURE_NAMES}

    # 1. Naive string treated as UTC
    df_naive = pd.DataFrame([{**base_row, "StartTime": "2018-05-15 18:00:00"}])
    res_naive = engine.predict_batch_advisories(df_naive)
    assert res_naive["projected_unplug_time_utc"].iloc[0] == pd.Timestamp("2018-05-16 02:30:00", tz="UTC")

    # 2. Timezone-aware timestamp (Europe/London BST UTC+1: 11:00 BST -> 10:00 UTC)
    london_ts = pd.Timestamp("2018-06-15 11:00:00", tz="Europe/London")
    df_aware = pd.DataFrame([{**base_row, "StartTime": london_ts}])
    res_aware = engine.predict_batch_advisories(df_aware)
    # 10:00 UTC + 8.5h duration = 18:30 UTC
    assert res_aware["projected_unplug_time_utc"].iloc[0] == pd.Timestamp("2018-06-15 18:30:00", tz="UTC")


def test_target_leakage_invariance() -> None:
    """Verify that predictions and unplug timings are invariant to actual outcomes (StopTime, ConsumedkWh, ChargingDuration)."""
    # Deterministic model whose prediction depends on LastDuration and LastConsumedkWh
    class FeatureBasedEstimator:
        def predict(self, X: Any) -> np.ndarray:
            return (X["LastDuration"].to_numpy() * 1.5 + X["CarKW"].to_numpy() * 0.5).astype(np.float64)

    engine = DecisionSupportEngine(
        duration_model=FeatureBasedEstimator(),
        energy_model=FeatureBasedEstimator(),
    )

    base_features = {
        "StartTime": "2018-05-15 18:00:00",
        "weekday": 1, "weekend": 0, "holiday": 0, "month": 5,
        "StartCos": 0.5, "StartSin": 0.866, "StartHourNorm": 0.75,
        "TimeSinceLastStop": 10.0, "sessionsToday": 0, "LastDuration": 6.0,
        "LastConsumedkWh": 15.0, "CarKW": 7.0, "CarKWh": 40.0,
    }

    # DataFrame A has actual session outcomes
    df_a = pd.DataFrame([{
        **base_features,
        "StopTime": "2018-05-16 04:00:00+00:00",
        "ConsumedkWh": 25.5,
        "ChargingDuration": 180.0,
    }])

    # DataFrame B has drastically different actual outcomes (e.g. 10x energy, 4 days stop time)
    df_b = pd.DataFrame([{
        **base_features,
        "StopTime": "2018-05-20 12:00:00+00:00",
        "ConsumedkWh": 999.9,
        "ChargingDuration": 1500.0,
    }])

    res_a = engine.predict_batch_advisories(df_a)
    res_b = engine.predict_batch_advisories(df_b)

    # Verify identical duration and energy predictions
    assert np.isclose(res_a["predicted_Duration"].iloc[0], res_b["predicted_Duration"].iloc[0])
    assert np.isclose(res_a["predicted_ConsumedkWh"].iloc[0], res_b["predicted_ConsumedkWh"].iloc[0])

    # Verify identical projected unplug timestamps
    assert res_a["projected_unplug_time_utc"].iloc[0] == res_b["projected_unplug_time_utc"].iloc[0]

    # Verify identical validity flags
    assert res_a["duration_prediction_valid"].iloc[0] == res_b["duration_prediction_valid"].iloc[0]
    assert res_a["energy_prediction_valid"].iloc[0] == res_b["energy_prediction_valid"].iloc[0]
