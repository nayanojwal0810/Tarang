"""Unit tests for TARANG inference engine and model loading.

Validates:
- Target resolution and models directory routing
- Model loading, caching, and missing artifact error messaging
- Target-specific LastValue lag configuration
- Feature schema validation (13 canonical features, dtypes, non-finite rejection)
- Equivalence between single and batch predictions
- Index and row alignment preservation
- Isolation from non-predictor metadata columns
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from tarang.config import FEATURE_NAMES, MODEL_NAMES, TARGET_DURATION, TARGET_ENERGY
from tarang.inference.engine import (
    InferenceEngine,
    predict_batch,
    predict_single,
)
from tarang.inference.loading import (
    clear_model_cache,
    get_target_models_dir,
    load_model,
    normalize_target_name,
)
from tarang.inference.validation import validate_inference_features
from tarang.modeling.models import LastValue, MeanValue, MedianValue


@pytest.fixture(autouse=True)
def reset_cache() -> None:
    """Clear model cache before and after each test."""
    clear_model_cache()
    yield
    clear_model_cache()


@pytest.fixture
def sample_feature_row() -> pd.DataFrame:
    """Provide a single valid 13-feature DataFrame."""
    return pd.DataFrame(
        [
            {
                "weekday": 1,
                "weekend": 0,
                "holiday": 0,
                "month": 5,
                "StartCos": 0.5,
                "StartSin": 0.866,
                "StartHourNorm": 0.25,
                "TimeSinceLastStop": 12.0,
                "sessionsToday": 1,
                "LastDuration": 8.5,
                "LastConsumedkWh": 15.0,
                "CarKW": 7.0,
                "CarKWh": 40.0,
            }
        ]
    )


@pytest.fixture
def sample_batch_features(sample_feature_row: pd.DataFrame) -> pd.DataFrame:
    """Provide a multi-row DataFrame with custom index and extra metadata."""
    row1 = sample_feature_row.iloc[0].to_dict()
    row2 = dict(row1)
    row2["LastDuration"] = 4.0
    row2["LastConsumedkWh"] = 25.0
    row3 = dict(row1)
    row3["LastDuration"] = 14.0
    row3["LastConsumedkWh"] = 35.0

    df = pd.DataFrame([row1, row2, row3], index=[101, 102, 103])
    # Add non-predictor metadata columns
    df["ParticipantID"] = ["P01", "P01", "P02"]
    df["TransactionID"] = [1, 2, 3]
    return df


@pytest.fixture
def mock_models_dir(tmp_path: Path) -> Path:
    """Create a temporary models directory structure with mock fitted estimators."""
    duration_dir = tmp_path / "duration" / "models"
    energy_dir = tmp_path / "energy" / "models"
    duration_dir.mkdir(parents=True, exist_ok=True)
    energy_dir.mkdir(parents=True, exist_ok=True)

    # Dummy fitted LinearRegression
    X_dummy = pd.DataFrame(np.ones((5, 13)), columns=FEATURE_NAMES)
    y_duration = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    y_energy = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

    lr_dur = LinearRegression().fit(X_dummy, y_duration)
    lr_ene = LinearRegression().fit(X_dummy, y_energy)
    joblib.dump(lr_dur, duration_dir / "LinearRegression.joblib")
    joblib.dump(lr_ene, energy_dir / "LinearRegression.joblib")

    # Median and Mean models
    med_dur = MedianValue().fit(X_dummy, y_duration)
    joblib.dump(med_dur, duration_dir / "MedianValue.joblib")

    mean_dur = MeanValue().fit(X_dummy, y_duration)
    joblib.dump(mean_dur, duration_dir / "MeanValue.joblib")

    # LastValue models with target-specific lag features
    lv_dur = LastValue(lag_feature="LastDuration").fit(X_dummy, y_duration)
    lv_ene = LastValue(lag_feature="LastConsumedkWh").fit(X_dummy, y_energy)
    joblib.dump(lv_dur, duration_dir / "LastValue.joblib")
    joblib.dump(lv_ene, energy_dir / "LastValue.joblib")

    return tmp_path


def test_target_normalization() -> None:
    """Verify target name normalization across case and alias variations."""
    assert normalize_target_name("Duration") == TARGET_DURATION
    assert normalize_target_name("duration") == TARGET_DURATION
    assert normalize_target_name("DURATION") == TARGET_DURATION
    assert normalize_target_name("ConsumedkWh") == TARGET_ENERGY
    assert normalize_target_name("consumedkwh") == TARGET_ENERGY
    assert normalize_target_name("energy") == TARGET_ENERGY
    assert normalize_target_name("Energy") == TARGET_ENERGY

    with pytest.raises(ValueError, match="Unknown target"):
        normalize_target_name("invalid_target")


def test_model_loading_and_caching(mock_models_dir: Path) -> None:
    """Verify loading from disk, in-memory caching, and cache clearing."""
    m1 = load_model(
        target=TARGET_DURATION,
        model_name="LinearRegression",
        models_dir=mock_models_dir,
        use_cache=True,
    )
    m2 = load_model(
        target=TARGET_DURATION,
        model_name="LinearRegression",
        models_dir=mock_models_dir,
        use_cache=True,
    )
    # Must be the exact same cached object in memory
    assert m1 is m2

    # After clearing cache, loads a new object
    clear_model_cache()
    m3 = load_model(
        target=TARGET_DURATION,
        model_name="LinearRegression",
        models_dir=mock_models_dir,
        use_cache=True,
    )
    assert m3 is not m1


def test_missing_model_artifact_error(mock_models_dir: Path) -> None:
    """Verify clear FileNotFoundError when a model artifact does not exist."""
    with pytest.raises(FileNotFoundError, match="Model artifact not found"):
        load_model(
            target=TARGET_DURATION,
            model_name="QuantileRegressor",
            models_dir=mock_models_dir,
        )


def test_unknown_model_name_rejection(mock_models_dir: Path) -> None:
    """Verify ValueError when requesting an unregistered model architecture."""
    with pytest.raises(ValueError, match="Unknown model_name"):
        load_model(
            target=TARGET_DURATION,
            model_name="RandomForestRegressor",
            models_dir=mock_models_dir,
        )


def test_target_specific_last_value_behavior(
    mock_models_dir: Path, sample_feature_row: pd.DataFrame
) -> None:
    """Verify LastValue uses LastDuration for Duration and LastConsumedkWh for Energy."""
    pred_dur = predict_single(
        features=sample_feature_row,
        target=TARGET_DURATION,
        model_name="LastValue",
        models_dir=mock_models_dir,
    )
    assert pred_dur == sample_feature_row["LastDuration"].iloc[0]

    pred_ene = predict_single(
        features=sample_feature_row,
        target=TARGET_ENERGY,
        model_name="LastValue",
        models_dir=mock_models_dir,
    )
    assert pred_ene == sample_feature_row["LastConsumedkWh"].iloc[0]


def test_single_and_batch_prediction_consistency(
    mock_models_dir: Path, sample_batch_features: pd.DataFrame
) -> None:
    """Verify that predict_single and predict_batch yield identical results."""
    # Batch prediction
    batch_preds = predict_batch(
        features_df=sample_batch_features,
        target=TARGET_DURATION,
        model_name="LinearRegression",
        models_dir=mock_models_dir,
    )

    # Check series properties
    assert isinstance(batch_preds, pd.Series)
    assert list(batch_preds.index) == list(sample_batch_features.index)
    assert batch_preds.name == "predicted_Duration"

    # Single predictions for each row individually
    for idx in sample_batch_features.index:
        single_row = sample_batch_features.loc[[idx]]
        single_pred = predict_single(
            features=single_row,
            target=TARGET_DURATION,
            model_name="LinearRegression",
            models_dir=mock_models_dir,
        )
        assert pytest.approx(batch_preds.loc[idx], abs=1e-6) == single_pred


def test_inference_engine_class(
    mock_models_dir: Path, sample_feature_row: pd.DataFrame, sample_batch_features: pd.DataFrame
) -> None:
    """Verify InferenceEngine class methods."""
    engine = InferenceEngine(
        target=TARGET_DURATION,
        model_name="LinearRegression",
        models_dir=mock_models_dir,
    )
    single_res = engine.predict_single(sample_feature_row)
    batch_res = engine.predict_batch(sample_batch_features)

    assert isinstance(single_res, float)
    assert isinstance(batch_res, pd.Series)
    assert len(batch_res) == len(sample_batch_features)


def test_feature_validation_rejects_missing_column(sample_feature_row: pd.DataFrame) -> None:
    """Verify rejection when any of the 13 canonical features is absent."""
    incomplete = sample_feature_row.drop(columns=["TimeSinceLastStop"])
    with pytest.raises(ValueError, match="Missing required predictor"):
        validate_inference_features(incomplete)


def test_feature_validation_rejects_nulls(sample_feature_row: pd.DataFrame) -> None:
    """Verify rejection when feature values contain null or NaN."""
    with_nan = sample_feature_row.copy()
    with_nan.loc[0, "LastDuration"] = np.nan
    with pytest.raises(ValueError, match="missing/null values"):
        validate_inference_features(with_nan)


def test_feature_validation_rejects_infinite(sample_feature_row: pd.DataFrame) -> None:
    """Verify rejection when feature values contain inf or -inf."""
    with_inf = sample_feature_row.copy()
    with_inf.loc[0, "TimeSinceLastStop"] = np.inf
    with pytest.raises(ValueError, match="non-finite numbers"):
        validate_inference_features(with_inf)


def test_feature_validation_rejects_non_numeric(sample_feature_row: pd.DataFrame) -> None:
    """Verify rejection when feature columns have non-numeric dtype."""
    with_str = sample_feature_row.copy()
    with_str["CarKW"] = "fast"
    with pytest.raises(ValueError, match="non-numeric"):
        validate_inference_features(with_str)


def test_extra_columns_ignored_by_estimator(
    mock_models_dir: Path, sample_batch_features: pd.DataFrame
) -> None:
    """Verify that non-feature metadata columns (e.g. ParticipantID) do not affect predictions."""
    preds_with_meta = predict_batch(
        features_df=sample_batch_features,
        target=TARGET_DURATION,
        model_name="LinearRegression",
        models_dir=mock_models_dir,
    )

    clean_features = sample_batch_features[FEATURE_NAMES]
    preds_without_meta = predict_batch(
        features_df=clean_features,
        target=TARGET_DURATION,
        model_name="LinearRegression",
        models_dir=mock_models_dir,
    )

    pd.testing.assert_series_equal(preds_with_meta, preds_without_meta)

