"""Integration tests for TARANG inference pipeline.

Validates:
- End-to-end single session feature construction and prediction
- End-to-end batch prediction on CSV and Parquet datasets
- CLI script execution with argument parsing and file export
- Optional tests against local Stage 2 model artifacts when available
"""

from __future__ import annotations

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from scripts.run_inference import main as run_cli_main
from tarang.config import (
    DURATION_MODELING_DIR,
    FEATURE_NAMES,
    PROCESSED_FEATURES_FILE,
    TARGET_DURATION,
    TARGET_ENERGY,
)
from tarang.inference.engine import InferenceEngine, predict_batch, predict_single
from tarang.inference.feature_builder import build_inference_features


@pytest.fixture
def test_environment(tmp_path: Path) -> dict[str, Path]:
    """Set up temporary directory structure with trained mock models and processing summary."""
    duration_models = tmp_path / "duration" / "models"
    energy_models = tmp_path / "energy" / "models"
    duration_models.mkdir(parents=True, exist_ok=True)
    energy_models.mkdir(parents=True, exist_ok=True)

    summary_file = tmp_path / "processing_summary.json"
    summary_file.write_text(json.dumps({"max_start_hour": 23.95}), encoding="utf-8")

    # Fit a simple synthetic HGBR and LinearRegression
    np.random.seed(42)
    X_train = pd.DataFrame(np.random.rand(50, 13) * 10.0, columns=FEATURE_NAMES)
    y_duration = X_train["LastDuration"] * 0.8 + 2.0
    y_energy = X_train["CarKWh"] * 0.5 + 5.0

    hgbr_dur = HistGradientBoostingRegressor(random_state=0).fit(X_train, y_duration)
    lr_dur = LinearRegression().fit(X_train, y_duration)
    hgbr_ene = HistGradientBoostingRegressor(random_state=0).fit(X_train, y_energy)

    joblib.dump(hgbr_dur, duration_models / "HistGradientBoostingRegressor.joblib")
    joblib.dump(lr_dur, duration_models / "LinearRegression.joblib")
    joblib.dump(hgbr_ene, energy_models / "HistGradientBoostingRegressor.joblib")

    return {
        "base_dir": tmp_path,
        "summary_file": summary_file,
        "duration_models": duration_models,
        "energy_models": energy_models,
    }


def test_end_to_end_single_session_flow(test_environment: dict[str, Path]) -> None:
    """Verify feature building + prediction flow for a single charging session."""
    history = pd.DataFrame(
        {
            "StartTime": [
                pd.Timestamp("2018-06-01 07:00:00", tz="UTC"),
                pd.Timestamp("2018-06-01 12:00:00", tz="UTC"),
            ],
            "StopTime": [
                pd.Timestamp("2018-06-01 10:00:00", tz="UTC"),
                pd.Timestamp("2018-06-01 15:30:00", tz="UTC"),
            ],
            "ConsumedkWh": [11.2, 14.5],
        }
    )

    # Build features
    features = build_inference_features(
        start_time="2018-06-01 19:00:00",
        car_kw=7.0,
        car_kwh=40.0,
        history=history,
        summary_file=test_environment["summary_file"],
    )
    assert features.shape == (1, 13)
    assert list(features.columns) == FEATURE_NAMES

    # Predict single
    pred_val = predict_single(
        features=features,
        target=TARGET_DURATION,
        model_name="HistGradientBoostingRegressor",
        models_dir=test_environment["base_dir"],
    )
    assert isinstance(pred_val, float)
    assert np.isfinite(pred_val)

    # Use stateful InferenceEngine
    engine = InferenceEngine(
        target=TARGET_DURATION,
        model_name="HistGradientBoostingRegressor",
        models_dir=test_environment["base_dir"],
        summary_file=test_environment["summary_file"],
    )
    engine_pred = engine.build_and_predict(
        start_time="2018-06-01 19:00:00",
        car_kw=7.0,
        car_kwh=40.0,
        history=history,
    )
    assert pytest.approx(pred_val, abs=1e-6) == engine_pred


def test_cli_batch_inference_csv(test_environment: dict[str, Path], tmp_path: Path) -> None:
    """Verify batch inference CLI tool with CSV input and output."""
    # Create synthetic input CSV
    np.random.seed(123)
    input_data = pd.DataFrame(np.random.rand(10, 13) * 5.0, columns=FEATURE_NAMES)
    input_data["ParticipantID"] = [f"P_{i}" for i in range(10)]
    input_file = tmp_path / "batch_input.csv"
    output_file = tmp_path / "batch_output.csv"
    input_data.to_csv(input_file, index=False)

    # Run CLI
    argv = [
        "--input",
        str(input_file),
        "--output",
        str(output_file),
        "--target",
        "Duration",
        "--model",
        "LinearRegression",
        "--models-dir",
        str(test_environment["base_dir"]),
    ]
    exit_code = run_cli_main(argv)
    assert exit_code == 0

    assert output_file.is_file()
    result_df = pd.read_csv(output_file)
    assert len(result_df) == 10
    assert "predicted_Duration" in result_df.columns
    assert "ParticipantID" in result_df.columns
    assert not result_df["predicted_Duration"].isna().any()


def test_cli_batch_inference_parquet(test_environment: dict[str, Path], tmp_path: Path) -> None:
    """Verify batch inference CLI tool with Parquet input and output."""
    np.random.seed(456)
    input_data = pd.DataFrame(np.random.rand(8, 13) * 5.0, columns=FEATURE_NAMES)
    input_file = tmp_path / "batch_input.parquet"
    output_file = tmp_path / "batch_output.parquet"
    input_data.to_parquet(input_file, index=False)

    argv = [
        "--input",
        str(input_file),
        "--output",
        str(output_file),
        "--target",
        "ConsumedkWh",
        "--model",
        "HistGradientBoostingRegressor",
        "--models-dir",
        str(test_environment["base_dir"]),
    ]
    exit_code = run_cli_main(argv)
    assert exit_code == 0

    assert output_file.is_file()
    result_df = pd.read_parquet(output_file)
    assert len(result_df) == 8
    assert "predicted_ConsumedkWh" in result_df.columns


@pytest.mark.skipif(
    not (DURATION_MODELING_DIR / "models" / "HistGradientBoostingRegressor.joblib").exists(),
    reason="Local Stage 2 trained models not present in environment.",
)
def test_local_model_inference_against_canonical_sample() -> None:
    """Verify inference against real local Stage 2 HGBR model if available."""
    if not PROCESSED_FEATURES_FILE.is_file():
        pytest.skip("Local features.parquet not available.")

    canonical_df = pd.read_parquet(PROCESSED_FEATURES_FILE)
    sample_df = canonical_df.head(5)

    preds_dur = predict_batch(
        features_df=sample_df,
        target=TARGET_DURATION,
        model_name="HistGradientBoostingRegressor",
    )
    assert len(preds_dur) == 5
    assert isinstance(preds_dur, pd.Series)
    assert np.all(np.isfinite(preds_dur.values))

