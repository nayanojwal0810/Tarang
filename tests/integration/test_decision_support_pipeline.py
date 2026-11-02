"""Integration tests for TARANG Stage 5 Decision Support pipeline.

Validates:
- End-to-end execution of DecisionSupportEngine with artifact persistence.
- Output generation of session advisories (.parquet / .csv), prediction_summary.csv,
  flexibility_summary.csv, and decision_support_summary.md.
- CLI execution via scripts/run_decision_support.py with argument parsing.
- Strict enforcement of supported file formats (.parquet and .csv, case-insensitive).
- Correct handling when input dataset contains or lacks historical flexibility columns.
- End-to-end verification against local Stage 2 models and canonical dataset if present.
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

from scripts.run_decision_support import main as run_cli_main
from tarang.config import (
    DECISION_SUPPORT_REPORTS_DIR,
    DEFAULT_SESSION_ADVISORIES_FILE,
    FEATURE_NAMES,
    PROCESSED_FEATURES_FILE,
    TARGET_DURATION,
    TARGET_ENERGY,
)
from tarang.decision_support.engine import DecisionSupportEngine
from tarang.decision_support.flexibility import analyze_charging_flexibility
from tarang.decision_support.reporting import (
    generate_decision_support_markdown_report,
    generate_flexibility_summary_table,
    generate_prediction_summary_table,
    save_decision_support_artifacts,
)


@pytest.fixture
def mock_environment(tmp_path: Path) -> dict[str, Path]:
    """Set up temporary directory structure with trained mock models and synthetic dataset."""
    duration_models = tmp_path / "duration" / "models"
    energy_models = tmp_path / "energy" / "models"
    duration_models.mkdir(parents=True, exist_ok=True)
    energy_models.mkdir(parents=True, exist_ok=True)

    # Train simple linear models for duration and energy
    np.random.seed(42)
    n_samples = 60
    X_train = pd.DataFrame(np.random.rand(n_samples, 13) * 10.0, columns=FEATURE_NAMES)
    y_duration = X_train["LastDuration"] * 0.8 + 2.0
    y_energy = X_train["CarKWh"] * 0.5 + 5.0

    lr_dur = LinearRegression().fit(X_train, y_duration)
    lr_ene = LinearRegression().fit(X_train, y_energy)
    hgbr_dur = HistGradientBoostingRegressor(random_state=0).fit(X_train, y_duration)
    hgbr_ene = HistGradientBoostingRegressor(random_state=0).fit(X_train, y_energy)

    joblib.dump(lr_dur, duration_models / "LinearRegression.joblib")
    joblib.dump(lr_ene, energy_models / "LinearRegression.joblib")
    joblib.dump(hgbr_dur, duration_models / "HistGradientBoostingRegressor.joblib")
    joblib.dump(hgbr_ene, energy_models / "HistGradientBoostingRegressor.joblib")

    # Create synthetic test dataset
    df_test = X_train.copy()
    start_times = pd.date_range("2018-05-01 08:00:00", periods=n_samples, freq="4h", tz="UTC")
    df_test["StartTime"] = start_times
    df_test["StopTime"] = start_times + pd.Timedelta(hours=8)
    df_test["ChargingDuration"] = 180.0  # 3 hours active charging

    input_file = tmp_path / "test_features.parquet"
    df_test.to_parquet(input_file, index=False)

    return {
        "base_dir": tmp_path,
        "input_file": input_file,
        "duration_models": duration_models,
        "energy_models": energy_models,
    }


def test_end_to_end_decision_support_engine(mock_environment: dict[str, Path]) -> None:
    """Test full decision support engine execution, metric calculation, and artifact persistence."""
    input_file = mock_environment["input_file"]
    base_dir = mock_environment["base_dir"]
    report_dir = base_dir / "reports"
    output_advisories = base_dir / "advisories.parquet"

    df = pd.read_parquet(input_file)

    engine = DecisionSupportEngine(
        model_name="LinearRegression",
        models_dir=base_dir,
    )
    advisories = engine.predict_batch_advisories(df)

    assert len(advisories) == len(df)
    assert "predicted_Duration" in advisories.columns
    assert "predicted_ConsumedkWh" in advisories.columns
    assert "projected_unplug_time_utc" in advisories.columns
    assert "duration_prediction_valid" in advisories.columns
    assert "energy_prediction_valid" in advisories.columns

    flex_stats = analyze_charging_flexibility(df)
    assert flex_stats["available"] is True

    # Persist artifacts
    artifacts = save_decision_support_artifacts(
        advisories_df=advisories,
        flexibility_stats=flex_stats,
        output_advisories_path=output_advisories,
        report_dir=report_dir,
        model_name="LinearRegression",
        source_file=input_file,
    )

    assert artifacts["advisories"].is_file()
    assert artifacts["prediction_summary"].is_file()
    assert artifacts["flexibility_summary"].is_file()
    assert artifacts["report"].is_file()

    # Verify report content contains updated factual headers
    report_text = artifacts["report"].read_text(encoding="utf-8")
    assert "TARANG — Stage 5: Decision Support Summary Report" in report_text
    assert "Predictive Session Advisories Distribution" in report_text
    assert "Smart-Charging Flexibility Ratio Analysis" in report_text
    assert "Valid Ratios" in report_text
    assert "Ratios in [0, 1] Histogram" in report_text
    assert "Mean Ratio (All Valid)" in report_text
    assert "Median Ratio (All Valid)" in report_text
    assert "Descriptive Flexibility Observations" in report_text
    assert "Operational Boundaries & Non-Dispatch Scope" in report_text
    assert "LinearRegression" in report_text


def test_cli_execution_synthetic(mock_environment: dict[str, Path]) -> None:
    """Test CLI script execution via run_cli_main."""
    input_file = mock_environment["input_file"]
    base_dir = mock_environment["base_dir"]
    report_dir = base_dir / "cli_reports"
    output_advisories = base_dir / "cli_advisories.parquet"

    cli_args = [
        "--input", str(input_file),
        "--output", str(output_advisories),
        "--report-dir", str(report_dir),
        "--model", "HistGradientBoostingRegressor",
        "--models-dir", str(base_dir),
    ]

    ret = run_cli_main(cli_args)
    assert ret == 0

    assert output_advisories.is_file()
    assert (report_dir / "prediction_summary.csv").is_file()
    assert (report_dir / "flexibility_summary.csv").is_file()
    assert (report_dir / "decision_support_summary.md").is_file()


def test_cli_unsupported_file_formats(mock_environment: dict[str, Path], tmp_path: Path) -> None:
    """Verify CLI rejects unsupported input and output extensions."""
    base_dir = mock_environment["base_dir"]
    valid_input = mock_environment["input_file"]

    # Case 1: Unsupported input extension (.txt)
    unsupported_in = tmp_path / "data.txt"
    unsupported_in.write_text("dummy", encoding="utf-8")
    ret_in = run_cli_main([
        "--input", str(unsupported_in),
        "--output", str(tmp_path / "out.parquet"),
        "--models-dir", str(base_dir),
    ])
    assert ret_in == 1

    # Case 2: Unsupported output extension (.json)
    ret_out = run_cli_main([
        "--input", str(valid_input),
        "--output", str(tmp_path / "out.json"),
        "--models-dir", str(base_dir),
    ])
    assert ret_out == 1


def test_save_artifacts_unsupported_format(mock_environment: dict[str, Path], tmp_path: Path) -> None:
    """Verify save_decision_support_artifacts rejects unsupported output formats."""
    base_dir = mock_environment["base_dir"]
    df = pd.read_parquet(mock_environment["input_file"])
    engine = DecisionSupportEngine(model_name="LinearRegression", models_dir=base_dir)
    advisories = engine.predict_batch_advisories(df)

    with pytest.raises(ValueError, match="Unsupported output advisories file format"):
        save_decision_support_artifacts(
            advisories_df=advisories,
            output_advisories_path=tmp_path / "out.xlsx",
            report_dir=tmp_path / "reports",
        )


def test_cli_case_insensitive_supported_extensions(mock_environment: dict[str, Path], tmp_path: Path) -> None:
    """Verify uppercase extensions (.PARQUET, .CSV) work cleanly in CLI."""
    base_dir = mock_environment["base_dir"]
    orig_input = mock_environment["input_file"]

    # Create uppercase .PARQUET input file
    upper_input = tmp_path / "features.PARQUET"
    upper_input.write_bytes(orig_input.read_bytes())

    upper_output = tmp_path / "advisories.CSV"
    ret = run_cli_main([
        "--input", str(upper_input),
        "--output", str(upper_output),
        "--report-dir", str(tmp_path / "reports"),
        "--models-dir", str(base_dir),
    ])
    assert ret == 0
    assert upper_output.is_file()


def test_decision_support_without_flexibility_columns(mock_environment: dict[str, Path]) -> None:
    """Verify that omitting flexibility columns still generates valid advisories and reports."""
    input_file = mock_environment["input_file"]
    base_dir = mock_environment["base_dir"]
    report_dir = base_dir / "no_flex_reports"

    df = pd.read_parquet(input_file)
    # Drop flexibility columns
    df_no_flex = df.drop(columns=["ChargingDuration", "StopTime"])

    engine = DecisionSupportEngine(
        model_name="LinearRegression",
        models_dir=base_dir,
    )
    advisories = engine.predict_batch_advisories(df_no_flex)
    flex_stats = analyze_charging_flexibility(df_no_flex)

    assert flex_stats["available"] is False
    assert flex_stats["missing_columns"] == ["ChargingDuration", "StopTime"]

    artifacts = save_decision_support_artifacts(
        advisories_df=advisories,
        flexibility_stats=flex_stats,
        report_dir=report_dir,
        model_name="LinearRegression",
    )

    assert artifacts["prediction_summary"].is_file()
    assert "flexibility_summary" not in artifacts
    assert artifacts["report"].is_file()

    report_text = artifacts["report"].read_text(encoding="utf-8")
    assert "Flexibility analysis is unavailable: Missing required historical column(s)" in report_text


@pytest.mark.skipif(
    not PROCESSED_FEATURES_FILE.is_file(),
    reason="Local features.parquet is required for canonical integration test",
)
def test_canonical_dataset_decision_support(tmp_path: Path) -> None:
    """Run decision support against the actual local canonical features dataset."""
    df_canonical = pd.read_parquet(PROCESSED_FEATURES_FILE)
    engine = DecisionSupportEngine(model_name="HistGradientBoostingRegressor")

    # Evaluate on a slice for speed in integration tests
    sample_df = df_canonical.head(200).copy()
    advisories = engine.predict_batch_advisories(sample_df)

    assert len(advisories) == 200
    assert advisories["duration_prediction_valid"].all()
    assert advisories["energy_prediction_valid"].all()

    report_dir = tmp_path / "canonical_reports"
    artifacts = save_decision_support_artifacts(
        advisories_df=advisories,
        flexibility_stats=analyze_charging_flexibility(sample_df),
        report_dir=report_dir,
        model_name="HistGradientBoostingRegressor",
        source_file=PROCESSED_FEATURES_FILE,
    )

    assert artifacts["prediction_summary"].is_file()
    assert artifacts["flexibility_summary"].is_file()
    assert artifacts["report"].is_file()


def test_output_path_collision_guards(mock_environment: dict[str, Path], tmp_path: Path) -> None:
    """Verify both API and CLI reject output paths that overwrite input data or report files."""
    base_dir = mock_environment["base_dir"]
    input_file = mock_environment["input_file"]
    df = pd.read_parquet(input_file)
    engine = DecisionSupportEngine(model_name="LinearRegression", models_dir=base_dir)
    advisories = engine.predict_batch_advisories(df)
    report_dir = tmp_path / "collision_reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    # API Case 1: Overwriting source dataset
    with pytest.raises(ValueError, match="cannot overwrite source dataset"):
        save_decision_support_artifacts(
            advisories_df=advisories,
            output_advisories_path=input_file,
            report_dir=report_dir,
            source_file=input_file,
        )

    # API Case 2: Overwriting generated report summary
    with pytest.raises(ValueError, match="conflicts with generated report file"):
        save_decision_support_artifacts(
            advisories_df=advisories,
            output_advisories_path=report_dir / "prediction_summary.csv",
            report_dir=report_dir,
        )

    # CLI Case 1: --output matches --input
    ret_self = run_cli_main([
        "--input", str(input_file),
        "--output", str(input_file),
        "--models-dir", str(base_dir),
    ])
    assert ret_self == 1

    # CLI Case 2: --output matches report file
    ret_rep = run_cli_main([
        "--input", str(input_file),
        "--output", str(report_dir / "prediction_summary.csv"),
        "--report-dir", str(report_dir),
        "--models-dir", str(base_dir),
    ])
    assert ret_rep == 1


def test_stale_flexibility_summary_removal(mock_environment: dict[str, Path], tmp_path: Path) -> None:
    """Verify that a stale flexibility_summary.csv is deleted when flexibility is unavailable."""
    base_dir = mock_environment["base_dir"]
    df = pd.read_parquet(mock_environment["input_file"])
    engine = DecisionSupportEngine(model_name="LinearRegression", models_dir=base_dir)
    advisories = engine.predict_batch_advisories(df)
    report_dir = tmp_path / "stale_reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy pre-existing flexibility_summary.csv
    stale_csv = report_dir / "flexibility_summary.csv"
    stale_csv.write_text("category,total_sessions\nstale_data,999\n", encoding="utf-8")
    assert stale_csv.is_file()

    # Run save_decision_support_artifacts with unavailable flexibility
    artifacts = save_decision_support_artifacts(
        advisories_df=advisories,
        flexibility_stats={"available": False, "reason": "Missing columns"},
        report_dir=report_dir,
        model_name="LinearRegression",
    )

    # Assert stale CSV was removed
    assert not stale_csv.exists()
    assert "flexibility_summary" not in artifacts
