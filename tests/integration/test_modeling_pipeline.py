"""Integration tests for TARANG Stage 2 modeling pipeline.

Verifies end-to-end training and evaluation across both targets (Duration and ConsumedkWh),
all six benchmark models, artifact persistence, and prediction row counts.
Gracefully skips when the local processed Parquet file is absent (e.g., in CI).
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import pytest

from tarang.config import (
    CANONICAL_COLUMNS,
    DURATION_MODELING_DIR,
    ENERGY_MODELING_DIR,
    MODEL_NAMES,
    PROCESSED_FEATURES_FILE,
    TARGET_DURATION,
    TARGET_ENERGY,
)
from tarang.modeling.experiment import run_experiment


def test_modeling_workflow_both_targets_on_canonical_features(tmp_path: Path) -> None:
    """Run full modeling workflow on canonical feature dataset for both targets."""
    if not PROCESSED_FEATURES_FILE.is_file():
        pytest.skip(
            f"Canonical features file not found at {PROCESSED_FEATURES_FILE}; "
            "skipping integration test."
        )

    # Read canonical dataset
    df = pd.read_parquet(PROCESSED_FEATURES_FILE)
    assert len(df) == 59520, f"Expected 59520 canonical sessions, found {len(df)}"

    # Use first 300 rows for rapid integration validation of all 6 models and both targets
    sample_size = 300
    df_sample = df.head(sample_size).copy()

    for target in [TARGET_DURATION, TARGET_ENERGY]:
        target_dir = tmp_path / target.lower()
        perf_df, pred_df = run_experiment(
            sessions=df_sample,
            target=target,
            output_dir=target_dir,
            saved=False,
        )

        # 1. Prediction row count matches input exactly
        assert len(pred_df) == sample_size, (
            f"Predictions count {len(pred_df)} != input count {sample_size}"
        )

        # 2. Dataset split column present with 80/20 chronological split
        expected_train_count = int(sample_size * 0.8)
        expected_test_count = sample_size - expected_train_count

        assert (pred_df["dataset"].iloc[:expected_train_count] == "Train").all()
        assert (pred_df["dataset"].iloc[expected_train_count:] == "Test").all()

        # 3. All original canonical columns retained
        for col in CANONICAL_COLUMNS:
            assert col in pred_df.columns, f"Canonical column '{col}' missing from predictions"

        # 4. All six model prediction columns present and non-null
        for model_name in MODEL_NAMES:
            assert model_name in pred_df.columns, f"Model column '{model_name}' missing from predictions"
            assert not pred_df[model_name].isna().any(), f"Model column '{model_name}' contains NaN"

        # 5. Saved files on disk
        assert (target_dir / "performance.csv").is_file()
        assert (target_dir / "predictions.csv").is_file()
        for model_name in MODEL_NAMES:
            joblib_path = target_dir / "models" / f"{model_name}.joblib"
            assert joblib_path.is_file(), f"Joblib file '{joblib_path}' missing"

        # 6. Performance structure verification
        assert "ALL" in perf_df.index
        assert perf_df.loc["ALL", ("number_sessions", "Train", "none")] == expected_train_count
        assert perf_df.loc["ALL", ("number_sessions", "Test", "none")] == expected_test_count
        assert perf_df.columns.nlevels == 3


def test_production_modeling_artifacts_integrity() -> None:
    """Verify saved production artifacts when full real-data benchmark run has executed."""
    if not (DURATION_MODELING_DIR / "performance.csv").is_file() or not (
        ENERGY_MODELING_DIR / "performance.csv"
    ).is_file():
        pytest.skip(
            "Production modeling artifacts not yet generated. "
            "Execute 'python scripts/run_modeling.py' to generate."
        )

    for target, target_dir in [
        (TARGET_DURATION, DURATION_MODELING_DIR),
        (TARGET_ENERGY, ENERGY_MODELING_DIR),
    ]:
        perf_csv = target_dir / "performance.csv"
        pred_csv = target_dir / "predictions.csv"

        assert perf_csv.is_file()
        assert pred_csv.is_file()

        # Load performance CSV with MultiIndex header
        perf = pd.read_csv(perf_csv, index_col=0, header=[0, 1, 2])
        assert "ALL" in perf.index
        assert len(perf.index) == 268  # ALL + 267 participants

        # Train/Test session counts
        assert perf.loc["ALL", ("number_sessions", "Train", "none")] == 47616
        assert perf.loc["ALL", ("number_sessions", "Test", "none")] == 11904

        # Load predictions CSV
        pred = pd.read_csv(pred_csv, index_col=0)
        assert len(pred) == 59520
        assert (pred["dataset"] == "Train").sum() == 47616
        assert (pred["dataset"] == "Test").sum() == 11904

        for model_name in MODEL_NAMES:
            assert model_name in pred.columns
            assert (target_dir / "models" / f"{model_name}.joblib").is_file()

