"""Unit tests for TARANG Stage 3 TreeSHAP explainability."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

from tarang.config import FEATURE_NAMES
from tarang.evaluation.explainability import compute_tree_shap_importance


@pytest.fixture
def synthetic_hgbr_and_data(tmp_path):
    """Create a trained small HGBR model and test data for fast SHAP tests."""
    n_samples = 60
    np.random.seed(42)

    X_data = {feat: np.random.uniform(0, 10, size=n_samples) for feat in FEATURE_NAMES}
    df = pd.DataFrame(X_data)
    # Target strongly correlated with LastDuration and StartHourNorm
    y = 2.0 * df["LastDuration"] + 0.5 * df["StartHourNorm"] + np.random.normal(0, 0.1, size=n_samples)
    df["Duration"] = y
    df["ParticipantID"] = [f"P{i % 4 + 1:02d}" for i in range(n_samples)]

    model = HistGradientBoostingRegressor(max_iter=10, random_state=0)
    model.fit(df[FEATURE_NAMES], df["Duration"])

    return model, df


def test_compute_tree_shap_importance_structure(synthetic_hgbr_and_data, tmp_path):
    """Test global and vehicle-level SHAP importance outputs."""
    model, df = synthetic_hgbr_and_data

    df_global, df_vehicle = compute_tree_shap_importance(
        model=model,
        test_df=df,
        feature_names=FEATURE_NAMES,
        output_dir=tmp_path,
    )

    # 1. Global DataFrame assertions
    assert isinstance(df_global, pd.DataFrame)
    assert list(df_global.columns) == ["feature", "importance_mean_abs_shap"]
    assert len(df_global) == len(FEATURE_NAMES)
    # Check descending sort
    assert df_global["importance_mean_abs_shap"].is_monotonic_decreasing
    # LastDuration should be ranked highest due to strong coefficient
    assert df_global.iloc[0]["feature"] == "LastDuration"

    # 2. Vehicle DataFrame assertions
    assert isinstance(df_vehicle, pd.DataFrame)
    assert df_vehicle.index.name == "ParticipantID"
    assert len(df_vehicle) == 4
    for feat in FEATURE_NAMES:
        assert feat in df_vehicle.columns

    # 3. CSV file persistence assertions
    global_csv = tmp_path / "shap_global_importance.csv"
    vehicle_csv = tmp_path / "shap_vehicle_importance.csv"
    assert global_csv.exists()
    assert vehicle_csv.exists()


def test_compute_tree_shap_missing_columns_raises(synthetic_hgbr_and_data):
    """Test error handling when test DataFrame is missing required features."""
    model, df = synthetic_hgbr_and_data
    df_broken = df.drop(columns=["LastDuration"])

    with pytest.raises(ValueError, match="missing required features"):
        compute_tree_shap_importance(model=model, test_df=df_broken)

    df_no_pid = df.drop(columns=["ParticipantID"])
    with pytest.raises(ValueError, match="ParticipantID"):
        compute_tree_shap_importance(model=model, test_df=df_no_pid)

