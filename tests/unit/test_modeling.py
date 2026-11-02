"""Unit tests for TARANG Stage 2 predictive modeling benchmark.

Tests benchmark models, baseline estimators, evaluation metrics, whisker formulas,
chronological splitting, participant metrics, and validation integrity.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

from tarang.config import (
    CANONICAL_COLUMNS,
    FEATURE_NAMES,
    MODEL_NAMES,
    TARGET_DURATION,
    TARGET_ENERGY,
    LAG_FEATURE_DURATION,
    LAG_FEATURE_ENERGY,
)
from tarang.modeling.models import (
    EmptyModel,
    MeanValue,
    MedianValue,
    LastValue,
    get_benchmark_models,
)
from tarang.modeling.metrics import (
    absolute_error,
    absolute_percentage_error,
    squared_error,
    symmetric_mean_absolute_percentage_error,
    calculate_metrics,
)
from tarang.modeling.experiment import (
    validate_modeling_input,
    run_experiment,
)


# ==============================================================================
# Helper to build synthetic canonical DataFrame
# ==============================================================================

def make_synthetic_sessions(n_rows: int = 100) -> pd.DataFrame:
    """Create a minimal synthetic DataFrame conforming to CANONICAL_COLUMNS."""
    rng = np.random.RandomState(42)
    start_times = pd.date_range("2017-01-01", periods=n_rows, freq="2h", tz="UTC")
    durations = rng.uniform(1.0, 15.0, size=n_rows)
    stop_times = start_times + pd.to_timedelta(durations, unit="h")

    pids = ["P100"] * (n_rows // 2) + ["P200"] * (n_rows - n_rows // 2)

    data = {
        "StartTime": start_times,
        "StopTime": stop_times,
        "ParticipantID": pids,
        "ConsumedkWh": rng.uniform(5.0, 60.0, size=n_rows),
        "CarKW": [7.0] * n_rows,
        "CarKWh": [30.0] * n_rows,
        "PluggedInTime": durations,
        "ChargingDuration": durations * 0.7,
        "Hot_Unplug": [0] * n_rows,
        "Duration": durations,
        "weekday": [1.0] * n_rows,
        "weekend": [0.0] * n_rows,
        "holiday": [0.0] * n_rows,
        "Mon": [1.0] * n_rows,
        "Tue": [0.0] * n_rows,
        "Wed": [0.0] * n_rows,
        "Thu": [0.0] * n_rows,
        "Fri": [0.0] * n_rows,
        "Sat": [0.0] * n_rows,
        "Sun": [0.0] * n_rows,
        "month": [1.0] * n_rows,
        "StartHour": [12.0] * n_rows,
        "StartHourNorm": [0.5] * n_rows,
        "StartCos": [1.0] * n_rows,
        "StartSin": [0.0] * n_rows,
        "TimeSinceLastStop": rng.uniform(1.0, 24.0, size=n_rows),
        "sessionsToday": [1.0] * n_rows,
        "LastDuration": np.roll(durations, 1),
        "LastConsumedkWh": rng.uniform(5.0, 60.0, size=n_rows),
        "id": [1] * (n_rows // 2) + [2] * (n_rows - n_rows // 2),
    }
    df = pd.DataFrame(data)
    # Ensure LastDuration first row has valid numeric value
    df.loc[0, "LastDuration"] = 5.0
    return df


# ==============================================================================
# Model Factory and Hyperparameters Tests
# ==============================================================================

def test_model_names_order_and_hyperparameters() -> None:
    """Verify exact six benchmark model names, order, and hyperparameter specifications."""
    models = get_benchmark_models(TARGET_DURATION)
    assert len(models) == 6
    names = [m[0] for m in models]
    assert names == [
        "LinearRegression",
        "QuantileRegressor",
        "HistGradientBoostingRegressor",
        "MedianValue",
        "MeanValue",
        "LastValue",
    ]
    assert names == MODEL_NAMES

    # Model 1: LinearRegression
    lr = models[0][1]
    assert isinstance(lr, LinearRegression)

    # Model 2: QuantileRegressor
    qr = models[1][1]
    assert isinstance(qr, QuantileRegressor)
    assert qr.quantile == 0.5
    assert qr.alpha == 0.0
    assert qr.solver == "highs"

    # Model 3: HistGradientBoostingRegressor
    hgb = models[2][1]
    assert isinstance(hgb, HistGradientBoostingRegressor)
    assert hgb.early_stopping is True
    assert hgb.learning_rate == 0.1
    assert hgb.min_samples_leaf == 100
    assert hgb.max_iter == 100
    assert hgb.loss == "absolute_error"
    assert isinstance(hgb.random_state, np.random.RandomState)

    # Model 4: MedianValue
    assert isinstance(models[3][1], MedianValue)

    # Model 5: MeanValue
    assert isinstance(models[4][1], MeanValue)

    # Model 6: LastValue
    assert isinstance(models[5][1], LastValue)
    assert models[5][1].lag_feature == LAG_FEATURE_DURATION


def test_fresh_random_state_per_target_call() -> None:
    """Confirm a fresh RandomState(0) is created for each target model factory call."""
    models_dur = get_benchmark_models(TARGET_DURATION)
    models_nrg = get_benchmark_models(TARGET_ENERGY)

    hgb_dur = models_dur[2][1]
    hgb_nrg = models_nrg[2][1]

    # Both must start from identical fresh random state
    rand_seq1 = hgb_dur.random_state.rand(5)
    rand_seq2 = hgb_nrg.random_state.rand(5)
    np.testing.assert_array_equal(rand_seq1, rand_seq2)


def test_baseline_mean_and_median_models() -> None:
    """Verify MeanValue and MedianValue fit computation and predictions."""
    y_train = np.array([10.0, 20.0, 30.0, 100.0])
    X_train = pd.DataFrame({"dummy": [1, 2, 3, 4]})
    X_test = pd.DataFrame({"dummy": [5, 6, 7]})

    mean_mod = MeanValue()
    assert mean_mod.fit(X_train, y_train) is mean_mod
    assert mean_mod.mean_ == pytest.approx(40.0)
    preds_mean = mean_mod.predict(X_test)
    assert len(preds_mean) == 3
    assert np.all(preds_mean == 40.0)

    med_mod = MedianValue()
    assert med_mod.fit(X_train, y_train) is med_mod
    assert med_mod.median_ == pytest.approx(25.0)
    preds_med = med_mod.predict(X_test)
    assert len(preds_med) == 3
    assert np.all(preds_med == 25.0)

    # Unfitted check
    with pytest.raises(ValueError, match="not fitted yet"):
        MeanValue().predict(X_test)
    with pytest.raises(ValueError, match="not fitted yet"):
        MedianValue().predict(X_test)


def test_baseline_last_value_lag_selection() -> None:
    """Verify LastValue correctly selects target-specific lag predictor."""
    df_test = pd.DataFrame({
        "LastDuration": [1.5, 2.5, 3.5],
        "LastConsumedkWh": [10.0, 20.0, 30.0],
    })

    # Duration target
    dur_last = LastValue(lag_feature=LAG_FEATURE_DURATION)
    preds_dur = dur_last.predict(df_test)
    np.testing.assert_array_equal(preds_dur, [1.5, 2.5, 3.5])

    # Energy target
    nrg_last = LastValue(lag_feature=LAG_FEATURE_ENERGY)
    preds_nrg = nrg_last.predict(df_test)
    np.testing.assert_array_equal(preds_nrg, [10.0, 20.0, 30.0])

    # Missing column check
    bad_df = pd.DataFrame({"WrongColumn": [1, 2, 3]})
    with pytest.raises(KeyError, match="not found in input columns"):
        dur_last.predict(bad_df)


def test_feature_names_order() -> None:
    """Verify the exact 13 predictors are defined in canonical sequence."""
    expected_order = [
        "weekday",
        "weekend",
        "holiday",
        "month",
        "StartCos",
        "StartSin",
        "StartHourNorm",
        "TimeSinceLastStop",
        "sessionsToday",
        "LastDuration",
        "LastConsumedkWh",
        "CarKW",
        "CarKWh",
    ]
    assert FEATURE_NAMES == expected_order


# ==============================================================================
# Metric and Whisker Calculation Tests
# ==============================================================================

def test_ordinary_finite_metrics_against_reference_formulas() -> None:
    """Verify all 16 metrics on ordinary finite values match reference calculations exactly."""
    y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    y_pred = np.array([12.0, 18.0, 33.0, 39.0, 55.0])

    # Element-wise error tests
    ae = absolute_error(y_true, y_pred)
    np.testing.assert_array_almost_equal(ae, [2.0, 2.0, 3.0, 1.0, 5.0])

    ape = absolute_percentage_error(y_true, y_pred)
    np.testing.assert_array_almost_equal(ape, [0.2, 0.1, 0.1, 0.025, 0.1])

    se = squared_error(y_true, y_pred)
    np.testing.assert_array_almost_equal(se, [4.0, 4.0, 9.0, 1.0, 25.0])

    smape = symmetric_mean_absolute_percentage_error(y_true, y_pred)
    expected_smape = float(
        np.sum(2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))
        / len(y_true)
    )
    assert smape == pytest.approx(expected_smape)

    # Complete metrics dict
    metrics = calculate_metrics(y_true, y_pred)
    expected_keys = [
        "medae", "mae", "medape", "mape", "rmse", "smape",
        "q1-ae", "q2-ae", "q3-ae", "wl-ae", "wh-ae",
        "q1-ape", "q2-ape", "q3-ape", "wl-ape", "wh-ape",
    ]
    for k in expected_keys:
        assert k in metrics, f"Missing metric key '{k}'"

    assert metrics["medae"] == pytest.approx(2.0)
    assert metrics["mae"] == pytest.approx(2.6)
    assert metrics["medape"] == pytest.approx(0.1)
    assert metrics["mape"] == pytest.approx(0.105)
    assert metrics["rmse"] == pytest.approx(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    assert metrics["smape"] == pytest.approx(expected_smape)

    # Exact whisker check against reference pandas Series behavior
    for name, err_arr in [("ae", ae), ("ape", ape)]:
        s = pd.Series(err_arr)
        q = np.quantile(s, [0.25, 0.5, 0.75])
        iqr = q[2] - q[0]
        wl = s[s > (q[0] - 1.5 * iqr)].min()
        wh = s[s < (q[0] + 1.5 * iqr)].max()

        assert metrics[f"q1-{name}"] == pytest.approx(q[0])
        assert metrics[f"q2-{name}"] == pytest.approx(q[1])
        assert metrics[f"q3-{name}"] == pytest.approx(q[2])
        assert metrics[f"wl-{name}"] == pytest.approx(wl)
        assert metrics[f"wh-{name}"] == pytest.approx(wh)


def test_zero_actual_and_zero_predicted_smape_nan() -> None:
    """Verify sMAPE propagates NaN when actual and predicted are both zero."""
    y_true = np.array([0.0, 10.0])
    y_pred = np.array([0.0, 12.0])

    smape = symmetric_mean_absolute_percentage_error(y_true, y_pred)
    assert np.isnan(smape), "sMAPE must be NaN when 0/0 division occurs"

    metrics = calculate_metrics(y_true, y_pred)
    assert np.isnan(metrics["smape"]), "calculate_metrics['smape'] must be NaN"


def test_constant_absolute_errors_empty_whisker_returns_nan() -> None:
    """Verify empty strict whisker selection returns NaN matching Pandas behavior."""
    # Case A: Constant error across multiple samples
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([12.0, 22.0, 32.0, 42.0])  # All AE = 2.0

    metrics = calculate_metrics(y_true, y_pred)
    assert metrics["medae"] == pytest.approx(2.0)
    assert metrics["mae"] == pytest.approx(2.0)
    assert metrics["q1-ae"] == pytest.approx(2.0)
    assert metrics["q2-ae"] == pytest.approx(2.0)
    assert metrics["q3-ae"] == pytest.approx(2.0)
    # With IQR=0, error > lower_bound and error < upper_bound are empty
    assert np.isnan(metrics["wl-ae"]), "Empty whisker selection must return NaN"
    assert np.isnan(metrics["wh-ae"]), "Empty whisker selection must return NaN"

    # Case B: Single sample (N=1)
    metrics_single = calculate_metrics(np.array([15.0]), np.array([10.0]))
    assert metrics_single["medae"] == pytest.approx(5.0)
    assert np.isnan(metrics_single["wl-ae"]), "Single sample whisker must be NaN"
    assert np.isnan(metrics_single["wh-ae"]), "Single sample whisker must be NaN"


def test_nan_percentage_errors_propagation() -> None:
    """Verify NaN percentage errors preserve reference NaN propagation in quantiles/whiskers."""
    y_true = np.array([0.0, 10.0, 20.0])
    y_pred = np.array([0.0, 12.0, 22.0])

    metrics = calculate_metrics(y_true, y_pred)

    # Percentage error has 0/0 = NaN, so percentage-based statistics propagate NaN
    assert np.isnan(metrics["medape"]), "MedAPE must be NaN when APE contains NaN"
    assert np.isnan(metrics["mape"]), "MAPE must be NaN when APE contains NaN"
    assert np.isnan(metrics["smape"]), "sMAPE must be NaN when APE contains NaN"
    assert np.isnan(metrics["q1-ape"]), "q1-ape must be NaN"
    assert np.isnan(metrics["q2-ape"]), "q2-ape must be NaN"
    assert np.isnan(metrics["q3-ape"]), "q3-ape must be NaN"
    assert np.isnan(metrics["wl-ape"]), "wl-ape must be NaN"
    assert np.isnan(metrics["wh-ape"]), "wh-ape must be NaN"

    # Absolute error metrics remain finite
    assert metrics["medae"] == pytest.approx(2.0)
    assert metrics["mae"] == pytest.approx(4.0 / 3.0)


# ==============================================================================
# Experiment Workflow and Persistence Tests
# ==============================================================================

def test_chronological_split_and_predictions_coverage() -> None:
    """Verify chronological 80/20 split and full predictions coverage."""
    df = make_synthetic_sessions(n_rows=100)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        perf_df, pred_df = run_experiment(
            sessions=df,
            target=TARGET_DURATION,
            output_dir=output_dir,
            saved=False,
        )

        # Predictions verification
        assert len(pred_df) == 100
        assert (pred_df["dataset"].iloc[:80] == "Train").all()
        assert (pred_df["dataset"].iloc[80:] == "Test").all()

        for model_name in MODEL_NAMES:
            assert model_name in pred_df.columns
            assert not pred_df[model_name].isna().any()

        # Performance verification
        assert "ALL" in perf_df.index
        assert perf_df.loc["ALL", ("number_sessions", "Train", "none")] == 80
        assert perf_df.loc["ALL", ("number_sessions", "Test", "none")] == 20

        # Participants P100 and P200 present
        assert "P100" in perf_df.index
        assert "P200" in perf_df.index

        # Saved files check
        assert (output_dir / "performance.csv").exists()
        assert (output_dir / "predictions.csv").exists()
        for model_name in MODEL_NAMES:
            assert (output_dir / "models" / f"{model_name}.joblib").exists()


def test_saved_model_reuse() -> None:
    """Verify that run_experiment loads cached joblib models when saved=True."""
    df = make_synthetic_sessions(n_rows=50)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        # First run creates models
        run_experiment(df, target=TARGET_DURATION, output_dir=output_dir, saved=False)

        # Record timestamp of a model file
        lr_file = output_dir / "models" / "LinearRegression.joblib"
        mtime_before = lr_file.stat().st_mtime

        # Second run with saved=True
        run_experiment(df, target=TARGET_DURATION, output_dir=output_dir, saved=True)
        mtime_after = lr_file.stat().st_mtime

        assert mtime_before == mtime_after, "Model was refitted despite saved=True"


def test_input_validation_catches_violations() -> None:
    """Verify input validator detects missing columns, NaNs, and non-monotonicity."""
    df = make_synthetic_sessions(n_rows=20)

    # Valid check
    validate_modeling_input(df, TARGET_DURATION)
    validate_modeling_input(df, TARGET_ENERGY)

    # Invalid target
    with pytest.raises(ValueError, match="Invalid target"):
        validate_modeling_input(df, "InvalidTarget")

    # Missing predictor
    df_missing_col = df.drop(columns=["StartCos"])
    with pytest.raises(ValueError, match="missing from modeling dataset"):
        validate_modeling_input(df_missing_col, TARGET_DURATION)

    # Missing target value
    df_nan_target = df.copy()
    df_nan_target.loc[0, "Duration"] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        validate_modeling_input(df_nan_target, TARGET_DURATION)

    # Missing feature value
    df_nan_feat = df.copy()
    df_nan_feat.loc[0, "CarKW"] = np.nan
    with pytest.raises(ValueError, match="Predictors contain missing values"):
        validate_modeling_input(df_nan_feat, TARGET_DURATION)

    # Non-monotonic StartTime
    df_unsorted = df.copy()
    df_unsorted.iloc[0, df_unsorted.columns.get_loc("StartTime")] = pd.Timestamp("2020-01-01", tz="UTC")
    with pytest.raises(ValueError, match="not sorted chronologically"):
        validate_modeling_input(df_unsorted, TARGET_DURATION)
