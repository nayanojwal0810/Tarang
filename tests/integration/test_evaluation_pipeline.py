"""Integration tests for the complete TARANG Stage 3 evaluation pipeline."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

from tarang.config import (
    FEATURE_NAMES,
    MODEL_NAMES,
    TARGET_DURATION,
)
from tarang.evaluation import (
    compute_data_in_splits,
    compute_error_over_time,
    compute_model_agreement,
    compute_seasonality,
    compute_tree_shap_importance,
    compute_vehicle_characteristics,
    generate_evaluation_markdown_report,
    generate_performance_table,
    plot_best_model_distribution,
    plot_duration_heatmaps,
    plot_error_over_time_histogram,
    plot_model_agreement,
    plot_performance_boxplots,
    plot_performance_over_features,
    plot_seasonality,
    plot_shap_summary,
    plot_subgroup_comparison,
    run_subgroup_analysis,
)


@pytest.fixture
def synthetic_pipeline_env(tmp_path):
    """Construct minimal synthetic datasets and fixtures for end-to-end integration test."""
    n_sessions = 80
    dates = pd.date_range("2017-03-01 06:00:00", periods=n_sessions, freq="14h")
    pids = [f"P{i % 4 + 1:03d}" for i in range(n_sessions)]

    durations = np.random.uniform(2.0, 12.0, size=n_sessions)

    df_sessions = pd.DataFrame(
        {
            "StartTime": dates,
            "StopTime": dates + pd.to_timedelta(durations, unit="h"),
            "ParticipantID": pids,
            "Duration": durations,
            "ConsumedkWh": durations * 3.6,
            "CarKW": 7.0,
            "CarKWh": 40.0,
            "StartHourNorm": 0.45,
            "dataset": ["Train"] * 64 + ["Test"] * 16,
            "HistGradientBoostingRegressor": durations + np.random.normal(0, 0.5, size=n_sessions),
            "LinearRegression": durations + np.random.normal(0, 1.0, size=n_sessions),
            "QuantileRegressor": durations + np.random.normal(0, 0.8, size=n_sessions),
            "MedianValue": [durations.mean()] * n_sessions,
            "MeanValue": [durations.mean()] * n_sessions,
            "LastValue": durations,
        }
    )

    for feat in FEATURE_NAMES:
        if feat not in df_sessions.columns:
            df_sessions[feat] = np.random.uniform(0, 5, size=n_sessions)

    # MultiIndex performance
    cols = []
    for m in MODEL_NAMES:
        for s in ["Train", "Test"]:
            for sc in ["mae", "medae", "mape", "medape", "smape", "wl-ae", "q1-ae", "q2-ae", "q3-ae", "wh-ae", "wl-ape", "q1-ape", "q2-ape", "q3-ape", "wh-ape"]:
                cols.append((m, s, sc))

    col_idx = pd.MultiIndex.from_tuples(cols, names=["model", "dataset", "score"])

    # Aggregate row 'ALL'
    all_dict = {c: 5.0 if "ae" in c[2] else 25.0 for c in cols}
    perf_all = pd.Series(all_dict, name="ALL")

    # Participant rows
    part_matrix = []
    unique_pids = sorted(list(set(pids)))
    for i in range(len(unique_pids)):
        row = [5.0 + i * 0.2 if "ae" in c[2] else 20.0 + i * 2.0 for c in cols]
        part_matrix.append(row)

    perf_part = pd.DataFrame(part_matrix, index=unique_pids, columns=col_idx)
    perf_part.index.name = "ParticipantID"

    # Fit a small HGBR model
    model = HistGradientBoostingRegressor(max_iter=10, random_state=0)
    model.fit(df_sessions[FEATURE_NAMES], df_sessions["Duration"])

    return {
        "sessions": df_sessions,
        "perf_all": perf_all,
        "perf_part": perf_part,
        "model": model,
        "output_dir": tmp_path / "reports",
        "data_dir": tmp_path / "data",
    }


def test_full_evaluation_pipeline_end_to_end(synthetic_pipeline_env):
    """Verify that all pipeline components execute and create all required artifacts."""
    env = synthetic_pipeline_env
    output_dir = env["output_dir"]
    data_dir = env["data_dir"]
    sessions = env["sessions"]
    perf_all = env["perf_all"]
    perf_part = env["perf_part"]
    model = env["model"]

    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1. Performance Table
    perf_table = generate_performance_table(perf_part, perf_all, dataset="Test")
    perf_csv = output_dir / "model_performance.csv"
    perf_table.to_csv(perf_csv)
    assert perf_csv.exists()

    # 2. Behavioral Characteristics
    veh_chars = compute_vehicle_characteristics(
        perf_part, sessions, model_name="HistGradientBoostingRegressor", dataset="Test"
    )
    assert "duration_variability" in veh_chars

    # 3. Model Agreement
    agreement = compute_model_agreement(perf_part, dataset="Test", score="mae")
    assert len(agreement) > 0

    # 4. Temporal Error Change
    error_over_time = compute_error_over_time(
        sessions, TARGET_DURATION, model_name="HistGradientBoostingRegressor", dataset="Test"
    )
    assert "mean_relative_change" in error_over_time

    # 5. Seasonality
    seasonality = compute_seasonality(
        sessions, TARGET_DURATION, model_name="HistGradientBoostingRegressor"
    )
    assert "Train" in seasonality

    # 6. Split Availability
    data_splits = compute_data_in_splits(sessions)
    assert "Train" in data_splits

    # 7. TreeSHAP
    test_preds = sessions[sessions["dataset"] == "Test"]
    df_global, df_vehicle = compute_tree_shap_importance(
        model=model,
        test_df=test_preds,
        output_dir=output_dir,
    )
    assert (output_dir / "shap_global_importance.csv").exists()
    assert (output_dir / "shap_vehicle_importance.csv").exists()

    # 8. Subgroup Analysis
    df_subgroups = run_subgroup_analysis(
        sessions=sessions,
        target=TARGET_DURATION,
        performance_all=perf_all,
        output_dir=output_dir,
        evaluation_data_dir=data_dir,
    )
    assert (output_dir / "same_day_overnight_metrics.csv").exists()

    # 9. Visualizations
    plot_performance_boxplots(perf_part, perf_all, TARGET_DURATION, output_dir / "performance.png")
    plot_best_model_distribution(sessions, perf_part, TARGET_DURATION, output_dir / "best_model.png")
    plot_performance_over_features(veh_chars, TARGET_DURATION, output_dir / "performance_over_features.png")
    plot_model_agreement(perf_part, TARGET_DURATION, output_dir / "model_agreement.png")
    plot_error_over_time_histogram(error_over_time, output_dir / "ae_over_time_hist.png")
    plot_seasonality(seasonality, output_dir / "seasonality.png")
    plot_shap_summary(df_global, df_vehicle, output_dir / "shap_values.png")
    plot_subgroup_comparison(df_subgroups, TARGET_DURATION, output_dir / "same_day_overnight_comparison.png")
    heatmaps = plot_duration_heatmaps(sessions, perf_part, output_dir)

    for fig_name in [
        "performance.png",
        "best_model.png",
        "performance_over_features.png",
        "model_agreement.png",
        "ae_over_time_hist.png",
        "seasonality.png",
        "shap_values.png",
        "same_day_overnight_comparison.png",
    ]:
        fig_file = output_dir / fig_name
        assert fig_file.exists(), f"Figure {fig_name} was not created."
        assert fig_file.stat().st_size > 0

    assert len(heatmaps) > 0
    for hm in heatmaps:
        assert hm.exists()
        assert hm.stat().st_size > 0

    # 10. Markdown Report
    md_content = generate_evaluation_markdown_report(
        target=TARGET_DURATION,
        table_df=perf_table,
        vehicle_characteristics=veh_chars,
        agreement_dict=agreement,
        error_over_time_dict=error_over_time,
        data_in_splits=data_splits,
        df_global_shap=df_global,
        df_subgroups=df_subgroups,
        has_heatmaps=True,
    )
    md_file = output_dir / "evaluation_summary.md"
    md_file.write_text(md_content, encoding="utf-8")
    assert md_file.exists()
    assert md_file.stat().st_size > 500

