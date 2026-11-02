#!/usr/bin/env python
"""CLI Runner for TARANG Stage 3 — Model Evaluation, Behavioral Analysis & Explainability.

Executes comprehensive evaluation workflows for:
- Plugged-in Duration (`Duration`)
- Energy Consumption (`ConsumedkWh`)

Produces publication-ready figures, quantitative metrics CSV tables, and detailed
Markdown summary reports under `reports/modeling/{duration,energy}/`.

Usage:
    python scripts/run_evaluation.py
    python scripts/run_evaluation.py --target Duration
    python scripts/run_evaluation.py --target ConsumedkWh
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
import time

from tarang.config import (
    DURATION_EVALUATION_DATA_DIR,
    DURATION_EVALUATION_REPORT_DIR,
    ENERGY_EVALUATION_DATA_DIR,
    ENERGY_EVALUATION_REPORT_DIR,
    TARGET_DURATION,
    TARGET_ENERGY,
)
from tarang.evaluation import (
    compute_data_in_splits,
    compute_error_over_time,
    compute_seasonality,
    compute_tree_shap_importance,
    compute_vehicle_characteristics,
    generate_evaluation_markdown_report,
    generate_performance_table,
    load_evaluation_data,
    normalize_target_name,
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_evaluation")


def evaluate_target(
    target: str,
    output_dir: Path,
    evaluation_data_dir: Path,
) -> None:
    """Execute complete Stage 3 evaluation pipeline for a specified target."""
    canonical_target = normalize_target_name(target)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation_data_dir = Path(evaluation_data_dir)
    evaluation_data_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    logger.info("==================================================================")
    logger.info("STARTING STAGE 3 EVALUATION FOR TARGET: %s", canonical_target)
    logger.info("Output report directory: %s", output_dir)
    logger.info("==================================================================")

    # 1. Load evaluation artifacts
    eval_data = load_evaluation_data(canonical_target)
    perf_all = eval_data.performance_all
    perf_part = eval_data.performance_participants
    predictions = eval_data.predictions
    hgbr_model = eval_data.hgbr_model
    features_df = eval_data.features

    test_preds = predictions[predictions["dataset"] == "Test"]
    logger.info("Loaded artifacts: %d sessions (%d Test), %d participants", len(predictions), len(test_preds), len(perf_part))

    # 2. Performance Table
    perf_table = generate_performance_table(perf_part, perf_all, dataset="Test")
    perf_table_csv = output_dir / "model_performance.csv"
    perf_table.to_csv(perf_table_csv)
    logger.info("Saved benchmark performance table to %s", perf_table_csv)

    # 3. Behavioral Analysis (OLS Regressions against MdAPE)
    logger.info("Computing user charging characteristics and regressions...")
    veh_chars = compute_vehicle_characteristics(perf_part, features_df, model_name="HistGradientBoostingRegressor", dataset="Test")

    # 4. Temporal Stability (Error over time)
    logger.info("Computing temporal error change across test split...")
    error_over_time = compute_error_over_time(predictions, canonical_target, model_name="HistGradientBoostingRegressor", dataset="Test")

    # 5. Seasonality (Monthly MAPE)
    logger.info("Computing monthly seasonality trends...")
    seasonality_data = compute_seasonality(predictions, canonical_target, model_name="HistGradientBoostingRegressor")

    # 6. Split Data Availability
    logger.info("Computing data availability in splits...")
    data_in_splits = compute_data_in_splits(predictions)

    # 7. TreeSHAP Explainability
    logger.info("Computing TreeSHAP feature importances...")
    df_global_shap, df_vehicle_shap = compute_tree_shap_importance(
        model=hgbr_model,
        test_df=test_preds,
        output_dir=output_dir,
    )

    # 8. Subgroup Analysis (Same-day vs Overnight)
    logger.info("Running subgroup analysis for same-day vs overnight sessions...")
    df_subgroups = run_subgroup_analysis(
        sessions=predictions,
        target=canonical_target,
        performance_all=perf_all,
        output_dir=output_dir,
        evaluation_data_dir=evaluation_data_dir,
    )

    # 9. Generate Visualizations
    logger.info("Generating publication-quality figures...")
    plot_performance_boxplots(perf_part, perf_all, canonical_target, output_dir / "performance.png")
    plot_best_model_distribution(predictions, perf_part, canonical_target, output_dir / "best_model.png")
    plot_performance_over_features(veh_chars, canonical_target, output_dir / "performance_over_features.png")
    plot_model_agreement(perf_part, canonical_target, output_dir / "model_agreement.png")
    plot_error_over_time_histogram(error_over_time, output_dir / "ae_over_time_hist.png")
    plot_seasonality(seasonality_data, output_dir / "seasonality.png")
    plot_shap_summary(df_global_shap, df_vehicle_shap, output_dir / "shap_values.png")
    plot_subgroup_comparison(df_subgroups, canonical_target, output_dir / "same_day_overnight_comparison.png")

    has_heatmaps = False
    if canonical_target == TARGET_DURATION:
        logger.info("Generating charging occupancy heatmaps for top and bottom vehicles...")
        plot_duration_heatmaps(predictions, perf_part, output_dir)
        has_heatmaps = True

    # 10. Markdown Summary Report
    logger.info("Synthesizing evaluation summary report...")
    dataset_counts = {
        "total_sessions": len(predictions),
        "n_participants": int(predictions["ParticipantID"].nunique()),
        "n_train": int((predictions["dataset"] == "Train").sum()),
        "n_test": int((predictions["dataset"] == "Test").sum()),
    }
    report_md = generate_evaluation_markdown_report(
        target=canonical_target,
        table_df=perf_table,
        vehicle_characteristics=veh_chars,
        error_over_time_dict=error_over_time,
        data_in_splits=data_in_splits,
        df_global_shap=df_global_shap,
        df_subgroups=df_subgroups,
        dataset_counts=dataset_counts,
        has_heatmaps=has_heatmaps,
    )
    report_path = output_dir / "evaluation_summary.md"
    report_path.write_text(report_md, encoding="utf-8")
    logger.info("Saved evaluation summary report to %s", report_path)

    elapsed = time.time() - t0
    logger.info("COMPLETED EVALUATION FOR %s IN %.2f SECONDS.", canonical_target, elapsed)
    logger.info("==================================================================")


def main() -> None:
    """CLI Entry point."""
    parser = argparse.ArgumentParser(description="TARANG Stage 3 Evaluation Runner")
    parser.add_argument(
        "--target",
        type=str,
        default="all",
        choices=["all", "Duration", "duration", "ConsumedkWh", "consumedkwh", "energy"],
        help="Prediction target to evaluate ('Duration', 'ConsumedkWh', or 'all').",
    )
    args = parser.parse_args()

    targets_to_run = []
    if args.target.lower() == "all":
        targets_to_run = [TARGET_DURATION, TARGET_ENERGY]
    else:
        targets_to_run = [normalize_target_name(args.target)]

    for target in targets_to_run:
        if target == TARGET_DURATION:
            out_dir = DURATION_EVALUATION_REPORT_DIR
            data_dir = DURATION_EVALUATION_DATA_DIR
        else:
            out_dir = ENERGY_EVALUATION_REPORT_DIR
            data_dir = ENERGY_EVALUATION_DATA_DIR

        evaluate_target(target=target, output_dir=out_dir, evaluation_data_dir=data_dir)


if __name__ == "__main__":
    main()

