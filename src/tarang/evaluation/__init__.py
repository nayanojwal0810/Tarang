"""TARANG Stage 3 — Model Evaluation, Behavioral Analysis & Explainability.

Provides comprehensive tools for:
- Benchmark performance tables and summary statistics
- User charging behavioral regressions (OLS against MdAPE)
- Pairwise model agreement and robustness profiling
- Temporal stability and seasonality tracking
- TreeSHAP global and participant-level feature explainability
- Subgroup analysis (same-day vs overnight sessions)
- 4-state charging occupancy heatmaps
"""

from tarang.evaluation.analysis import (
    compute_data_in_splits,
    compute_error_over_time,
    compute_participant_summary_stats,
    compute_seasonality,
    compute_vehicle_characteristics,
    format_performance_markdown_table,
    format_sci,
    generate_performance_table,
)
from tarang.evaluation.explainability import compute_tree_shap_importance
from tarang.evaluation.loading import (
    EvaluationData,
    load_evaluation_data,
    load_hgbr_model,
    load_performance_data,
    load_predictions_data,
    normalize_target_name,
)
from tarang.evaluation.reporting import generate_evaluation_markdown_report
from tarang.evaluation.subgroup_analysis import (
    partition_sessions_by_timing,
    run_subgroup_analysis,
    train_and_evaluate_subgroup_hgbr,
)
from tarang.evaluation.visualizations import (
    plot_best_model_distribution,
    plot_duration_heatmaps,
    plot_error_over_time_histogram,
    plot_model_agreement,
    plot_performance_boxplots,
    plot_performance_over_features,
    plot_seasonality,
    plot_shap_summary,
    plot_subgroup_comparison,
    save_figure,
)

__all__ = [
    "EvaluationData",
    "load_evaluation_data",
    "load_performance_data",
    "load_predictions_data",
    "load_hgbr_model",
    "normalize_target_name",
    "format_sci",
    "generate_performance_table",
    "format_performance_markdown_table",
    "compute_participant_summary_stats",
    "compute_vehicle_characteristics",
    "compute_error_over_time",
    "compute_seasonality",
    "compute_data_in_splits",
    "compute_tree_shap_importance",
    "partition_sessions_by_timing",
    "train_and_evaluate_subgroup_hgbr",
    "run_subgroup_analysis",
    "generate_evaluation_markdown_report",
    "save_figure",
    "plot_performance_boxplots",
    "plot_best_model_distribution",
    "plot_performance_over_features",
    "plot_model_agreement",
    "plot_error_over_time_histogram",
    "plot_seasonality",
    "plot_shap_summary",
    "plot_subgroup_comparison",
    "plot_duration_heatmaps",
]

