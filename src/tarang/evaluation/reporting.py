"""Automated Markdown report generator for TARANG Stage 3 evaluation.

Synthesizes benchmark performance, behavioral regressions, model agreement,
temporal stability, TreeSHAP feature importances, and subgroup analysis into
a publication-quality evaluation summary document.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from tarang.config import TARGET_DURATION, TARGET_ENERGY
from tarang.evaluation.analysis import (
    format_performance_markdown_table,
    format_sci,
)

logger = logging.getLogger("tarang.evaluation.reporting")


def format_p_value(p: float) -> str:
    """Format p-value avoiding rounded p = 0.0000.

    Args:
        p: Floating-point p-value.

    Returns:
        Formatted string (e.g. 'p < 0.0001' or 'p = 0.0053').
    """
    if p is None or pd.isna(p):
        return "N/A"
    if p < 0.0001:
        return "p < 0.0001"
    return f"p = {p:.4f}"


def summarize_benchmark_winners(table_df: pd.DataFrame, unit: str) -> str:
    """Dynamically determine metric winners from the computed benchmark table."""
    metrics = ["MAE", "MdAE", "MAPE", "MdAPE", "SMAPE"]
    all_winners = {m: table_df[("All data", m)].idxmin() for m in metrics}
    veh_winners = {m: table_df[("Per vehicle mean", m)].idxmin() for m in metrics}

    hgbr_all_mae = format_sci(table_df.loc["HistGradientBoostingRegressor", ("All data", "MAE")])
    hgbr_all_medae = format_sci(table_df.loc["HistGradientBoostingRegressor", ("All data", "MdAE")])
    hgbr_all_mape = format_sci(table_df.loc["HistGradientBoostingRegressor", ("All data", "MAPE")])
    hgbr_all_smape = format_sci(table_df.loc["HistGradientBoostingRegressor", ("All data", "SMAPE")])

    # Check if HGBR wins on every metric
    hgbr_wins_all = all(w == "HistGradientBoostingRegressor" for w in all_winners.values()) and all(
        w == "HistGradientBoostingRegressor" for w in veh_winners.values()
    )

    if hgbr_wins_all:
        return (
            f"- **Best Performing Model:** `HistGradientBoostingRegressor` (HGBR) achieves the lowest error "
            f"across all evaluated metrics on the held-out test split (All data MAE: **{hgbr_all_mae} {unit}**, "
            f"MdAE: **{hgbr_all_medae} {unit}**, MAPE: **{hgbr_all_mape}%**, SMAPE: **{hgbr_all_smape}%**)."
        )
    else:
        hgbr_won_metrics = [m for m, w in all_winners.items() if w == "HistGradientBoostingRegressor"]
        other_winners = {m: w for m, w in all_winners.items() if w != "HistGradientBoostingRegressor"}

        parts = []
        if hgbr_won_metrics:
            m_str = ", ".join(hgbr_won_metrics)
            parts.append(
                f"- **Model Performance Summary:** `HistGradientBoostingRegressor` (HGBR) achieves the lowest error "
                f"on {m_str} on the held-out test split (MAE: **{hgbr_all_mae} {unit}**, MdAE: **{hgbr_all_medae} {unit}**, "
                f"SMAPE: **{hgbr_all_smape}%**)"
            )
        for m, model in other_winners.items():
            val_all = format_sci(table_df.loc[model, ("All data", m)])
            val_veh = format_sci(table_df.loc[model, ("Per vehicle mean", m)])
            parts.append(
                f"`{model}` achieves the lowest aggregate {m} (**{val_all}%**) and per-vehicle mean {m} (**{val_veh}%**)"
            )
        return "; while ".join(parts) + "."


def generate_evaluation_markdown_report(
    target: str,
    table_df: pd.DataFrame,
    vehicle_characteristics: Dict[str, Any],
    error_over_time_dict: Dict[str, Any],
    data_in_splits: Dict[str, Any],
    df_global_shap: pd.DataFrame,
    df_subgroups: pd.DataFrame,
    dataset_counts: Dict[str, int],
    has_heatmaps: bool = False,
) -> str:
    """Generate Markdown report summarizing all Stage 3 findings.

    Args:
        target: Prediction target ('Duration' or 'ConsumedkWh').
        table_df: Benchmark performance table.
        vehicle_characteristics: OLS regression results against user characteristics.
        error_over_time_dict: Temporal degradation shift statistics.
        data_in_splits: Session count and timespan statistics in train and test splits.
        df_global_shap: Global TreeSHAP importance ranking.
        df_subgroups: Same-day vs overnight subgroup metrics table.
        dataset_counts: Dictionary of evaluated dataset counts (total_sessions,
            n_participants, n_train, n_test).
        has_heatmaps: True if occupancy heatmaps were generated (Duration target).

    Returns:
        Formatted Markdown text.
    """
    unit = "h" if target == TARGET_DURATION else "kWh"
    target_display = "Plugged-in Duration (target_duration_hours)" if target == TARGET_DURATION else "Energy Consumption (target_energy_kwh)"

    total_sessions = dataset_counts.get("total_sessions", 0)
    n_participants = dataset_counts.get("n_participants", 0)
    n_train = dataset_counts.get("n_train", 0)
    n_test = dataset_counts.get("n_test", 0)

    hgbr_veh_mae = format_sci(table_df.loc["HistGradientBoostingRegressor", ("Per vehicle mean", "MAE")])
    hgbr_veh_medae = format_sci(table_df.loc["HistGradientBoostingRegressor", ("Per vehicle mean", "MdAE")])
    hgbr_veh_mape = format_sci(table_df.loc["HistGradientBoostingRegressor", ("Per vehicle mean", "MAPE")])

    # Top 3 features from SHAP
    top_features = df_global_shap.head(3)["feature"].tolist()
    top_feature_str = ", ".join([f"`{f}`" for f in top_features])

    md_lines = [
        f"# TARANG — Model Evaluation, Behavioral Analysis & Explainability",
        f"**Target:** {target_display}  ",
        f"**Evaluation Split:** 20% Chronological Held-out Test Set  ",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "",
        f"This report presents the model evaluation, participant-level behavioral analysis, and TreeSHAP explainability "
        f"for **TARANG** predicting **{target_display}**.",
        "",
        f"- **Dataset Evaluated:** {total_sessions:,} canonical sessions across {n_participants:,} unique participants "
        f"({n_train:,} Train sessions / {n_test:,} Test sessions in chronological 80/20 split).",
        summarize_benchmark_winners(table_df, unit),
        f"- **Participant-Level Performance:** The reported per-vehicle mean MAE is **{hgbr_veh_mae} {unit}** "
        f"(MdAE: **{hgbr_veh_medae} {unit}**, MAPE: **{hgbr_veh_mape}%**), reflecting the distribution of "
        f"individual participant charging patterns across the cohort.",
        f"- **Dominant Feature Drivers:** TreeSHAP analysis identifies {top_feature_str} as the primary "
        f"global predictors driving model decisions.",
        "",
        "---",
        "",
        "## 2. Benchmark Model Performance",
        "",
        "Evaluation results across all six benchmark models comparing aggregate session performance "
        "('All data') against the average per-vehicle performance ('Per vehicle mean'):",
        "",
        format_performance_markdown_table(table_df),
        "",
        "![Model Performance Comparison](performance.png)",
        "",
        "---",
        "",
        "## 3. Best Model Error Density & Distribution",
        "",
        f"Comparison of absolute error density between all sessions and the vehicle-level mean distribution "
        f"for `HistGradientBoostingRegressor`:",
        "",
        "![HGBR Error Distribution](best_model.png)",
        "",
        "---",
        "",
        "## 4. Behavioral & Vehicle-Level Analysis",
        "",
        "Ordinary Least Squares (OLS) regression analyzing the dependence of participant-level Median Absolute "
        "Percentage Error (MdAPE) on user charging behaviors:",
        "",
    ]

    for key in ["duration_variability", "battery_size", "start_hour"]:
        info = vehicle_characteristics[key]
        p_formatted = format_p_value(info["p_value"])
        sig_str = "Statistically significant ($p < 0.05$)" if info["significant"] else "Not statistically significant"
        md_lines.append(
            f"- **{info['name']}**: Slope $a = {info['slope']:.3f}$, Intercept $b = {info['intercept']:.3f}$, "
            f"$r^2 = {info['r2']:.3f}$, ${p_formatted}$ ({sig_str})"
        )

    md_lines.extend(
        [
            "",
            "![Performance Over Features](performance_over_features.png)",
            "",
            "---",
            "",
            "## 5. Model Agreement & Multi-Model Comparison",
            "",
            "Visual comparison of participant-level Test MAE values between benchmark models "
            "(HGBR versus Linear Regression, HGBR versus Quantile Regressor, and Linear Regression versus Quantile Regressor). "
            "The main scatter plot displays agreement across all vehicles, while the inset axis zooms into the $[0, 20]$ error range:",
            "",
            "![Model Agreement](model_agreement.png)",
            "",
            "---",
            "",
            "## 6. Temporal Stability & Seasonality Analysis",
            "",
            f"- **Temporal Error Shift Across Test Split:** The relative change in MAE from the first to second "
            f"half of the test set is calculated per participant as $(\\text{{mean}}(\\text{{first-half AE}}) - "
            f"\\text{{mean}}(\\text{{second-half AE}})) / \\text{{mean}}(\\text{{all participant Test AE}})$. "
            f"Across evaluated vehicles, the mean relative change is **{error_over_time_dict['mean_relative_change']:.3f} "
            f"\\pm {error_over_time_dict['std_relative_change']:.3f}**. Under this formula, a negative value indicates "
            f"that prediction error is higher in the second half of the participant's test sessions than in the first half.",
            f"- **Split Data Availability:**",
            f"  - **Train split:** {data_in_splits['Train']['session_count_mean']:.1f} \\pm {data_in_splits['Train']['session_count_std']:.1f} sessions "
            f"over {data_in_splits['Train']['timespan_days_mean']:.1f} \\pm {data_in_splits['Train']['timespan_days_std']:.1f} days per vehicle.",
            f"  - **Test split:** {data_in_splits['Test']['session_count_mean']:.1f} \\pm {data_in_splits['Test']['session_count_std']:.1f} sessions "
            f"over {data_in_splits['Test']['timespan_days_mean']:.1f} \\pm {data_in_splits['Test']['timespan_days_std']:.1f} days per vehicle.",
            "",
            "![Error Over Time Histogram](ae_over_time_hist.png)",
            "",
            "![Seasonality](seasonality.png)",
            "",
            "---",
            "",
            "## 7. TreeSHAP Explainability & Feature Importance",
            "",
            "Global feature importance ranking evaluated via TreeSHAP on test sessions:",
            "",
            "| Rank | Feature Name | Mean Absolute SHAP Value |",
            "| :---: | :--- | :---: |",
        ]
    )

    for idx, row in df_global_shap.iterrows():
        md_lines.append(f"| {idx + 1} | `{row['feature']}` | {row['importance_mean_abs_shap']:.4f} |")

    md_lines.extend(
        [
            "",
            "Data files: [shap_global_importance.csv](shap_global_importance.csv), [shap_vehicle_importance.csv](shap_vehicle_importance.csv)",
            "",
            "![SHAP Values](shap_values.png)",
            "",
            "---",
            "",
            "## 8. Subgroup Analysis: Same-day vs. Overnight Charging",
            "",
            "Independent evaluation comparing same-day charging sessions (where StopTime and StartTime fall "
            "on the same UTC calendar date, day difference == 0) against overnight charging sessions "
            "(where StopTime falls on a subsequent UTC calendar date, day difference > 0):",
            "",
            "| Subgroup | Sessions | Train $N$ | Test $N$ | MAE | MdAE | MAPE (%) | MdAPE (%) | SMAPE (%) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )

    for _, row in df_subgroups.iterrows():
        md_lines.append(
            f"| **{row['subgroup']}** | {int(row['n_sessions']):,} | {int(row['n_train']):,} | {int(row['n_test']):,} | "
            f"{format_sci(row['mae'])} | {format_sci(row['medae'])} | {format_sci(row['mape'])} | {format_sci(row['medape'])} | {format_sci(row['smape'])} |"
        )

    md_lines.extend(
        [
            "",
            "Data file: [same_day_overnight_metrics.csv](same_day_overnight_metrics.csv)",
            "",
            "![Same-Day vs Overnight Comparison](same_day_overnight_comparison.png)",
            "",
        ]
    )

    if has_heatmaps:
        md_lines.extend(
            [
                "---",
                "",
                "## 9. Occupancy Heatmaps (Top Best & Worst Vehicles)",
                "",
                "Visual analysis of 15-minute resolution charging occupancy comparing real and predicted "
                "plugged-in states (White: True Negative, Orange: False Negative, Red: False Positive, Blue: True Positive):",
                "",
                "### Best Performing Vehicles",
                "![Best Vehicle 0](heatmap-best-0.png)  ",
                "![Best Vehicle 1](heatmap-best-1.png)  ",
                "![Best Vehicle 2](heatmap-best-2.png)  ",
                "",
                "### Worst Performing Vehicles",
                "![Worst Vehicle 0](heatmap-worst-0.png)  ",
                "![Worst Vehicle 1](heatmap-worst-1.png)  ",
                "![Worst Vehicle 2](heatmap-worst-2.png)  ",
                "",
            ]
        )

    md_lines.extend(
        [
            "---",
            "",
            "*Report generated automatically by TARANG Stage 3 Model Evaluation Suite.*",
        ]
    )

    return "\n".join(md_lines)
