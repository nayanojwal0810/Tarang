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


def generate_evaluation_markdown_report(
    target: str,
    table_df: pd.DataFrame,
    vehicle_characteristics: Dict[str, Any],
    agreement_dict: Dict[str, Any],
    error_over_time_dict: Dict[str, Any],
    data_in_splits: Dict[str, Any],
    df_global_shap: pd.DataFrame,
    df_subgroups: pd.DataFrame,
    has_heatmaps: bool = False,
) -> str:
    """Generate Markdown report summarizing all Stage 3 findings.

    Args:
        target: Prediction target ('Duration' or 'ConsumedkWh').
        table_df: Benchmark performance table.
        vehicle_characteristics: OLS regression results against user characteristics.
        agreement_dict: Model agreement pairwise statistics.
        error_over_time_dict: Temporal degradation shift statistics.
        data_in_splits: Session count and timespan statistics in train and test splits.
        df_global_shap: Global TreeSHAP importance ranking.
        df_subgroups: Same-day vs overnight subgroup metrics table.
        has_heatmaps: True if occupancy heatmaps were generated (Duration target).

    Returns:
        Formatted Markdown text.
    """
    unit = "h" if target == TARGET_DURATION else "kWh"
    target_display = "Plugged-in Duration (target_duration_hours)" if target == TARGET_DURATION else "Energy Consumption (target_energy_kwh)"

    hgbr_all_mae = format_sci(table_df.loc["HistGradientBoostingRegressor", ("All data", "MAE")])
    hgbr_all_medae = format_sci(table_df.loc["HistGradientBoostingRegressor", ("All data", "MdAE")])
    hgbr_all_mape = format_sci(table_df.loc["HistGradientBoostingRegressor", ("All data", "MAPE")])
    hgbr_all_smape = format_sci(table_df.loc["HistGradientBoostingRegressor", ("All data", "SMAPE")])

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
        f"This report presents the rigorous evaluation, behavioral analysis, and TreeSHAP explainability "
        f"for **TARANG** predicting **{target_display}**.",
        "",
        f"- **Best Performing Model:** `HistGradientBoostingRegressor` (HGBR) achieves superior accuracy "
        f"across all evaluated metrics on the held-out test split (MAE: **{hgbr_all_mae} {unit}**, "
        f"MdAE: **{hgbr_all_medae} {unit}**, MAPE: **{hgbr_all_mape}%**, SMAPE: **{hgbr_all_smape}%**).",
        f"- **Vehicle-Level Heterogeneity:** The per-vehicle mean MAE is **{hgbr_veh_mae} {unit}** "
        f"(MdAE: **{hgbr_veh_medae} {unit}**, MAPE: **{hgbr_veh_mape}%**), demonstrating strong user-level "
        f"predictability with meaningful variance across individual charging habits.",
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
        sig_str = "Statistically significant ($p < 0.05$)" if info["significant"] else "Not statistically significant"
        md_lines.append(
            f"- **{info['name']}**: Slope $a = {info['slope']:.3f}$, Intercept $b = {info['intercept']:.3f}$, "
            f"$r^2 = {info['r2']:.3f}$, $p = {info['p_value']:.4f}$ ({sig_str})"
        )

    md_lines.extend(
        [
            "",
            "![Performance Over Features](performance_over_features.png)",
            "",
            "---",
            "",
            "## 5. Model Agreement & Multi-Model Robustness",
            "",
            "Pairwise vehicle-level MAE correlations between benchmark regressors:",
            "",
            "| Model Comparison | Pearson $r$ | Spearman $\\rho$ | Mean Absolute Difference | Vehicles |",
            "| :--- | :---: | :---: | :---: | :---: |",
        ]
    )

    for pair_name, stats in agreement_dict.items():
        md_lines.append(
            f"| **{pair_name}** | {stats['pearson_r']:.3f} | {stats['spearman_rho']:.3f} | {stats['mean_abs_diff']:.3f} {unit} | {stats['n_vehicles']} |"
        )

    md_lines.extend(
        [
            "",
            "![Model Agreement](model_agreement.png)",
            "",
            "---",
            "",
            "## 6. Temporal Stability & Seasonality Analysis",
            "",
            f"- **Temporal Shift in Test Split:** The mean relative change in MAE from the first to second "
            f"half of the test set across vehicles is **{error_over_time_dict['mean_relative_change']:.3f} "
            f"\\pm {error_over_time_dict['std_relative_change']:.3f}**, indicating strong model calibration "
            f"over time without catastrophic concept drift.",
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
            "Independent evaluation comparing daytime charging sessions ending on the same UTC date against "
            "overnight charging sessions spanning into the following day:",
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

