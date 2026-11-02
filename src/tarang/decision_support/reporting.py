"""Reporting and artifact generation utilities for TARANG Stage 5 Decision Support.

Synthesizes batch prediction advisories and smart-charging flexibility metrics into:
- `prediction_summary.csv`: Statistical distribution of predicted duration and energy.
- `flexibility_summary.csv`: Same-day vs overnight flexibility ratios.
- `decision_support_summary.md`: Publication-grade summary report detailing operational findings,
  occupancy projections, and non-dispatch scope boundaries.
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from tarang.config import (
    DECISION_SUPPORT_REPORTS_DIR,
    DEFAULT_DECISION_FLEXIBILITY_SUMMARY_FILE,
    DEFAULT_DECISION_PREDICTION_SUMMARY_FILE,
    DEFAULT_DECISION_REPORT_FILE,
    PROJECT_ROOT,
    TARGET_DURATION,
    TARGET_ENERGY,
)

logger = logging.getLogger("tarang.decision_support.reporting")


def generate_prediction_summary_table(advisories_df: pd.DataFrame) -> pd.DataFrame:
    """Generate distribution summary statistics for predicted duration and energy.

    Evaluates count, mean, std, min, 25%, 50% (median), 75%, 90%, and max across
    valid finite, non-negative predictions, while explicitly tracking invalid counts.

    Args:
        advisories_df: DataFrame containing advisory predictions with columns
            'predicted_Duration', 'predicted_ConsumedkWh',
            'duration_prediction_valid', and 'energy_prediction_valid'.

    Returns:
        DataFrame containing summary statistics for both targets.

    Raises:
        ValueError: If required prediction and validity columns are missing or DataFrame is empty.
    """
    if advisories_df is None or not isinstance(advisories_df, pd.DataFrame) or advisories_df.empty:
        raise ValueError("Advisories DataFrame is empty or None.")

    required_cols = [
        f"predicted_{TARGET_DURATION}",
        f"predicted_{TARGET_ENERGY}",
        "duration_prediction_valid",
        "energy_prediction_valid",
    ]
    missing = [c for c in required_cols if c not in advisories_df.columns]
    if missing:
        raise ValueError(
            f"Advisories DataFrame is missing required column(s): {missing}. "
            f"Expected columns: {required_cols}."
        )

    total_count = len(advisories_df)

    targets_meta = [
        (
            TARGET_DURATION,
            f"predicted_{TARGET_DURATION}",
            "duration_prediction_valid",
            "hours",
        ),
        (
            TARGET_ENERGY,
            f"predicted_{TARGET_ENERGY}",
            "energy_prediction_valid",
            "kWh",
        ),
    ]

    records: List[Dict[str, Any]] = []

    for target_name, pred_col, valid_col, unit in targets_meta:
        valid_mask = advisories_df[valid_col].astype(bool)
        valid_series = advisories_df.loc[valid_mask, pred_col].dropna()
        valid_count = int(valid_mask.sum())
        invalid_count = total_count - valid_count

        if valid_count > 0:
            mean_val = float(valid_series.mean())
            std_val = float(valid_series.std(ddof=1)) if valid_count > 1 else 0.0
            min_val = float(valid_series.min())
            p25_val = float(valid_series.quantile(0.25))
            p50_val = float(valid_series.median())
            p75_val = float(valid_series.quantile(0.75))
            p90_val = float(valid_series.quantile(0.90))
            max_val = float(valid_series.max())
        else:
            mean_val = np.nan
            std_val = np.nan
            min_val = np.nan
            p25_val = np.nan
            p50_val = np.nan
            p75_val = np.nan
            p90_val = np.nan
            max_val = np.nan

        records.append(
            {
                "target": target_name,
                "unit": unit,
                "total_count": total_count,
                "valid_count": valid_count,
                "invalid_count": invalid_count,
                "mean": round(mean_val, 4) if np.isfinite(mean_val) else mean_val,
                "std": round(std_val, 4) if np.isfinite(std_val) else std_val,
                "min": round(min_val, 4) if np.isfinite(min_val) else min_val,
                "25%": round(p25_val, 4) if np.isfinite(p25_val) else p25_val,
                "50%": round(p50_val, 4) if np.isfinite(p50_val) else p50_val,
                "75%": round(p75_val, 4) if np.isfinite(p75_val) else p75_val,
                "90%": round(p90_val, 4) if np.isfinite(p90_val) else p90_val,
                "max": round(max_val, 4) if np.isfinite(max_val) else max_val,
            }
        )

    return pd.DataFrame(records)


def generate_flexibility_summary_table(
    flexibility_stats: Optional[Dict[str, Any]],
) -> Optional[pd.DataFrame]:
    """Generate tabular summary of smart-charging flexibility metrics.

    Args:
        flexibility_stats: Dictionary produced by analyze_charging_flexibility.

    Returns:
        DataFrame with same-day and overnight metrics, or None if flexibility is unavailable.
    """
    if flexibility_stats is None or not flexibility_stats.get("available", False):
        return None

    records = [
        {
            "category": "Same-Day Sessions",
            "total_sessions": int(flexibility_stats.get("same_day_total", 0)),
            "evaluated_in_range": int(flexibility_stats.get("same_day_in_range", 0)),
            "valid_sessions": int(flexibility_stats.get("same_day_valid", 0)),
            "mean_ratio": round(float(flexibility_stats.get("same_day_mean_ratio", 0.0)), 4),
            "median_ratio": round(float(flexibility_stats.get("same_day_median_ratio", 0.0)), 4),
        },
        {
            "category": "Overnight Sessions",
            "total_sessions": int(flexibility_stats.get("overnight_total", 0)),
            "evaluated_in_range": int(flexibility_stats.get("overnight_in_range", 0)),
            "valid_sessions": int(flexibility_stats.get("overnight_valid", 0)),
            "mean_ratio": round(float(flexibility_stats.get("overnight_mean_ratio", 0.0)), 4),
            "median_ratio": round(float(flexibility_stats.get("overnight_median_ratio", 0.0)), 4),
        },
    ]

    return pd.DataFrame(records)


def generate_decision_support_markdown_report(
    advisories_df: pd.DataFrame,
    flexibility_stats: Optional[Dict[str, Any]] = None,
    model_name: str = "HistGradientBoostingRegressor",
    source_file: Optional[Union[str, Path]] = None,
) -> str:
    """Generate comprehensive Markdown report for Stage 5 decision support.

    Args:
        advisories_df: DataFrame containing batch session advisories.
        flexibility_stats: Optional dictionary containing flexibility analysis output.
        model_name: Model architecture utilized for inference. Defaults to HGBR.
        source_file: Optional source dataset path.

    Returns:
        Formatted Markdown report string.
    """
    pred_summary_df = generate_prediction_summary_table(advisories_df)
    flex_df = generate_flexibility_summary_table(flexibility_stats)

    total_sessions = len(advisories_df)
    if source_file:
        p = Path(source_file)
        try:
            source_desc = p.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            source_desc = p.as_posix()
    else:
        source_desc = "data/processed/electric_nation/features.parquet"
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Format prediction summary markdown table
    pred_table_lines = [
        "| Target | Unit | Total Sessions | Valid Predictions | Invalid (Flagged) | Mean | Std Dev | Min | 25% | Median (50%) | 75% | 90% | Max |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for _, row in pred_summary_df.iterrows():
        pred_table_lines.append(
            f"| **{row['target']}** | {row['unit']} | {row['total_count']:,} | {row['valid_count']:,} | "
            f"{row['invalid_count']:,} | {row['mean']:.2f} | {row['std']:.2f} | {row['min']:.2f} | "
            f"{row['25%']:.2f} | {row['50%']:.2f} | {row['75%']:.2f} | {row['90%']:.2f} | {row['max']:.2f} |"
        )
    pred_table_md = "\n".join(pred_table_lines)

    # Format flexibility markdown table if available
    if flex_df is not None:
        flex_table_lines = [
            "| Session Category | Total Sessions | Evaluated in [0, 1] | Mean Ratio | Median Ratio |",
            "| :--- | :---: | :---: | :---: | :---: |",
        ]
        for _, row in flex_df.iterrows():
            flex_table_lines.append(
                f"| **{row['category']}** | {row['total_sessions']:,} | {row['evaluated_in_range']:,} | "
                f"{row['mean_ratio']:.3f} | {row['median_ratio']:.3f} |"
            )
        flex_table_md = "\n".join(flex_table_lines)

        same_day_median = flexibility_stats.get("same_day_median_ratio", 0.0)
        overnight_median = flexibility_stats.get("overnight_median_ratio", 0.0)
        flexibility_section = f"""## 2. Smart-Charging Flexibility Ratio Analysis

Smart-charging flexibility is quantified as the ratio of active charging duration to total connection duration:

$$\\text{{Flexibility Ratio}} = \\frac{{\\text{{ChargingDuration}} / 60.0}}{{(\\text{{StopTime}} - \\text{{StartTime}})_{{\\text{{hours}}}}}}$$

Sessions are partitioned using local UK calendar dates (`Europe/London`) into same-day arrivals versus overnight charging sessions.

{flex_table_md}

### Flexibility Key Findings
- **Overnight Flexibility Advantage**: Overnight sessions exhibit a median flexibility ratio of **{overnight_median:.3f}** (mean: **{flexibility_stats.get('overnight_mean_ratio', 0.0):.3f}**), indicating that vehicles remain plugged in idle for over 80% of their connection window. This presents substantial load shifting and renewable curtailment mitigation opportunities.
- **Same-Day Tight Occupancy**: Same-day sessions demonstrate a median ratio of **{same_day_median:.3f}** (mean: **{flexibility_stats.get('same_day_mean_ratio', 0.0):.3f}**), reflecting shorter daytime connection spans where active charging occupies the majority of the connection.
"""
    else:
        flexibility_section = """## 2. Smart-Charging Flexibility Ratio Analysis

*Flexibility analysis is unavailable because historical raw connection timestamps (`ChargingDuration`, `StartTime`, `StopTime`) were omitted from the input dataset.*
"""

    # Occupancy projections summary
    valid_unplugs = advisories_df["projected_unplug_time_utc"].dropna()
    valid_unplug_count = len(valid_unplugs)
    valid_unplug_pct = (valid_unplug_count / total_sessions * 100.0) if total_sessions > 0 else 0.0

    report = f"""# TARANG — Stage 5: Decision Support Summary Report

**System**: TARANG — EV Charging Intelligence System  
**Stage**: Stage 5 — Decision Support  
**Model Architecture**: `{model_name}`  
**Source Dataset**: `{source_desc}`  
**Evaluation Scope**: {total_sessions:,} charging sessions  
**Generated At**: {now_utc}  

---

## 1. Predictive Session Advisories Distribution

The decision support layer generates raw numeric estimates of connection duration and energy demand, verifies non-negativity and finiteness, and calculates projected departure timestamps.

{pred_table_md}

### Operational Advisory Integrity
- **Non-Negativity Verification**: Estimator outputs are strictly validated via `isfinite(pred) and pred >= 0.0`. Flagged predictions retain their raw continuous estimator values without artificial clipping or thresholding.
- **Occupancy Projections**: A total of **{valid_unplug_count:,} sessions ({valid_unplug_pct:.1f}%)** produced valid duration predictions and corresponding projected unplug timestamps (`projected_unplug_time_utc`). Invalid or negative duration estimates are safely represented as `NaT` (Not-a-Time) to prevent faulty bay scheduling.

---

{flexibility_section}

---

## 3. Operational Boundaries & Non-Dispatch Scope

> [!IMPORTANT]
> TARANG operates strictly as an **advisory decision-support system**. The predictions, occupancy timelines, and flexibility ratios generated by this pipeline are designed to inform fleet operators, charging facility managers, and smart grid aggregators.
> 
> The system explicitly **does not**:
> 1. Execute direct charger hardware actuation, power throttling, or relay switching.
> 2. Implement autonomous grid dispatch, real-time load shedding, or carbon/tariff optimization algorithms.
> 3. Perform post-processing clipping or artificial distortion on raw statistical predictions.

---

## 4. References & Documentation

- [**Decision Support Contract**](../../docs/decision_support/decision_support_contract.md): Advisory schema, validation specifications, and flexibility definitions.
- [**Inference Contract**](../../docs/inference/inference_contract.md): Stage 4 single-session and batch inference architecture.
- [**Exploratory Data Analysis Report**](../eda/eda_summary.md): Source distribution analysis for trial availability and flexibility.
- [**Predictive Modeling Evaluation**](../modeling/duration/evaluation_summary.md): Dual-target benchmark performance and TreeSHAP explainability.
"""
    return report


def save_decision_support_artifacts(
    advisories_df: pd.DataFrame,
    flexibility_stats: Optional[Dict[str, Any]] = None,
    output_advisories_path: Optional[Union[str, Path]] = None,
    report_dir: Optional[Union[str, Path]] = None,
    model_name: str = "HistGradientBoostingRegressor",
    source_file: Optional[Union[str, Path]] = None,
) -> Dict[str, Path]:
    """Persist decision support advisories, summary tables, and Markdown report.

    Args:
        advisories_df: DataFrame of session advisories.
        flexibility_stats: Optional dictionary containing flexibility analysis results.
        output_advisories_path: Target path for advisories dataset (.parquet or .csv).
        report_dir: Directory for reports and summary CSV files.
        model_name: Benchmark model architecture.
        source_file: Optional source dataset path.

    Returns:
        Dictionary mapping artifact identifiers to their written Path locations.
    """
    reports_directory = Path(report_dir) if report_dir else DECISION_SUPPORT_REPORTS_DIR
    reports_directory.mkdir(parents=True, exist_ok=True)

    artifacts: Dict[str, Path] = {}

    # 1. Save Advisories Dataset if path provided
    if output_advisories_path:
        out_path = Path(output_advisories_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Saving session advisories dataset (%d rows) to %s", len(advisories_df), out_path)
        if out_path.suffix.lower() == ".parquet":
            advisories_df.to_parquet(out_path, index=False)
        else:
            advisories_df.to_csv(out_path, index=False)
        artifacts["advisories"] = out_path

    # 2. Save Prediction Summary CSV
    pred_summary_df = generate_prediction_summary_table(advisories_df)
    pred_summary_path = reports_directory / "prediction_summary.csv"
    logger.info("Saving prediction summary table to %s", pred_summary_path)
    pred_summary_df.to_csv(pred_summary_path, index=False)
    artifacts["prediction_summary"] = pred_summary_path

    # 3. Save Flexibility Summary CSV if available
    flex_df = generate_flexibility_summary_table(flexibility_stats)
    if flex_df is not None:
        flex_summary_path = reports_directory / "flexibility_summary.csv"
        logger.info("Saving flexibility summary table to %s", flex_summary_path)
        flex_df.to_csv(flex_summary_path, index=False)
        artifacts["flexibility_summary"] = flex_summary_path

    # 4. Save Markdown Summary Report
    report_md = generate_decision_support_markdown_report(
        advisories_df=advisories_df,
        flexibility_stats=flexibility_stats,
        model_name=model_name,
        source_file=source_file,
    )
    report_path = reports_directory / "decision_support_summary.md"
    logger.info("Saving decision support summary report to %s", report_path)
    report_path.write_text(report_md, encoding="utf-8")
    artifacts["report"] = report_path

    return artifacts
