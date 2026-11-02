"""Analytical routines for TARANG Stage 3 model evaluation.

Implements model performance table construction, participant-level summary
statistics, behavioral characteristic regressions, model agreement metrics,
temporal stability analysis, seasonality profiling, and split data availability.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

from tarang.config import MODEL_NAMES, TARGET_DURATION, TARGET_ENERGY

logger = logging.getLogger("tarang.evaluation.analysis")

SCORE_KEYS: List[str] = ["mae", "medae", "mape", "medape", "smape"]
SCORE_LABELS: List[str] = ["MAE", "MdAE", "MAPE (%)", "MdAPE (%)", "SMAPE (%)"]


def format_sci(score: float) -> str:
    """Format numeric score in compact 3-significant-figure scientific notation.

    Faithfully reproduces the reference notation formatter.

    Args:
        score: Floating-point metric score.

    Returns:
        Formatted string (e.g. '5.53', '201', '0.042').
    """
    if score is None or pd.isna(score) or np.isinf(score):
        return "N/A"
    try:
        return f"{float(f'{score:.3g}'):g}"
    except (ValueError, TypeError):
        return str(score)


def generate_performance_table(
    performance_participants: pd.DataFrame,
    performance_all: pd.Series,
    dataset: str = "Test",
    models: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Construct benchmark performance comparison table across all models.

    Columns include aggregate error across all sessions ('All data') and
    vehicle-level average error ('Per vehicle mean') for MAE, MdAE, MAPE,
    MdAPE, and SMAPE.

    Args:
        performance_participants: Participant-level MultiIndex performance DataFrame.
        performance_all: Aggregate 'ALL' MultiIndex performance Series.
        dataset: Split name ('Test' or 'Train'). Defaults to 'Test'.
        models: Ordered model list. Defaults to canonical MODEL_NAMES.

    Returns:
        MultiIndex DataFrame with levels: (Scope, Metric) and Model index.
    """
    if models is None:
        models = [m for m in MODEL_NAMES if m in performance_all.index.get_level_values(0)]

    headers = [
        ("All data", "MAE"),
        ("All data", "MdAE"),
        ("All data", "MAPE"),
        ("All data", "MdAPE"),
        ("All data", "SMAPE"),
        ("Per vehicle mean", "MAE"),
        ("Per vehicle mean", "MdAE"),
        ("Per vehicle mean", "MAPE"),
        ("Per vehicle mean", "MdAPE"),
        ("Per vehicle mean", "SMAPE"),
    ]
    col_index = pd.MultiIndex.from_tuples(headers, names=["Scope", "Metric"])
    table_data = []

    for model in models:
        row = []
        # All data metrics
        for score in SCORE_KEYS:
            val = performance_all.get((model, dataset, score), np.nan)
            row.append(val)

        # Per vehicle mean metrics
        for score in SCORE_KEYS:
            if (model, dataset, score) in performance_participants.columns:
                series = performance_participants.loc[:, (model, dataset, score)]
                val = float(series.mean(skipna=True))
            else:
                val = np.nan
            row.append(val)

        table_data.append(row)

    df_table = pd.DataFrame(table_data, index=models, columns=col_index)
    df_table.index.name = "Model"
    return df_table


def format_performance_markdown_table(table_df: pd.DataFrame) -> str:
    """Format benchmark performance DataFrame into GitHub-flavored Markdown.

    Args:
        table_df: DataFrame output by generate_performance_table.

    Returns:
        Markdown table string formatted with 3-sig-fig numbers.
    """
    lines = [
        "| Model | All MAE | All MdAE | All MAPE (%) | All MdAPE (%) | All SMAPE (%) | Vehicle MAE | Vehicle MdAE | Vehicle MAPE (%) | Vehicle MdAPE (%) | Vehicle SMAPE (%) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for model, row in table_df.iterrows():
        all_vals = [format_sci(row[("All data", s)]) for s in ["MAE", "MdAE", "MAPE", "MdAPE", "SMAPE"]]
        veh_vals = [format_sci(row[("Per vehicle mean", s)]) for s in ["MAE", "MdAE", "MAPE", "MdAPE", "SMAPE"]]
        line = f"| **{model}** | " + " | ".join(all_vals) + " | " + " | ".join(veh_vals) + " |"
        lines.append(line)

    return "\n".join(lines)


def compute_participant_summary_stats(
    performance_participants: pd.DataFrame,
    dataset: str = "Test",
    model_name: str = "HistGradientBoostingRegressor",
) -> pd.DataFrame:
    """Compute summary statistics for participant-level metric distributions.

    Args:
        performance_participants: Participant-level performance table.
        dataset: Target split ('Test' or 'Train').
        model_name: Regressor model name.

    Returns:
        DataFrame containing distribution quantiles (mean, std, min, p25, median, p75, max).
    """
    stats_dict = {}
    for score in SCORE_KEYS:
        col = (model_name, dataset, score)
        if col in performance_participants.columns:
            s = performance_participants[col].dropna()
            stats_dict[score] = {
                "mean": float(s.mean()),
                "std": float(s.std(ddof=1)),
                "min": float(s.min()),
                "q25": float(s.quantile(0.25)),
                "median": float(s.median()),
                "q75": float(s.quantile(0.75)),
                "max": float(s.max()),
            }

    return pd.DataFrame.from_dict(stats_dict, orient="index")


def compute_vehicle_characteristics(
    performance_participants: pd.DataFrame,
    df: pd.DataFrame,
    model_name: str = "HistGradientBoostingRegressor",
    dataset: str = "Test",
    score: str = "medape",
) -> Dict[str, Any]:
    """Compute user charging characteristics and fit linear regressions vs error.

    Calculates:
    1. Duration variability: (P90 - P10) / Median session duration
    2. Battery capacity: Mean CarKWh
    3. Session start hour: Mean StartHourNorm * 24.0

    Performs Ordinary Least Squares (OLS) regression against model MdAPE.

    Args:
        performance_participants: MultiIndex performance table by participant.
        df: Full session DataFrame with canonical features.
        model_name: Name of model to analyze. Defaults to 'HistGradientBoostingRegressor'.
        dataset: Dataset split. Defaults to 'Test'.
        score: Error score to evaluate. Defaults to 'medape'.

    Returns:
        Dictionary containing per-participant feature values and OLS fit statistics.
    """
    participants = performance_participants.index.tolist()
    y_series = performance_participants.loc[:, (model_name, dataset, score)]

    # Compute characteristics per participant
    grouped = df.groupby("ParticipantID")

    # 1. Variation of session duration (P90-P10 / Median)
    duration_stats = grouped["Duration"].agg(
        p10=lambda s: s.quantile(0.10),
        p90=lambda s: s.quantile(0.90),
        med=lambda s: s.median(),
    )
    # Avoid zero division
    duration_var = (duration_stats["p90"] - duration_stats["p10"]) / duration_stats["med"].replace(0, np.nan)
    duration_var = duration_var.reindex(participants)

    # 2. Battery size (CarKWh)
    battery_size = grouped["CarKWh"].mean().reindex(participants)

    # 3. Session start hour (StartHour) = StartHourNorm * 24
    if "StartHourNorm" in df.columns:
        start_hour = (grouped["StartHourNorm"].mean() * 24.0).reindex(participants)
    else:
        start_hour = grouped["StartHour"].mean().reindex(participants)

    features_meta = [
        ("duration_variability", "Variation of session duration (P90-P10 / Median)", duration_var),
        ("battery_size", "Battery size (CarKWh)", battery_size),
        ("start_hour", "Session start hour (StartHour)", start_hour),
    ]

    regression_results = {}
    for key, name, x_series in features_meta:
        valid_mask = x_series.notna() & y_series.notna()
        x_clean = x_series[valid_mask].astype(float)
        y_clean = y_series[valid_mask].astype(float)

        ols_summary = {
            "name": name,
            "x": x_clean,
            "y": y_clean,
            "slope": np.nan,
            "intercept": np.nan,
            "r2": np.nan,
            "p_value": np.nan,
            "significant": False,
        }

        if len(x_clean) > 2 and x_clean.nunique() > 1:
            try:
                X_mat = sm.add_constant(x_clean)
                ols = sm.OLS(y_clean, X_mat).fit()
                ols_summary["slope"] = float(ols.params.iloc[1])
                ols_summary["intercept"] = float(ols.params.iloc[0])
                ols_summary["r2"] = float(ols.rsquared)
                ols_summary["p_value"] = float(ols.f_pvalue)
                ols_summary["significant"] = bool(ols.f_pvalue < 0.05)
            except Exception as e:
                logger.warning("OLS regression failed for %s: %s", key, e)

        regression_results[key] = ols_summary

    return regression_results


def compute_error_over_time(
    predictions: pd.DataFrame,
    target: str,
    model_name: str = "HistGradientBoostingRegressor",
    dataset: str = "Test",
) -> Dict[str, Any]:
    """Calculate relative change in MAE from first to second half of test set.

    Reproduces reference temporal degradation analysis:
    Per participant, splits test sessions chronologically into first and second half:
    relative_change = (first.mean() - last.mean()) / df.mean()

    Args:
        predictions: Predictions DataFrame.
        target: Target name ('Duration' or 'ConsumedkWh').
        model_name: Model name.
        dataset: Split name ('Test').

    Returns:
        Dict with relative_changes array, mean, std, and participant counts.
    """
    test_preds = predictions[predictions["dataset"] == dataset].copy()
    test_preds = test_preds.sort_values(by="StartTime")
    test_preds["error"] = np.abs(test_preds[model_name] - test_preds[target])

    relative_changes = []
    pids = []

    for pid, dfp in test_preds.groupby("ParticipantID"):
        err = dfp["error"].values
        n = len(err)
        if n >= 2:
            split = n // 2
            first_half = err[:split]
            second_half = err[split:]
            mean_all = np.mean(err)
            if mean_all > 0:
                rel = (np.mean(first_half) - np.mean(second_half)) / mean_all
                relative_changes.append(float(rel))
                pids.append(pid)

    rel_arr = np.array(relative_changes, dtype=float)
    return {
        "relative_changes": rel_arr,
        "participants": pids,
        "mean_relative_change": float(np.nanmean(rel_arr)) if len(rel_arr) > 0 else 0.0,
        "std_relative_change": float(np.nanstd(rel_arr)) if len(rel_arr) > 0 else 0.0,
        "n_evaluated_vehicles": len(rel_arr),
    }


def compute_seasonality(
    predictions: pd.DataFrame,
    target: str,
    model_name: str = "HistGradientBoostingRegressor",
) -> Dict[str, Dict[str, Any]]:
    """Compute monthly average MAPE across Train and Test datasets.

    Args:
        predictions: Predictions DataFrame with StartTime and target values.
        target: Prediction target.
        model_name: Model to evaluate.

    Returns:
        Dict mapping split ('Train', 'Test') to monthly dates and mean MAPE values.
    """
    df_sorted = predictions.copy()
    df_sorted = df_sorted.set_index("StartTime").sort_index()

    actual = df_sorted[target].values
    predicted = df_sorted[model_name].values
    with np.errstate(divide="ignore", invalid="ignore"):
        ape = np.abs(actual - predicted) / np.abs(actual) * 100.0

    df_sorted["tmp_ape"] = ape

    out = {}
    for dataset in ["Train", "Test"]:
        sub = df_sorted[df_sorted["dataset"] == dataset]["tmp_ape"]
        dates = []
        means = []
        for (month, year), group in sub.groupby([sub.index.month, sub.index.year], sort=False):
            dates.append(dt.datetime(year=year, month=month, day=1))
            means.append(float(np.nanmean(group.values)))

        # Sort chronologically by date
        sorted_pairs = sorted(zip(dates, means), key=lambda x: x[0])
        out[dataset] = {
            "dates": [p[0] for p in sorted_pairs],
            "means": [p[1] for p in sorted_pairs],
        }

    return out


def compute_data_in_splits(predictions: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Compute average data availability (session count and timespan) in splits.

    Faithfully reproduces reference average_data_in_split routine:
    Calculates mean ± std (ddof=0) across vehicles for session count and timespan days.

    Args:
        predictions: Predictions DataFrame containing ParticipantID, StartTime, dataset.

    Returns:
        Dict with 'Train' and 'Test' statistics: session_count_mean, session_count_std,
        timespan_days_mean, timespan_days_std.
    """
    stats = {}
    for dataset in ["Train", "Test"]:
        sub = predictions[predictions["dataset"] == dataset]
        session_counts = []
        timespans = []
        for _, dfp in sub.groupby("ParticipantID"):
            session_counts.append(len(dfp))
            delta = (dfp["StartTime"].max() - dfp["StartTime"].min()).total_seconds() / 86400.0
            timespans.append(delta)

        counts_arr = np.array(session_counts, dtype=float)
        span_arr = np.array(timespans, dtype=float)

        stats[dataset] = {
            "session_count_mean": float(np.mean(counts_arr)) if len(counts_arr) > 0 else 0.0,
            "session_count_std": float(np.std(counts_arr)) if len(counts_arr) > 0 else 0.0,
            "timespan_days_mean": float(np.mean(span_arr)) if len(span_arr) > 0 else 0.0,
            "timespan_days_std": float(np.std(span_arr)) if len(span_arr) > 0 else 0.0,
            "n_vehicles": len(session_counts),
        }

    return stats

