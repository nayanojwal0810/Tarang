"""Subgroup analysis comparing same-day versus overnight charging sessions.

Implements UTC date partitioning:
- Same-day: StopTime.dt.date == StartTime.dt.date (day difference == 0)
- Overnight: StopTime.dt.date > StartTime.dt.date (day difference > 0)

Trains fresh HistGradientBoostingRegressor models for each subgroup using
chronological 80/20 train/test splits, computes benchmark evaluation metrics,
and saves comparison tables, subgroup models, and held-out test predictions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

from tarang.config import (
    EVALUATION_DATA_DIR,
    FEATURE_NAMES,
    MODEL_RANDOM_STATE,
    SPLIT_RANDOM_STATE,
    TARGET_DURATION,
    TARGET_ENERGY,
    TEST_SPLIT_SIZE,
)
from tarang.modeling.metrics import calculate_metrics

logger = logging.getLogger("tarang.evaluation.subgroup_analysis")


def partition_sessions_by_timing(sessions: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Partition session records into same-day and overnight subsets.

    Criteria:
    - Same-day: (StopTime.dt.date - StartTime.dt.date).dt.days == 0
    - Overnight: (StopTime.dt.date - StartTime.dt.date).dt.days > 0

    Args:
        sessions: DataFrame containing StartTime and StopTime datetime columns.

    Returns:
        Tuple of (df_sameday, df_overnight).
    """
    if not pd.api.types.is_datetime64_any_dtype(sessions["StartTime"]):
        sessions["StartTime"] = pd.to_datetime(sessions["StartTime"])
    if not pd.api.types.is_datetime64_any_dtype(sessions["StopTime"]):
        sessions["StopTime"] = pd.to_datetime(sessions["StopTime"])

    days_diff = (sessions["StopTime"].dt.date - sessions["StartTime"].dt.date).apply(
        lambda d: d.days if hasattr(d, "days") else int(d)
    )

    mask_sameday = days_diff == 0
    mask_overnight = days_diff > 0

    df_sameday = sessions[mask_sameday].copy()
    df_overnight = sessions[mask_overnight].copy()

    logger.info(
        "Timing partition: Total=%d | Same-day=%d (%.1f%%) | Overnight=%d (%.1f%%)",
        len(sessions),
        len(df_sameday),
        100.0 * len(df_sameday) / len(sessions) if len(sessions) else 0.0,
        len(df_overnight),
        100.0 * len(df_overnight) / len(sessions) if len(sessions) else 0.0,
    )
    return df_sameday, df_overnight


def train_and_evaluate_subgroup_hgbr(
    sessions_subgroup: pd.DataFrame,
    target: str,
    subgroup_name: str,
    models_dir: Optional[Path] = None,
) -> Tuple[Dict[str, float], HistGradientBoostingRegressor, pd.DataFrame]:
    """Train fresh HGBR on subgroup and evaluate performance on test split.

    Args:
        sessions_subgroup: Filtered subgroup sessions.
        target: Target column ('Duration' or 'ConsumedkWh').
        subgroup_name: Name identifier ('same_day' or 'overnight').
        models_dir: Optional path to save trained joblib model.

    Returns:
        Tuple of (metrics_dict, trained_model, predictions_df).
    """
    # Chronological sort
    df_sorted = sessions_subgroup.sort_values(by="StartTime").copy()

    X = df_sorted.copy()
    y = df_sorted[target].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT_SIZE, random_state=SPLIT_RANDOM_STATE, shuffle=False
    )

    model = HistGradientBoostingRegressor(
        early_stopping=True,
        learning_rate=0.1,
        min_samples_leaf=100,
        max_iter=100,
        loss="absolute_error",
        random_state=np.random.RandomState(MODEL_RANDOM_STATE),
    )

    X_train_feat = X_train[FEATURE_NAMES]
    X_test_feat = X_test[FEATURE_NAMES]

    model.fit(X_train_feat, y_train)
    y_pred_test = model.predict(X_test_feat)

    # Calculate metrics on the held-out test split
    raw_metrics = calculate_metrics(y_true=y_test.values, y_pred=y_pred_test)

    # Convert percentage scores to percentage scale
    test_metrics = {
        "n_sessions": len(df_sorted),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "mae": float(raw_metrics["mae"]),
        "medae": float(raw_metrics["medae"]),
        "mape": float(raw_metrics["mape"] * 100.0),
        "medape": float(raw_metrics["medape"] * 100.0),
        "smape": float(raw_metrics["smape"] * 100.0),
    }

    # Save model artifact if dir provided
    if models_dir is not None:
        models_dir = Path(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        model_save_path = models_dir / f"HistGradientBoostingRegressor_{subgroup_name}.joblib"
        joblib.dump(model, model_save_path)
        logger.info("Saved subgroup model to %s", model_save_path)

    # Prepare held-out test predictions DataFrame (no training records included)
    preds_df = X_test.copy()
    if "dataset" in preds_df.columns:
        preds_df = preds_df.drop(columns=["dataset"])
    preds_df["subgroup"] = subgroup_name
    preds_df["subgroup_split"] = "Test"
    preds_df[f"actual_{target}"] = y_test.values
    preds_df[f"predicted_{target}"] = y_pred_test

    return test_metrics, model, preds_df


def run_subgroup_analysis(
    sessions: pd.DataFrame,
    target: str,
    performance_all: Optional[pd.Series] = None,
    output_dir: Optional[Path] = None,
    evaluation_data_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Execute complete subgroup analysis comparing same-day vs overnight sessions.

    Trains fresh HGBR models for same-day and overnight charging sessions,
    evaluates performance on each subgroup's held-out test split, persists
    models and prediction CSVs, and exports comparison summary metrics.

    Args:
        sessions: Full canonical sessions DataFrame.
        target: Prediction target ('Duration' or 'ConsumedkWh').
        performance_all: MultiIndex performance Series for aggregate benchmark metrics.
        output_dir: Path to save same_day_overnight_metrics.csv.
        evaluation_data_dir: Path to save subgroup joblib models and test prediction CSVs.

    Returns:
        Comparison DataFrame summarizing sample sizes and evaluation metrics.
    """
    logger.info("Running subgroup analysis for target '%s'", target)

    df_sameday, df_overnight = partition_sessions_by_timing(sessions)

    target_data_dir: Optional[Path] = None
    if evaluation_data_dir is not None:
        target_data_dir = Path(evaluation_data_dir)
    elif EVALUATION_DATA_DIR is not None:
        sub_folder = "duration" if target == TARGET_DURATION else "energy"
        target_data_dir = EVALUATION_DATA_DIR / sub_folder

    models_dir = target_data_dir / "models" if target_data_dir is not None else None

    metrics_sameday, model_sameday, preds_sameday = train_and_evaluate_subgroup_hgbr(
        df_sameday, target, "same_day", models_dir=models_dir
    )
    metrics_overnight, model_overnight, preds_overnight = train_and_evaluate_subgroup_hgbr(
        df_overnight, target, "overnight", models_dir=models_dir
    )

    # Persist subgroup held-out test predictions
    if target_data_dir is not None:
        target_data_dir.mkdir(parents=True, exist_ok=True)
        same_day_csv = target_data_dir / "same_day_predictions.csv"
        overnight_csv = target_data_dir / "overnight_predictions.csv"
        preds_sameday.to_csv(same_day_csv, index=False)
        preds_overnight.to_csv(overnight_csv, index=False)
        logger.info(
            "Saved same-day held-out test predictions (%d records) to %s",
            len(preds_sameday),
            same_day_csv,
        )
        logger.info(
            "Saved overnight held-out test predictions (%d records) to %s",
            len(preds_overnight),
            overnight_csv,
        )

    # Retrieve all-session benchmark metrics
    if performance_all is not None:
        mae_all = float(performance_all.get(("HistGradientBoostingRegressor", "Test", "mae"), np.nan))
        medae_all = float(performance_all.get(("HistGradientBoostingRegressor", "Test", "medae"), np.nan))
        mape_all = float(performance_all.get(("HistGradientBoostingRegressor", "Test", "mape"), np.nan))
        medape_all = float(performance_all.get(("HistGradientBoostingRegressor", "Test", "medape"), np.nan))
        smape_all = float(performance_all.get(("HistGradientBoostingRegressor", "Test", "smape"), np.nan))
    else:
        mae_all, medae_all, mape_all, medape_all, smape_all = np.nan, np.nan, np.nan, np.nan, np.nan

    n_total = len(sessions)
    n_test_all = int(np.round(n_total * TEST_SPLIT_SIZE))
    n_train_all = n_total - n_test_all

    records = [
        {
            "subgroup": "All Sessions",
            "n_sessions": n_total,
            "n_train": n_train_all,
            "n_test": n_test_all,
            "mae": mae_all,
            "medae": medae_all,
            "mape": mape_all,
            "medape": medape_all,
            "smape": smape_all,
        },
        {
            "subgroup": "Same-day Sessions",
            "n_sessions": metrics_sameday["n_sessions"],
            "n_train": metrics_sameday["n_train"],
            "n_test": metrics_sameday["n_test"],
            "mae": metrics_sameday["mae"],
            "medae": metrics_sameday["medae"],
            "mape": metrics_sameday["mape"],
            "medape": metrics_sameday["medape"],
            "smape": metrics_sameday["smape"],
        },
        {
            "subgroup": "Overnight Sessions",
            "n_sessions": metrics_overnight["n_sessions"],
            "n_train": metrics_overnight["n_train"],
            "n_test": metrics_overnight["n_test"],
            "mae": metrics_overnight["mae"],
            "medae": metrics_overnight["medae"],
            "mape": metrics_overnight["mape"],
            "medape": metrics_overnight["medape"],
            "smape": metrics_overnight["smape"],
        },
    ]

    df_metrics = pd.DataFrame(records)

    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        csv_path = out_path / "same_day_overnight_metrics.csv"
        df_metrics.to_csv(csv_path, index=False)
        logger.info("Saved subgroup metrics table to %s", csv_path)

    return df_metrics
