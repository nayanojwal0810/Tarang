"""Experiment execution engine for TARANG Stage 2 predictive modeling.

Handles input validation, chronological train/test splitting, model training,
prediction generation, aggregate and participant-level metric calculations,
and artifact persistence matching established benchmark specifications.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from tarang.config import (
    CANONICAL_COLUMNS,
    DURATION_MODELING_DIR,
    ENERGY_MODELING_DIR,
    FEATURE_NAMES,
    MODEL_NAMES,
    SPLIT_RANDOM_STATE,
    TARGET_DURATION,
    TARGET_ENERGY,
    TEST_SPLIT_SIZE,
)
from tarang.modeling.metrics import calculate_metrics
from tarang.modeling.models import get_benchmark_models

logger = logging.getLogger("tarang.modeling.experiment")


def validate_modeling_input(df: pd.DataFrame, target: str) -> None:
    """Validate canonical columns, feature predictors, and row integrity.

    Args:
        df: Canonical features DataFrame.
        target: Target column name ('Duration' or 'ConsumedkWh').

    Raises:
        ValueError: If input DataFrame violates required schema or ordering.
    """
    if df is None or df.empty:
        raise ValueError("Input DataFrame is empty or None.")

    # Validate target presence
    if target not in [TARGET_DURATION, TARGET_ENERGY]:
        raise ValueError(
            f"Invalid target '{target}'. Expected '{TARGET_DURATION}' or '{TARGET_ENERGY}'."
        )
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame columns.")

    # Validate canonical columns
    missing_canonical = [c for c in CANONICAL_COLUMNS if c not in df.columns]
    if missing_canonical:
        raise ValueError(
            f"Canonical columns missing from modeling dataset: {missing_canonical}"
        )

    # Validate 13 authoritative predictors in exact sequence
    missing_features = [f for f in FEATURE_NAMES if f not in df.columns]
    if missing_features:
        raise ValueError(
            f"Required predictors missing from modeling dataset: {missing_features}"
        )

    # Validate target values
    if df[target].isna().any():
        n_missing = int(df[target].isna().sum())
        raise ValueError(
            f"Target '{target}' contains {n_missing} missing values; imputation is disallowed."
        )

    # Validate feature values
    missing_feature_counts = df[FEATURE_NAMES].isna().sum()
    if (missing_feature_counts > 0).any():
        bad_cols = missing_feature_counts[missing_feature_counts > 0].to_dict()
        raise ValueError(
            f"Predictors contain missing values; imputation is disallowed: {bad_cols}"
        )

    # Validate chronological ordering
    if not df["StartTime"].is_monotonic_increasing:
        raise ValueError(
            "DataFrame is not sorted chronologically by StartTime. "
            "Canonical ordering must be preserved."
        )


def run_experiment(
    sessions: pd.DataFrame,
    target: str = TARGET_DURATION,
    output_dir: Optional[Path] = None,
    saved: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Train and evaluate the six benchmark models for the specified target.

    Follows the reference methodology:
    1. Validates canonical input dataset.
    2. Executes chronological 80/20 train/test split without shuffling.
    3. Trains/loads six benchmark models on the 13 canonical predictors.
    4. Records train and test predictions in predictions.csv.
    5. Calculates 16 performance metrics for ALL sessions and per participant.
    6. Persists trained joblib models, performance.csv, and predictions.csv.

    Args:
        sessions: Canonical features DataFrame.
        target: Prediction target ('Duration' or 'ConsumedkWh').
        output_dir: Directory where models and CSV files are stored. Defaults to
            target-specific folder under data/processed/electric_nation/modeling/.
        saved: If True and model file exists, load rather than refit.

    Returns:
        Tuple of (performance_df, predictions_df).
    """
    validate_modeling_input(sessions, target)

    if output_dir is None:
        output_dir = (
            DURATION_MODELING_DIR if target == TARGET_DURATION else ENERGY_MODELING_DIR
        )
    output_dir = Path(output_dir)
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Executing experiment for target '%s' -> %s", target, output_dir)

    # Chronological 80/20 split preserving original order
    X = sessions.copy()
    y = sessions[target].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT_SIZE, random_state=SPLIT_RANDOM_STATE, shuffle=False
    )
    logger.info(
        "Chronological split complete: Train=%d (80%%), Test=%d (20%%)",
        len(X_train),
        len(X_test),
    )

    # Restrict regressor features to exact 13 predictors
    X_train_features = X_train[FEATURE_NAMES].copy()
    X_test_features = X_test[FEATURE_NAMES].copy()

    # Predictions DataFrame preserving canonical columns + split + predictions
    predictions = sessions.copy()
    predictions["dataset"] = ""
    predictions.loc[X_train.index, "dataset"] = "Train"
    predictions.loc[X_test.index, "dataset"] = "Test"

    # MultiIndex performance tracking dictionary: perf_dict[pid][(model, dataset, metric)]
    perf_dict: Dict[str, Dict[Tuple[str, str, str], float]] = {}

    def set_perf(pid: str, col_tuple: Tuple[str, str, str], val: float) -> None:
        if pid not in perf_dict:
            perf_dict[pid] = {}
        perf_dict[pid][col_tuple] = val

    # Benchmark models in established order
    benchmark_models = get_benchmark_models(target=target)

    for model_name, model in benchmark_models:
        model_path = models_dir / f"{model_name}.joblib"
        if saved and model_path.exists():
            logger.info(" - Loading model '%s' from %s", model_name, model_path)
            model = joblib.load(model_path)
        else:
            logger.info(" - Training model '%s'...", model_name)
            model.fit(X_train_features, y_train)
            joblib.dump(model, model_path)

        # Generate predictions for both Train and Test
        y_pred_train = model.predict(X_train_features)
        y_pred_test = model.predict(X_test_features)

        # Ensure 1D series aligned with original indices
        y_pred_train_series = pd.Series(
            np.asarray(y_pred_train, dtype=np.float64), index=y_train.index
        )
        y_pred_test_series = pd.Series(
            np.asarray(y_pred_test, dtype=np.float64), index=y_test.index
        )

        predictions.loc[y_pred_train_series.index, model_name] = y_pred_train_series
        predictions.loc[y_pred_test_series.index, model_name] = y_pred_test_series

        # Evaluate on Train and Test
        for dataset_name, x_split, y_true_split, y_pred_split in [
            ("Train", X_train, y_train, y_pred_train_series),
            ("Test", X_test, y_test, y_pred_test_series),
        ]:
            # Aggregate evaluation ('ALL')
            all_metrics = calculate_metrics(y_true_split, y_pred_split)
            for m_name, m_val in all_metrics.items():
                set_perf("ALL", (model_name, dataset_name, m_name), m_val)
            set_perf(
                "ALL",
                ("number_sessions", dataset_name, "none"),
                float(len(x_split.index)),
            )

            # Per-participant evaluation
            for pid, dfp in x_split.groupby("ParticipantID"):
                p_true = y_true_split.loc[dfp.index]
                p_pred = y_pred_split.loc[dfp.index]
                p_metrics = calculate_metrics(p_true, p_pred)
                for m_name, m_val in p_metrics.items():
                    set_perf(str(pid), (model_name, dataset_name, m_name), m_val)
                set_perf(
                    str(pid),
                    ("number_sessions", dataset_name, "none"),
                    float(len(dfp.index)),
                )

        train_medape = perf_dict.get("ALL", {}).get(
            (model_name, "Train", "medape"), float("nan")
        )
        test_medape = perf_dict.get("ALL", {}).get(
            (model_name, "Test", "medape"), float("nan")
        )
        logger.info(
            "   Completed '%s' -> Train MedAPE: %.4f | Test MedAPE: %.4f",
            model_name,
            train_medape,
            test_medape,
        )

    # Construct performance DataFrame from dictionary
    # Order rows: 'ALL' first, followed by sorted ParticipantIDs
    pids = [p for p in sorted(perf_dict.keys()) if p != "ALL"]
    row_order = ["ALL"] + pids

    # Construct complete list of 3-level MultiIndex columns in consistent order
    metric_order = [
        "medae",
        "mae",
        "medape",
        "mape",
        "rmse",
        "smape",
        "q1-ae",
        "q2-ae",
        "q3-ae",
        "wl-ae",
        "wh-ae",
        "q1-ape",
        "q2-ape",
        "q3-ape",
        "wl-ape",
        "wh-ape",
    ]
    column_tuples: List[Tuple[str, str, str]] = []
    for m_name in MODEL_NAMES:
        for ds_name in ["Train", "Test"]:
            for sc in metric_order:
                column_tuples.append((m_name, ds_name, sc))
    column_tuples.append(("number_sessions", "Train", "none"))
    column_tuples.append(("number_sessions", "Test", "none"))

    multi_cols = pd.MultiIndex.from_tuples(column_tuples)
    performance = pd.DataFrame(index=row_order, columns=multi_cols, dtype=np.float64)

    for pid in row_order:
        p_row = perf_dict.get(pid, {})
        for col_tup in column_tuples:
            if col_tup in p_row:
                performance.loc[pid, col_tup] = p_row[col_tup]

    # Save performance.csv and predictions.csv
    perf_path = output_dir / "performance.csv"
    pred_path = output_dir / "predictions.csv"

    performance.to_csv(perf_path)
    predictions.to_csv(pred_path)
    logger.info("Saved performance matrix to: %s", perf_path)
    logger.info("Saved predictions to: %s", pred_path)

    return performance, predictions

