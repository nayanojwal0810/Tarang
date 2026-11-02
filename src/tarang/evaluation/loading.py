"""Artifact loading and validation utilities for TARANG Stage 3 evaluation.

Provides validated loaders for Stage 2 modeling outputs:
- MultiIndex performance tables (`performance.csv`)
- Canonical and model prediction records (`predictions.csv`)
- Trained HGBR model artifacts (`HistGradientBoostingRegressor.joblib`)
- Processed feature datasets (`features.parquet`)
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import pandas as pd

from tarang.config import (
    CANONICAL_COLUMNS,
    DURATION_MODELING_DIR,
    ENERGY_MODELING_DIR,
    FEATURE_NAMES,
    MODEL_NAMES,
    PROCESSED_FEATURES_FILE,
    TARGET_DURATION,
    TARGET_ENERGY,
)

logger = logging.getLogger("tarang.evaluation.loading")

PERCENTAGE_SCORES: List[str] = [
    "mape",
    "medape",
    "smape",
    "q1-ape",
    "q2-ape",
    "q3-ape",
    "wl-ape",
    "wh-ape",
]


@dataclass
class EvaluationData:
    """Encapsulates loaded artifacts for model evaluation and analysis."""

    target: str
    performance_all: pd.Series
    performance_participants: pd.DataFrame
    predictions: pd.DataFrame
    hgbr_model: Any
    features: pd.DataFrame


def normalize_target_name(target: str) -> str:
    """Normalize input target name to canonical constant.

    Args:
        target: Target name, case-insensitive (e.g. 'duration', 'Duration',
            'energy', 'consumedkwh', 'ConsumedkWh').

    Returns:
        Canonical target string ('Duration' or 'ConsumedkWh').

    Raises:
        ValueError: If target name cannot be recognized.
    """
    clean = target.strip().lower()
    if clean in ["duration", "pluggedintime"]:
        return TARGET_DURATION
    if clean in ["consumedkwh", "energy", "consumption"]:
        return TARGET_ENERGY
    raise ValueError(
        f"Unknown target '{target}'. Expected '{TARGET_DURATION}' or '{TARGET_ENERGY}'."
    )


def get_target_modeling_dir(target: str, modeling_dir: Optional[Path] = None) -> Path:
    """Determine the modeling artifact directory for a given target."""
    canonical_target = normalize_target_name(target)
    if modeling_dir is not None:
        return Path(modeling_dir)
    return (
        DURATION_MODELING_DIR
        if canonical_target == TARGET_DURATION
        else ENERGY_MODELING_DIR
    )


def load_performance_data(
    perf_path: Path,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Load and format MultiIndex performance results.

    Args:
        perf_path: Path to performance.csv.

    Returns:
        Tuple of (performance_all: pd.Series, performance_participants: pd.DataFrame).
        Percentage scores are scaled from fractions to percentage values (0-100%).

    Raises:
        FileNotFoundError: If performance file does not exist.
        ValueError: If file format or MultiIndex levels are invalid.
    """
    if not perf_path.exists():
        raise FileNotFoundError(f"Performance file not found at: {perf_path}")

    perf_df = pd.read_csv(perf_path, index_col=0, header=[0, 1, 2])
    if "ALL" not in perf_df.index:
        raise ValueError("performance.csv must contain an 'ALL' aggregate row.")

    # Scale percentage metrics to percentage scale (x 100)
    for score in PERCENTAGE_SCORES:
        if score in perf_df.columns.get_level_values(2):
            perf_df.loc[:, (slice(None), slice(None), score)] *= 100.0

    perf_all = perf_df.loc["ALL", :]
    perf_participants = perf_df.drop("ALL").copy()

    return perf_all, perf_participants


def load_predictions_data(pred_path: Path) -> pd.DataFrame:
    """Load and validate predictions dataset.

    Args:
        pred_path: Path to predictions.csv.

    Returns:
        Parsed DataFrame with datetime indices and required columns.

    Raises:
        FileNotFoundError: If predictions file does not exist.
        ValueError: If required columns or splits are missing.
    """
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions file not found at: {pred_path}")

    preds = pd.read_csv(
        pred_path,
        index_col=0,
        parse_dates=["StartTime", "StopTime"],
    )

    required_cols = ["StartTime", "StopTime", "ParticipantID", "dataset"]
    missing = [c for c in required_cols if c not in preds.columns]
    if missing:
        raise ValueError(f"Predictions file missing required columns: {missing}")

    if not {"Train", "Test"}.issubset(set(preds["dataset"].unique())):
        raise ValueError("Predictions file must contain both 'Train' and 'Test' records.")

    return preds


def load_hgbr_model(model_dir: Path) -> Any:
    """Load trained HistGradientBoostingRegressor model artifact.

    Args:
        model_dir: Directory containing models/HistGradientBoostingRegressor.joblib.

    Returns:
        Loaded sklearn HistGradientBoostingRegressor model.

    Raises:
        FileNotFoundError: If model file does not exist.
    """
    model_path = model_dir / "HistGradientBoostingRegressor.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"HGBR model artifact not found at: {model_path}")
    return joblib.load(model_path)


def load_evaluation_data(
    target: str,
    modeling_dir: Optional[Path] = None,
    features_file: Optional[Path] = None,
) -> EvaluationData:
    """Load and validate all artifacts required for Stage 3 evaluation.

    Args:
        target: Prediction target ('Duration' or 'ConsumedkWh').
        modeling_dir: Optional override for modeling output folder.
        features_file: Optional override for processed features file.

    Returns:
        Populated EvaluationData instance.
    """
    canonical_target = normalize_target_name(target)
    target_dir = get_target_modeling_dir(canonical_target, modeling_dir)

    perf_path = target_dir / "performance.csv"
    pred_path = target_dir / "predictions.csv"
    models_dir = target_dir / "models"

    feat_path = Path(features_file) if features_file else PROCESSED_FEATURES_FILE

    logger.info("Loading Stage 3 evaluation data for target '%s'", canonical_target)
    logger.info(" - Target directory: %s", target_dir)

    perf_all, perf_participants = load_performance_data(perf_path)
    predictions = load_predictions_data(pred_path)
    hgbr_model = load_hgbr_model(models_dir)

    if feat_path.exists():
        features_df = pd.read_parquet(feat_path)
    else:
        logger.warning(
            "Features parquet file not found at %s. Using canonical subset from predictions.",
            feat_path,
        )
        canonical_present = [c for c in CANONICAL_COLUMNS if c in predictions.columns]
        features_df = predictions[canonical_present].copy()

    return EvaluationData(
        target=canonical_target,
        performance_all=perf_all,
        performance_participants=perf_participants,
        predictions=predictions,
        hgbr_model=hgbr_model,
        features=features_df,
    )

