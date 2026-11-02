"""TreeSHAP explainability engine for TARANG Stage 3 evaluation.

Computes exact TreeSHAP feature importances on the test split for trained
HistGradientBoostingRegressor models, generating global importance rankings
and participant-level feature attribution distributions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import shap

from tarang.config import FEATURE_NAMES

logger = logging.getLogger("tarang.evaluation.explainability")

FEATURE_DISPLAY_MAP: Dict[str, str] = {
    "weekday": "weekday",
    "weekend": "weekend",
    "holiday": "holiday",
    "month": "month",
    "StartHourNorm": "startHourNorm",
    "StartCos": "startCos",
    "StartSin": "startSin",
    "CarKWh": "carKWh",
    "CarKW": "carKW",
    "TimeSinceLastStop": "timeSinceLastStop",
    "sessionsToday": "sessionsToday",
    "LastDuration": "lastDuration",
    "LastConsumedkWh": "lastConsumedKWh",
}


def compute_tree_shap_importance(
    model: Any,
    test_df: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
    output_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate global and participant-level TreeSHAP feature importance.

    Computes:
    1. Global importance: Mean absolute SHAP value across all test sessions.
    2. Vehicle-level importance: Mean absolute SHAP value per participant.

    Args:
        model: Trained HistGradientBoostingRegressor model artifact.
        test_df: Test split DataFrame containing 13 predictors and ParticipantID.
        feature_names: Feature names to evaluate. Defaults to canonical FEATURE_NAMES.
        output_dir: Optional folder to persist shap_global_importance.csv and
            shap_vehicle_importance.csv.

    Returns:
        Tuple of (df_global_importance, df_vehicle_importance).
    """
    if feature_names is None:
        feature_names = FEATURE_NAMES

    missing = [f for f in feature_names if f not in test_df.columns]
    if missing:
        raise ValueError(f"Test dataset is missing required features: {missing}")

    if "ParticipantID" not in test_df.columns:
        raise ValueError("Test dataset must contain 'ParticipantID' column.")

    X_test = test_df[feature_names].copy()
    logger.info("Computing TreeSHAP values for %d test sessions...", len(X_test))

    explainer = shap.Explainer(model)
    shap_explanation = explainer(X_test)
    raw_shap_matrix = shap_explanation.values  # Shape: (N, 13)

    # 1. Global feature importance across all test sessions
    global_mean_abs = np.mean(np.abs(raw_shap_matrix), axis=0)
    df_global = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean_abs_shap": global_mean_abs,
        }
    ).sort_values(by="importance_mean_abs_shap", ascending=False).reset_index(drop=True)

    # 2. Vehicle-level importance distribution
    test_df_reset = test_df.reset_index(drop=True)
    participants = test_df_reset["ParticipantID"].unique()

    vehicle_rows = []
    vehicle_pids = []

    for pid in participants:
        mask = (test_df_reset["ParticipantID"] == pid).values
        if mask.any():
            veh_shap = raw_shap_matrix[mask]
            veh_mean_abs = np.mean(np.abs(veh_shap), axis=0)
            vehicle_rows.append(veh_mean_abs)
            vehicle_pids.append(pid)

    df_vehicle = pd.DataFrame(
        vehicle_rows,
        index=vehicle_pids,
        columns=feature_names,
    )
    df_vehicle.index.name = "ParticipantID"

    # Persist CSVs if output directory provided
    if output_dir is not None:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        global_csv = out_path / "shap_global_importance.csv"
        vehicle_csv = out_path / "shap_vehicle_importance.csv"

        df_global.to_csv(global_csv, index=False)
        df_vehicle.to_csv(vehicle_csv)
        logger.info("Saved SHAP global importance to %s", global_csv)
        logger.info("Saved SHAP vehicle importance to %s", vehicle_csv)

    return df_global, df_vehicle

