"""Inference execution engine for TARANG predictive models.

Provides single-session and batch prediction APIs supporting both Duration and ConsumedkWh
targets and all six benchmark model estimators.
Strictly returns raw estimator predictions without post-processing, clipping, or thresholding.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd

from tarang.config import TARGET_DURATION
from tarang.inference.feature_builder import build_inference_features
from tarang.inference.loading import load_model, normalize_target_name
from tarang.inference.validation import validate_inference_features

logger = logging.getLogger("tarang.inference.engine")


def predict_single(
    features: Union[pd.DataFrame, pd.Series, Dict[str, Any]],
    target: str = TARGET_DURATION,
    model_name: str = "HistGradientBoostingRegressor",
    models_dir: Optional[Path] = None,
    model: Optional[Any] = None,
) -> float:
    """Generate a single numeric prediction from a feature row or dictionary.

    Args:
        features: Input feature row as DataFrame, Series, or dict containing the
            13 canonical predictors.
        target: Prediction target ('Duration' or 'ConsumedkWh'). Defaults to 'Duration'.
        model_name: Benchmark model identifier. Defaults to 'HistGradientBoostingRegressor'.
        models_dir: Optional custom model artifacts directory.
        model: Optional pre-loaded estimator instance. If None, loads from disk or cache.

    Returns:
        Raw floating-point prediction from the model.

    Raises:
        ValueError: If input schema is invalid, row count != 1, or target/model is unrecognized.
        FileNotFoundError: If the requested model artifact is missing.
    """
    canonical_target = normalize_target_name(target)
    X = validate_inference_features(features)

    if len(X) != 1:
        raise ValueError(
            f"predict_single expects exactly 1 feature row, received {len(X)} rows. "
            "Use predict_batch for multi-row inference."
        )

    if model is None:
        estimator = load_model(
            target=canonical_target,
            model_name=model_name,
            models_dir=models_dir,
        )
    else:
        estimator = model

    raw_pred = estimator.predict(X)
    pred_arr = np.asarray(raw_pred, dtype=np.float64).ravel()
    return float(pred_arr[0])


def predict_batch(
    features_df: pd.DataFrame,
    target: str = TARGET_DURATION,
    model_name: str = "HistGradientBoostingRegressor",
    models_dir: Optional[Path] = None,
    model: Optional[Any] = None,
) -> pd.Series:
    """Generate aligned predictions for a batch of session feature rows.

    Preserves input index and row order. Extracts only the 13 canonical predictors
    for model evaluation, ignoring any additional metadata columns.

    Args:
        features_df: DataFrame containing at least the 13 canonical predictors.
        target: Prediction target ('Duration' or 'ConsumedkWh'). Defaults to 'Duration'.
        model_name: Benchmark model identifier. Defaults to 'HistGradientBoostingRegressor'.
        models_dir: Optional custom model artifacts directory.
        model: Optional pre-loaded estimator instance. If None, loads from disk or cache.

    Returns:
        Series of raw numeric predictions with index matching features_df.index.

    Raises:
        ValueError: If features_df is empty, missing predictors, or contains invalid values.
        FileNotFoundError: If the requested model artifact is missing.
    """
    canonical_target = normalize_target_name(target)
    X = validate_inference_features(features_df)

    if model is None:
        estimator = load_model(
            target=canonical_target,
            model_name=model_name,
            models_dir=models_dir,
        )
    else:
        estimator = model

    raw_preds = estimator.predict(X)
    pred_arr = np.asarray(raw_preds, dtype=np.float64).ravel()

    return pd.Series(
        pred_arr,
        index=features_df.index,
        name=f"predicted_{canonical_target}",
        dtype=np.float64,
    )


class InferenceEngine:
    """Stateful, multi-session prediction engine encapsulating loaded models and configuration."""

    def __init__(
        self,
        target: str = TARGET_DURATION,
        model_name: str = "HistGradientBoostingRegressor",
        models_dir: Optional[Path] = None,
        max_start_hour: Optional[float] = None,
        summary_file: Optional[Path] = None,
    ) -> None:
        """Initialize inference engine for a given target and model.

        Args:
            target: Prediction target ('Duration' or 'ConsumedkWh').
            model_name: Benchmark model name.
            models_dir: Optional directory containing model artifacts.
            max_start_hour: Optional explicit normalization max_start_hour.
            summary_file: Optional path to processing summary.
        """
        self.target = normalize_target_name(target)
        self.model_name = model_name
        self.models_dir = models_dir
        self.max_start_hour = max_start_hour
        self.summary_file = summary_file

        # Eagerly load model to validate artifact existence
        self.model = load_model(
            target=self.target,
            model_name=self.model_name,
            models_dir=self.models_dir,
        )

    def predict_single(
        self,
        features: Union[pd.DataFrame, pd.Series, Dict[str, Any]],
    ) -> float:
        """Predict for a single feature row using the preloaded model."""
        return predict_single(
            features=features,
            target=self.target,
            model_name=self.model_name,
            models_dir=self.models_dir,
            model=self.model,
        )

    def predict_batch(
        self,
        features_df: pd.DataFrame,
    ) -> pd.Series:
        """Predict for a batch of feature rows using the preloaded model."""
        return predict_batch(
            features_df=features_df,
            target=self.target,
            model_name=self.model_name,
            models_dir=self.models_dir,
            model=self.model,
        )

    def build_and_predict(
        self,
        start_time: Any,
        car_kw: float,
        car_kwh: float,
        history: pd.DataFrame,
    ) -> float:
        """Construct canonical features from session inputs and history, then predict.

        Args:
            start_time: New session arrival timestamp.
            car_kw: Vehicle charger power in kW.
            car_kwh: Vehicle battery capacity in kWh.
            history: Participant completed prior sessions DataFrame.

        Returns:
            Scalar float prediction.
        """
        features_row = build_inference_features(
            start_time=start_time,
            car_kw=car_kw,
            car_kwh=car_kwh,
            history=history,
            max_start_hour=self.max_start_hour,
            summary_file=self.summary_file,
        )
        return self.predict_single(features_row)

