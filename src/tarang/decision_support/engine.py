"""Decision support engine for EV charging session occupancy and advisory intelligence.

Provides single-session and batch advisory APIs generating predicted connection durations,
energy consumption, numeric validity checks, and projected vehicle unplug timestamps in UTC.
Preserves raw estimator integrity without post-processing, artificial thresholding, or clipping.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd

from tarang.config import FEATURE_NAMES, TARGET_DURATION, TARGET_ENERGY
from tarang.inference.engine import predict_batch, predict_single
from tarang.inference.feature_builder import build_inference_features
from tarang.inference.loading import load_model
from tarang.inference.validation import normalize_utc_timestamp, validate_inference_features

logger = logging.getLogger("tarang.decision_support.engine")


@dataclass(frozen=True)
class SessionAdvisory:
    """Immutable data record encapsulating decision support advisory metrics for a charging session.

    Attributes:
        arrival_time_utc: Normalized session connection arrival timestamp in UTC.
        predicted_duration_hours: Raw model estimate of total plugged-in duration in hours.
        predicted_energy_kwh: Raw model estimate of active energy consumption in kWh.
        projected_unplug_time_utc: Projected departure timestamp in UTC (or pd.NaT if duration is invalid).
        duration_prediction_valid: Boolean flag indicating if duration estimate is finite and >= 0.0.
        energy_prediction_valid: Boolean flag indicating if energy estimate is finite and >= 0.0.
    """

    arrival_time_utc: pd.Timestamp
    predicted_duration_hours: float
    predicted_energy_kwh: float
    projected_unplug_time_utc: pd.Timestamp
    duration_prediction_valid: bool
    energy_prediction_valid: bool


def is_prediction_valid(val: Any) -> bool:
    """Determine whether a predicted numeric value is valid (finite and non-negative).

    Args:
        val: Numeric value or float-castable object.

    Returns:
        True if value is finite and >= 0.0, False otherwise.
    """
    try:
        fval = float(val)
        return bool(np.isfinite(fval) and fval >= 0.0)
    except (TypeError, ValueError):
        return False


def calculate_projected_unplug_time(
    arrival_time_utc: pd.Timestamp,
    predicted_duration_hours: float,
    is_valid: bool,
) -> pd.Timestamp:
    """Calculate projected unplug timestamp in UTC, or return pd.NaT if duration is invalid.

    Args:
        arrival_time_utc: Session arrival timestamp in UTC.
        predicted_duration_hours: Predicted connection duration in hours.
        is_valid: Boolean flag indicating duration prediction validity.

    Returns:
        Projected unplug timestamp in UTC, or pd.NaT if duration is invalid.
    """
    if is_valid:
        return arrival_time_utc + pd.Timedelta(hours=float(predicted_duration_hours))
    return pd.NaT


class DecisionSupportEngine:
    """High-level decision support engine for generating single-session and batch advisories."""

    def __init__(
        self,
        model_name: str = "HistGradientBoostingRegressor",
        models_dir: Optional[Path] = None,
        duration_model: Optional[Any] = None,
        energy_model: Optional[Any] = None,
        max_start_hour: Optional[float] = None,
        summary_file: Optional[Path] = None,
    ) -> None:
        """Initialize the decision support engine with pre-trained models and configuration.

        Args:
            model_name: Benchmark model identifier. Defaults to 'HistGradientBoostingRegressor'.
            models_dir: Optional custom model artifacts directory.
            duration_model: Optional pre-loaded Duration model instance.
            energy_model: Optional pre-loaded ConsumedkWh model instance.
            max_start_hour: Optional explicit normalization max_start_hour.
            summary_file: Optional path to processing summary.
        """
        self.model_name = model_name
        self.models_dir = models_dir
        self.max_start_hour = max_start_hour
        self.summary_file = summary_file

        self.duration_model = (
            duration_model
            if duration_model is not None
            else load_model(
                target=TARGET_DURATION,
                model_name=self.model_name,
                models_dir=self.models_dir,
            )
        )
        self.energy_model = (
            energy_model
            if energy_model is not None
            else load_model(
                target=TARGET_ENERGY,
                model_name=self.model_name,
                models_dir=self.models_dir,
            )
        )

    def predict_single_advisory(
        self,
        start_time: Optional[Union[str, pd.Timestamp, datetime, pd.DataFrame, pd.Series, Dict[str, Any]]] = None,
        car_kw: Optional[float] = None,
        car_kwh: Optional[float] = None,
        history: Optional[pd.DataFrame] = None,
        features: Optional[Union[pd.DataFrame, pd.Series, Dict[str, Any]]] = None,
    ) -> SessionAdvisory:
        """Generate a complete decision support advisory for a single charging session.

        Accepts either:
        1. Session arrival parameters: `start_time`, `car_kw`, `car_kwh`, and `history`.
        2. Pre-engineered feature vector: `features` (with optional `start_time`).

        Args:
            start_time: Arrival timestamp (string, Timestamp, datetime), or feature input if
                car_kw is None.
            car_kw: Vehicle charger rating in kW (strictly positive).
            car_kwh: Vehicle battery capacity in kWh (strictly positive).
            history: DataFrame of completed prior sessions for the participant.
            features: Pre-built feature row containing the 13 canonical predictors.

        Returns:
            SessionAdvisory dataclass with duration, energy, validity flags, and unplug timestamp.

        Raises:
            ValueError: If inputs are invalid or insufficient to evaluate features and arrival time.
        """
        # Detect if features was passed as the first positional argument
        if features is None and isinstance(start_time, (pd.DataFrame, pd.Series, dict)) and car_kw is None:
            features = start_time
            start_time = None

        if features is not None:
            feat_df = validate_inference_features(features)
            # Resolve arrival time from start_time or features
            if start_time is not None:
                arrival_time_utc = normalize_utc_timestamp(start_time)
            elif isinstance(features, pd.DataFrame) and "StartTime" in features.columns:
                arrival_time_utc = normalize_utc_timestamp(features["StartTime"].iloc[0])
            elif isinstance(features, pd.Series) and "StartTime" in features.index:
                arrival_time_utc = normalize_utc_timestamp(features["StartTime"])
            elif isinstance(features, dict) and "StartTime" in features:
                arrival_time_utc = normalize_utc_timestamp(features["StartTime"])
            else:
                raise ValueError(
                    "Session arrival timestamp ('start_time' or 'StartTime' feature column) "
                    "is required to calculate projected unplug time."
                )
        else:
            if start_time is None or car_kw is None or car_kwh is None or history is None:
                raise ValueError(
                    "Either 'features' or all session arrival arguments ('start_time', 'car_kw', "
                    "'car_kwh', 'history') must be provided."
                )
            arrival_time_utc = normalize_utc_timestamp(start_time)
            feat_df = build_inference_features(
                start_time=start_time,
                car_kw=car_kw,
                car_kwh=car_kwh,
                history=history,
                max_start_hour=self.max_start_hour,
                summary_file=self.summary_file,
            )

        # Generate predictions for both targets
        raw_dur = predict_single(
            features=feat_df,
            target=TARGET_DURATION,
            model_name=self.model_name,
            models_dir=self.models_dir,
            model=self.duration_model,
        )
        raw_energy = predict_single(
            features=feat_df,
            target=TARGET_ENERGY,
            model_name=self.model_name,
            models_dir=self.models_dir,
            model=self.energy_model,
        )

        dur_valid = is_prediction_valid(raw_dur)
        energy_valid = is_prediction_valid(raw_energy)
        projected_unplug = calculate_projected_unplug_time(
            arrival_time_utc=arrival_time_utc,
            predicted_duration_hours=raw_dur,
            is_valid=dur_valid,
        )

        return SessionAdvisory(
            arrival_time_utc=arrival_time_utc,
            predicted_duration_hours=float(raw_dur),
            predicted_energy_kwh=float(raw_energy),
            projected_unplug_time_utc=projected_unplug,
            duration_prediction_valid=dur_valid,
            energy_prediction_valid=energy_valid,
        )

    def predict_batch_advisories(
        self,
        features_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Generate aligned session advisories for a batch of session feature rows.

        Validates arrival timestamps before model invocation, rejects missing/NaT/malformed
        timestamps, and preserves all input columns and original index (including non-consecutive
        or duplicate indices).

        Appends:
        - `predicted_Duration`: Raw numeric prediction in hours.
        - `predicted_ConsumedkWh`: Raw numeric prediction in kWh.
        - `projected_unplug_time_utc`: UTC Timestamp of estimated unplug time (NaT if duration invalid).
        - `duration_prediction_valid`: bool indicating finite, non-negative duration prediction.
        - `energy_prediction_valid`: bool indicating finite, non-negative energy prediction.

        Args:
            features_df: DataFrame containing at least 'StartTime' and the 13 canonical predictors.

        Returns:
            DataFrame copy containing all original columns and index plus the 5 advisory columns.

        Raises:
            ValueError: If input is empty, missing 'StartTime', contains invalid/NaT timestamps,
                or fails feature schema validation.
        """
        if features_df is None or not isinstance(features_df, pd.DataFrame):
            raise ValueError(
                f"Unsupported features_df type: {type(features_df).__name__}. Expected pd.DataFrame."
            )
        if features_df.empty:
            raise ValueError("Input features DataFrame is empty.")

        # 1. Validate presence and parse arrival timestamps before invoking models
        if "StartTime" not in features_df.columns:
            raise ValueError(
                "Input features DataFrame must contain 'StartTime' column to compute session advisories."
            )

        start_col = features_df["StartTime"]
        if start_col.isna().any():
            raise ValueError("StartTime column contains missing or null/NaT timestamps.")

        try:
            start_utc = pd.to_datetime(start_col, utc=True)
        except Exception as exc:
            raise ValueError(f"Failed to parse 'StartTime' timestamps to UTC: {exc}") from exc

        if start_utc.isna().any():
            raise ValueError("StartTime column contains unparseable or NaT timestamps.")

        # 2. Validate predictors
        _ = validate_inference_features(features_df)

        # 3. Generate batch predictions for both targets
        dur_preds = predict_batch(
            features_df=features_df,
            target=TARGET_DURATION,
            model_name=self.model_name,
            models_dir=self.models_dir,
            model=self.duration_model,
        )
        energy_preds = predict_batch(
            features_df=features_df,
            target=TARGET_ENERGY,
            model_name=self.model_name,
            models_dir=self.models_dir,
            model=self.energy_model,
        )

        # 4. Preserve input DataFrame and append advisory columns using position-aligned arrays
        result_df = features_df.copy()
        raw_dur_arr = np.asarray(dur_preds.to_numpy(), dtype=np.float64)
        raw_energy_arr = np.asarray(energy_preds.to_numpy(), dtype=np.float64)

        result_df["predicted_Duration"] = raw_dur_arr
        result_df["predicted_ConsumedkWh"] = raw_energy_arr

        # Validity flags: finite and non-negative
        dur_valid = np.isfinite(raw_dur_arr) & (raw_dur_arr >= 0.0)
        energy_valid = np.isfinite(raw_energy_arr) & (raw_energy_arr >= 0.0)

        # Projected unplug timestamp: arrival + predicted duration in hours (NaT if duration invalid)
        dur_delta = pd.to_timedelta(np.where(dur_valid, raw_dur_arr, 0.0), unit="h")
        projected_unplug = (start_utc + dur_delta).where(dur_valid, pd.NaT)

        result_df["projected_unplug_time_utc"] = projected_unplug
        result_df["duration_prediction_valid"] = dur_valid
        result_df["energy_prediction_valid"] = energy_valid

        return result_df


def generate_session_advisory(
    start_time: Optional[Union[str, pd.Timestamp, datetime, pd.DataFrame, pd.Series, Dict[str, Any]]] = None,
    car_kw: Optional[float] = None,
    car_kwh: Optional[float] = None,
    history: Optional[pd.DataFrame] = None,
    features: Optional[Union[pd.DataFrame, pd.Series, Dict[str, Any]]] = None,
    model_name: str = "HistGradientBoostingRegressor",
    models_dir: Optional[Path] = None,
    duration_model: Optional[Any] = None,
    energy_model: Optional[Any] = None,
    max_start_hour: Optional[float] = None,
    summary_file: Optional[Path] = None,
) -> SessionAdvisory:
    """Convenience function to generate a decision support advisory for a single session."""
    engine = DecisionSupportEngine(
        model_name=model_name,
        models_dir=models_dir,
        duration_model=duration_model,
        energy_model=energy_model,
        max_start_hour=max_start_hour,
        summary_file=summary_file,
    )
    return engine.predict_single_advisory(
        start_time=start_time,
        car_kw=car_kw,
        car_kwh=car_kwh,
        history=history,
        features=features,
    )


def generate_batch_advisories(
    features_df: pd.DataFrame,
    model_name: str = "HistGradientBoostingRegressor",
    models_dir: Optional[Path] = None,
    duration_model: Optional[Any] = None,
    energy_model: Optional[Any] = None,
) -> pd.DataFrame:
    """Convenience function to generate decision support advisories for a batch of sessions."""
    engine = DecisionSupportEngine(
        model_name=model_name,
        models_dir=models_dir,
        duration_model=duration_model,
        energy_model=energy_model,
    )
    return engine.predict_batch_advisories(features_df=features_df)
