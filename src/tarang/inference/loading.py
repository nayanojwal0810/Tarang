"""Artifact loading, normalization, and caching utilities for TARANG inference.

Provides functions to locate, validate, and load trained model artifacts (Stage 2)
and runtime summary parameters (Stage 1) needed for real-time and batch predictions.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np

from tarang.config import (
    DEPLOYMENT_ARTIFACTS_DIR,
    DURATION_MODELING_DIR,
    ENERGY_MODELING_DIR,
    FEATURE_NAMES,
    MODEL_NAMES,
    PROCESSING_SUMMARY_FILE,
    TARGET_DURATION,
    TARGET_ENERGY,
    TARGET_LAG_MAP,
)

logger = logging.getLogger("tarang.inference.loading")

# In-memory model artifact cache: (target, model_name, str(models_dir)) -> loaded estimator
_MODEL_CACHE: Dict[Tuple[str, str, str], Any] = {}


def _sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of a file's contents."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_deployment_artifacts() -> bool:
    """Verify SHA-256 checksums for the deployment artifact bundle.

    Returns True if all checksums match the manifest. Logs a warning on mismatch
    but does not raise so that development environments (data/processed/) still work.
    """
    manifest_path = DEPLOYMENT_ARTIFACTS_DIR / "manifest.json"
    if not manifest_path.is_file():
        return False

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    for key, meta in manifest.get("models", {}).items():
        artifact_path = DEPLOYMENT_ARTIFACTS_DIR / meta["path"]
        if not artifact_path.is_file():
            logger.warning("Deployment artifact missing: %s", artifact_path)
            return False
        actual = _sha256_file(artifact_path)
        expected = meta.get("sha256", "")
        if actual != expected:
            logger.warning(
                "SHA-256 mismatch for %s: expected %s, got %s",
                artifact_path.name,
                expected,
                actual,
            )
            return False

    logger.info("Deployment artifact checksums verified.")
    return True


def normalize_target_name(target: str) -> str:
    """Normalize input target name to canonical constant.

    Args:
        target: Target name, case-insensitive (e.g. 'Duration', 'duration',
            'ConsumedkWh', 'consumedkwh', 'energy').

    Returns:
        Canonical target string ('Duration' or 'ConsumedkWh').

    Raises:
        ValueError: If target name is not recognized.
    """
    if not isinstance(target, str):
        raise ValueError(f"Target must be a string, got {type(target).__name__}.")

    clean = target.strip().lower()
    if clean in ["duration", "pluggedintime", "plugged_in_time"]:
        return TARGET_DURATION
    if clean in ["consumedkwh", "energy", "consumption", "consumed_kwh"]:
        return TARGET_ENERGY

    raise ValueError(
        f"Unknown target '{target}'. Expected '{TARGET_DURATION}' or '{TARGET_ENERGY}'."
    )


def get_target_models_dir(target: str, base_dir: Optional[Path] = None) -> Path:
    """Resolve the directory containing saved joblib models for the specified target.

    Args:
        target: Target name ('Duration' or 'ConsumedkWh').
        base_dir: Optional custom base modeling directory for testing.

    Returns:
        Path to the target's models directory.
    """
    canonical_target = normalize_target_name(target)
    target_sub = "duration" if canonical_target == TARGET_DURATION else "energy"

    if base_dir is not None:
        target_dir = Path(base_dir)
        # If user passed directory already ending in 'models', use it directly
        if target_dir.name == "models":
            return target_dir
        # If user passed root modeling dir containing duration/ and energy/
        if (target_dir / target_sub / "models").exists():
            return target_dir / target_sub / "models"
        if (target_dir / target_sub).is_dir():
            if (target_dir / target_sub / "models").is_dir():
                return target_dir / target_sub / "models"
            return target_dir / target_sub
        if (target_dir / "models").is_dir():
            return target_dir / "models"
        return target_dir

    if canonical_target == TARGET_DURATION:
        deploy_dir = DEPLOYMENT_ARTIFACTS_DIR / "duration" / "models"
        if deploy_dir.is_dir():
            return deploy_dir
        return DURATION_MODELING_DIR / "models"

    deploy_dir = DEPLOYMENT_ARTIFACTS_DIR / "energy" / "models"
    if deploy_dir.is_dir():
        return deploy_dir
    return ENERGY_MODELING_DIR / "models"


def load_max_start_hour(summary_file: Optional[Path] = None) -> float:
    """Load and validate the dataset-wide maximum start hour from Stage 1 processing summary.

    This value is required to normalize StartHour identically to Stage 1. Imputation,
    recalculation, or fallback to 24.0 is strictly disallowed.

    Args:
        summary_file: Optional path to processing_summary.json. Defaults to
            PROCESSING_SUMMARY_FILE in config.

    Returns:
        Authoritative max_start_hour float.

    Raises:
        FileNotFoundError: If summary file does not exist.
        ValueError: If max_start_hour is missing, non-numeric, non-positive, or non-finite.
    """
    summary_path = Path(summary_file) if summary_file is not None else PROCESSING_SUMMARY_FILE

    if not summary_path.is_file():
        raise FileNotFoundError(
            f"Processing summary file not found at '{summary_path}'. "
            "Stage 1 data preparation must be executed to produce the canonical summary."
        )

    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        raise ValueError(
            f"Failed to read JSON processing summary from '{summary_path}': {exc}"
        ) from exc

    if "max_start_hour" not in data:
        raise ValueError(
            f"Required field 'max_start_hour' not found in processing summary '{summary_path}'. "
            "Stage 1 summary is incomplete or corrupted."
        )

    val = data["max_start_hour"]
    try:
        max_hour = float(val)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid 'max_start_hour' value in '{summary_path}': {val!r}. "
            "Value must be a valid positive float."
        ) from exc

    if not np.isfinite(max_hour) or max_hour <= 0.0:
        raise ValueError(
            f"Invalid 'max_start_hour' value in '{summary_path}': {max_hour}. "
            "Value must be a positive finite float."
        )

    return max_hour


def load_model(
    target: str = TARGET_DURATION,
    model_name: str = "HistGradientBoostingRegressor",
    models_dir: Optional[Path] = None,
    use_cache: bool = True,
) -> Any:
    """Load a trained Stage 2 model artifact with caching and schema validation.

    Args:
        target: Prediction target ('Duration' or 'ConsumedkWh').
        model_name: Name of benchmark model from MODEL_NAMES.
        models_dir: Optional custom models directory.
        use_cache: Whether to return cached estimator instances.

    Returns:
        Fitted model estimator ready for inference.

    Raises:
        ValueError: If target or model_name is invalid or incompatible.
        FileNotFoundError: If the model artifact does not exist on disk.
    """
    canonical_target = normalize_target_name(target)

    if model_name not in MODEL_NAMES:
        raise ValueError(
            f"Unknown model_name '{model_name}'. "
            f"Supported benchmark models are: {MODEL_NAMES}."
        )

    target_models_dir = get_target_models_dir(canonical_target, base_dir=models_dir)
    cache_key = (canonical_target, model_name, str(target_models_dir.resolve()))

    if use_cache and cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    model_file = target_models_dir / f"{model_name}.joblib"
    if not model_file.is_file():
        raise FileNotFoundError(
            f"Model artifact not found at '{model_file}'. "
            f"Ensure Stage 2 predictive modeling has been executed to train and "
            f"persist '{model_name}' for target '{canonical_target}'."
        )

    try:
        model = joblib.load(model_file)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load model artifact from '{model_file}': {exc}"
        ) from exc

    # Validate feature count compatibility if exposed by scikit-learn estimator
    if hasattr(model, "n_features_in_"):
        expected_n = len(FEATURE_NAMES)
        actual_n = getattr(model, "n_features_in_")
        if actual_n != expected_n:
            raise ValueError(
                f"Model '{model_name}' expects {actual_n} features, but canonical "
                f"TARANG schema specifies {expected_n} features: {FEATURE_NAMES}."
            )

    # Validate feature names compatibility if exposed
    if hasattr(model, "feature_names_in_"):
        actual_names = list(getattr(model, "feature_names_in_"))
        if actual_names != FEATURE_NAMES:
            logger.warning(
                "Model feature names %s differ from canonical FEATURE_NAMES %s",
                actual_names,
                FEATURE_NAMES,
            )

    # Validate LastValue lag feature configuration
    if model_name == "LastValue" and hasattr(model, "lag_feature"):
        expected_lag = TARGET_LAG_MAP[canonical_target]
        if getattr(model, "lag_feature") != expected_lag:
            raise ValueError(
                f"LastValue model for target '{canonical_target}' is configured with "
                f"lag feature '{getattr(model, 'lag_feature')}', expected '{expected_lag}'."
            )

    if use_cache:
        _MODEL_CACHE[cache_key] = model

    return model


def clear_model_cache() -> None:
    """Clear in-memory model cache."""
    _MODEL_CACHE.clear()
