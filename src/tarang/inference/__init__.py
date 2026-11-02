"""Inference package for TARANG — EV Charging Intelligence System.

Provides high-performance single-session and batch inference pipelines, strict 13-predictor
schema enforcement, dynamic parameter loading, and seamless model loading across all benchmark
models and prediction targets.
"""

from __future__ import annotations

from tarang.inference.engine import (
    InferenceEngine,
    predict_batch,
    predict_single,
)
from tarang.inference.feature_builder import build_inference_features
from tarang.inference.loading import (
    clear_model_cache,
    get_target_models_dir,
    load_max_start_hour,
    load_model,
    normalize_target_name,
)
from tarang.inference.validation import (
    normalize_utc_timestamp,
    validate_inference_features,
    validate_session_inputs,
)

__all__ = [
    "InferenceEngine",
    "predict_single",
    "predict_batch",
    "build_inference_features",
    "load_model",
    "load_max_start_hour",
    "clear_model_cache",
    "normalize_target_name",
    "get_target_models_dir",
    "validate_inference_features",
    "validate_session_inputs",
    "normalize_utc_timestamp",
]

