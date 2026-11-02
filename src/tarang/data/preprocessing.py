"""Preprocessing pipeline compatibility facade for TARANG.

Provides backwards-compatible aliases pointing to the canonical screening
and data preparation implementations.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd

from tarang.config import (
    FEATURE_NAMES as WARM_START_FEATURES,
    PROCESSED_FEATURES_FILE,
    PROCESSING_SUMMARY_FILE,
)
from tarang.data.preparation import run_data_preparation
from tarang.data.screening import (
    screen_and_prepare_dataset,
    screen_participant_history,
    screen_participant_sessions,
)

__all__ = [
    "WARM_START_FEATURES",
    "screen_participant_sessions",
    "screen_participant_history",
    "screen_and_prepare_dataset",
    "run_data_preparation",
    "run_preprocessing",
]


def run_preprocessing(
    raw_dir: Optional[Union[str, Any]] = None,
    output_dir: Optional[Union[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute data preparation pipeline (backward compatibility alias)."""
    return run_data_preparation()
