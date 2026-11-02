"""Charging flexibility analysis wrapper for TARANG decision support.

Computes smart-charging flexibility metrics representing the proportion of plugged-in time
actively spent delivering energy versus idle occupancy:
    Flexibility Ratio = (ChargingDuration / 60.0) / ((StopTime - StartTime) in hours)

Sessions are classified using local calendar dates in `Europe/London`:
- Same-day: Session connection and disconnection occur on the same local date.
- Overnight: Session connection spans across midnight into a subsequent local date.

Wraps the established exploratory data analysis methodology, providing safe schema validation
and graceful fallback when historical columns are unavailable.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List
import numpy as np
import pandas as pd

from tarang.analysis.exploratory_data_analysis import calculate_charging_flexibility

logger = logging.getLogger("tarang.decision_support.flexibility")

REQUIRED_FLEXIBILITY_COLUMNS: List[str] = [
    "ChargingDuration",
    "StartTime",
    "StopTime",
]


def analyze_charging_flexibility(
    df: pd.DataFrame,
    n_bins: int = 25,
) -> Dict[str, Any]:
    """Analyze smart-charging flexibility distributions across same-day and overnight sessions.

    Wraps the established EDA flexibility calculation with explicit schema validation and
    availability status flags.

    Args:
        df: Input DataFrame containing charging session records. For flexibility evaluation,
            the DataFrame must contain 'ChargingDuration', 'StartTime', and 'StopTime'.
        n_bins: Number of histogram bins across the [0.0, 1.0] interval. Defaults to 25.

    Returns:
        Dictionary containing:
        - 'available': bool indicating whether flexibility analysis could be performed.
        - 'reason': str or None detailing why analysis was unavailable if available is False.
        - 'bin_edges': List[float] of histogram bin boundaries.
        - 'same_day_pct': List[float] of percentage frequencies across bins for same-day sessions.
        - 'overnight_pct': List[float] of percentage frequencies across bins for overnight sessions.
        - 'same_day_total': int total count of same-day sessions.
        - 'overnight_total': int total count of overnight sessions.
        - 'same_day_valid': int count of same-day sessions with positive duration and non-null charging.
        - 'overnight_valid': int count of overnight sessions with positive duration and non-null charging.
        - 'same_day_in_range': int count of same-day session ratios falling within [0.0, 1.0].
        - 'overnight_in_range': int count of overnight session ratios falling within [0.0, 1.0].
        - 'same_day_count': int count matching same_day_in_range.
        - 'overnight_count': int count matching overnight_in_range.
        - 'same_day_median_ratio': float median flexibility ratio for same-day sessions.
        - 'overnight_median_ratio': float median flexibility ratio for overnight sessions.
        - 'same_day_mean_ratio': float mean flexibility ratio for same-day sessions.
        - 'overnight_mean_ratio': float mean flexibility ratio for overnight sessions.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1).tolist()

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        logger.info("Flexibility analysis unavailable: DataFrame is empty or None.")
        return {
            "available": False,
            "reason": "DataFrame is empty or None.",
            "bin_edges": bins,
            "same_day_pct": [0.0] * n_bins,
            "overnight_pct": [0.0] * n_bins,
            "same_day_total": 0,
            "overnight_total": 0,
            "same_day_valid": 0,
            "overnight_valid": 0,
            "same_day_in_range": 0,
            "overnight_in_range": 0,
            "same_day_count": 0,
            "overnight_count": 0,
            "same_day_median_ratio": 0.0,
            "overnight_median_ratio": 0.0,
            "same_day_mean_ratio": 0.0,
            "overnight_mean_ratio": 0.0,
        }

    missing_cols = [col for col in REQUIRED_FLEXIBILITY_COLUMNS if col not in df.columns]
    if missing_cols:
        logger.info(
            "Flexibility analysis unavailable: missing required column(s) %s.",
            missing_cols,
        )
        return {
            "available": False,
            "reason": f"Missing required column(s): {missing_cols}.",
            "bin_edges": bins,
            "same_day_pct": [0.0] * n_bins,
            "overnight_pct": [0.0] * n_bins,
            "same_day_total": 0,
            "overnight_total": 0,
            "same_day_valid": 0,
            "overnight_valid": 0,
            "same_day_in_range": 0,
            "overnight_in_range": 0,
            "same_day_count": 0,
            "overnight_count": 0,
            "same_day_median_ratio": 0.0,
            "overnight_median_ratio": 0.0,
            "same_day_mean_ratio": 0.0,
            "overnight_mean_ratio": 0.0,
        }

    # Delegate to established exploratory data analysis calculation
    eda_result = calculate_charging_flexibility(df=df, n_bins=n_bins)
    result = dict(eda_result)
    result["available"] = True
    result["reason"] = None
    return result
