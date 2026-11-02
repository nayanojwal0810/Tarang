"""Feature engineering orchestration and authoritative feature specifications for TARANG.

Maintains the authoritative list of 13 regression predictors and provides helper
functions to validate and extract features from processed datasets.
"""

from __future__ import annotations

from typing import List
import pandas as pd

from tarang.config import FEATURE_NAMES


def get_feature_names() -> List[str]:
    """Return the authoritative list of 13 regression predictor names in standard order.

    Features:
    1. weekday
    2. weekend
    3. holiday
    4. month
    5. StartCos
    6. StartSin
    7. StartHourNorm
    8. TimeSinceLastStop
    9. sessionsToday
    10. LastDuration
    11. LastConsumedkWh
    12. CarKW
    13. CarKWh

    Returns:
        List of 13 feature column names.
    """
    return list(FEATURE_NAMES)


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract the 13 authoritative predictor columns from a processed dataset.

    Args:
        df: DataFrame containing all 13 predictor columns.

    Returns:
        DataFrame containing only the 13 features in authoritative order.

    Raises:
        ValueError: If any required feature columns are missing from df.
    """
    missing = [f for f in FEATURE_NAMES if f not in df.columns]
    if missing:
        raise ValueError(
            f"Cannot extract features: missing required predictor column(s): {missing}. "
            f"Available columns: {list(df.columns)}"
        )
    return df[FEATURE_NAMES].copy()
