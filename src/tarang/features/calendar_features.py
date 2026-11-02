"""Calendar feature engineering module for TARANG.

Provides functions to compute calendar indicators, day-of-week indicators,
holiday flags, and month features.
"""

from __future__ import annotations

from typing import Optional, Set
import pandas as pd

from tarang.config import UK_BANK_HOLIDAYS_REFERENCE


DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def calculate_calendar_features(
    df: pd.DataFrame,
    time_col: str = "StartTime",
    holidays: Optional[Set] = None,
) -> pd.DataFrame:
    """Compute calendar features for transaction sessions.

    Engineered columns:
    - 'weekday': 1 if Monday-Friday and not a holiday, else 0
    - 'weekend': 1 if Saturday-Sunday and not a holiday, else 0
    - 'holiday': 1 if session start date matches a designated bank holiday, else 0
    - 'month': Calendar month (1-12)
    - 7 individual day flags: 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun' (binary 0/1)

    Note: On holidays, both 'weekday' and 'weekend' are set to 0.

    Args:
        df: Input DataFrame containing datetime column.
        time_col: Name of the timestamp column. Defaults to 'StartTime'.
        holidays: Set of holiday date objects. Defaults to UK_BANK_HOLIDAYS_REFERENCE.

    Returns:
        New DataFrame with calendar feature columns added. Does not mutate input.
    """
    if time_col not in df.columns:
        raise ValueError(f"Timestamp column '{time_col}' not found in DataFrame.")

    holiday_set = holidays if holidays is not None else UK_BANK_HOLIDAYS_REFERENCE

    out = df.copy()
    if out.empty:
        for d in DAY_NAMES:
            out[d] = pd.Series(dtype=int)
        out["weekday"] = pd.Series(dtype=int)
        out["weekend"] = pd.Series(dtype=int)
        out["holiday"] = pd.Series(dtype=int)
        out["month"] = pd.Series(dtype=int)
        return out

    timestamps = pd.to_datetime(out[time_col])
    session_dates = timestamps.dt.date
    dow = timestamps.dt.dayofweek

    is_holiday = session_dates.isin(holiday_set)

    # Both weekday and weekend are 0 on holidays
    out["weekday"] = ((dow < 5) & (~is_holiday)).astype(int)
    out["weekend"] = ((dow >= 5) & (~is_holiday)).astype(int)
    out["holiday"] = is_holiday.astype(int)

    # 7 day indicators
    for i, name in enumerate(DAY_NAMES):
        out[name] = (dow == i).astype(int)

    out["month"] = timestamps.dt.month.astype(int)

    return out

