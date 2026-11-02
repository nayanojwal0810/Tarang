"""Historical session and lag feature engineering module for TARANG.

Provides functions to compute participant-specific lag features:
- TimeSinceLastStop: Hours elapsed since the end of the immediately preceding session.
- sessionsToday: Running count of sessions initiated by the participant on the current calendar date.
- LastDuration: Plugged-in duration of the immediately preceding retained session.
- LastConsumedkWh: Energy consumed in the immediately preceding retained session.
"""

from __future__ import annotations

import pandas as pd


def calculate_time_since_last_stop(
    df_participant: pd.DataFrame,
    start_col: str = "StartTime",
    stop_col: str = "StopTime",
) -> pd.Series:
    """Calculate elapsed hours since the immediately preceding session's StopTime.

    Formula: (StartTime - StopTime.shift(1)) in hours.

    Args:
        df_participant: Chronologically sorted DataFrame for a single participant.
        start_col: Start timestamp column name. Defaults to 'StartTime'.
        stop_col: Stop timestamp column name. Defaults to 'StopTime'.

    Returns:
        Series of float hours since last stop. First session has NaN.
    """
    if df_participant.empty:
        return pd.Series(dtype=float)

    start_ts = pd.to_datetime(df_participant[start_col])
    stop_ts = pd.to_datetime(df_participant[stop_col])

    return (start_ts - stop_ts.shift(1)).dt.total_seconds() / 3600.0


def calculate_sessions_today(
    df_participant: pd.DataFrame,
    time_col: str = "StartTime",
) -> pd.Series:
    """Calculate running count of sessions started on the same date for a single participant.

    Formula: cumcount() grouped by calendar date.

    Args:
        df_participant: Chronologically sorted DataFrame for a single participant.
        time_col: Start timestamp column name. Defaults to 'StartTime'.

    Returns:
        Series of integer cumulative session counts (0-indexed for the first of the day).
    """
    if df_participant.empty:
        return pd.Series(dtype=int)

    dates = pd.to_datetime(df_participant[time_col]).dt.date
    return df_participant.groupby(dates).cumcount()


def calculate_last_duration(
    df_participant: pd.DataFrame,
    duration_col: str = "Duration",
) -> pd.Series:
    """Extract plugged-in duration of the immediately preceding session.

    Formula: Duration.shift(1).

    Args:
        df_participant: Chronologically sorted DataFrame for a single participant.
        duration_col: Duration column name. Defaults to 'Duration'.

    Returns:
        Series of previous session duration values.
    """
    if df_participant.empty or duration_col not in df_participant.columns:
        return pd.Series(dtype=float)

    return df_participant[duration_col].shift(1)


def calculate_last_consumed_kwh(
    df_participant: pd.DataFrame,
    energy_col: str = "ConsumedkWh",
) -> pd.Series:
    """Extract consumed energy of the immediately preceding session.

    Takes the value directly from the immediately previous remaining session without
    skipping missing values to search backward.

    Formula: ConsumedkWh.shift(1).

    Args:
        df_participant: Chronologically sorted DataFrame for a single participant.
        energy_col: Consumed energy column name. Defaults to 'ConsumedkWh'.

    Returns:
        Series of previous session energy values (preserving NaN where prior value was NaN).
    """
    if df_participant.empty or energy_col not in df_participant.columns:
        return pd.Series(dtype=float)

    return df_participant[energy_col].shift(1)
