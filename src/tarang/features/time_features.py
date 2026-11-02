"""Time-of-day and cyclical feature engineering module for TARANG.

Provides functions to compute continuous session start hour (including seconds),
dataset-wide maximum normalization, and trigonometric sine/cosine cyclical features.
"""

from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd


def calculate_start_hour(
    df: pd.DataFrame,
    time_col: str = "StartTime",
) -> pd.Series:
    """Calculate continuous start hour including minute and second fractions.

    Formula: hour + minute / 60.0 + second / 3600.0

    Args:
        df: Input DataFrame containing timestamp column.
        time_col: Name of the timestamp column. Defaults to 'StartTime'.

    Returns:
        Series of continuous start hour values.
    """
    if time_col not in df.columns:
        raise ValueError(f"Timestamp column '{time_col}' not found in DataFrame.")

    if df.empty:
        return pd.Series(dtype=float)

    ts = pd.to_datetime(df[time_col])
    return ts.dt.hour + ts.dt.minute / 60.0 + ts.dt.second / 3600.0


def calculate_cyclical_time_features(
    df: pd.DataFrame,
    start_hour_max: Optional[float] = None,
    time_col: str = "StartTime",
) -> pd.DataFrame:
    """Calculate normalized start hour and cyclical trigonometric transformations.

    Formulas:
    - StartHour = hour + minute / 60.0 + second / 3600.0
    - StartHourNorm = StartHour / max(StartHour)
    - StartCos = cos(2 * pi * StartHourNorm)
    - StartSin = sin(2 * pi * StartHourNorm)

    Args:
        df: Input DataFrame containing timestamp column.
        start_hour_max: Dataset-wide maximum StartHour used for normalization.
            If None, defaults to df['StartHour'].max().
        time_col: Name of timestamp column. Defaults to 'StartTime'.

    Returns:
        New DataFrame with StartHour, StartHourNorm, StartCos, and StartSin columns added.
    """
    out = df.copy()
    if out.empty:
        out["StartHour"] = pd.Series(dtype=float)
        out["StartHourNorm"] = pd.Series(dtype=float)
        out["StartCos"] = pd.Series(dtype=float)
        out["StartSin"] = pd.Series(dtype=float)
        return out

    if "StartHour" not in out.columns:
        out["StartHour"] = calculate_start_hour(out, time_col=time_col)

    norm_max = start_hour_max if start_hour_max is not None else float(out["StartHour"].max())
    if norm_max <= 0:
        norm_max = 24.0

    out["StartHourNorm"] = out["StartHour"] / norm_max
    out["StartCos"] = np.cos(2.0 * np.pi * out["StartHourNorm"])
    out["StartSin"] = np.sin(2.0 * np.pi * out["StartHourNorm"])

    return out

