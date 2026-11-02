"""Participant and session screening module for TARANG.

Implements the research-grade screening pipeline to filter raw charging sessions
and extract clean, leakage-safe participant cohorts.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from tarang.config import (
    CANONICAL_COLUMNS,
    MAX_DURATION_HOURS,
    MIN_DURATION_HOURS,
    MIN_RETAINED_SESSIONS,
    RAW_SELECTED_COLUMNS,
)
from tarang.data.quality_audit import parse_utc_timestamps
from tarang.features.calendar_features import calculate_calendar_features
from tarang.features.historical_features import (
    calculate_last_consumed_kwh,
    calculate_last_duration,
    calculate_sessions_today,
    calculate_time_since_last_stop,
)
from tarang.features.time_features import calculate_cyclical_time_features

logger = logging.getLogger(__name__)


def screen_and_prepare_dataset(
    df_tx: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Execute the complete multi-step screening and feature engineering sequence.

    Exact execution sequence:
    1. Filter out Trial 3 sessions (Trial != 3).
    2. Select the 9 canonical raw transaction fields.
    3. Ensure StartTime and StopTime are parsed as UTC.
    4. Compute plugged-in Duration in hours = (StopTime - StartTime).
    5. Exclude short sessions (< 5 minutes).
    6. Cap Duration at 336.0 hours (preserving original StopTime).
    7. Compute calendar features (weekday, weekend, holiday, month, Mon..Sun).
    8. Compute continuous StartHour and normalize by dataset-wide maximum, plus StartCos/StartSin.
    9. Process each participant independently:
       a. Sort chronologically by StartTime.
       b. Compute TimeSinceLastStop from immediately preceding StopTime.
       c. Discard records where TimeSinceLastStop < 0 without recalculating subsequent gaps.
       d. Drop the first remaining session per participant.
       e. Exclude participants with fewer than 10 remaining sessions.
       f. Exclude participants where retained session count < (max_stop - min_start).days / 7.0.
       g. Assign sequential integer participant id.
       h. Compute lag features: sessionsToday, LastDuration, LastConsumedkWh.
    10. Concatenate all retained participant cohorts, sort globally by StartTime, and reset index.

    Args:
        df_tx: Raw transactions DataFrame.

    Returns:
        Tuple of (canonical_features_df, screening_metrics_dict).
    """
    total_raw = len(df_tx)
    if df_tx.empty or "ParticipantID" not in df_tx.columns or "StartTime" not in df_tx.columns or "StopTime" not in df_tx.columns:
        empty_metrics = _build_empty_screening_metrics(total_raw)
        empty_df = pd.DataFrame(columns=CANONICAL_COLUMNS)
        return empty_df, empty_metrics

    # Parse timestamps safely and check initial validity
    start_utc, stop_utc, _ = parse_utc_timestamps(df_tx, "StartTime", "StopTime")
    valid_ts_mask = df_tx["ParticipantID"].notna() & start_utc.notna() & stop_utc.notna()
    initial_valid = int(valid_ts_mask.sum())

    df_p = df_tx[valid_ts_mask].copy()
    df_p["StartTime"] = start_utc.loc[df_p.index]
    df_p["StopTime"] = stop_utc.loc[df_p.index]

    initial_pids_count = int(df_p["ParticipantID"].dropna().nunique())

    if df_p.empty:
        empty_metrics = _build_empty_screening_metrics(total_raw)
        empty_metrics["initial_valid_records"] = initial_valid
        empty_metrics["initial_participants"] = initial_pids_count
        empty_df = pd.DataFrame(columns=CANONICAL_COLUMNS)
        return empty_df, empty_metrics

    # Step 1: Trial != 3
    if "Trial" in df_p.columns:
        t3_mask = df_p["Trial"] == 3.0
        trial_3_excluded = int(t3_mask.sum())
        df_t = df_p[~t3_mask].copy()
    else:
        trial_3_excluded = 0
        df_t = df_p.copy()

    remaining_after_t3 = len(df_t)
    participants_after_t3 = int(df_t["ParticipantID"].dropna().nunique())

    # Populate any missing raw columns with NaN for synthetic tests
    for col in RAW_SELECTED_COLUMNS:
        if col not in df_t.columns:
            df_t[col] = np.nan

    df_sel = df_t[RAW_SELECTED_COLUMNS].copy()

    # Step 3: Compute Duration
    duration_hours = (df_sel["StopTime"] - df_sel["StartTime"]).dt.total_seconds() / 3600.0
    df_sel["Duration"] = duration_hours

    # Step 4: Exclude short sessions (< 5 minutes)
    valid_duration_mask = df_sel["Duration"] >= MIN_DURATION_HOURS
    short_excluded = int((~valid_duration_mask).sum())
    df_valid = df_sel[valid_duration_mask].copy()

    remaining_after_short = len(df_valid)
    participants_after_short = int(df_valid["ParticipantID"].dropna().nunique())

    # Step 5: Cap Duration at 336 hours (preserve original StopTime)
    df_valid["Duration"] = df_valid["Duration"].clip(upper=MAX_DURATION_HOURS)

    # Step 6: Calendar features
    df_valid = calculate_calendar_features(df_valid, time_col="StartTime")

    # Step 7: Continuous StartHour and cyclical features (dataset-wide max)
    dataset_max_start_hour = float(
        (df_valid["StartTime"].dt.hour + df_valid["StartTime"].dt.minute / 60.0 + df_valid["StartTime"].dt.second / 3600.0).max()
    ) if not df_valid.empty else 24.0

    df_valid = calculate_cyclical_time_features(
        df_valid,
        start_hour_max=dataset_max_start_hour,
        time_col="StartTime",
    )

    # Step 8: Per-participant processing
    unique_pids = sorted(df_valid["ParticipantID"].dropna().unique())
    pids_before_grouping = len(unique_pids)

    retained_participant_dfs: List[pd.DataFrame] = []
    int_id = 0

    negative_gaps_removed = 0
    sessions_after_neg_gap = 0
    dropped_first_sessions = 0
    sessions_after_first_drop = 0
    participants_zero_retained = 0
    participants_lt_10 = 0
    sessions_in_lt_10 = 0
    sessions_before_density = 0
    participants_before_density = 0
    participants_excluded_density = 0
    sessions_excluded_density = 0

    retained_spans = []
    retained_spws = []
    retained_intervals = []

    for pid in unique_pids:
        pdf = df_valid[df_valid["ParticipantID"] == pid].sort_values(by="StartTime").copy()

        # Immediate preceding session stop gap
        pdf["TimeSinceLastStop"] = calculate_time_since_last_stop(pdf)

        # Discard records with negative gap without recalculating subsequent gaps
        neg_mask = pdf["TimeSinceLastStop"] < 0
        neg_count = int(neg_mask.sum())
        negative_gaps_removed += neg_count
        pdf = pdf[~neg_mask].copy()
        sessions_after_neg_gap += len(pdf)

        # Drop first remaining session
        if len(pdf) > 0:
            dropped_first_sessions += 1
            pdf = pdf.iloc[1:].copy()
            if len(pdf) == 0:
                participants_zero_retained += 1
        else:
            participants_zero_retained += 1

        sessions_after_first_drop += len(pdf)

        # Minimum history condition: >= 10 retained sessions
        if len(pdf) < MIN_RETAINED_SESSIONS:
            participants_lt_10 += 1
            sessions_in_lt_10 += len(pdf)
            continue

        participants_before_density += 1
        sessions_before_density += len(pdf)

        # Observation density condition: sessions >= span_days / 7.0
        span_days = (pdf["StopTime"].max() - pdf["StartTime"].min()).days
        required_sessions = span_days / 7.0
        if len(pdf) < required_sessions:
            participants_excluded_density += 1
            sessions_excluded_density += len(pdf)
            continue

        # Retained participant
        pdf["id"] = int_id
        int_id += 1

        # Calculate lag features
        pdf["sessionsToday"] = calculate_sessions_today(pdf)
        pdf["LastDuration"] = calculate_last_duration(pdf)
        pdf["LastConsumedkWh"] = calculate_last_consumed_kwh(pdf)

        # Availability stats tracking
        spw = len(pdf) / (span_days / 7.0) if span_days > 0 else len(pdf)
        retained_spans.append(span_days)
        retained_spws.append(spw)
        intervals = (pdf["StartTime"] - pdf["StartTime"].shift(1)).dropna().dt.total_seconds() / 86400.0
        if not intervals.empty:
            retained_intervals.append(float(intervals.mean()))

        retained_participant_dfs.append(pdf)

    # Step 9: Concatenation and global sorting
    if retained_participant_dfs:
        final_df = pd.concat(retained_participant_dfs, ignore_index=True)
        final_df = final_df.sort_values(by="StartTime").reset_index(drop=True)
        # Ensure column ordering matches CANONICAL_COLUMNS
        final_df = final_df[[c for c in CANONICAL_COLUMNS if c in final_df.columns]].copy()
    else:
        final_df = pd.DataFrame(columns=CANONICAL_COLUMNS)

    final_participants = int_id
    final_sessions = len(final_df)

    mean_span = float(np.mean(retained_spans)) if retained_spans else 0.0
    mean_spw = float(np.mean(retained_spws)) if retained_spws else 0.0
    med_spw = float(np.median(retained_spws)) if retained_spws else 0.0
    mean_interval = float(np.mean(retained_intervals)) if retained_intervals else 0.0

    metrics: Dict[str, Any] = {
        "initial_valid_records": initial_valid,
        "initial_participants": initial_pids_count,
        "trial_3_excluded": trial_3_excluded,
        "remaining_after_t3": remaining_after_t3,
        "participants_after_t3": participants_after_t3,
        "short_sessions_excluded": short_excluded,
        "remaining_after_short": remaining_after_short,
        "participants_after_short": participants_after_short,
        "negative_gaps_removed": negative_gaps_removed,
        "screened_overlaps": negative_gaps_removed,  # for audit consistency
        "remaining_after_overlaps": sessions_after_neg_gap,
        "participants_after_overlaps": pids_before_grouping,
        "dropped_first_sessions": dropped_first_sessions,
        "retained_sessions_total": sessions_after_first_drop,
        "participants_after_dropped_first": pids_before_grouping - participants_zero_retained,
        "participants_zero_retained_after_dropped_first": participants_zero_retained,
        "participants_lt_10_retained": participants_lt_10,
        "sessions_in_lt_10_retained": sessions_in_lt_10,
        "participants_before_density": participants_before_density,
        "sessions_before_density": sessions_before_density,
        "participants_excluded_density": participants_excluded_density,
        "sessions_excluded_density": sessions_excluded_density,
        "final_eligible_participants": final_participants,
        "final_eligible_sessions": final_sessions,
        "max_start_hour": dataset_max_start_hour,
        "density_mean_span_days": round(mean_span, 2),
        "density_mean_sessions_per_week": round(mean_spw, 2),
        "density_median_sessions_per_week": round(med_spw, 2),
        "density_mean_avg_interval_days": round(mean_interval, 2),
    }

    return final_df, metrics


def screen_participant_sessions(
    df_tx: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Execute participant screening returning screened raw, eligible final dataset, and metrics.

    Args:
        df_tx: Raw transactions DataFrame.

    Returns:
        Tuple of (df_screened, df_final, metrics).
    """
    df_final, metrics = screen_and_prepare_dataset(df_tx)
    return df_final, df_final, metrics


def screen_participant_history(df_tx: pd.DataFrame) -> Dict[str, Any]:
    """Execute participant screening returning the screening metrics dictionary.

    Args:
        df_tx: Transactions DataFrame.

    Returns:
        Dictionary detailing counts at each screening step.
    """
    _, metrics = screen_and_prepare_dataset(df_tx)
    return metrics


def _build_empty_screening_metrics(total_raw: int) -> Dict[str, Any]:
    """Construct a clean metrics dictionary for empty or invalid DataFrames."""
    return {
        "initial_valid_records": total_raw,
        "initial_participants": 0,
        "trial_3_excluded": 0,
        "remaining_after_t3": 0,
        "participants_after_t3": 0,
        "short_sessions_excluded": 0,
        "remaining_after_short": 0,
        "participants_after_short": 0,
        "negative_gaps_removed": 0,
        "screened_overlaps": 0,
        "remaining_after_overlaps": 0,
        "participants_after_overlaps": 0,
        "dropped_first_sessions": 0,
        "retained_sessions_total": 0,
        "participants_after_dropped_first": 0,
        "participants_zero_retained_after_dropped_first": 0,
        "participants_lt_10_retained": 0,
        "sessions_in_lt_10_retained": 0,
        "participants_before_density": 0,
        "sessions_before_density": 0,
        "participants_excluded_density": 0,
        "sessions_excluded_density": 0,
        "final_eligible_participants": 0,
        "final_eligible_sessions": 0,
        "max_start_hour": 24.0,
        "density_mean_span_days": 0.0,
        "density_mean_sessions_per_week": 0.0,
        "density_median_sessions_per_week": 0.0,
        "density_mean_avg_interval_days": 0.0,
    }
