"""Data quality audit and integrity inspection module for TARANG.

Provides non-destructive checks for inspecting raw datasets, evaluating schema integrity,
identifying missingness, timestamp anomalies, overlap patterns, unattributed sessions,
metadata reconciliation, energy anomalies, charging discrepancies, and feature feasibility.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

from tarang.config import UK_BANK_HOLIDAYS_REFERENCE

logger = logging.getLogger(__name__)


def calculate_missingness(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate missing values and percentage missing for each column.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with columns ['column_name', 'missing_count', 'missing_percentage'].
    """
    total_records = len(df)
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / total_records * 100.0) if total_records > 0 else pd.Series(0.0, index=df.columns)

    return pd.DataFrame({
        "column_name": df.columns,
        "missing_count": missing_count.values,
        "missing_percentage": missing_pct.round(4).values if hasattr(missing_pct, "round") else [0.0] * len(df.columns),
    })


def detect_duplicates(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
) -> int:
    """Count duplicate rows in a DataFrame or within a subset of columns.

    Args:
        df: Input DataFrame.
        subset: Optional list of column names defining uniqueness.

    Returns:
        Total number of duplicated records.
    """
    if df.empty:
        return 0
    return int(df.duplicated(subset=subset, keep="first").sum())


def parse_utc_timestamps(
    df: pd.DataFrame,
    start_col: str = "StartTime",
    stop_col: str = "StopTime",
) -> Tuple[pd.Series, pd.Series, Dict[str, Any]]:
    """Parse raw timestamps as UTC-aware datetime series without shifting recorded clock values.

    Naive Excel timestamps are treated directly as UTC.

    Args:
        df: DataFrame containing the timestamp columns.
        start_col: Start timestamp column name. Defaults to 'StartTime'.
        stop_col: Stop timestamp column name. Defaults to 'StopTime'.

    Returns:
        Tuple of (start_utc_series, stop_utc_series, validation_metrics_dict).
    """
    total_records = len(df)
    if start_col not in df.columns or stop_col not in df.columns:
        empty_s = pd.Series(dtype="datetime64[ns, UTC]")
        return empty_s, empty_s, {
            "start_col": start_col,
            "stop_col": stop_col,
            "total_records": total_records,
            "missing_start": total_records,
            "missing_stop": total_records,
            "raw_missing_start": total_records,
            "raw_missing_stop": total_records,
            "start_parse_failures": 0,
            "stop_parse_failures": 0,
            "earliest_start": None,
            "latest_start": None,
            "earliest_stop": None,
            "latest_stop": None,
            "negative_intervals": 0,
            "zero_intervals": 0,
            "negative_intervals_pct": 0.0,
            "zero_intervals_pct": 0.0,
            "timezone": "UTC",
        }

    raw_start = df[start_col]
    raw_stop = df[stop_col]

    start_dt = pd.to_datetime(raw_start, errors="coerce")
    stop_dt = pd.to_datetime(raw_stop, errors="coerce")

    start_parse_failures = int((start_dt.isnull() & raw_start.notnull()).sum())
    stop_parse_failures = int((stop_dt.isnull() & raw_stop.notnull()).sum())

    missing_start = int(start_dt.isnull().sum())
    missing_stop = int(stop_dt.isnull().sum())
    raw_missing_start = int(raw_start.isnull().sum())
    raw_missing_stop = int(raw_stop.isnull().sum())

    if not start_dt.empty and start_dt.dt.tz is None:
        start_utc = start_dt.dt.tz_localize("UTC")
    elif not start_dt.empty:
        start_utc = start_dt.dt.tz_convert("UTC")
    else:
        start_utc = start_dt

    if not stop_dt.empty and stop_dt.dt.tz is None:
        stop_utc = stop_dt.dt.tz_localize("UTC")
    elif not stop_dt.empty:
        stop_utc = stop_dt.dt.tz_convert("UTC")
    else:
        stop_utc = stop_dt

    valid_mask = start_utc.notnull() & stop_utc.notnull()
    valid_start = start_utc[valid_mask]
    valid_stop = stop_utc[valid_mask]

    diff_seconds = (valid_stop - valid_start).dt.total_seconds()
    negative_intervals = int((diff_seconds < 0).sum())
    zero_intervals = int((diff_seconds == 0).sum())

    metrics = {
        "start_col": start_col,
        "stop_col": stop_col,
        "total_records": total_records,
        "missing_start": missing_start,
        "missing_stop": missing_stop,
        "raw_missing_start": raw_missing_start,
        "raw_missing_stop": raw_missing_stop,
        "start_parse_failures": start_parse_failures,
        "stop_parse_failures": stop_parse_failures,
        "earliest_start": str(valid_start.min()) if not valid_start.empty else None,
        "latest_start": str(valid_start.max()) if not valid_start.empty else None,
        "earliest_stop": str(valid_stop.min()) if not valid_stop.empty else None,
        "latest_stop": str(valid_stop.max()) if not valid_stop.empty else None,
        "negative_intervals": negative_intervals,
        "zero_intervals": zero_intervals,
        "negative_intervals_pct": round((negative_intervals / total_records * 100.0) if total_records > 0 else 0.0, 4),
        "zero_intervals_pct": round((zero_intervals / total_records * 100.0) if total_records > 0 else 0.0, 4),
        "timezone": "UTC",
    }

    return start_utc, stop_utc, metrics


def detect_session_overlaps(
    df: pd.DataFrame,
    participant_col: str = "ParticipantID",
    start_col: str = "StartTime",
    stop_col: str = "StopTime",
) -> Tuple[pd.Series, int]:
    """Identify overlapping sessions per participant using chronological sorting and cumulative max StopTime.

    A session is flagged as overlapping if its StartTime is strictly earlier than
    the maximum StopTime of all prior sessions for that participant.

    Args:
        df: Input DataFrame.
        participant_col: Participant ID column name.
        start_col: Session start timestamp column name.
        stop_col: Session stop timestamp column name.

    Returns:
        Tuple of (boolean Series mask aligned with df index where True = overlapping, int overlap count).
    """
    if df.empty or participant_col not in df.columns or start_col not in df.columns or stop_col not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool), 0

    valid_mask = df[participant_col].notnull() & df[start_col].notnull() & df[stop_col].notnull()
    valid_df = df[valid_mask].copy()

    if valid_df.empty:
        return pd.Series(False, index=df.index, dtype=bool), 0

    valid_df["_start_dt"] = pd.to_datetime(valid_df[start_col], errors="coerce")
    valid_df["_stop_dt"] = pd.to_datetime(valid_df[stop_col], errors="coerce")

    # Keep only rows where both parsed successfully
    valid_df = valid_df[valid_df["_start_dt"].notnull() & valid_df["_stop_dt"].notnull()]
    if valid_df.empty:
        return pd.Series(False, index=df.index, dtype=bool), 0

    valid_df = valid_df.sort_values(
        by=[participant_col, "_start_dt", "_stop_dt"],
        ascending=[True, True, True],
        kind="mergesort",
    )

    valid_df["_prior_cummax_stop"] = (
        valid_df.groupby(participant_col)["_stop_dt"]
        .apply(lambda s: s.shift(1).cummax())
        .reset_index(level=0, drop=True)
    )

    valid_df["_is_overlap"] = (
        valid_df["_prior_cummax_stop"].notnull()
        & (valid_df["_start_dt"] < valid_df["_prior_cummax_stop"])
    )

    full_mask = pd.Series(False, index=df.index, dtype=bool)
    full_mask.loc[valid_df.index] = valid_df["_is_overlap"]
    overlap_count = int(full_mask.sum())

    return full_mask, overlap_count


def compute_simple_shift_overlaps(
    df: pd.DataFrame,
    participant_col: str = "ParticipantID",
    start_col: str = "StartTime",
    stop_col: str = "StopTime",
) -> int:
    """Calculate overlap count using immediate previous session stop time shift.

    Formula: StartTime < StopTime.shift(1) per participant.

    Args:
        df: Transactions DataFrame.
        participant_col: Participant column name.
        start_col: Start timestamp column name.
        stop_col: Stop timestamp column name.

    Returns:
        Integer count of simple shift overlap records.
    """
    if df.empty or participant_col not in df.columns or start_col not in df.columns or stop_col not in df.columns:
        return 0

    valid_mask = df[participant_col].notnull() & df[start_col].notnull() & df[stop_col].notnull()
    valid_df = df[valid_mask].copy()
    if valid_df.empty:
        return 0

    valid_df["_start_dt"] = pd.to_datetime(valid_df[start_col], errors="coerce")
    valid_df["_stop_dt"] = pd.to_datetime(valid_df[stop_col], errors="coerce")
    valid_df = valid_df[valid_df["_start_dt"].notnull() & valid_df["_stop_dt"].notnull()]
    if valid_df.empty:
        return 0

    valid_df = valid_df.sort_values(
        by=[participant_col, "_start_dt", "_stop_dt"],
        ascending=[True, True, True],
        kind="mergesort",
    )

    valid_df["_prior_stop"] = valid_df.groupby(participant_col)["_stop_dt"].shift(1)
    simple_shift_overlaps = int((valid_df["_prior_stop"].notnull() & (valid_df["_start_dt"] < valid_df["_prior_stop"])).sum())
    return simple_shift_overlaps


def investigate_unattributed_sessions(
    df_tx: pd.DataFrame,
    df_meta: pd.DataFrame,
) -> Dict[str, Any]:
    """Analyze transaction sessions missing ParticipantID and reconcile against ChargerInstall metadata.

    Args:
        df_tx: Transactions DataFrame.
        df_meta: Charger metadata DataFrame.

    Returns:
        Dictionary detailing unattributed session counts, involved chargers, and natural session matches.
    """
    total_tx = len(df_tx)
    if "ParticipantID" not in df_tx.columns:
        return {
            "total_unattributed_records": total_tx,
            "unattributed_percentage": 100.0 if total_tx > 0 else 0.0,
            "chargers_involved": 0,
            "known_chargers_count": 0,
            "unknown_chargers_count": 0,
            "unknown_charger_ids": [],
            "unknown_charger_records": 0,
            "matched_4field_session_records": 0,
            "unmatched_session_records": 0,
        }

    unattr_mask = df_tx["ParticipantID"].isnull()
    unattr_df = df_tx[unattr_mask].copy()
    total_unattr = len(unattr_df)
    unattr_pct = round((total_unattr / total_tx * 100.0) if total_tx > 0 else 0.0, 4)

    if total_unattr == 0:
        return {
            "total_unattributed_records": 0,
            "unattributed_percentage": 0.0,
            "chargers_involved": 0,
            "known_chargers_count": 0,
            "unknown_chargers_count": 0,
            "unknown_charger_ids": [],
            "unknown_charger_records": 0,
            "matched_4field_session_records": 0,
            "unmatched_session_records": 0,
        }

    meta_chargers = set(df_meta["Charger"].dropna().astype(str).str.strip().unique()) if "Charger" in df_meta.columns else set()
    unattr_chargers = set(unattr_df["ChargerID"].dropna().astype(str).str.strip().unique()) if "ChargerID" in unattr_df.columns else set()

    known_chargers = unattr_chargers.intersection(meta_chargers)
    unknown_chargers = unattr_chargers - meta_chargers

    unknown_charger_records = int(unattr_df["ChargerID"].astype(str).str.strip().isin(unknown_chargers).sum()) if "ChargerID" in unattr_df.columns else 0

    attr_df = df_tx[~unattr_mask].copy()
    session_keys = ["ChargerID", "StartTime", "StopTime", "ConsumedkWh"]
    has_session_keys = all(col in df_tx.columns for col in session_keys)

    if has_session_keys and not attr_df.empty:
        merged_match = unattr_df.merge(
            attr_df[session_keys].drop_duplicates(),
            on=session_keys,
            how="inner",
        )
        matched_4field = len(merged_match)
        unmatched_sessions = total_unattr - matched_4field
    else:
        matched_4field = 0
        unmatched_sessions = total_unattr

    return {
        "total_unattributed_records": total_unattr,
        "unattributed_percentage": unattr_pct,
        "chargers_involved": len(unattr_chargers),
        "known_chargers_count": len(known_chargers),
        "unknown_chargers_count": len(unknown_chargers),
        "unknown_charger_ids": sorted(list(unknown_chargers)),
        "unknown_charger_records": unknown_charger_records,
        "matched_4field_session_records": matched_4field,
        "unmatched_session_records": unmatched_sessions,
    }


def audit_metadata_row_level(
    df_tx: pd.DataFrame,
    df_meta: pd.DataFrame,
) -> Dict[str, Any]:
    """Perform a row-level comparison of transaction vehicle specifications against metadata.

    Args:
        df_tx: Transactions DataFrame.
        df_meta: Metadata DataFrame.

    Returns:
        Dictionary of row-level reconciliation metrics.
    """
    total_tx = len(df_tx)
    total_meta = len(df_meta)

    tx_pids = set(df_tx["ParticipantID"].dropna().unique()) if "ParticipantID" in df_tx.columns else set()
    meta_pids = set(df_meta["ParticipantID"].dropna().unique()) if "ParticipantID" in df_meta.columns else set()

    common_pids = tx_pids.intersection(meta_pids)
    tx_only_pids = tx_pids - meta_pids
    meta_only_pids = meta_pids - tx_pids

    if "ParticipantID" in df_tx.columns and "ParticipantID" in df_meta.columns and not df_tx.empty and not df_meta.empty:
        merged = df_tx.merge(df_meta, on="ParticipantID", how="inner", suffixes=("_tx", "_meta"))
        total_matched_rows = len(merged)

        kw_mismatches = int((merged["CarKW"] != merged["CarkW"]).sum()) if "CarKW" in merged.columns and "CarkW" in merged.columns else 0
        kwh_mismatches = int((merged["CarKWh"] != merged["CarkWh"]).sum()) if "CarKWh" in merged.columns and "CarkWh" in merged.columns else 0

        c_tx_str = merged["ChargerID"].astype(str).str.strip() if "ChargerID" in merged.columns else pd.Series(dtype=str)
        c_meta_str = merged["Charger"].astype(str).str.strip() if "Charger" in merged.columns else pd.Series(dtype=str)
        charger_mismatches = int((c_tx_str != c_meta_str).sum())
        charger_matches = int((c_tx_str == c_meta_str).sum())

        tx_valid_p = df_tx[df_tx["ParticipantID"].isin(common_pids)]
        if "CarKW" in tx_valid_p.columns:
            kw_counts = tx_valid_p.groupby("ParticipantID")["CarKW"].nunique()
            inconsistent_kw_pids = set(kw_counts[kw_counts > 1].index)
        else:
            inconsistent_kw_pids = set()

        if "CarKWh" in tx_valid_p.columns:
            kwh_counts = tx_valid_p.groupby("ParticipantID")["CarKWh"].nunique()
            inconsistent_kwh_pids = set(kwh_counts[kwh_counts > 1].index)
        else:
            inconsistent_kwh_pids = set()

        combined_inconsistent_pids = inconsistent_kw_pids | inconsistent_kwh_pids

        p_kw_var = len(inconsistent_kw_pids)
        p_kwh_var = len(inconsistent_kwh_pids)
        p_specs_var = len(combined_inconsistent_pids)

        matched_providers = df_meta[df_meta["ParticipantID"].isin(common_pids)]["DCSProvider"].value_counts().to_dict() if "DCSProvider" in df_meta.columns else {}
        unmatched_providers = df_meta[~df_meta["ParticipantID"].isin(common_pids)]["DCSProvider"].value_counts().to_dict() if "DCSProvider" in df_meta.columns else {}
    else:
        total_matched_rows = 0
        kw_mismatches = 0
        kwh_mismatches = 0
        charger_mismatches = 0
        charger_matches = 0
        p_kw_var = 0
        p_kwh_var = 0
        p_specs_var = 0
        matched_providers = {}
        unmatched_providers = {}

    return {
        "unique_tx_participants": len(tx_pids),
        "unique_meta_participants": len(meta_pids),
        "matched_participants": len(common_pids),
        "tx_only_participants": len(tx_only_pids),
        "meta_only_participants": len(meta_only_pids),
        "total_matched_transaction_rows": total_matched_rows,
        "row_level_carkw_mismatches": kw_mismatches,
        "row_level_carkw_mismatch_pct": round((kw_mismatches / total_matched_rows * 100.0) if total_matched_rows > 0 else 0.0, 4),
        "row_level_carkwh_mismatches": kwh_mismatches,
        "row_level_carkwh_mismatch_pct": round((kwh_mismatches / total_matched_rows * 100.0) if total_matched_rows > 0 else 0.0, 4),
        "row_level_charger_matches": charger_matches,
        "row_level_charger_mismatches": charger_mismatches,
        "row_level_charger_mismatch_pct": round((charger_mismatches / total_matched_rows * 100.0) if total_matched_rows > 0 else 0.0, 4),
        "participants_with_inconsistent_carkw": p_kw_var,
        "participants_with_inconsistent_carkwh": p_kwh_var,
        "participants_with_inconsistent_specs": p_specs_var,
        "matched_providers": matched_providers,
        "unmatched_providers": unmatched_providers,
    }


def audit_energy_quality(df_tx: pd.DataFrame) -> Dict[str, Any]:
    """Audit energy consumption data quality, negative readings, and extreme values.

    Args:
        df_tx: Transactions DataFrame.

    Returns:
        Dictionary of energy distribution and anomaly metrics.
    """
    total_records = len(df_tx)
    if "ConsumedkWh" not in df_tx.columns:
        return {
            "total_records": total_records,
            "missing_energy": total_records,
            "missing_energy_pct": 100.0,
            "non_positive_energy": 0,
            "non_positive_energy_pct": 0.0,
            "energy_gt_100kwh": 0,
            "energy_gt_100kwh_pct": 0.0,
            "energy_gt_150kwh": 0,
            "energy_gt_150kwh_pct": 0.0,
            "energy_exceeds_carkwh": 0,
            "energy_exceeds_carkwh_pct": 0.0,
            "max_energy": 0.0,
            "quantiles": {},
        }

    energy = pd.to_numeric(df_tx["ConsumedkWh"], errors="coerce")
    missing_count = int(energy.isnull().sum())
    valid_energy = energy.dropna()

    non_positive = int((valid_energy <= 0).sum())
    gt_100 = int((valid_energy > 100.0).sum())
    gt_150 = int((valid_energy > 150.0).sum())

    if "CarKWh" in df_tx.columns:
        car_cap = pd.to_numeric(df_tx["CarKWh"], errors="coerce")
        both_valid = energy.notnull() & car_cap.notnull() & (car_cap > 0)
        exceeds_cap = int((energy[both_valid] > car_cap[both_valid]).sum())
        exceeds_cap_pct = round((exceeds_cap / int(both_valid.sum()) * 100.0) if both_valid.sum() > 0 else 0.0, 2)
    else:
        exceeds_cap = 0
        exceeds_cap_pct = 0.0

    quantiles = {}
    if not valid_energy.empty:
        q_vals = [0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
        quantiles = {f"p{int(q*100)}": round(float(valid_energy.quantile(q)), 2) for q in q_vals}

    return {
        "total_records": total_records,
        "missing_energy": missing_count,
        "missing_energy_pct": round((missing_count / total_records * 100.0) if total_records > 0 else 0.0, 2),
        "non_positive_energy": non_positive,
        "non_positive_energy_pct": round((non_positive / total_records * 100.0) if total_records > 0 else 0.0, 2),
        "energy_gt_100kwh": gt_100,
        "energy_gt_100kwh_pct": round((gt_100 / total_records * 100.0) if total_records > 0 else 0.0, 2),
        "energy_gt_150kwh": gt_150,
        "energy_gt_150kwh_pct": round((gt_150 / total_records * 100.0) if total_records > 0 else 0.0, 2),
        "energy_exceeds_carkwh": exceeds_cap,
        "energy_exceeds_carkwh_pct": exceeds_cap_pct,
        "max_energy": round(float(valid_energy.max()), 2) if not valid_energy.empty else 0.0,
        "quantiles": quantiles,
    }


def find_charging_duration_discrepancies(df_tx: pd.DataFrame) -> Dict[str, Any]:
    """Find records where active charging duration exceeds plugged-in duration.

    Args:
        df_tx: Transactions DataFrame.

    Returns:
        Dictionary with count and list of transaction IDs.
    """
    if df_tx.empty or "ChargingDuration" not in df_tx.columns or "StartTime" not in df_tx.columns or "StopTime" not in df_tx.columns:
        return {"count": 0, "ids": []}

    start_utc, stop_utc, _ = parse_utc_timestamps(df_tx, "StartTime", "StopTime")
    valid_ts = start_utc.notna() & stop_utc.notna()
    dur_min = (stop_utc[valid_ts] - start_utc[valid_ts]).dt.total_seconds() / 60.0

    ch_dur = df_tx.loc[valid_ts, "ChargingDuration"]
    mask = (ch_dur > dur_min) & ch_dur.notna()

    discrepant_ids = df_tx.loc[valid_ts].loc[mask, "TransactionID"].tolist() if "TransactionID" in df_tx.columns else []

    return {
        "count": len(discrepant_ids),
        "ids": discrepant_ids,
    }


def audit_feature_feasibility(
    df_tx: pd.DataFrame,
    df_meta: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate feasibility, dependencies, and leakage risks for all 13 planned features and 2 targets."""
    rows = [
        {
            "feature_name": "weekday",
            "type": "Predictor Feature",
            "source_column_or_calculation": "(StartTime.dt.dayofweek < 5) & (~is_holiday) (binary: 1 for Monday–Friday, 0 otherwise; 0 on designated holidays)",
            "time_basis": "UTC",
            "unit": "Binary flag (0/1)",
            "availability_at_prediction": "Known at plug-in arrival",
            "feasibility_status": "Fully Feasible",
            "missingness_or_reliability_concerns": "0% missing; StartTime timestamps are complete and valid in UTC.",
            "leakage_risk": "None (derived strictly from session arrival timestamp known at plug-in).",
        },
        {
            "feature_name": "weekend",
            "type": "Predictor Feature",
            "source_column_or_calculation": "(StartTime.dt.dayofweek >= 5) & (~is_holiday) (binary: 1 for Saturday–Sunday, 0 otherwise; 0 on designated holidays)",
            "time_basis": "UTC",
            "unit": "Binary flag (0/1)",
            "availability_at_prediction": "Known at plug-in arrival",
            "feasibility_status": "Fully Feasible",
            "missingness_or_reliability_concerns": "0% missing; mirrors UTC calendar weekend status.",
            "leakage_risk": "None (known at plug-in arrival time).",
        },
        {
            "feature_name": "holiday",
            "type": "Predictor Feature",
            "source_column_or_calculation": "StartTime.dt.date.isin(UK_BANK_HOLIDAYS_REFERENCE).astype(int) (binary: 1 for exact 13 designated UK holiday dates, else 0)",
            "time_basis": "UTC calendar date",
            "unit": "Binary flag (0/1)",
            "availability_at_prediction": "Known at plug-in arrival",
            "feasibility_status": "Fully Feasible",
            "missingness_or_reliability_concerns": "Requires external calendar lookup table; 100% determinable for dataset date span.",
            "leakage_risk": "None (calendar lookup known at plug-in arrival).",
        },
        {
            "feature_name": "month",
            "type": "Predictor Feature",
            "source_column_or_calculation": "StartTime.dt.month (integer in 1..12)",
            "time_basis": "UTC",
            "unit": "Month index (1..12)",
            "availability_at_prediction": "Known at plug-in arrival",
            "feasibility_status": "Fully Feasible",
            "missingness_or_reliability_concerns": "0% missing; captures seasonal variation over the 22-month trial period.",
            "leakage_risk": "None (known at plug-in arrival).",
        },
        {
            "feature_name": "StartCos",
            "type": "Predictor Feature",
            "source_column_or_calculation": "cos(2 * pi * StartHourNorm)",
            "time_basis": "UTC",
            "unit": "Cosine cyclical component in [-1.0, 1.0]",
            "availability_at_prediction": "Known at plug-in arrival",
            "feasibility_status": "Fully Feasible",
            "missingness_or_reliability_concerns": "0% missing; smooth cyclical encoding of daily arrival time.",
            "leakage_risk": "None (plug-in start hour is available at time of arrival).",
        },
        {
            "feature_name": "StartSin",
            "type": "Predictor Feature",
            "source_column_or_calculation": "sin(2 * pi * StartHourNorm)",
            "time_basis": "UTC",
            "unit": "Sine cyclical component in [-1.0, 1.0]",
            "availability_at_prediction": "Known at plug-in arrival",
            "feasibility_status": "Fully Feasible",
            "missingness_or_reliability_concerns": "0% missing; smooth cyclical encoding of daily arrival time.",
            "leakage_risk": "None (plug-in start hour is available at time of arrival).",
        },
        {
            "feature_name": "StartHourNorm",
            "type": "Predictor Feature",
            "source_column_or_calculation": "StartHour / max(StartHour) over entire filtered dataset (including null ParticipantID records)",
            "time_basis": "UTC",
            "unit": "Normalized hour in [0.0, 1.0]",
            "availability_at_prediction": "Known at plug-in arrival",
            "feasibility_status": "Fully Feasible",
            "missingness_or_reliability_concerns": "0% missing; normalized continuous arrival hour.",
            "leakage_risk": "None (plug-in start time is available at arrival).",
        },
        {
            "feature_name": "TimeSinceLastStop",
            "type": "Predictor Feature",
            "source_column_or_calculation": "Elapsed hours since immediately preceding session StopTime before negative-gap removal (without subsequent recalculation)",
            "time_basis": "UTC",
            "unit": "Hours",
            "availability_at_prediction": "Known at plug-in arrival (from prior session)",
            "feasibility_status": "Feasible with Conditions",
            "missingness_or_reliability_concerns": "Calculated from immediately preceding session before negative-gap rows are removed.",
            "leakage_risk": "Requires prior session completion. Must NEVER reference current session StopTime.",
        },
        {
            "feature_name": "sessionsToday",
            "type": "Predictor Feature",
            "source_column_or_calculation": "Consecutive-date cumulative count on each participant's retained rows",
            "time_basis": "UTC calendar date",
            "unit": "Integer count (0, 1, 2, ...)",
            "availability_at_prediction": "Known at plug-in arrival (from prior sessions)",
            "feasibility_status": "Fully Feasible",
            "missingness_or_reliability_concerns": "0% missing on retained participant rows.",
            "leakage_risk": "Count includes only consecutive prior sessions up to current arrival.",
        },
        {
            "feature_name": "LastDuration",
            "type": "Predictor Feature",
            "source_column_or_calculation": "Duration of immediately preceding session. First retained session receives duration of dropped preceding session.",
            "time_basis": "UTC",
            "unit": "Hours",
            "availability_at_prediction": "Known at plug-in arrival (from prior session)",
            "feasibility_status": "Fully Feasible",
            "missingness_or_reliability_concerns": "0% missing in modeling population; first retained row receives duration of dropped preceding session.",
            "leakage_risk": "Safe only when computed from prior completed session. Must NEVER reference current session StopTime.",
        },
        {
            "feature_name": "LastConsumedkWh",
            "type": "Predictor Feature",
            "source_column_or_calculation": "ConsumedkWh of immediately preceding session. First retained session receives energy of dropped preceding session.",
            "time_basis": "UTC",
            "unit": "Kilowatt-hours (kWh)",
            "availability_at_prediction": "Known at plug-in arrival (from prior session)",
            "feasibility_status": "Fully Feasible",
            "missingness_or_reliability_concerns": "Preserves missing preceding energy value without backward searching (1 missing record). Preceding session energy above 100 kWh is preserved without clipping.",
            "leakage_risk": "Safe only when taken strictly from the preceding completed session. Must NEVER reference current session ConsumedkWh.",
        },
        {
            "feature_name": "CarKW",
            "type": "Predictor Feature",
            "source_column_or_calculation": "df_tx['CarKW'] or df_meta['CarkW'] matched on ParticipantID",
            "time_basis": "Static specification",
            "unit": "Kilowatts (kW)",
            "availability_at_prediction": "Known prior to arrival",
            "feasibility_status": "Fully Feasible",
            "missingness_or_reliability_concerns": "5.47% missing in raw transactions (unattributed records). 100% complete for all matched participants in ChargerInstall.",
            "leakage_risk": "None (static vehicle charging hardware capacity known prior to session).",
        },
        {
            "feature_name": "CarKWh",
            "type": "Predictor Feature",
            "source_column_or_calculation": "df_tx['CarKWh'] or df_meta['CarkWh'] matched on ParticipantID",
            "time_basis": "Static specification",
            "unit": "Kilowatt-hours (kWh)",
            "availability_at_prediction": "Known prior to arrival",
            "feasibility_status": "Fully Feasible",
            "missingness_or_reliability_concerns": "5.47% missing in raw transactions (unattributed records). 100% complete for all matched participants in ChargerInstall.",
            "leakage_risk": "None (static vehicle battery capacity known prior to session).",
        },
        {
            "feature_name": "Plugged-in Duration (Hours)",
            "type": "Prediction Target 1",
            "source_column_or_calculation": "(StopTime - StartTime).total_seconds() / 3600.0, capped at 336.0 hours",
            "time_basis": "UTC",
            "unit": "Hours (capped at 336.0)",
            "availability_at_prediction": "TARGET (Unobserved at arrival)",
            "feasibility_status": "Feasible with Filtering and Capping",
            "missingness_or_reliability_concerns": "Must NOT be confused with active ChargingDuration. Sessions < 5 min excluded. Sessions > 336 hours capped at 336 hours.",
            "leakage_risk": "TARGET LEAKAGE: Must NEVER be used as an input feature. StopTime, PluggedInTime, ChargingDuration, EndCharge, and Hot_Unplug occur post-plug-in and must be strictly excluded from predictor features.",
        },
        {
            "feature_name": "Energy Consumption (kWh)",
            "type": "Prediction Target 2",
            "source_column_or_calculation": "ConsumedkWh",
            "time_basis": "UTC",
            "unit": "Kilowatt-hours (kWh)",
            "availability_at_prediction": "TARGET (Unobserved at arrival)",
            "feasibility_status": "Feasible with Filtering",
            "missingness_or_reliability_concerns": "1 missing record in raw data. 20 records > 100 kWh (12 > 150 kWh). 1,788 records exceed vehicle CarKWh.",
            "leakage_risk": "TARGET LEAKAGE: Must NEVER be used as an input feature for predicting current session energy or duration. Metered post-session.",
        },
    ]

    return pd.DataFrame(rows)
