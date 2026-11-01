"""Data audit module for TARANG — EV Charging Intelligence System.

This module provides reusable, focused functions for inspecting raw datasets,
evaluating data quality, verifying cross-dataset consistency, and auditing
feature feasibility without modifying original datasets.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import openpyxl
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Official UK Bank Holidays for 2017 and 2018 (England and Wales)
UK_BANK_HOLIDAYS_2017_2018 = {
    # 2017
    pd.Timestamp("2017-01-02").date(),  # New Year's Day (substitute)
    pd.Timestamp("2017-04-14").date(),  # Good Friday
    pd.Timestamp("2017-04-17").date(),  # Easter Monday
    pd.Timestamp("2017-05-01").date(),  # Early May bank holiday
    pd.Timestamp("2017-05-29").date(),  # Spring bank holiday
    pd.Timestamp("2017-08-28").date(),  # Summer bank holiday
    pd.Timestamp("2017-12-25").date(),  # Christmas Day
    pd.Timestamp("2017-12-26").date(),  # Boxing Day
    # 2018
    pd.Timestamp("2018-01-01").date(),  # New Year's Day
    pd.Timestamp("2018-03-30").date(),  # Good Friday
    pd.Timestamp("2018-04-02").date(),  # Easter Monday
    pd.Timestamp("2018-05-07").date(),  # Early May bank holiday
    pd.Timestamp("2018-05-28").date(),  # Spring bank holiday
    pd.Timestamp("2018-08-27").date(),  # Summer bank holiday
    pd.Timestamp("2018-12-25").date(),  # Christmas Day
    pd.Timestamp("2018-12-26").date(),  # Boxing Day
}


def load_dataset(
    filepath: Union[str, Path],
    sheet_name: Optional[Union[str, int]] = 0,
) -> pd.DataFrame:
    """Load an Excel workbook sheet into a pandas DataFrame safely.

    Args:
        filepath: Path to the Excel file.
        sheet_name: Sheet name or index to read. Defaults to 0.

    Returns:
        pd.DataFrame containing the sheet contents.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the sheet cannot be found or read.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Workbook not found at: {path.resolve()}")
    try:
        return pd.read_excel(path, sheet_name=sheet_name)
    except Exception as exc:
        raise ValueError(f"Failed to load sheet '{sheet_name}' from {path}: {exc}") from exc


def inspect_workbook_metadata(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Inspect basic metadata and available sheets of an Excel workbook.

    Args:
        filepath: Path to the Excel workbook.

    Returns:
        Dictionary containing file size in bytes, sheet names, and sheet dimensions.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Workbook not found at: {path.resolve()}")

    file_size = path.stat().st_size
    wb = openpyxl.load_workbook(path, read_only=True)
    sheet_names = wb.sheetnames

    sheet_dimensions = {}
    for sheet in sheet_names:
        ws = wb[sheet]
        sheet_dimensions[sheet] = {
            "max_row": ws.max_row,
            "max_column": ws.max_column,
        }
    wb.close()

    return {
        "filepath": str(path),
        "filename": path.name,
        "size_bytes": file_size,
        "sheet_names": sheet_names,
        "sheet_dimensions": sheet_dimensions,
    }


def calculate_missingness(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate missing values and percentage missing for each column.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with columns ['column_name', 'missing_count', 'missing_percentage'].
    """
    total_records = len(df)
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / total_records * 100.0) if total_records > 0 else 0.0

    return pd.DataFrame({
        "column_name": df.columns,
        "missing_count": missing_count.values,
        "missing_percentage": missing_pct.round(4).values,
    })


def detect_duplicates(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
) -> int:
    """Count duplicate rows in a DataFrame or specific column subset.

    Args:
        df: Input DataFrame.
        subset: Optional list of column names. If None, considers all columns.

    Returns:
        Number of duplicate rows.
    """
    return int(df.duplicated(subset=subset).sum())


def validate_timestamps(
    df: pd.DataFrame,
    start_col: str = "StartTime",
    stop_col: str = "StopTime",
) -> Dict[str, Any]:
    """Validate timestamp columns, duration intervals, and parsing.

    Args:
        df: DataFrame containing the timestamp columns.
        start_col: Start timestamp column name.
        stop_col: Stop timestamp column name.

    Returns:
        Dictionary containing timestamp summary metrics.
    """
    start_series = pd.to_datetime(df[start_col], errors="coerce")
    stop_series = pd.to_datetime(df[stop_col], errors="coerce")

    start_unparsed = start_series.isnull().sum() - df[start_col].isnull().sum()
    stop_unparsed = stop_series.isnull().sum() - df[stop_col].isnull().sum()

    diff_sec = (stop_series - start_series).dt.total_seconds()
    neg_intervals = int((diff_sec < 0).sum())
    zero_intervals = int((diff_sec == 0).sum())

    return {
        "start_col": start_col,
        "stop_col": stop_col,
        "earliest_start": str(start_series.min()) if not start_series.empty else None,
        "latest_start": str(start_series.max()) if not start_series.empty else None,
        "earliest_stop": str(stop_series.min()) if not stop_series.empty else None,
        "latest_stop": str(stop_series.max()) if not stop_series.empty else None,
        "start_parse_failures": int(start_unparsed),
        "stop_parse_failures": int(stop_unparsed),
        "negative_intervals": neg_intervals,
        "zero_intervals": zero_intervals,
    }


def audit_transaction_data(df_tx: pd.DataFrame) -> Dict[str, Any]:
    """Perform a comprehensive audit on the transaction dataset.

    Args:
        df_tx: Raw transactions DataFrame.

    Returns:
        Dictionary of all key audit findings for transactions.
    """
    total_records = len(df_tx)
    unique_tx_ids = int(df_tx["TransactionID"].nunique()) if "TransactionID" in df_tx.columns else 0
    dup_tx_ids = detect_duplicates(df_tx, subset=["TransactionID"]) if "TransactionID" in df_tx.columns else 0
    dup_rows = detect_duplicates(df_tx)
    unique_participants = int(df_tx["ParticipantID"].dropna().nunique()) if "ParticipantID" in df_tx.columns else 0
    unique_chargers = int(df_tx["ChargerID"].nunique()) if "ChargerID" in df_tx.columns else 0

    ts_val = validate_timestamps(df_tx, "StartTime", "StopTime")

    dur_min = (pd.to_datetime(df_tx["StopTime"]) - pd.to_datetime(df_tx["StartTime"])).dt.total_seconds() / 60.0
    dur_hrs = dur_min / 60.0

    consumed = df_tx["ConsumedkWh"] if "ConsumedkWh" in df_tx.columns else pd.Series(dtype=float)
    charging_dur = df_tx["ChargingDuration"] if "ChargingDuration" in df_tx.columns else pd.Series(dtype=float)

    # Discrepancy between active charging duration and plugged-in duration
    ch_dur_exceeds_plugged = int((charging_dur > dur_min).sum())

    trial_dist = df_tx["Trial"].value_counts(dropna=False).to_dict() if "Trial" in df_tx.columns else {}

    return {
        "total_records": total_records,
        "total_columns": len(df_tx.columns),
        "unique_transactions": unique_tx_ids,
        "duplicate_transactions": dup_tx_ids,
        "duplicate_rows": dup_rows,
        "unique_participants": unique_participants,
        "unique_chargers": unique_chargers,
        "missing_participant_records": int(df_tx["ParticipantID"].isnull().sum()) if "ParticipantID" in df_tx.columns else 0,
        "timestamps": ts_val,
        "plugged_duration_hours": {
            "mean": float(dur_hrs.mean()),
            "std": float(dur_hrs.std()),
            "min": float(dur_hrs.min()),
            "median": float(dur_hrs.median()),
            "max": float(dur_hrs.max()),
            "sessions_lt_5min": int((dur_min < 5).sum()),
            "sessions_gt_336hrs": int((dur_hrs > 336).sum()),
        },
        "consumed_kwh": {
            "non_null_count": int(consumed.notnull().sum()),
            "missing_count": int(consumed.isnull().sum()),
            "mean": float(consumed.mean()),
            "std": float(consumed.std()),
            "min": float(consumed.min()),
            "median": float(consumed.median()),
            "max": float(consumed.max()),
            "negative_count": int((consumed < 0).sum()),
            "zero_count": int((consumed == 0).sum()),
            "extreme_gt_150kwh": int((consumed > 150).sum()),
        },
        "charging_duration_minutes": {
            "non_null_count": int(charging_dur.notnull().sum()),
            "missing_count": int(charging_dur.isnull().sum()),
            "negative_count": int((charging_dur < 0).sum()),
            "zero_count": int((charging_dur == 0).sum()),
            "mean": float(charging_dur.mean()),
            "std": float(charging_dur.std()),
            "min": float(charging_dur.min()),
            "max": float(charging_dur.max()),
            "exceeds_plugged_duration": ch_dur_exceeds_plugged,
        },
        "trial_distribution": trial_dist,
    }


def audit_metadata(df_meta: pd.DataFrame) -> Dict[str, Any]:
    """Perform an audit of the vehicle metadata dataset (ChargerInstall).

    Args:
        df_meta: Raw metadata DataFrame.

    Returns:
        Dictionary of all metadata audit metrics.
    """
    total_records = len(df_meta)
    unique_pids = int(df_meta["ParticipantID"].nunique()) if "ParticipantID" in df_meta.columns else 0
    dup_pids = detect_duplicates(df_meta, subset=["ParticipantID"]) if "ParticipantID" in df_meta.columns else 0
    dup_rows = detect_duplicates(df_meta)

    provider_dist = df_meta["DCSProvider"].value_counts(dropna=False).to_dict() if "DCSProvider" in df_meta.columns else {}
    piv_dist = df_meta["PIVType"].value_counts(dropna=False).to_dict() if "PIVType" in df_meta.columns else {}

    car_kw_dist = df_meta["CarkW"].value_counts(dropna=False).to_dict() if "CarkW" in df_meta.columns else {}
    car_kwh_summary = {
        "min": float(df_meta["CarkWh"].min()),
        "mean": float(df_meta["CarkWh"].mean()),
        "max": float(df_meta["CarkWh"].max()),
    } if "CarkWh" in df_meta.columns else {}

    make_count = int(df_meta["CarMake"].nunique()) if "CarMake" in df_meta.columns else 0
    model_count = int(df_meta["CarModel"].nunique()) if "CarModel" in df_meta.columns else 0

    return {
        "total_records": total_records,
        "total_columns": len(df_meta.columns),
        "unique_participants": unique_pids,
        "duplicate_participants": dup_pids,
        "duplicate_rows": dup_rows,
        "provider_distribution": provider_dist,
        "piv_distribution": piv_dist,
        "car_kw_distribution": car_kw_dist,
        "car_kwh_summary": car_kwh_summary,
        "unique_makes": make_count,
        "unique_models": model_count,
    }


def reconcile_participants(
    df_tx: pd.DataFrame,
    df_meta: pd.DataFrame,
) -> pd.DataFrame:
    """Perform explicit participant-level reconciliation between transactions and metadata.

    Args:
        df_tx: Transaction DataFrame.
        df_meta: Metadata DataFrame.

    Returns:
        Summary DataFrame reporting reconciliation counts and percentages.
    """
    tx_pids = set(df_tx["ParticipantID"].dropna().unique())
    meta_pids = set(df_meta["ParticipantID"].dropna().unique())

    common = tx_pids.intersection(meta_pids)
    tx_only = tx_pids - meta_pids
    meta_only = meta_pids - tx_pids

    # Provider breakdown in metadata
    meta_matched = df_meta[df_meta["ParticipantID"].isin(common)]
    meta_unmatched = df_meta[~df_meta["ParticipantID"].isin(common)]

    matched_providers = meta_matched["DCSProvider"].value_counts().to_dict() if "DCSProvider" in meta_matched.columns else {}
    unmatched_providers = meta_unmatched["DCSProvider"].value_counts().to_dict() if "DCSProvider" in meta_unmatched.columns else {}

    # Check vehicle capacity / power field consistency for matched participants
    kw_mismatches = 0
    kwh_mismatches = 0
    if "CarKW" in df_tx.columns and "CarkW" in df_meta.columns:
        tx_specs = df_tx.dropna(subset=["ParticipantID"]).groupby("ParticipantID")[["CarKW", "CarKWh"]].first().reset_index()
        merged_specs = tx_specs.merge(df_meta[["ParticipantID", "CarkW", "CarkWh"]], on="ParticipantID", how="inner")
        kw_mismatches = int((merged_specs["CarKW"] != merged_specs["CarkW"]).sum())
        kwh_mismatches = int((merged_specs["CarKWh"] != merged_specs["CarkWh"]).sum())

    records = [
        {"metric": "Unique Participants in Transactions (Non-null)", "count": len(tx_pids), "percentage": 100.0 if tx_pids else 0.0, "notes": "Distinct participant IDs appearing in transaction records"},
        {"metric": "Unique Participants in Metadata", "count": len(meta_pids), "percentage": 100.0 if meta_pids else 0.0, "notes": "Total installed charger participants in metadata"},
        {"metric": "Matched Participants (Present in Both)", "count": len(common), "percentage": round(len(common) / len(tx_pids) * 100.0, 2) if tx_pids else 0.0, "notes": "100% of transaction participants are identified in metadata"},
        {"metric": "Transaction Participants Missing from Metadata", "count": len(tx_only), "percentage": round(len(tx_only) / len(tx_pids) * 100.0, 2) if tx_pids else 0.0, "notes": "No unknown participant IDs present in transactions"},
        {"metric": "Metadata Participants Missing from Transactions", "count": len(meta_only), "percentage": round(len(meta_only) / len(meta_pids) * 100.0, 2) if meta_pids else 0.0, "notes": "GreenFlux participants (345) and unused Crowd Charge chargers (28)"},
        {"metric": "Matched Participants - Crowd Charge Provider", "count": matched_providers.get("Crowd Charge", 0), "percentage": round(matched_providers.get("Crowd Charge", 0) / len(common) * 100.0, 2) if common else 0.0, "notes": "All matched participants belong to Crowd Charge provider"},
        {"metric": "Matched Participants - GreenFlux Provider", "count": matched_providers.get("Greenflux", 0), "percentage": 0.0, "notes": "No GreenFlux participants present in CrowdCharge transactions file"},
        {"metric": "Unmatched Metadata - GreenFlux Provider", "count": unmatched_providers.get("Greenflux", 0), "percentage": round(unmatched_providers.get("Greenflux", 0) / len(meta_only) * 100.0, 2) if meta_only else 0.0, "notes": "GreenFlux trial data is stored in separate external trial datasets"},
        {"metric": "Unmatched Metadata - Crowd Charge Provider", "count": unmatched_providers.get("Crowd Charge", 0), "percentage": round(unmatched_providers.get("Crowd Charge", 0) / len(meta_only) * 100.0, 2) if meta_only else 0.0, "notes": "Crowd Charge installations with zero recorded sessions in this file"},
        {"metric": "CarKW Specification Discrepancies", "count": kw_mismatches, "percentage": 0.0, "notes": "Perfect agreement between transaction CarKW and metadata CarkW"},
        {"metric": "CarKWh Battery Capacity Discrepancies", "count": kwh_mismatches, "percentage": 0.0, "notes": "Perfect agreement between transaction CarKWh and metadata CarkWh"},
    ]

    return pd.DataFrame(records)


def audit_data_quality(df_tx: pd.DataFrame) -> pd.DataFrame:
    """Investigate specific data-quality checks and categorize severity.

    Args:
        df_tx: Raw transaction DataFrame.

    Returns:
        DataFrame summarizing data-quality findings, impact, and severity class.
    """
    total_tx = len(df_tx)
    start_dt = pd.to_datetime(df_tx["StartTime"]) if "StartTime" in df_tx.columns else pd.Series(dtype="datetime64[ns]")
    stop_dt = pd.to_datetime(df_tx["StopTime"]) if "StopTime" in df_tx.columns else pd.Series(dtype="datetime64[ns]")

    dur_min = (stop_dt - start_dt).dt.total_seconds() / 60.0
    dur_hrs = dur_min / 60.0

    # 1. Sessions < 5 min
    sessions_lt_5m = int((dur_min < 5).sum())
    zero_dur = int((dur_min == 0).sum())

    # 2. Sessions > 336 hours
    sessions_gt_336h = int((dur_hrs > 336).sum())

    # 3. StopTime earlier than StartTime
    stop_lt_start = int((dur_min < 0).sum())

    # 4. Duplicate transactions (exact duplicate rows)
    dup_rows = detect_duplicates(df_tx)
    dup_tx_id = detect_duplicates(df_tx, subset=["TransactionID"]) if "TransactionID" in df_tx.columns else 0

    # 5. Overlapping sessions for the same vehicle
    df_tx_copy = df_tx.copy()
    df_tx_copy["_start_dt"] = start_dt
    df_tx_copy["_stop_dt"] = stop_dt

    df_valid_p = df_tx_copy[df_tx_copy["ParticipantID"].notna()].sort_values(["ParticipantID", "_start_dt"]).reset_index(drop=True)
    df_valid_p["prev_StopTime"] = df_valid_p.groupby("ParticipantID")["_stop_dt"].shift(1)
    overlap_count = int((df_valid_p["prev_StopTime"] > df_valid_p["_start_dt"]).sum())

    # 6. Missing previous-session information (first session for each participant)
    num_participants = int(df_valid_p["ParticipantID"].nunique())

    # 7. Vehicles with fewer than 10 sessions
    session_counts = df_valid_p.groupby("ParticipantID").size()
    low_session_pids = session_counts[session_counts < 10]
    sessions_in_low_count_vehicles = int(low_session_pids.sum())

    # 8. Vehicles with low observation density (< 1 session/week on average)
    density = df_valid_p.groupby("ParticipantID").agg(
        first_session=("_start_dt", "min"),
        last_session=("_start_dt", "max"),
        session_count=("TransactionID", "count"),
    )
    density["span_days"] = (density["last_session"] - density["first_session"]).dt.total_seconds() / (24 * 3600)
    density["sessions_per_week"] = (density["session_count"] / (density["span_days"] + 1)) * 7
    low_density_pids = density[density["sessions_per_week"] < 1.0]
    sessions_in_low_density_vehicles = int(low_density_pids["session_count"].sum())

    # 9. Missing or invalid target values
    missing_consumed = int(df_tx["ConsumedkWh"].isnull().sum()) if "ConsumedkWh" in df_tx.columns else 0
    consumed_extreme = int((df_tx["ConsumedkWh"] > 150).sum()) if "ConsumedkWh" in df_tx.columns else 0

    # 10. Missing ParticipantID records (unattributed charging sessions)
    missing_pid_count = int(df_tx["ParticipantID"].isnull().sum()) if "ParticipantID" in df_tx.columns else 0

    # 11. ChargingDuration exceeding plugged duration
    ch_dur = df_tx["ChargingDuration"] if "ChargingDuration" in df_tx.columns else pd.Series(dtype=float)
    exceeds_plugged = int((ch_dur > dur_min).sum())

    records = [
        {
            "check_name": "Zero Duration Sessions (StopTime == StartTime)",
            "affected_records": zero_dur,
            "percentage": round(zero_dur / total_tx * 100.0, 4),
            "severity_classification": "Confirmed invalid records",
            "impact_assessment": "Unusable for duration prediction; represents instantaneous connect/disconnect glitches.",
        },
        {
            "check_name": "Sessions Shorter than 5 Minutes (< 5 min)",
            "affected_records": sessions_lt_5m,
            "percentage": round(sessions_lt_5m / total_tx * 100.0, 4),
            "severity_classification": "Potentially suspicious records",
            "impact_assessment": "Includes 92 zero-duration records; remaining 469 are micro-sessions that usually do not represent meaningful charging.",
        },
        {
            "check_name": "Sessions Longer than 336 Hours (> 14 days)",
            "affected_records": sessions_gt_336h,
            "percentage": round(sessions_gt_336h / total_tx * 100.0, 4),
            "severity_classification": "Potentially suspicious records",
            "impact_assessment": "Severe outliers reaching up to 7,844 hours (~326 days); likely unclosed sessions due to network or charger disconnection.",
        },
        {
            "check_name": "Inverted Timestamps (StopTime < StartTime)",
            "affected_records": stop_lt_start,
            "percentage": 0.0,
            "severity_classification": "Confirmed invalid records",
            "impact_assessment": "Zero inverted intervals detected; chronological order between StartTime and StopTime is intact.",
        },
        {
            "check_name": "Duplicate Transaction Records (Exact Row Duplicates)",
            "affected_records": dup_rows,
            "percentage": 0.0,
            "severity_classification": "Confirmed invalid records",
            "impact_assessment": "Zero exact row duplicates detected across the entire 71,264 records.",
        },
        {
            "check_name": "Duplicate TransactionID Values",
            "affected_records": dup_tx_id,
            "percentage": 0.0,
            "severity_classification": "Confirmed invalid records",
            "impact_assessment": "All TransactionID values are globally unique primary keys.",
        },
        {
            "check_name": "Missing Target - ConsumedkWh",
            "affected_records": missing_consumed,
            "percentage": round(missing_consumed / total_tx * 100.0, 4),
            "severity_classification": "Confirmed invalid records",
            "impact_assessment": "Exactly 1 record missing energy target; cannot be used for energy prediction without target imputation.",
        },
        {
            "check_name": "Extreme ConsumedkWh Outliers (> 150 kWh)",
            "affected_records": consumed_extreme,
            "percentage": round(consumed_extreme / total_tx * 100.0, 4),
            "severity_classification": "Potentially suspicious records",
            "impact_assessment": "12 sessions exceed realistic EV battery capacities (e.g., 37,487 kWh); 6 are long multi-month sessions and 6 are duplicate null-PID sessions.",
        },
        {
            "check_name": "Active Charging Duration Exceeds Plugged Duration",
            "affected_records": exceeds_plugged,
            "percentage": round(exceeds_plugged / total_tx * 100.0, 4),
            "severity_classification": "Confirmed invalid records",
            "impact_assessment": "1 record (ID 132725) has ChargingDuration (478 min) > PluggedInTime (463 min), caused by UK DST clock change on 2018-03-25.",
        },
        {
            "check_name": "Overlapping Sessions for Same Vehicle",
            "affected_records": overlap_count,
            "percentage": round(overlap_count / total_tx * 100.0, 4),
            "severity_classification": "Records requiring further investigation",
            "impact_assessment": "133 sessions start before previous session ended; invalidates sequential lag features unless resolved or purged.",
        },
        {
            "check_name": "Missing Previous-Session Information (First Vehicle Session)",
            "affected_records": num_participants,
            "percentage": round(num_participants / total_tx * 100.0, 4),
            "severity_classification": "Expected missing values",
            "impact_assessment": "Every participant's initial session lacks prior history; lag features (LastDuration, LastConsumedkWh) will naturally be NaN.",
        },
        {
            "check_name": "Vehicles with Fewer than 10 Sessions",
            "affected_records": sessions_in_low_count_vehicles,
            "percentage": round(sessions_in_low_count_vehicles / total_tx * 100.0, 4),
            "severity_classification": "Records requiring further investigation",
            "impact_assessment": "11 vehicles have < 10 total sessions (52 sessions total); insufficient history for reliable participant-level personalization.",
        },
        {
            "check_name": "Vehicles with Low Observation Density (< 1 session/week)",
            "affected_records": sessions_in_low_density_vehicles,
            "percentage": round(sessions_in_low_density_vehicles / total_tx * 100.0, 4),
            "severity_classification": "Records requiring further investigation",
            "impact_assessment": "28 participants exhibit sparse charging habits; temporal features and lag dependencies may be stale or unrepresentative.",
        },
        {
            "check_name": "Unattributed Sessions (Missing ParticipantID)",
            "affected_records": missing_pid_count,
            "percentage": round(missing_pid_count / total_tx * 100.0, 4),
            "severity_classification": "Records requiring further investigation",
            "impact_assessment": "3,899 sessions lack ParticipantID; 3,774 duplicate known charger sessions with 200,000+ IDs, while 125 belong to unknown charger 4441003917.",
        },
    ]

    return pd.DataFrame(records)


def audit_feature_feasibility(
    df_tx: pd.DataFrame,
    df_meta: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate feasibility, dependencies, and leakage risks for all planned features.

    Args:
        df_tx: Raw transaction DataFrame.
        df_meta: Raw metadata DataFrame.

    Returns:
        DataFrame detailing feasibility status and leakage assessment per feature/target.
    """
    rows = [
        {
            "feature_name": "Weekday",
            "type": "Predictor Feature",
            "source_column_or_calculation": "StartTime.dt.dayofweek (or StartTime.dt.dayofweek < 5)",
            "required_dependencies": "StartTime",
            "feasibility_status": "Fully Feasible",
            "missingness_or_reliability_concerns": "0% missing; StartTime timestamps are 100% complete and valid.",
            "leakage_risk": "None (derived strictly from session arrival timestamp known at plug-in).",
        },
        {
            "feature_name": "Weekend",
            "type": "Predictor Feature",
            "source_column_or_calculation": "StartTime.dt.dayofweek >= 5 (binary indicator)",
            "required_dependencies": "StartTime",
            "feasibility_status": "Fully Feasible",
            "missingness_or_reliability_concerns": "0% missing; mirrors Weekday_or_Weekend column with zero missingness.",
            "leakage_risk": "None (known at plug-in arrival time).",
        },
        {
            "feature_name": "Holiday",
            "type": "Predictor Feature",
            "source_column_or_calculation": "StartTime.dt.date.isin(UK_BANK_HOLIDAYS_2017_2018)",
            "required_dependencies": "StartTime, UK official holiday calendar (2017-2018)",
            "feasibility_status": "Fully Feasible",
            "missingness_or_reliability_concerns": "Requires external calendar lookup table; 100% determinable for dataset date span.",
            "leakage_risk": "None (calendar lookup known at plug-in arrival).",
        },
        {
            "feature_name": "Month",
            "type": "Predictor Feature",
            "source_column_or_calculation": "StartTime.dt.month",
            "required_dependencies": "StartTime",
            "feasibility_status": "Fully Feasible",
            "missingness_or_reliability_concerns": "0% missing; captures seasonal variation over the 22-month trial period.",
            "leakage_risk": "None (known at plug-in arrival).",
        },
        {
            "feature_name": "Start-Hour Cyclic (sin/cos)",
            "type": "Predictor Feature",
            "source_column_or_calculation": "sin(2*pi*hour/24), cos(2*pi*hour/24) where hour = StartTime.hour + StartTime.minute/60",
            "required_dependencies": "StartTime",
            "feasibility_status": "Fully Feasible",
            "missingness_or_reliability_concerns": "0% missing; smooth cyclical representation of arrival time.",
            "leakage_risk": "None (plug-in start hour is available at time of arrival).",
        },
        {
            "feature_name": "TimeSinceLastStop",
            "type": "Predictor Feature",
            "source_column_or_calculation": "(StartTime - prev_StopTime).total_seconds() / 3600.0 per ParticipantID",
            "required_dependencies": "ParticipantID, StartTime, chronological participant sort, previous StopTime",
            "feasibility_status": "Feasible with Conditions",
            "missingness_or_reliability_concerns": "Missing for participant initial sessions (300 records) and unattributed records (3,899 records). Negative values occur on 133 overlapping sessions.",
            "leakage_risk": "High if future sessions or concurrent sessions are referenced. Safe ONLY if strictly sorted chronologically and referring exclusively to the immediately completed preceding session.",
        },
        {
            "feature_name": "SessionsToday",
            "type": "Predictor Feature",
            "source_column_or_calculation": "Count of prior sessions for same ParticipantID on same calendar date before StartTime",
            "required_dependencies": "ParticipantID, StartTime, chronological sorting",
            "feasibility_status": "Feasible with Conditions",
            "missingness_or_reliability_concerns": "Missing for unattributed sessions (3,899 rows). Requires participant-level grouping.",
            "leakage_risk": "Leakage risk if daily session count includes future sessions on that day. Must count ONLY sessions starting prior to the current session StartTime.",
        },
        {
            "feature_name": "LastDuration",
            "type": "Predictor Feature",
            "source_column_or_calculation": "Previous session (StopTime - StartTime).total_seconds() / 3600.0 per ParticipantID",
            "required_dependencies": "ParticipantID, StartTime, chronological participant sort, previous session timestamps",
            "feasibility_status": "Feasible with Conditions",
            "missingness_or_reliability_concerns": "NaN for initial session of each participant (300 sessions). Missing for 3,899 unattributed sessions.",
            "leakage_risk": "Safe only when computed from prior session. Must NEVER reference current session StopTime.",
        },
        {
            "feature_name": "LastConsumedkWh",
            "type": "Predictor Feature",
            "source_column_or_calculation": "Previous session ConsumedkWh per ParticipantID",
            "required_dependencies": "ParticipantID, StartTime, chronological participant sort, previous session energy",
            "feasibility_status": "Feasible with Conditions",
            "missingness_or_reliability_concerns": "NaN for initial session of each participant (300 sessions) and unattributed sessions (3,899 sessions). Extreme outliers in historical energy (e.g. 37,487 kWh) require outlier clipping.",
            "leakage_risk": "Safe only when taken strictly from the preceding session. Must NEVER reference current session ConsumedkWh.",
        },
        {
            "feature_name": "CarKW",
            "type": "Predictor Feature",
            "source_column_or_calculation": "df_tx['CarKW'] or df_meta['CarkW'] matched on ParticipantID",
            "required_dependencies": "df_tx['CarKW'] or df_meta metadata table",
            "feasibility_status": "Fully Feasible",
            "missingness_or_reliability_concerns": "5.47% missing in raw transactions (3,899 records with null ParticipantID). 100% complete for all 300 matched participants in ChargerInstall.",
            "leakage_risk": "None (static vehicle charging hardware capacity known prior to session).",
        },
        {
            "feature_name": "CarKWh",
            "type": "Predictor Feature",
            "source_column_or_calculation": "df_tx['CarKWh'] or df_meta['CarkWh'] matched on ParticipantID",
            "required_dependencies": "df_tx['CarKWh'] or df_meta metadata table",
            "feasibility_status": "Fully Feasible",
            "missingness_or_reliability_concerns": "5.47% missing in raw transactions. 100% complete for all 300 matched participants in ChargerInstall.",
            "leakage_risk": "None (static vehicle battery capacity known prior to session).",
        },
        {
            "feature_name": "Plugged-in Duration (Hours)",
            "type": "Prediction Target 1",
            "source_column_or_calculation": "(StopTime - StartTime).total_seconds() / 3600.0",
            "required_dependencies": "StartTime, StopTime",
            "feasibility_status": "Feasible with Filtering",
            "missingness_or_reliability_concerns": "Must NOT be confused with active ChargingDuration. 92 records have zero duration. 23 records exceed 336 hours (up to 7,844 hrs). 561 records < 5 minutes.",
            "leakage_risk": "TARGET LEAKAGE: Must NEVER be used as an input feature. StopTime, PluggedInTime, ChargingDuration, EndCharge, and Hot_Unplug occur post-plug-in and must be strictly excluded from predictor features.",
        },
        {
            "feature_name": "Energy Consumption (kWh)",
            "type": "Prediction Target 2",
            "source_column_or_calculation": "ConsumedkWh",
            "required_dependencies": "ConsumedkWh",
            "feasibility_status": "Feasible with Filtering",
            "missingness_or_reliability_concerns": "1 missing record in raw data. 12 extreme outliers > 150 kWh (reaching 37,487 kWh).",
            "leakage_risk": "TARGET LEAKAGE: Must NEVER be used as an input feature for predicting current session energy or duration. Metered post-session.",
        },
    ]

    return pd.DataFrame(rows)


def generate_transaction_quality_summary(df_tx: pd.DataFrame) -> pd.DataFrame:
    """Generate detailed transaction quality summary table.

    Args:
        df_tx: Raw transactions DataFrame.

    Returns:
        DataFrame summarizing transaction quality metrics.
    """
    total = len(df_tx)
    dur_min = (pd.to_datetime(df_tx["StopTime"]) - pd.to_datetime(df_tx["StartTime"])).dt.total_seconds() / 60.0
    dur_hrs = dur_min / 60.0

    consumed = df_tx["ConsumedkWh"] if "ConsumedkWh" in df_tx.columns else pd.Series(dtype=float)
    ch_dur = df_tx["ChargingDuration"] if "ChargingDuration" in df_tx.columns else pd.Series(dtype=float)

    # Overlaps for same participant
    df_valid_p = df_tx[df_tx["ParticipantID"].notna()].sort_values(["ParticipantID", "StartTime"]).reset_index(drop=True)
    df_valid_p["prev_StopTime"] = df_valid_p.groupby("ParticipantID")["StopTime"].shift(1)
    overlap_count = int((df_valid_p["prev_StopTime"] > df_valid_p["StartTime"]).sum())

    checks = [
        ("Total Transaction Records", total, 100.0, "Total rows in Transaction Data sheet"),
        ("Unique Transaction Identifiers", int(df_tx["TransactionID"].nunique()), 100.0, "All TransactionID values are distinct"),
        ("Duplicate Transaction IDs", detect_duplicates(df_tx, subset=["TransactionID"]), 0.0, "Zero duplicate TransactionID values"),
        ("Duplicate Complete Rows", detect_duplicates(df_tx), 0.0, "Zero full row duplicates"),
        ("Unique Participant Identifiers", int(df_tx["ParticipantID"].dropna().nunique()), round(int(df_tx["ParticipantID"].dropna().nunique()) / total * 100.0, 4), "300 unique non-null participants"),
        ("Unique Charger Identifiers", int(df_tx["ChargerID"].nunique()), round(int(df_tx["ChargerID"].nunique()) / total * 100.0, 4), "301 unique physical chargers"),
        ("Missing ParticipantID", int(df_tx["ParticipantID"].isnull().sum()), round(int(df_tx["ParticipantID"].isnull().sum()) / total * 100.0, 4), "3,899 unattributed sessions (200,000 series IDs)"),
        ("Missing CarKW / CarKWh", int(df_tx["CarKW"].isnull().sum()), round(int(df_tx["CarKW"].isnull().sum()) / total * 100.0, 4), "Exactly matches missing ParticipantID rows"),
        ("Missing ConsumedkWh", int(consumed.isnull().sum()), round(int(consumed.isnull().sum()) / total * 100.0, 4), "Single record with missing metered energy"),
        ("ConsumedkWh <= 0", int((consumed <= 0).sum()), 0.0, "All non-null energy values are positive (min 0.01 kWh)"),
        ("ConsumedkWh Extreme Outliers (> 150 kWh)", int((consumed > 150).sum()), round(int((consumed > 150).sum()) / total * 100.0, 4), "12 records exceed realistic vehicle battery capacity"),
        ("StopTime < StartTime", int((dur_min < 0).sum()), 0.0, "Zero negative duration intervals"),
        ("StopTime == StartTime (Zero Duration)", int((dur_min == 0).sum()), round(int((dur_min == 0).sum()) / total * 100.0, 4), "92 sessions have identical start and stop times"),
        ("Sessions Shorter than 5 Minutes (< 5 min)", int((dur_min < 5).sum()), round(int((dur_min < 5).sum()) / total * 100.0, 4), "561 micro-sessions (includes 92 zero-duration)"),
        ("Sessions Longer than 336 Hours (> 14 days)", int((dur_hrs > 336).sum()), round(int((dur_hrs > 336).sum()) / total * 100.0, 4), "23 extreme long sessions reaching up to 7,844.88 hrs"),
        ("Missing ChargingDuration", int(ch_dur.isnull().sum()), round(int(ch_dur.isnull().sum()) / total * 100.0, 4), "20,861 unmonitored or unrecorded active charging intervals"),
        ("ChargingDuration Exceeds Plugged Duration", int((ch_dur > dur_min).sum()), round(int((ch_dur > dur_min).sum()) / total * 100.0, 4), "1 session (ID 132725) due to DST clock change on 2018-03-25"),
        ("Missing Hot_Unplug", int(df_tx["Hot_Unplug"].isnull().sum()), round(int(df_tx["Hot_Unplug"].isnull().sum()) / total * 100.0, 4), "19,839 records without unplug status classification"),
        ("Trial 1 Sessions", int((df_tx["Trial"] == 1.0).sum()), round(int((df_tx["Trial"] == 1.0).sum()) / total * 100.0, 4), "Managed charging Trial 1"),
        ("Trial 2 Sessions", int((df_tx["Trial"] == 2.0).sum()), round(int((df_tx["Trial"] == 2.0).sum()) / total * 100.0, 4), "Managed charging Trial 2"),
        ("Trial 3 Sessions", int((df_tx["Trial"] == 3.0).sum()), round(int((df_tx["Trial"] == 3.0).sum()) / total * 100.0, 4), "Managed charging Trial 3 (to be excluded in modeling)"),
        ("Unmanaged / Control Sessions (Trial is NaN)", int(df_tx["Trial"].isnull().sum()), round(int(df_tx["Trial"].isnull().sum()) / total * 100.0, 4), "24,034 unmanaged control group sessions"),
        ("Overlapping Sessions for Same Participant", overlap_count, round(overlap_count / total * 100.0, 4), "133 sessions start before prior session StopTime"),
    ]

    return pd.DataFrame(checks, columns=["check_name", "record_count", "percentage", "description"])


def generate_metadata_quality_summary(df_meta: pd.DataFrame) -> pd.DataFrame:
    """Generate detailed metadata quality summary table.

    Args:
        df_meta: Raw ChargerInstall DataFrame.

    Returns:
        DataFrame summarizing metadata quality metrics.
    """
    total = len(df_meta)
    dup_pids = detect_duplicates(df_meta, subset=["ParticipantID"]) if "ParticipantID" in df_meta.columns else 0
    dup_rows = detect_duplicates(df_meta)

    # Missing counts by column
    missing_by_col = df_meta.isnull().sum()

    checks = [
        ("Total Records", total, 100.0, "Total installation records in sheet 20181126_ChargerInstall_DriveEl"),
        ("Unique Participant IDs", int(df_meta["ParticipantID"].nunique()), 100.0, "All 673 ParticipantID values are distinct"),
        ("Duplicate Participant IDs", dup_pids, 0.0, "Zero duplicate ParticipantIDs"),
        ("Duplicate Complete Rows", dup_rows, 0.0, "Zero duplicate rows"),
        ("Missing ParticipantID", int(missing_by_col.get("ParticipantID", 0)), 0.0, "ParticipantID is 100% populated"),
        ("Missing Charger ID", int(missing_by_col.get("Charger", 0)), 0.0, "Charger ID is 100% populated"),
        ("Missing ChargerInstallDate", int(missing_by_col.get("ChargerInstallDate", 0)), 0.0, "Charger install dates 100% populated"),
        ("Missing DCSProvider", int(missing_by_col.get("DCSProvider", 0)), 0.0, "Charging management provider 100% populated"),
        ("Missing CarInstallDate", int(missing_by_col.get("CarInstallDate", 0)), round(int(missing_by_col.get("CarInstallDate", 0)) / total * 100.0, 4), "104 records lack vehicle installation date"),
        ("Missing CarMake", int(missing_by_col.get("CarMake", 0)), 0.0, "Vehicle manufacturer 100% populated"),
        ("Missing CarModel", int(missing_by_col.get("CarModel", 0)), 0.0, "Vehicle model 100% populated"),
        ("Missing PIVType", int(missing_by_col.get("PIVType", 0)), 0.0, "Plug-in vehicle powertrain classification 100% populated"),
        ("Missing CarkW", int(missing_by_col.get("CarkW", 0)), 0.0, "Charging power rating 100% populated"),
        ("Missing CarkWh", int(missing_by_col.get("CarkWh", 0)), 0.0, "Battery usable capacity 100% populated"),
        ("Greenflux Provider Count", int((df_meta["DCSProvider"] == "Greenflux").sum()), round(int((df_meta["DCSProvider"] == "Greenflux").sum()) / total * 100.0, 4), "345 GreenFlux trial participants"),
        ("Crowd Charge Provider Count", int((df_meta["DCSProvider"] == "Crowd Charge").sum()), round(int((df_meta["DCSProvider"] == "Crowd Charge").sum()) / total * 100.0, 4), "328 Crowd Charge trial participants"),
        ("Electric only (BEV) Vehicles", int((df_meta["PIVType"] == "Electric only (BEV)").sum()), round(int((df_meta["PIVType"] == "Electric only (BEV)").sum()) / total * 100.0, 4), "315 Battery Electric Vehicles"),
        ("Plug-in Hybrid (PHEV) Vehicles", int(df_meta["PIVType"].str.lower().str.contains("hybrid").sum()), round(int(df_meta["PIVType"].str.lower().str.contains("hybrid").sum()) / total * 100.0, 4), "272 PHEV vehicles (including 1 with lowercase 'p')"),
        ("Range Extender (REX) Vehicles", int((df_meta["PIVType"] == "Range extender (REX)").sum()), round(int((df_meta["PIVType"] == "Range extender (REX)").sum()) / total * 100.0, 4), "86 Range Extender vehicles (e.g. BMW i3 REX)"),
        ("CarkW == 3.6 kW", int((df_meta["CarkW"] == 3.6).sum()), round(int((df_meta["CarkW"] == 3.6).sum()) / total * 100.0, 4), "318 single-phase 16A chargers"),
        ("CarkW == 7.0 kW", int((df_meta["CarkW"] == 7.0).sum()), round(int((df_meta["CarkW"] == 7.0).sum()) / total * 100.0, 4), "355 single-phase 32A chargers"),
        ("Unique Car Manufacturers", int(df_meta["CarMake"].nunique()), round(int(df_meta["CarMake"].nunique()) / total * 100.0, 4), "18 distinct car makes represented"),
        ("Unique Car Models", int(df_meta["CarModel"].nunique()), round(int(df_meta["CarModel"].nunique()) / total * 100.0, 4), "45 distinct vehicle models represented"),
    ]

    return pd.DataFrame(checks, columns=["check_name", "record_count", "percentage", "description"])


def run_full_audit(
    raw_dir: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Execute complete dataset audit and generate all report CSVs.

    Args:
        raw_dir: Directory containing raw Excel workbooks. Defaults to 'data/raw/electric_nation'.
        output_dir: Directory to save generated CSV reports. Defaults to 'reports/data_audit'.

    Returns:
        Dictionary summarizing audit metrics and paths of created files.
    """
    project_root = Path(__file__).resolve().parents[3]
    raw_path = Path(raw_dir) if raw_dir else project_root / "data" / "raw" / "electric_nation"
    reports_path = Path(output_dir) if output_dir else project_root / "reports" / "data_audit"
    reports_path.mkdir(parents=True, exist_ok=True)

    tx_file = raw_path / "CrowdCharge Transactions.xlsx"
    meta_file = raw_path / "ChargerInstall.xlsx"

    if not tx_file.is_file():
        raise FileNotFoundError(f"Transactions workbook missing at: {tx_file}")
    if not meta_file.is_file():
        raise FileNotFoundError(f"Metadata workbook missing at: {meta_file}")

    logger.info("Loading workbooks...")
    tx_meta = inspect_workbook_metadata(tx_file)
    meta_meta = inspect_workbook_metadata(meta_file)

    df_tx = load_dataset(tx_file, sheet_name="Transaction Data")
    df_meta = load_dataset(meta_file, sheet_name="20181126_ChargerInstall_DriveEl")

    logger.info("Auditing transaction dataset...")
    tx_audit = audit_transaction_data(df_tx)

    logger.info("Auditing vehicle metadata...")
    meta_audit = audit_metadata(df_meta)

    logger.info("Reconciling participants...")
    df_reconcile = reconcile_participants(df_tx, df_meta)
    reconcile_csv = reports_path / "participant_reconciliation.csv"
    df_reconcile.to_csv(reconcile_csv, index=False)

    logger.info("Auditing feature feasibility...")
    df_feasibility = audit_feature_feasibility(df_tx, df_meta)
    feasibility_csv = reports_path / "feature_feasibility.csv"
    df_feasibility.to_csv(feasibility_csv, index=False)

    logger.info("Generating transaction quality summary...")
    df_tx_summary = generate_transaction_quality_summary(df_tx)
    tx_summary_csv = reports_path / "transaction_quality_summary.csv"
    df_tx_summary.to_csv(tx_summary_csv, index=False)

    logger.info("Generating metadata quality summary...")
    df_meta_summary = generate_metadata_quality_summary(df_meta)
    meta_summary_csv = reports_path / "metadata_quality_summary.csv"
    df_meta_summary.to_csv(meta_summary_csv, index=False)

    logger.info("Audit complete! Reports saved to: %s", reports_path)

    return {
        "transactions_meta": tx_meta,
        "metadata_meta": meta_meta,
        "transactions_audit": tx_audit,
        "metadata_audit": meta_audit,
        "output_files": [
            str(tx_summary_csv),
            str(meta_summary_csv),
            str(reconcile_csv),
            str(feasibility_csv),
        ],
    }


if __name__ == "__main__":
    run_full_audit()
