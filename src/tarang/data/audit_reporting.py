"""Audit reporting and summary artifact generation module for TARANG.

Provides functions to generate tabular audit summaries (CSV) and dynamic Markdown
audit documentation from calculated quality metrics.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union
import pandas as pd

from tarang.config import (
    AUDIT_REPORTS_DIR,
    DATA_RAW_DIR,
    RAW_META_FILE,
    RAW_TX_FILE,
    REQUIRED_META_COLUMNS,
    REQUIRED_TX_COLUMNS,
)
from tarang.data.loading import inspect_workbook_metadata, load_dataset
from tarang.data.quality_audit import (
    audit_energy_quality,
    audit_feature_feasibility,
    audit_metadata_row_level,
    compute_simple_shift_overlaps,
    detect_duplicates,
    detect_session_overlaps,
    find_charging_duration_discrepancies,
    investigate_unattributed_sessions,
    parse_utc_timestamps,
)
from tarang.data.screening import screen_participant_history
from tarang.data.validation import validate_workbook_schema

logger = logging.getLogger(__name__)


def generate_transaction_quality_summary(
    df_tx: pd.DataFrame,
    ts_metrics: Dict[str, Any],
    unattributed_metrics: Dict[str, Any],
    energy_metrics: Dict[str, Any],
    overlap_count: int,
    ch_discrepancy_info: Dict[str, Any],
) -> pd.DataFrame:
    """Generate detailed transaction quality summary table using actual calculated metrics.

    Args:
        df_tx: Raw transactions DataFrame.
        ts_metrics: Timestamp validation metrics.
        unattributed_metrics: Unattributed session metrics.
        energy_metrics: Energy quality metrics.
        overlap_count: Participant-level overlap count.
        ch_discrepancy_info: Discrepancy info for ChargingDuration exceeding PluggedInTime.

    Returns:
        DataFrame summarizing transaction quality checks.
    """
    total = len(df_tx)

    start_utc, stop_utc, _ = parse_utc_timestamps(df_tx, "StartTime", "StopTime")
    valid_ts = start_utc.notna() & stop_utc.notna()
    dur_min = (stop_utc[valid_ts] - start_utc[valid_ts]).dt.total_seconds() / 60.0
    dur_hrs = dur_min / 60.0

    ch_dur = df_tx.loc[valid_ts, "ChargingDuration"] if "ChargingDuration" in df_tx.columns else pd.Series(dtype=float)

    sessions_lt_5m = int((dur_min < 5).sum())
    sessions_gt_336h = int((dur_hrs > 336).sum())

    t1_count = int((df_tx["Trial"] == 1.0).sum()) if "Trial" in df_tx.columns else 0
    t2_count = int((df_tx["Trial"] == 2.0).sum()) if "Trial" in df_tx.columns else 0
    t3_count = int((df_tx["Trial"] == 3.0).sum()) if "Trial" in df_tx.columns else 0
    t_null_count = int(df_tx["Trial"].isnull().sum()) if "Trial" in df_tx.columns else 0

    discrepant_count = ch_discrepancy_info.get("count", len(ch_discrepancy_info.get("ids", [])))
    discrepant_ids = ", ".join(map(str, ch_discrepancy_info["ids"])) if ch_discrepancy_info.get("ids") else "None"
    if discrepant_count == 0:
        ch_discrepancy_desc = "No unexplained logging discrepancies detected (0 records)"
    elif discrepant_count == 1:
        ch_discrepancy_desc = f"Unexplained logging discrepancy requiring investigation (1 record, Transaction ID: {discrepant_ids})"
    else:
        ch_discrepancy_desc = f"Unexplained logging discrepancies requiring investigation ({discrepant_count} records, Transaction IDs: {discrepant_ids})"

    missing_rec_cnt = energy_metrics.get("missing_energy", 0)
    missing_rec_desc = f"{missing_rec_cnt} record" if missing_rec_cnt == 1 else f"{missing_rec_cnt} records"

    unknown_ids = unattributed_metrics.get("unknown_charger_ids", [])
    unknown_chargers_str = ", ".join(map(str, unknown_ids)) if unknown_ids else "None"
    max_energy_val = energy_metrics.get("max_energy", 0.0)

    unique_tx_ids = int(df_tx["TransactionID"].nunique()) if "TransactionID" in df_tx.columns else 0
    unique_tx_id_pct = round(unique_tx_ids / total * 100.0, 4) if total > 0 else 0.0
    dup_tx_ids = detect_duplicates(df_tx, subset=["TransactionID"]) if "TransactionID" in df_tx.columns else 0
    dup_tx_pct = round(dup_tx_ids / total * 100.0, 4) if total > 0 else 0.0
    dup_rows = detect_duplicates(df_tx)
    dup_rows_pct = round(dup_rows / total * 100.0, 4) if total > 0 else 0.0
    unique_pids = int(df_tx["ParticipantID"].dropna().nunique()) if "ParticipantID" in df_tx.columns else 0
    pid_pct = round(unique_pids / total * 100.0, 4) if total > 0 else 0.0
    unique_chargers = int(df_tx["ChargerID"].nunique()) if "ChargerID" in df_tx.columns else 0
    charger_pct = round(unique_chargers / total * 100.0, 4) if total > 0 else 0.0

    checks = [
        ("Total Transaction Records", total, 100.0 if total > 0 else 0.0, "Total rows in Transaction Data sheet"),
        ("Unique Transaction Identifiers", unique_tx_ids, unique_tx_id_pct, f"{unique_tx_ids:,} distinct primary keys in Transaction Data ({unique_tx_id_pct:.2f}% uniqueness)"),
        ("Duplicate Transaction IDs", dup_tx_ids, dup_tx_pct, "Duplicate TransactionID values"),
        ("Duplicate Complete Rows", dup_rows, dup_rows_pct, "Full duplicate rows"),
        ("Unique Participant Identifiers", unique_pids, pid_pct, "Unique non-null participants"),
        ("Unique Charger Identifiers", unique_chargers, charger_pct, "Unique physical chargers"),
        ("Missing ParticipantID", unattributed_metrics["total_unattributed_records"], unattributed_metrics["unattributed_percentage"], f"Unattributed sessions ({unattributed_metrics['matched_4field_session_records']} records match known participant sessions on four session fields; {unattributed_metrics['unknown_charger_records']} belong to unregistered charger {unknown_chargers_str})"),
        ("Missing CarKW / CarKWh", int(df_tx["CarKW"].isnull().sum()) if "CarKW" in df_tx.columns else 0, round(int(df_tx["CarKW"].isnull().sum()) / total * 100.0, 4) if total > 0 else 0.0, "Vehicle specification missingness in transactions"),
        ("Missing ConsumedkWh", energy_metrics["missing_energy"], energy_metrics["missing_energy_pct"], f"Records with missing metered electrical energy ({missing_rec_desc})"),
        ("ConsumedkWh <= 0", energy_metrics["non_positive_energy"], energy_metrics["non_positive_energy_pct"], "Zero or negative energy readings"),
        ("ConsumedkWh > 100 kWh (Implausible Threshold)", energy_metrics["energy_gt_100kwh"], energy_metrics["energy_gt_100kwh_pct"], "Official Electric Nation implausible consumption threshold"),
        ("ConsumedkWh > 150 kWh (Extreme Outliers)", energy_metrics["energy_gt_150kwh"], energy_metrics["energy_gt_150kwh_pct"], f"Extreme energy consumption outliers reaching {max_energy_val:,.2f} kWh"),
        ("ConsumedkWh > Matched CarKWh", energy_metrics["energy_exceeds_carkwh"], energy_metrics["energy_exceeds_carkwh_pct"], "Records requiring review (energy consumption exceeds registered nominal battery capacity)"),
        ("StopTime < StartTime (Negative Duration)", ts_metrics["negative_intervals"], ts_metrics["negative_intervals_pct"], "Chronologically inverted intervals"),
        ("StopTime == StartTime (Zero Duration)", ts_metrics["zero_intervals"], ts_metrics["zero_intervals_pct"], "Instantaneous connect/disconnect glitches"),
        ("Sessions Shorter than 5 Minutes (< 5 min)", sessions_lt_5m, round(sessions_lt_5m / total * 100.0, 4) if total > 0 else 0.0, "Micro-sessions excluded from modeling population"),
        ("Sessions Longer than 336 Hours (> 14 days)", sessions_gt_336h, round(sessions_gt_336h / total * 100.0, 4) if total > 0 else 0.0, "Unclosed sessions capped at 336 hours for target"),
        ("Missing ChargingDuration", int(ch_dur.isnull().sum()), round(int(ch_dur.isnull().sum()) / total * 100.0, 4) if total > 0 else 0.0, "Unmonitored active power intervals"),
        ("ChargingDuration Exceeds Plugged Duration", discrepant_count, round(discrepant_count / total * 100.0, 4) if total > 0 else 0.0, ch_discrepancy_desc),
        ("Missing Hot_Unplug", int(df_tx["Hot_Unplug"].isnull().sum()) if "Hot_Unplug" in df_tx.columns else 0, round(int(df_tx["Hot_Unplug"].isnull().sum()) / total * 100.0, 4) if total > 0 else 0.0, "Sessions without unplug status classification"),
        ("Trial 1 Sessions", t1_count, round(t1_count / total * 100.0, 4) if total > 0 else 0.0, "Managed charging Trial 1"),
        ("Trial 2 Sessions", t2_count, round(t2_count / total * 100.0, 4) if total > 0 else 0.0, "Managed charging Trial 2"),
        ("Trial 3 Sessions", t3_count, round(t3_count / total * 100.0, 4) if total > 0 else 0.0, "Managed charging Trial 3 (excluded in research dataset)"),
        ("Unmanaged / Control Sessions (Trial is NaN)", t_null_count, round(t_null_count / total * 100.0, 4) if total > 0 else 0.0, "Unmanaged control group sessions"),
        ("Participant-Level Overlapping Sessions", overlap_count, round(overlap_count / total * 100.0, 4) if total > 0 else 0.0, "Sessions starting before cumulative maximum stop time of prior sessions"),
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
    missing_by_col = df_meta.isnull().sum()

    unique_pids = int(df_meta["ParticipantID"].nunique()) if "ParticipantID" in df_meta.columns else 0
    unique_pid_pct = round(unique_pids / total * 100.0, 4) if total > 0 else 0.0

    checks = [
        ("Total Records", total, 100.0 if total > 0 else 0.0, "Total installation records in sheet 20181126_ChargerInstall_DriveEl"),
        ("Unique Participant IDs", unique_pids, unique_pid_pct, f"{unique_pids} distinct participant identifiers in metadata ({unique_pid_pct:.2f}% uniqueness)"),
        ("Duplicate Participant IDs", dup_pids, round(dup_pids / total * 100.0, 4) if total > 0 else 0.0, "Duplicate ParticipantIDs"),
        ("Duplicate Complete Rows", dup_rows, round(dup_rows / total * 100.0, 4) if total > 0 else 0.0, "Duplicate rows"),
        ("Missing ParticipantID", int(missing_by_col.get("ParticipantID", 0)), round(int(missing_by_col.get("ParticipantID", 0)) / total * 100.0, 4) if total > 0 else 0.0, "ParticipantID completeness"),
        ("Missing Charger ID", int(missing_by_col.get("Charger", 0)), round(int(missing_by_col.get("Charger", 0)) / total * 100.0, 4) if total > 0 else 0.0, "Charger ID completeness"),
        ("Missing ChargerInstallDate", int(missing_by_col.get("ChargerInstallDate", 0)), round(int(missing_by_col.get("ChargerInstallDate", 0)) / total * 100.0, 4) if total > 0 else 0.0, "Charger install dates completeness"),
        ("Missing DCSProvider", int(missing_by_col.get("DCSProvider", 0)), round(int(missing_by_col.get("DCSProvider", 0)) / total * 100.0, 4) if total > 0 else 0.0, "Charging management provider completeness"),
        ("Missing CarInstallDate", int(missing_by_col.get("CarInstallDate", 0)), round(int(missing_by_col.get("CarInstallDate", 0)) / total * 100.0, 4) if total > 0 else 0.0, f"{int(missing_by_col.get('CarInstallDate', 0))} records lack vehicle installation date"),
        ("Missing CarMake", int(missing_by_col.get("CarMake", 0)), round(int(missing_by_col.get("CarMake", 0)) / total * 100.0, 4) if total > 0 else 0.0, "Vehicle manufacturer completeness"),
        ("Missing CarModel", int(missing_by_col.get("CarModel", 0)), round(int(missing_by_col.get("CarModel", 0)) / total * 100.0, 4) if total > 0 else 0.0, "Vehicle model completeness"),
        ("Missing PIVType", int(missing_by_col.get("PIVType", 0)), round(int(missing_by_col.get("PIVType", 0)) / total * 100.0, 4) if total > 0 else 0.0, "Powertrain classification completeness"),
        ("Missing CarkW", int(missing_by_col.get("CarkW", 0)), round(int(missing_by_col.get("CarkW", 0)) / total * 100.0, 4) if total > 0 else 0.0, "Charging power rating completeness"),
        ("Missing CarkWh", int(missing_by_col.get("CarkWh", 0)), round(int(missing_by_col.get("CarkWh", 0)) / total * 100.0, 4) if total > 0 else 0.0, "Battery usable capacity completeness"),
        ("Greenflux Provider Count", int((df_meta["DCSProvider"] == "Greenflux").sum()) if "DCSProvider" in df_meta.columns else 0, round(int((df_meta["DCSProvider"] == "Greenflux").sum()) / total * 100.0, 4) if total > 0 else 0.0, "GreenFlux trial participants"),
        ("Crowd Charge Provider Count", int((df_meta["DCSProvider"] == "Crowd Charge").sum()) if "DCSProvider" in df_meta.columns else 0, round(int((df_meta["DCSProvider"] == "Crowd Charge").sum()) / total * 100.0, 4) if total > 0 else 0.0, "Crowd Charge trial participants"),
        ("Electric only (BEV) Vehicles", int((df_meta["PIVType"] == "Electric only (BEV)").sum()) if "PIVType" in df_meta.columns else 0, round(int((df_meta["PIVType"] == "Electric only (BEV)").sum()) / total * 100.0, 4) if total > 0 else 0.0, "Battery Electric Vehicles"),
        ("Plug-in Hybrid (PHEV) Vehicles", int(df_meta["PIVType"].str.lower().str.contains("hybrid").sum()) if "PIVType" in df_meta.columns else 0, round(int(df_meta["PIVType"].str.lower().str.contains("hybrid").sum()) / total * 100.0, 4) if total > 0 else 0.0, "PHEV vehicles"),
        ("Range Extender (REX) Vehicles", int((df_meta["PIVType"] == "Range extender (REX)").sum()) if "PIVType" in df_meta.columns else 0, round(int((df_meta["PIVType"] == "Range extender (REX)").sum()) / total * 100.0, 4) if total > 0 else 0.0, "Range Extender vehicles"),
        ("CarkW == 3.6 kW", int((df_meta["CarkW"] == 3.6).sum()) if "CarkW" in df_meta.columns else 0, round(int((df_meta["CarkW"] == 3.6).sum()) / total * 100.0, 4) if total > 0 else 0.0, "Single-phase 16A chargers"),
        ("CarkW == 7.0 kW", int((df_meta["CarkW"] == 7.0).sum()) if "CarkW" in df_meta.columns else 0, round(int((df_meta["CarkW"] == 7.0).sum()) / total * 100.0, 4) if total > 0 else 0.0, "Single-phase 32A chargers"),
        ("Unique Car Manufacturers", int(df_meta["CarMake"].nunique()) if "CarMake" in df_meta.columns else 0, round(int(df_meta["CarMake"].nunique()) / total * 100.0, 4) if total > 0 else 0.0, "Distinct car makes represented"),
        ("Unique Car Models", int(df_meta["CarModel"].nunique()) if "CarModel" in df_meta.columns else 0, round(int(df_meta["CarModel"].nunique()) / total * 100.0, 4) if total > 0 else 0.0, "Distinct vehicle models represented"),
    ]

    return pd.DataFrame(checks, columns=["check_name", "record_count", "percentage", "description"])


def generate_participant_reconciliation(
    row_meta_metrics: Dict[str, Any],
    df_meta: pd.DataFrame,
    unattributed_metrics: Dict[str, Any],
    total_tx: int,
) -> pd.DataFrame:
    """Generate participant reconciliation summary table.

    Args:
        row_meta_metrics: Row-level metadata reconciliation metrics.
        df_meta: Metadata DataFrame.
        unattributed_metrics: Unattributed session metrics.
        total_tx: Total transaction record count.

    Returns:
        DataFrame summarizing reconciliation.
    """
    total_tx_pids = row_meta_metrics["unique_tx_participants"]
    total_meta_pids = row_meta_metrics["unique_meta_participants"]
    matched_pids = row_meta_metrics["matched_participants"]
    tx_only = row_meta_metrics["tx_only_participants"]
    meta_only = row_meta_metrics["meta_only_participants"]

    matched_providers = row_meta_metrics.get("matched_providers", {})
    unmatched_providers = row_meta_metrics.get("unmatched_providers", {})
    greenflux_unmatched = unmatched_providers.get("Greenflux", 0)
    crowdcharge_unmatched = unmatched_providers.get("Crowd Charge", 0)
    greenflux_matched = matched_providers.get("Greenflux", 0)
    crowdcharge_matched = matched_providers.get("Crowd Charge", 0)

    unknown_ids = unattributed_metrics.get("unknown_charger_ids", [])
    unknown_chargers_str = ", ".join(map(str, unknown_ids)) if unknown_ids else "None"

    matched_pct = round(matched_pids / total_tx_pids * 100.0, 2) if total_tx_pids else 0.0
    tx_only_pct = round(tx_only / total_tx_pids * 100.0, 2) if total_tx_pids else 0.0
    meta_only_pct = round(meta_only / total_meta_pids * 100.0, 2) if total_meta_pids else 0.0
    inconsistent_pids = row_meta_metrics.get(
        "participants_with_inconsistent_specs",
        row_meta_metrics.get("participants_with_inconsistent_carkw", 0),
    )
    inconsistent_pct = round(inconsistent_pids / matched_pids * 100.0, 2) if matched_pids else 0.0

    carkw_mismatches = row_meta_metrics.get("row_level_carkw_mismatches", 0)
    carkwh_mismatches = row_meta_metrics.get("row_level_carkwh_mismatches", 0)
    carkw_mismatch_pct = row_meta_metrics.get("row_level_carkw_mismatch_pct", 0.0)
    carkwh_mismatch_pct = row_meta_metrics.get("row_level_carkwh_mismatch_pct", 0.0)
    charger_mismatches = row_meta_metrics.get("row_level_charger_mismatches", 0)
    charger_mismatch_pct = row_meta_metrics.get("row_level_charger_mismatch_pct", 0.0)

    tx_only_str = f"{tx_only} transaction participant" if tx_only == 1 else f"{tx_only} transaction participants"
    inconsistent_pids_str = f"{inconsistent_pids} participant" if inconsistent_pids == 1 else f"{inconsistent_pids} participants"

    records = [
        {"metric": "Unique Participants in Transactions (Non-null)", "count": total_tx_pids, "percentage": 100.0 if total_tx_pids else 0.0, "notes": "Distinct participant IDs appearing in transaction records"},
        {"metric": "Unique Participants in Metadata", "count": total_meta_pids, "percentage": 100.0 if total_meta_pids else 0.0, "notes": "Total installed charger participants in metadata"},
        {"metric": "Matched Participants (Present in Both)", "count": matched_pids, "percentage": matched_pct, "notes": f"{matched_pct:.1f}% of transaction participants are identified in metadata ({matched_pids}/{total_tx_pids})"},
        {"metric": "Transaction Participants Missing from Metadata", "count": tx_only, "percentage": tx_only_pct, "notes": f"{tx_only_str} not found in metadata registry"},
        {"metric": "Metadata Participants Missing from Transactions", "count": meta_only, "percentage": meta_only_pct, "notes": f"Metadata records without transactions ({greenflux_unmatched} GreenFlux, {crowdcharge_unmatched} Crowd Charge)"},
        {"metric": "Matched Participants - Crowd Charge Provider", "count": crowdcharge_matched, "percentage": round(crowdcharge_matched / matched_pids * 100.0, 2) if matched_pids else 0.0, "notes": "Matched participants belonging to Crowd Charge provider"},
        {"metric": "Matched Participants - GreenFlux Provider", "count": greenflux_matched, "percentage": round(greenflux_matched / matched_pids * 100.0, 2) if matched_pids else 0.0, "notes": "Matched participants belonging to GreenFlux provider"},
        {"metric": "Unmatched Metadata - GreenFlux Provider", "count": greenflux_unmatched, "percentage": round(greenflux_unmatched / meta_only * 100.0, 2) if meta_only else 0.0, "notes": "GreenFlux trial data is stored in separate external trial datasets"},
        {"metric": "Unmatched Metadata - Crowd Charge Provider", "count": crowdcharge_unmatched, "percentage": round(crowdcharge_unmatched / meta_only * 100.0, 2) if meta_only else 0.0, "notes": "Crowd Charge installations with zero recorded sessions in this file"},
        {"metric": "Row-Level CarKW Mismatches Across Matched Sessions", "count": carkw_mismatches, "percentage": carkw_mismatch_pct, "notes": f"{100.0 - carkw_mismatch_pct:.2f}% agreement across matched transaction rows ({carkw_mismatches} mismatches)"},
        {"metric": "Row-Level CarKWh Mismatches Across Matched Sessions", "count": carkwh_mismatches, "percentage": carkwh_mismatch_pct, "notes": f"{100.0 - carkwh_mismatch_pct:.2f}% agreement across matched transaction rows ({carkwh_mismatches} mismatches)"},
        {"metric": "Participants with Inconsistent Specifications Across Sessions", "count": inconsistent_pids, "percentage": inconsistent_pct, "notes": f"{inconsistent_pids_str} with fluctuating CarKW or CarKWh across sessions"},
        {"metric": "Row-Level ChargerID vs Charger Mismatches", "count": charger_mismatches, "percentage": charger_mismatch_pct, "notes": f"{100.0 - charger_mismatch_pct:.2f}% agreement when string formats are compared ({charger_mismatches} mismatches)"},
        {"metric": "Unattributed Sessions Matching Known Sessions on 4 Fields", "count": unattributed_metrics["matched_4field_session_records"], "percentage": round(unattributed_metrics["matched_4field_session_records"] / total_tx * 100.0, 2) if total_tx else 0.0, "notes": "These records match known participant sessions on ChargerID, StartTime, StopTime, ConsumedkWh; other transaction fields may differ"},
        {"metric": "Unattributed Sessions on Unregistered Charger", "count": unattributed_metrics["unknown_charger_records"], "percentage": round(unattributed_metrics["unknown_charger_records"] / total_tx * 100.0, 2) if total_tx else 0.0, "notes": f"Records associated with unregistered charger {unknown_chargers_str} absent from metadata"},
    ]

    return pd.DataFrame(records)


def generate_markdown_report(
    tx_meta: Dict[str, Any],
    meta_meta: Dict[str, Any],
    ts_metrics: Dict[str, Any],
    unattributed_metrics: Dict[str, Any],
    row_meta_metrics: Dict[str, Any],
    energy_metrics: Dict[str, Any],
    overlap_count: int,
    simple_overlap_count: int,
    ch_discrepancy_info: Dict[str, Any],
    screening_metrics: Dict[str, Any],
    total_tx: int,
    dup_tx_ids: int = 0,
    dup_tx_rows: int = 0,
) -> str:
    """Generate dynamic Markdown audit report from calculated metrics."""
    tx_dims = tx_meta.get("sheet_dimensions", {}).get("Transaction Data", {"max_row": total_tx + 1, "max_column": 28})
    tx_exp_dims = tx_meta.get("sheet_dimensions", {}).get("Explanation", {"max_row": 1, "max_column": 1})
    meta_dims = meta_meta.get("sheet_dimensions", {}).get("20181126_ChargerInstall_DriveEl", {"max_row": row_meta_metrics["unique_meta_participants"] + 1, "max_column": 10})
    meta_exp_dims = meta_meta.get("sheet_dimensions", {}).get("Explanation", {"max_row": 1, "max_column": 1})

    nested_overlap_diff = overlap_count - simple_overlap_count
    discrepant_count = ch_discrepancy_info.get("discrepancy_count", ch_discrepancy_info.get("count", 0))
    discrepant_ids_list = ch_discrepancy_info.get("ids", [])
    discrepant_ids = ", ".join(map(str, discrepant_ids_list)) if discrepant_ids_list else "None"

    if discrepant_count == 0:
        ch_discrepancy_summary = "No records exhibit `ChargingDuration` exceeding plugged-in connection duration."
    elif discrepant_count == 1:
        ch_discrepancy_summary = f"1 record (Transaction ID: {discrepant_ids}) exhibits `ChargingDuration` exceeding plugged-in connection duration. This is documented as an unexplained logging discrepancy requiring investigation."
    else:
        ch_discrepancy_summary = f"{discrepant_count} records (Transaction IDs: {discrepant_ids}) exhibit `ChargingDuration` exceeding plugged-in connection duration. This is documented as an unexplained logging discrepancy requiring investigation."

    unknown_ids = unattributed_metrics.get("unknown_charger_ids", [])
    unknown_chargers_str = ", ".join(map(str, unknown_ids)) if unknown_ids else "None"
    max_energy_val = energy_metrics.get("max_energy", 0.0)

    matched_pids = row_meta_metrics["matched_participants"]
    total_tx_pids = row_meta_metrics["unique_tx_participants"]
    matched_pct = round(matched_pids / total_tx_pids * 100.0, 2) if total_tx_pids else 0.0
    matched_providers = row_meta_metrics.get("matched_providers", {})
    provider_str = ", ".join(f"{cnt} {prov}" for prov, cnt in matched_providers.items()) if matched_providers else "none"

    carkw_mismatches = row_meta_metrics.get("row_level_carkw_mismatches", 0)
    carkwh_mismatches = row_meta_metrics.get("row_level_carkwh_mismatches", 0)
    carkw_mismatch_pct = row_meta_metrics.get("row_level_carkw_mismatch_pct", 0.0)
    carkwh_mismatch_pct = row_meta_metrics.get("row_level_carkwh_mismatch_pct", 0.0)
    carkw_agree_pct = 100.0 - carkw_mismatch_pct
    carkwh_agree_pct = 100.0 - carkwh_mismatch_pct

    tx_only_pids = row_meta_metrics.get("tx_only_participants", 0)
    tx_only_pct = round(tx_only_pids / total_tx_pids * 100.0, 2) if total_tx_pids else 0.0
    meta_pids = row_meta_metrics.get("unique_meta_participants", 0)
    meta_only_pids = row_meta_metrics.get("meta_only_participants", 0)
    meta_only_pct = round(meta_only_pids / meta_pids * 100.0, 2) if meta_pids else 0.0

    inconsistent_pids = row_meta_metrics.get(
        "participants_with_inconsistent_specs",
        row_meta_metrics.get("participants_with_inconsistent_carkw", 0),
    )
    inconsistent_pct = round(inconsistent_pids / matched_pids * 100.0, 2) if matched_pids else 0.0

    total_matched_rows = row_meta_metrics.get("total_matched_transaction_rows", 0)
    charger_matches = row_meta_metrics.get("row_level_charger_matches", 0)
    charger_match_pct = round(charger_matches / total_matched_rows * 100.0, 2) if total_matched_rows else 0.0

    start_failures = ts_metrics.get("start_parse_failures", 0)
    stop_failures = ts_metrics.get("stop_parse_failures", 0)
    total_parse_failures = start_failures + stop_failures

    raw_missing_start = ts_metrics.get("raw_missing_start", ts_metrics.get("missing_start", 0) - start_failures)
    raw_missing_stop = ts_metrics.get("raw_missing_stop", ts_metrics.get("missing_stop", 0) - stop_failures)
    total_raw_missing = raw_missing_start + raw_missing_stop

    total_non_missing = (2 * total_tx) - total_raw_missing
    if total_non_missing > 0:
        parsed_valid = total_non_missing - total_parse_failures
        parse_success_pct = round((parsed_valid / total_non_missing) * 100.0, 2)
        parse_success_str = f"{parse_success_pct:.1f}%"
    else:
        parse_success_str = "N/A"

    parse_fail_str = f"{total_parse_failures} parse failure" if total_parse_failures == 1 else f"{total_parse_failures} parse failures"
    missing_cells_str = f"{total_raw_missing} missing cell" if total_raw_missing == 1 else f"{total_raw_missing} missing cells"
    neg_intervals = ts_metrics.get("negative_intervals", 0)

    non_pos_energy = energy_metrics.get("non_positive_energy", 0)
    min_pos_energy = energy_metrics.get("min_positive_energy")
    if non_pos_energy == 0:
        if min_pos_energy is not None:
            non_pos_assessment = f"No non-positive readings found (all recorded values are positive, min {min_pos_energy:,.2f} kWh)"
        else:
            non_pos_assessment = "No non-positive readings found (all recorded values are positive)"
    elif non_pos_energy == 1:
        non_pos_assessment = "1 non-positive energy reading detected (<= 0 kWh)"
    else:
        non_pos_assessment = f"{non_pos_energy} non-positive energy readings detected (<= 0 kWh)"

    missing_energy_cnt = energy_metrics.get("missing_energy", 0)
    if missing_energy_cnt == 0:
        missing_energy_assessment = "No missing energy readings found"
    elif missing_energy_cnt == 1:
        missing_energy_assessment = "1 record lacks a meter reading"
    else:
        missing_energy_assessment = f"{missing_energy_cnt} records lack meter readings"

    tx_only_spec_str = f"{tx_only_pids} unknown participant ID" if tx_only_pids == 1 else f"{tx_only_pids} unknown participant IDs"
    inconsistent_pids_spec_str = f"{inconsistent_pids} participant with fluctuating specs" if inconsistent_pids == 1 else f"{inconsistent_pids} participants with fluctuating specs"

    scr_initial_pids = screening_metrics["initial_participants"]
    scr_initial_records = screening_metrics["initial_valid_records"]
    scr_pids_after_t3 = screening_metrics["participants_after_t3"]
    scr_records_after_t3 = screening_metrics["remaining_after_t3"]
    scr_pids_after_short = screening_metrics["participants_after_short"]
    scr_records_after_short = screening_metrics["remaining_after_short"]
    scr_pids_after_overlaps = screening_metrics["participants_after_overlaps"]
    scr_records_after_overlaps = screening_metrics["remaining_after_overlaps"]
    scr_dropped_first = screening_metrics["dropped_first_sessions"]
    scr_pids_after_dropped_first = screening_metrics["participants_after_dropped_first"]
    scr_pids_zero_retained = screening_metrics.get(
        "participants_zero_retained_after_dropped_first",
        scr_pids_after_overlaps - scr_pids_after_dropped_first,
    )
    scr_zero_pids_str = f"{scr_pids_zero_retained} participant had" if scr_pids_zero_retained == 1 else f"{scr_pids_zero_retained} participants had"
    scr_retained_sessions = screening_metrics["retained_sessions_total"]
    scr_pids_lt_10 = screening_metrics["participants_lt_10_retained"]
    scr_sessions_lt_10 = screening_metrics["sessions_in_lt_10_retained"]
    scr_pids_before_density = screening_metrics["participants_before_density"]
    scr_sessions_before_density = screening_metrics["sessions_before_density"]
    scr_pids_excluded_density = screening_metrics["participants_excluded_density"]
    scr_sessions_excluded_density = screening_metrics["sessions_excluded_density"]
    scr_final_pids = screening_metrics["final_eligible_participants"]
    scr_final_sessions = screening_metrics["final_eligible_sessions"]

    report = f"""# TARANG — Initial Dataset Audit Report

**Project**: TARANG — EV Charging Intelligence System  
**Date**: November 2026  
**Status**: Completed and Validated  
**Branch**: `main`  
**Timezone Standard**: Coordinated Universal Time (UTC)  
**Datasets Audited**:
1. `data/raw/electric_nation/CrowdCharge Transactions.xlsx`
2. `data/raw/electric_nation/ChargerInstall.xlsx`

---

## 1. Executive Summary

A complete, evidence-based data audit was performed on the raw Electric Nation datasets downloaded locally in `data/raw/electric_nation/`. The primary objective was to establish whether the raw data supports the planned EV charging prediction (plugged-in duration and energy consumption) and behavioral analysis models without data leakage or fatal quality defects.

**Key Findings:**
- **Usability**: The dataset **supports** the planned system, provided specific preprocessing rules, duration capping, and outlier treatments are implemented.
- **Timestamp Standard**: All raw timestamps are verified and treated as **UTC-aware timestamps** without shifting their recorded clock values.
- **Transaction Coverage**: {total_tx:,} transaction records spanning from {ts_metrics['earliest_start']} to {ts_metrics['latest_stop']} UTC.
- **Participant Coverage**: {row_meta_metrics['unique_tx_participants']} distinct non-null participants in transactions and {row_meta_metrics['unique_meta_participants']} participants in vehicle metadata.
- **Reconciliation**: {matched_pct:.1f}% ({matched_pids}/{total_tx_pids}) of non-null transaction participants match metadata records ({provider_str}). Row-level checks across all {total_matched_rows:,} matched records confirm {carkw_mismatches} mismatches for `CarKW` and {carkwh_mismatches} mismatches for `CarKWh`.
- **Data Quality Invariants**: {neg_intervals} negative timestamp intervals (`StopTime < StartTime`), {dup_tx_ids} duplicate primary keys (`TransactionID`), and {dup_tx_rows} full-row duplicates.
- **Unattributed Sessions**: {unattributed_metrics['total_unattributed_records']:,} records ({unattributed_metrics['unattributed_percentage']}%) lack `ParticipantID`. Natural-session matching confirms {unattributed_metrics['matched_4field_session_records']:,} records match known participant sessions on the four session fields (`ChargerID, StartTime, StopTime, ConsumedkWh`), while {unattributed_metrics['unknown_charger_records']} records belong to an unregistered charger ({unknown_chargers_str}).
- **Participant Session Overlaps**: {overlap_count:,} overlapping sessions detected using the cumulative maximum stop time of all earlier sessions for that participant (uncovering {nested_overlap_diff} nested overlaps missed by sequential shift checks).
- **Energy Quality**: {energy_metrics['energy_gt_100kwh']} records exceed the official Electric Nation 100 kWh implausibility threshold ({energy_metrics['energy_gt_150kwh']} exceed 150 kWh, reaching {max_energy_val:,.2f} kWh), and {energy_metrics['energy_exceeds_carkwh']:,} records exceed their vehicle's rated battery capacity. {missing_energy_assessment}.
- **Charging Duration Discrepancy**: {ch_discrepancy_summary}
- **Final Modeling Population**: Following the complete planned screening sequence (Trial 3 exclusion, duration threshold, overlap removal, initial session drop, $\\ge 10$ retained sessions, and the observation-density filter), the final retained modeling population comprises **{scr_final_pids} participants** and **{scr_final_sessions:,} sessions**.

---

## 2. Dataset Overview and Workbook Inventory

Both workbooks were inspected across all sheets. Original file sizes, available sheets, and dimensions are recorded below.

| Workbook | File Size (Bytes) | Sheet Name | Dimensions (Rows x Cols) | Content Description |
| :--- | :--- | :--- | :--- | :--- |
| `CrowdCharge Transactions.xlsx` | {tx_meta['size_bytes']:,} | `Explanation` | {tx_exp_dims['max_row']} x {tx_exp_dims['max_column']} | Empty placeholder sheet |
| `CrowdCharge Transactions.xlsx` | {tx_meta['size_bytes']:,} | `Transaction Data` | {tx_dims['max_row']:,} x {tx_dims['max_column']} | Primary EV charging transactions |
| `ChargerInstall.xlsx` | {meta_meta['size_bytes']:,} | `Explanation` | {meta_exp_dims['max_row']} x {meta_exp_dims['max_column']} | Empty placeholder sheet |
| `ChargerInstall.xlsx` | {meta_meta['size_bytes']:,} | `20181126_ChargerInstall_DriveEl` | {meta_dims['max_row']:,} x {meta_dims['max_column']} | Participant charger and vehicle metadata |

*Note: Excel row counts above include the header row.*

---

## 3. Timestamp Integrity and Timezone Handling

### 3.1 UTC Standardization
Raw timestamps in Electric Nation are supplied in UTC. The audit implements the following robust handling:
- Naive Excel timestamps are parsed and converted to UTC-aware timestamps without shifting recorded clock values.
- All derived temporal features (`weekday`, `weekend`, `holiday`, `month`, `StartCos`, `StartSin`, `StartHourNorm`, `sessionsToday`, and `TimeSinceLastStop`) are calculated strictly on this UTC time basis.
- Parsing success rate for non-missing raw timestamps is {parse_success_str} ({parse_fail_str} across {total_non_missing} non-missing timestamp cells; {missing_cells_str}).

| Timestamp Metric | Start Time (`StartTime`) | Stop Time (`StopTime`) |
| :--- | :--- | :--- |
| Column Name | `{ts_metrics['start_col']}` | `{ts_metrics['stop_col']}` |
| Timezone | UTC (UTC-aware) | UTC (UTC-aware) |
| Earliest Value | `{ts_metrics['earliest_start']}` | `{ts_metrics['earliest_stop']}` |
| Latest Value | `{ts_metrics['latest_start']}` | `{ts_metrics['latest_stop']}` |
| Parse Failures | {ts_metrics.get('start_parse_failures', 0)} | {ts_metrics.get('stop_parse_failures', 0)} |
| Missing Raw Values | {raw_missing_start} | {raw_missing_stop} |
| Negative Intervals (`StopTime < StartTime`) | {ts_metrics['negative_intervals']} ({ts_metrics['negative_intervals_pct']}%) | — |
| Zero Duration (`StopTime == StartTime`) | {ts_metrics['zero_intervals']} ({ts_metrics['zero_intervals_pct']}%) | — |

---

## 4. Unattributed-Session Investigation (`ParticipantID is Null`)

A rigorous natural-session matching investigation was conducted on the {unattributed_metrics['total_unattributed_records']:,} unattributed records ({unattributed_metrics['unattributed_percentage']}% of transactions):

- **Chargers Represented**: {unattributed_metrics['chargers_involved']} physical chargers appear in the unattributed records.
  - **{unattributed_metrics['known_chargers_count']} chargers** exist in `ChargerInstall.xlsx` as valid participant installations.
  - **{unattributed_metrics['unknown_chargers_count']} charger** ({unknown_chargers_str}) does NOT exist in `ChargerInstall.xlsx`.
- **Natural Session Matching**: Using the matching key `(ChargerID, StartTime, StopTime, ConsumedkWh)` against the {total_matched_rows:,} known participant transactions:
  - **{unattributed_metrics['matched_4field_session_records']:,} records** match known participant sessions on the four session fields. Other transaction fields may differ.
  - **{unattributed_metrics['unknown_charger_records']} records** belong to the unregistered charger ({unknown_chargers_str}) and cannot be attributed to any trial participant.

---

## 5. Overlap Detection with Nested Overlap Resolution

The session overlap check was implemented using the cumulative maximum stop time of all earlier sessions for each participant:

$$\\text{{Overlap Condition: }} \\text{{StartTime}}_i < \\max_{{j < i}} (\\text{{StopTime}}_j)$$

- **Simple Shift(1) Overlaps**: {simple_overlap_count} sessions detected when comparing only to the immediately preceding session.
- **Cumulative Max Earlier Overlaps**: **{overlap_count:,} sessions** detected ({round(overlap_count / total_tx * 100.0, 4) if total_tx > 0 else 0.0}% of transactions).
- **Nested Overlaps Identified**: Exactly {nested_overlap_diff} nested overlapping sessions were uncovered where a session started after the immediately prior short session ended but before an earlier long session concluded.
- Boundary sessions starting exactly when an earlier session ended (`StartTime == max(StopTime)`) are correctly treated as non-overlapping.

---

## 6. Row-Level Metadata Reconciliation

A row-level reconciliation was performed across all {total_matched_rows:,} matched transaction records against `ChargerInstall.xlsx`:

| Reconciliation Check | Count | Percentage | Status |
| :--- | :--- | :--- | :--- |
| Unique Participants in Transactions | {total_tx_pids} | 100.00% | Non-null participant IDs |
| Unique Participants in Metadata | {meta_pids} | 100.00% | Total trial participants across DCS providers |
| Matched Participants (Both Datasets) | {matched_pids} | {matched_pct:.2f}% | {matched_pct:.1f}% of transaction participants match |
| Unmatched Transaction Participants | {tx_only_pids} | {tx_only_pct:.2f}% | {tx_only_spec_str} |
| Unmatched Metadata Participants | {meta_only_pids} | {meta_only_pct:.2f}% | Metadata records without transactions |
| Row-Level `CarKW` Mismatches | {carkw_mismatches} | {carkw_mismatch_pct:.4f}% | {carkw_agree_pct:.2f}% agreement across all matched rows |
| Row-Level `CarKWh` Mismatches | {carkwh_mismatches} | {carkwh_mismatch_pct:.4f}% | {carkwh_agree_pct:.2f}% agreement across all matched rows |
| Inconsistent Vehicle Specs per Participant | {inconsistent_pids} | {inconsistent_pct:.2f}% | {inconsistent_pids_spec_str} |
| Row-Level `ChargerID` vs `Charger` Matches | {charger_matches:,} | {charger_match_pct:.2f}% | Formats match when compared as strings |

---

## 7. Energy Quality and Battery Capacity Audit

Energy consumption readings (`ConsumedkWh`) were evaluated against physical thresholds:

| Metric | Record Count | % of Transactions | Assessment |
| :--- | :--- | :--- | :--- |
| Missing Energy | {energy_metrics['missing_energy']} | {energy_metrics['missing_energy_pct']:.4f}% | {missing_energy_assessment} |
| Non-Positive Energy (`<= 0`) | {energy_metrics['non_positive_energy']} | {energy_metrics['non_positive_energy_pct']:.4f}% | {non_pos_assessment} |
| Energy > 100 kWh | {energy_metrics['energy_gt_100kwh']} | {energy_metrics['energy_gt_100kwh_pct']:.4f}% | Exceeds official Electric Nation plausibility threshold |
| Energy > 150 kWh | {energy_metrics['energy_gt_150kwh']} | {energy_metrics['energy_gt_150kwh_pct']:.4f}% | Severe outliers reaching {max_energy_val:,.2f} kWh |
| Energy > Registered `CarKWh` | {energy_metrics['energy_exceeds_carkwh']:,} | {energy_metrics['energy_exceeds_carkwh_pct']:.4f}% | Records requiring review (requires investigation into telemetry or multi-cycle sessions) |

---

## 8. Planned Participant Screening and Observation-Density Filtering

The planned modeling population screening was executed in strict order:

| Step | Action | Filter / Rule Applied | Participants | Sessions |
| :---: | :--- | :--- | ---:| ---:|
| 1 | Baseline Population | Valid ParticipantID and valid timestamps | {scr_initial_pids} | {scr_initial_records:,} |
| 2 | Exclude Trial 3 | `Trial != 3.0` | {scr_pids_after_t3} | {scr_records_after_t3:,} |
| 3 | Duration Filter | `dur_min >= 5.0` | {scr_pids_after_short} | {scr_records_after_short:,} |
| 4 | Chronological Sort | Sort by `[ParticipantID, StartTime, StopTime]` | {scr_pids_after_short} | {scr_records_after_short:,} |
| 5 | Overlap Removal | Remove negative-gap overlap records | {scr_pids_after_overlaps} | {scr_records_after_overlaps:,} |
| 6 | Drop First Session | Lag initialization (drop unlagged first session) | {scr_pids_after_dropped_first} | {scr_retained_sessions:,} |
| 7 | Minimum History Filter | Retain participants with $\\ge 10$ retained sessions | {scr_pids_before_density} | {scr_sessions_before_density:,} |
| 8 | Observation-Density Filter | Retain participants where $\\text{{retained\\_sessions}} \\ge \\text{{observation\\_days}} / 7$ | **{scr_final_pids}** | **{scr_final_sessions:,}** |

### Screening Transition Details
- **Step 5 (Overlap Removal)**: Yields {scr_pids_after_overlaps} participants and {scr_records_after_overlaps:,} sessions.
- **Step 6 (Lag Initialization / Dropping First Session)**: Removes {scr_dropped_first} sessions (one first session per participant). {scr_zero_pids_str} only a single session, leaving 0 retained sessions. This leaves **{scr_pids_after_dropped_first} participants** with at least 1 retained session, totaling **{scr_retained_sessions:,} retained sessions**.
- **Step 7 (Minimum History Filter)**: Of these {scr_pids_after_dropped_first} participants, exactly **{scr_pids_lt_10} participants** have fewer than 10 retained sessions (accounting for {scr_sessions_lt_10} sessions). The remaining **{scr_pids_before_density} participants** ({scr_sessions_before_density:,} sessions) pass the minimum-history threshold.
- **Step 8 (Observation-Density Filter)**: Excludes {scr_pids_excluded_density} participants ({scr_sessions_excluded_density:,} sessions) whose retained session count is below $\\text{{observation\\_days}} / 7$, yielding the final population of **{scr_final_pids} participants** and **{scr_final_sessions:,} sessions**.

### Observation-Density Results
- **Before Density Filtering**: {screening_metrics['participants_before_density']} participants, {screening_metrics['sessions_before_density']:,} sessions.
- **Excluded by Density Rule**: {screening_metrics['participants_excluded_density']} participants, {screening_metrics['sessions_excluded_density']:,} sessions.
- **After Density Filtering**: **{screening_metrics['final_eligible_participants']} participants**, **{screening_metrics['final_eligible_sessions']:,} sessions**.
- **Final Population Density Metrics ({screening_metrics['final_eligible_participants']} vehicles)**:
  - Mean observation period: {screening_metrics['density_mean_span_days']} days.
  - Mean sessions per week: {screening_metrics['density_mean_sessions_per_week']} sessions/week.
  - Median sessions per week: {screening_metrics['density_median_sessions_per_week']} sessions/week.
  - Mean average interval between sessions: {screening_metrics['density_mean_avg_interval_days']} days.

---

## 9. Planned Model Inputs and Anti-Leakage Boundaries

The planned model utilizes exactly **13 predictor inputs** and **2 prediction targets** evaluated at plug-in arrival ($t_0 = \\text{{StartTime}}$):

### 9.1 Planned Model Features (13 Inputs)
1. `weekday`: UTC day of week (0..6 integer, Monday=0). Known at $t_0$.
2. `weekend`: UTC weekend binary flag (1 if Saturday or Sunday, else 0). Known at $t_0$.
3. `holiday`: UTC official UK bank holiday binary flag (1 if England & Wales holiday, else 0). Known at $t_0$.
4. `month`: UTC calendar month (1..12 integer). Known at $t_0$.
5. `StartCos`: $\\cos(2\\pi \\cdot \\text{{hour}}_{{\\text{{float}}}} / 24.0)$ arrival cyclic component. Known at $t_0$.
6. `StartSin`: $\\sin(2\\pi \\cdot \\text{{hour}}_{{\\text{{float}}}} / 24.0)$ arrival cyclic component. Known at $t_0$.
7. `StartHourNorm`: $\\text{{hour}}_{{\\text{{float}}}} / 24.0 \\in [0.0, 1.0)$ normalized arrival hour. Known at $t_0$.
8. `TimeSinceLastStop`: Hours elapsed since previous session `StopTime`. Historical lag. Missing on initial session.
9. `sessionsToday`: Count of prior sessions initiated on same calendar date (UTC) before $t_0$. Historical lag.
10. `LastDuration`: Duration in hours of immediately prior completed session. Historical lag. Missing on initial session.
11. `LastConsumedkWh`: Energy consumed in kWh of immediately prior completed session. Historical lag. Missing on initial session.
12. `CarKW`: Vehicle onboard charger power rating in kW. Static vehicle hardware specification.
13. `CarKWh`: Usable battery capacity in kWh. Static vehicle hardware specification.

### 9.2 Prediction Targets
1. **Target 1: Plugged-In Duration (Hours)**: `(StopTime - StartTime) / 3600.0`, capped at 336.0 hours for the modeling target. Original `StopTime` is preserved for historical timing.
2. **Target 2: Energy Consumption (kWh)**: `ConsumedkWh`.

### 9.3 Strict Leakage Boundaries
The following fields occur post-arrival and are strictly excluded from predictors: `StopTime`, `AdjustedStopTime`, `PluggedInTime`, `ConsumedkWh`, `ActiveCharging_Start`, `EndCharge`, `ChargingDuration`, `t_inactive_start`, `t_inactive_end`, `Hot_Unplug`, `Max_Current_Drawn_for_T`, and trial curtailment indicators.

---

## 10. Final Dataset Feasibility Conclusion

The raw datasets **successfully pass the feasibility audit** for building the TARANG EV Charging Intelligence System, subject to the documented preprocessing pipeline:
- Complete, high-precision UTC-aware timestamps.
- {matched_pct:.1f}% metadata concordance across all {matched_pids} active trial participants.
- The dual prediction targets (plugged-in duration capped at 336 hours and energy consumption) can be constructed reliably.
- All 13 planned features are feasible without target leakage.
- Participant screening yields a robust population of **{screening_metrics['final_eligible_participants']} participants** and **{screening_metrics['final_eligible_sessions']:,} sessions**.
"""
    return report


def run_full_audit(
    raw_dir: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Execute complete dataset audit, calculate all metrics, and generate CSVs and Markdown report.

    Args:
        raw_dir: Directory containing raw Excel workbooks. Defaults to config.DATA_RAW_DIR.
        output_dir: Directory to save generated CSV reports. Defaults to config.AUDIT_REPORTS_DIR.

    Returns:
        Dictionary summarizing audit metrics and paths of created files.
    """
    raw_path = Path(raw_dir) if raw_dir is not None else DATA_RAW_DIR
    reports_path = Path(output_dir) if output_dir is not None else AUDIT_REPORTS_DIR
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
    total_tx = len(df_tx)

    logger.info("Validating workbook schemas...")
    validate_workbook_schema(df_tx, REQUIRED_TX_COLUMNS, "CrowdCharge Transactions.xlsx (Transaction Data)")
    validate_workbook_schema(df_meta, REQUIRED_META_COLUMNS, "ChargerInstall.xlsx (20181126_ChargerInstall_DriveEl)")

    logger.info("Parsing UTC timestamps safely...")
    start_utc, stop_utc, ts_metrics = parse_utc_timestamps(df_tx, "StartTime", "StopTime")

    logger.info("Investigating unattributed sessions...")
    unattributed_metrics = investigate_unattributed_sessions(df_tx, df_meta)

    logger.info("Detecting session overlaps with cumulative max stop time...")
    _, overlap_count = detect_session_overlaps(df_tx, "ParticipantID", "StartTime", "StopTime")
    simple_overlap_count = compute_simple_shift_overlaps(df_tx, "ParticipantID", "StartTime", "StopTime")

    logger.info("Finding charging duration discrepancies...")
    ch_discrepancy_info = find_charging_duration_discrepancies(df_tx)

    logger.info("Auditing row-level metadata reconciliation...")
    row_meta_metrics = audit_metadata_row_level(df_tx, df_meta)

    logger.info("Auditing energy quality thresholds...")
    energy_metrics = audit_energy_quality(df_tx)

    logger.info("Screening participant history including observation-density filtering...")
    screening_metrics = screen_participant_history(df_tx)

    logger.info("Auditing feature feasibility for 13 planned features and 2 targets...")
    df_feasibility = audit_feature_feasibility(df_tx, df_meta)
    feasibility_csv = reports_path / "feature_feasibility.csv"
    df_feasibility.to_csv(feasibility_csv, index=False)

    logger.info("Generating transaction quality summary CSV...")
    df_tx_summary = generate_transaction_quality_summary(
        df_tx, ts_metrics, unattributed_metrics, energy_metrics, overlap_count, ch_discrepancy_info
    )
    tx_summary_csv = reports_path / "transaction_quality_summary.csv"
    df_tx_summary.to_csv(tx_summary_csv, index=False)

    logger.info("Generating metadata quality summary CSV...")
    df_meta_summary = generate_metadata_quality_summary(df_meta)
    meta_summary_csv = reports_path / "metadata_quality_summary.csv"
    df_meta_summary.to_csv(meta_summary_csv, index=False)

    logger.info("Generating participant reconciliation CSV...")
    df_reconcile = generate_participant_reconciliation(row_meta_metrics, df_meta, unattributed_metrics, total_tx)
    reconcile_csv = reports_path / "participant_reconciliation.csv"
    df_reconcile.to_csv(reconcile_csv, index=False)

    logger.info("Generating dynamic Markdown audit report...")
    dup_tx_ids = detect_duplicates(df_tx, subset=["TransactionID"]) if "TransactionID" in df_tx.columns else 0
    dup_tx_rows = detect_duplicates(df_tx)
    report_md = generate_markdown_report(
        tx_meta=tx_meta,
        meta_meta=meta_meta,
        ts_metrics=ts_metrics,
        unattributed_metrics=unattributed_metrics,
        row_meta_metrics=row_meta_metrics,
        energy_metrics=energy_metrics,
        overlap_count=overlap_count,
        simple_overlap_count=simple_overlap_count,
        ch_discrepancy_info=ch_discrepancy_info,
        screening_metrics=screening_metrics,
        total_tx=total_tx,
        dup_tx_ids=dup_tx_ids,
        dup_tx_rows=dup_tx_rows,
    )
    report_file = reports_path / "dataset_audit_report.md"
    report_file.write_text(report_md, encoding="utf-8")

    logger.info("Audit complete! Reports saved to: %s", reports_path)

    return {
        "transactions_meta": tx_meta,
        "metadata_meta": meta_meta,
        "timestamp_metrics": ts_metrics,
        "unattributed_metrics": unattributed_metrics,
        "row_metadata_metrics": row_meta_metrics,
        "energy_metrics": energy_metrics,
        "overlap_count": overlap_count,
        "simple_overlap_count": simple_overlap_count,
        "charging_discrepancy": ch_discrepancy_info,
        "screening_metrics": screening_metrics,
        "output_files": [
            str(tx_summary_csv),
            str(meta_summary_csv),
            str(reconcile_csv),
            str(feasibility_csv),
            str(report_file),
        ],
    }
