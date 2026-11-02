"""Preprocessing and modeling dataset pipeline for TARANG — EV Charging Intelligence System.

This module provides reusable, leakage-safe functions to screen sessions,
engineer features, construct prediction targets, and generate both the
intermediate screened dataset and the final warm-start modeling dataset.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from tarang.data.audit import (
    REQUIRED_META_COLUMNS,
    REQUIRED_TX_COLUMNS,
    UK_BANK_HOLIDAYS_2017_2018,
    audit_metadata_row_level,
    load_dataset,
    screen_participant_sessions,
    validate_workbook_schema,
)

logger = logging.getLogger("tarang.data.preprocessing")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Exact feature allowlist in mandatory order
WARM_START_FEATURES: List[str] = [
    "weekday",
    "weekend",
    "holiday",
    "month",
    "StartCos",
    "StartSin",
    "StartHourNorm",
    "TimeSinceLastStop",
    "sessionsToday",
    "LastDuration",
    "LastConsumedkWh",
    "CarKW",
    "CarKWh",
]

# Exact target columns
TARGET_COLUMNS: List[str] = [
    "target_duration_hours",
    "target_energy_kwh",
]

# Metadata columns retained for traceability and grouping in model-ready data
MODEL_METADATA_COLUMNS: List[str] = [
    "TransactionID",
    "ParticipantID",
    "StartTime",
]

# Exact columns for the final model-ready dataset (18 columns total)
MODEL_READY_COLUMNS: List[str] = MODEL_METADATA_COLUMNS + WARM_START_FEATURES + TARGET_COLUMNS

# Prohibited post-session and raw target fields that must NEVER enter feature matrix
PROHIBITED_FEATURE_COLUMNS: Set[str] = {
    "ParticipantID",
    "TransactionID",
    "StartTime",
    "StopTime",
    "AdjustedStopTime",
    "PluggedInTime",
    "ConsumedkWh",
    "ChargingDuration",
    "ActiveCharging_Start",
    "EndCharge",
    "t_inactive_start",
    "t_inactive_end",
    "Hot_Unplug",
    "Max_Current_Drawn_for_T",
    "Restricton T1",
    "Restriction T2",
    "energy_gt_100kwh",
    "target_duration_hours",
    "target_energy_kwh",
}


def validate_input_datasets(df_tx: pd.DataFrame, df_meta: pd.DataFrame) -> None:
    """Validate workbook schemas, key uniqueness, participant concordance, and specification integrity.

    Args:
        df_tx: Raw transactions DataFrame.
        df_meta: Raw ChargerInstall metadata DataFrame.

    Raises:
        ValueError: If any schema, uniqueness, reconciliation, or specification check fails.
    """
    validate_workbook_schema(df_tx, REQUIRED_TX_COLUMNS, "CrowdCharge Transactions.xlsx (Transaction Data)")
    validate_workbook_schema(df_meta, REQUIRED_META_COLUMNS, "ChargerInstall.xlsx (20181126_ChargerInstall_DriveEl)")

    # 1. TransactionID non-null and unique
    if df_tx["TransactionID"].isna().any():
        raise ValueError("Validation failed: TransactionID contains null values in transactions dataset.")
    if df_tx["TransactionID"].duplicated().any():
        dup_count = int(df_tx["TransactionID"].duplicated().sum())
        raise ValueError(f"Validation failed: TransactionID contains {dup_count} duplicate values in transactions.")

    # 2. Metadata ParticipantID non-null and unique
    if df_meta["ParticipantID"].isna().any():
        raise ValueError("Validation failed: ParticipantID contains null values in metadata registry.")
    if df_meta["ParticipantID"].duplicated().any():
        dup_count = int(df_meta["ParticipantID"].duplicated().sum())
        raise ValueError(f"Validation failed: ParticipantID contains {dup_count} duplicate values in metadata.")

    # 3. All non-null transaction participant IDs match metadata
    tx_pids = set(df_tx["ParticipantID"].dropna().unique())
    meta_pids = set(df_meta["ParticipantID"].dropna().unique())
    unmatched = tx_pids - meta_pids
    if unmatched:
        raise ValueError(f"Validation failed: {len(unmatched)} transaction participants not found in metadata: {unmatched}")

    # 4. Transaction CarKW and CarKWh values agree with metadata for matched participants
    meta_metrics = audit_metadata_row_level(df_tx, df_meta)
    kw_mismatches = meta_metrics.get("row_level_carkw_mismatches", 0)
    kwh_mismatches = meta_metrics.get("row_level_carkwh_mismatches", 0)
    if kw_mismatches > 0 or kwh_mismatches > 0:
        raise ValueError(
            f"Validation failed: Vehicle specification mismatches between transactions and metadata: "
            f"CarKW mismatches={kw_mismatches}, CarKWh mismatches={kwh_mismatches}"
        )

    # 5. Model-eligible vehicle specifications are present and numeric
    valid_tx = df_tx[df_tx["ParticipantID"].isin(meta_pids)]
    if valid_tx["CarKW"].isna().any() or not pd.api.types.is_numeric_dtype(valid_tx["CarKW"]):
        raise ValueError("Validation failed: Non-numeric or missing CarKW in matched transactions.")
    if valid_tx["CarKWh"].isna().any() or not pd.api.types.is_numeric_dtype(valid_tx["CarKWh"]):
        raise ValueError("Validation failed: Non-numeric or missing CarKWh in matched transactions.")


def prepare_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate plugged-in duration and energy targets, duration capping, and outlier flag.

    Target 1: target_duration_hours = min(duration_hours_raw, 336.0)
    Target 2: target_energy_kwh = ConsumedkWh

    Args:
        df: DataFrame with UTC StartTime, StopTime, and ConsumedkWh.

    Returns:
        DataFrame with duration_hours_raw, target_duration_hours, target_energy_kwh,
        and energy_gt_100kwh added.
    """
    df_out = df.copy()

    # Raw connection duration in hours
    dur_hours_raw = (df_out["StopTime"] - df_out["StartTime"]).dt.total_seconds() / 3600.0
    df_out["duration_hours_raw"] = dur_hours_raw.astype("float64")

    # Target 1: Capped duration at 336.0 hours
    df_out["target_duration_hours"] = np.minimum(dur_hours_raw, 336.0).astype("float64")

    # Target 2: Observed energy consumption in kWh
    df_out["target_energy_kwh"] = pd.to_numeric(df_out["ConsumedkWh"], errors="coerce").astype("float64")

    # Energy quality review flag for readings exceeding 100 kWh
    df_out["energy_gt_100kwh"] = (df_out["target_energy_kwh"] > 100.0).astype(bool)

    return df_out


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer all 13 predictor features strictly aligned with TARANG data contract.

    Features engineered:
    1. weekday: UTC day of week (0..6 integer, Monday=0).
    2. weekend: Binary flag (1 if Saturday/Sunday, else 0).
    3. holiday: Binary flag (1 if England & Wales bank holiday, else 0).
    4. month: UTC calendar month (1..12 integer).
    5. StartCos: cos(2 * pi * StartHourNorm).
    6. StartSin: sin(2 * pi * StartHourNorm).
    7. StartHourNorm: (hour + minute / 60.0) / 24.0.
    8. TimeSinceLastStop: Elapsed hours since previous session uncapped StopTime.
    9. sessionsToday: Count of prior sessions initiated on same UTC date strictly before current session.
    10. LastDuration: Capped duration of immediately preceding session.
    11. LastConsumedkWh: Observed energy of immediately preceding session (if valid and >= 0.01 kWh).
    12. CarKW: Vehicle onboard charger power rating in kW.
    13. CarKWh: Vehicle usable battery capacity in kWh.

    Args:
        df: Post-overlap sessions DataFrame sorted chronologically per participant.

    Returns:
        DataFrame with all 13 features engineered and attached.
    """
    # Enforce stable chronological sorting per participant
    df_out = df.sort_values(["ParticipantID", "StartTime", "StopTime"], kind="stable").reset_index(drop=True).copy()

    # 1-4. Calendar features (UTC)
    weekday = df_out["StartTime"].dt.dayofweek.astype("int64")
    df_out["weekday"] = weekday
    df_out["weekend"] = weekday.isin([5, 6]).astype("int64")

    # Bank holidays lookup against England & Wales 2017-2018 schedule
    start_dates = df_out["StartTime"].dt.date
    df_out["holiday"] = start_dates.isin(UK_BANK_HOLIDAYS_2017_2018).astype("int64")
    df_out["month"] = df_out["StartTime"].dt.month.astype("int64")

    # 5-7. Daily cyclical arrival time features (fixed 24.0 hour cycle)
    hour_decimal = df_out["StartTime"].dt.hour + df_out["StartTime"].dt.minute / 60.0
    start_hour_norm = (hour_decimal / 24.0).astype("float64")
    df_out["StartHourNorm"] = start_hour_norm

    angle = 2.0 * np.pi * start_hour_norm
    df_out["StartCos"] = np.cos(angle).astype("float64")
    df_out["StartSin"] = np.sin(angle).astype("float64")

    # 8. TimeSinceLastStop: Hours elapsed since previous session's original uncapped StopTime
    prev_stop = df_out.groupby("ParticipantID", sort=False)["StopTime"].shift(1)
    df_out["TimeSinceLastStop"] = ((df_out["StartTime"] - prev_stop).dt.total_seconds() / 3600.0).astype("float64")

    # 9. sessionsToday: Count of prior sessions on same UTC calendar date strictly before current session
    df_out["sessionsToday"] = df_out.groupby(["ParticipantID", start_dates], sort=False).cumcount().astype("int64")

    # 10. LastDuration: Immediately preceding session's capped duration
    df_out["LastDuration"] = df_out.groupby("ParticipantID", sort=False)["target_duration_hours"].shift(1).astype("float64")

    # 11. LastConsumedkWh: Immediately preceding session's observed energy if numeric, finite, >= 0.01
    valid_energy = df_out["target_energy_kwh"].where(
        df_out["target_energy_kwh"].notna() & np.isfinite(df_out["target_energy_kwh"]) & (df_out["target_energy_kwh"] >= 0.01),
        np.nan,
    )
    df_out["_valid_energy_for_lag"] = valid_energy
    df_out["LastConsumedkWh"] = df_out.groupby("ParticipantID", sort=False)["_valid_energy_for_lag"].shift(1).astype("float64")
    df_out.drop(columns=["_valid_energy_for_lag"], inplace=True)

    # 12-13. Vehicle specifications directly from transaction records
    df_out["CarKW"] = df_out["CarKW"].astype("float64")
    df_out["CarKWh"] = df_out["CarKWh"].astype("float64")

    return df_out


def build_screened_and_model_ready_datasets(
    df_tx: pd.DataFrame,
    df_meta: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Build the intermediate screened dataset and final warm-start modeling dataset.

    Executes:
    1. Input validation against metadata and schemas.
    2. Session screening via shared screen_participant_sessions().
    3. Target construction and feature engineering on all post-overlap sessions.
    4. Construction of intermediate screened_sessions dataset (all 60,575 post-overlap sessions).
    5. Selection of final warm-start modeling population:
       - Drop unlagged first sessions.
       - Apply >= 10 retained sessions minimum history filter.
       - Apply observation density filter: retained_session_count >= observation_days / 7.0.
       - Exclude invalid/non-positive targets.
       - Enforce explicit 13-feature allowlist and metadata columns.
       - Chronological sort by [StartTime, TransactionID].

    Args:
        df_tx: Raw transactions DataFrame.
        df_meta: Raw metadata DataFrame.

    Returns:
        Tuple of (df_screened, df_model_ready, processing_summary_dict).
    """
    logger.info("Validating input datasets...")
    validate_input_datasets(df_tx, df_meta)

    logger.info("Executing participant screening sequence...")
    df_screened_raw, _, screening_metrics = screen_participant_sessions(df_tx)

    if df_screened_raw.empty:
        empty_summary = {
            "input_transactions_count": len(df_tx),
            "screened_sessions_count": 0,
            "model_ready_sessions_count": 0,
            "screening_metrics": screening_metrics,
        }
        return pd.DataFrame(), pd.DataFrame(), empty_summary

    logger.info("Preparing prediction targets on screened sessions (%d rows)...", len(df_screened_raw))
    df_screened_targets = prepare_targets(df_screened_raw)

    logger.info("Engineering 13 features across complete post-overlap session history...")
    df_screened = engineer_features(df_screened_targets)

    # Clean internal temporary calculation columns from screened dataset
    for col in ["_start_dt", "_stop_dt", "_dur_min"]:
        if col in df_screened.columns:
            df_screened.drop(columns=[col], inplace=True)

    # Standardize mixed Excel date/time formatting in Max_Current_Drawn_for_T to string for Parquet serialization
    if "Max_Current_Drawn_for_T" in df_screened.columns:
        df_screened["Max_Current_Drawn_for_T"] = df_screened["Max_Current_Drawn_for_T"].astype(str)

    # --- Construct Final Warm-Start Modeling Dataset ---
    logger.info("Constructing warm-start model-ready dataset...")
    # Step 6: Drop unlagged first session per participant
    df_screened["session_rank"] = df_screened.groupby("ParticipantID", sort=False).cumcount()
    first_sessions_count = int(df_screened["ParticipantID"].nunique())
    df_warm = df_screened[df_screened["session_rank"] > 0].copy()

    # Step 7: Minimum history (>= 10 retained sessions after dropping first session)
    retained_counts = df_warm.groupby("ParticipantID").size()
    pids_min_history = retained_counts[retained_counts >= 10].index
    df_warm = df_warm[df_warm["ParticipantID"].isin(pids_min_history)].copy()

    # Step 8: Observation-density filtering (retained_session_count >= observation_days / 7.0)
    density_calc = df_warm.groupby("ParticipantID").agg(
        earliest_start=("StartTime", "min"),
        latest_stop=("StopTime", "max"),
        retained_session_count=("TransactionID", "count"),
    )
    density_calc["observation_days"] = (density_calc["latest_stop"] - density_calc["earliest_start"]).dt.days
    density_calc["min_required_sessions"] = density_calc["observation_days"] / 7.0
    density_calc["fails_density"] = density_calc["retained_session_count"] < density_calc["min_required_sessions"]
    final_eligible_pids = density_calc[~density_calc["fails_density"]].index

    df_model_ready = df_warm[df_warm["ParticipantID"].isin(final_eligible_pids)].copy()

    # Exclude missing, non-numeric, non-finite, or non-positive targets
    valid_duration = (
        df_model_ready["target_duration_hours"].notna()
        & np.isfinite(df_model_ready["target_duration_hours"])
        & (df_model_ready["target_duration_hours"] > 0)
    )
    valid_energy = (
        df_model_ready["target_energy_kwh"].notna()
        & np.isfinite(df_model_ready["target_energy_kwh"])
        & (df_model_ready["target_energy_kwh"] > 0)
    )
    initial_model_ready_count = len(df_model_ready)
    df_model_ready = df_model_ready[valid_duration & valid_energy].copy()
    missing_target_exclusions = initial_model_ready_count - len(df_model_ready)

    # Restrict model-ready dataset exclusively to metadata, 13 features, and 2 targets
    df_model_ready = df_model_ready[MODEL_READY_COLUMNS].copy()

    # Verify no prohibited columns crept into the feature matrix
    violating_features = set(WARM_START_FEATURES).intersection(PROHIBITED_FEATURE_COLUMNS)
    if violating_features:
        raise ValueError(f"Leakage violation: Prohibited columns found in feature list: {violating_features}")

    # Chronologically sort by [StartTime, TransactionID] using stable sort
    df_model_ready = df_model_ready.sort_values(["StartTime", "TransactionID"], kind="stable").reset_index(drop=True)

    # Verify unique TransactionID
    if df_model_ready["TransactionID"].duplicated().any():
        raise ValueError("Integrity error: Duplicate TransactionID found in model_ready_sessions.")

    # Remove temporary session_rank from df_screened
    df_screened.drop(columns=["session_rank"], inplace=True)

    # Compile dynamic processing summary
    summary: Dict[str, Any] = {
        "input_transactions_count": len(df_tx),
        "initial_valid_records": screening_metrics.get("initial_valid_records", 0),
        "initial_valid_participants": screening_metrics.get("initial_participants", 0),
        "trial_3_excluded": screening_metrics.get("trial_3_excluded", 0),
        "remaining_after_trial_3": screening_metrics.get("remaining_after_t3", 0),
        "short_sessions_excluded": screening_metrics.get("short_sessions_excluded", 0),
        "remaining_after_short_sessions": screening_metrics.get("remaining_after_short", 0),
        "screened_overlaps": screening_metrics.get("screened_overlaps", 0),
        "sessions_after_overlap_removal": len(df_screened),
        "participants_after_overlap_removal": int(df_screened["ParticipantID"].nunique()),
        "first_sessions_excluded_warm_start": first_sessions_count,
        "participants_after_first_session_drop": screening_metrics.get("participants_after_dropped_first", 0),
        "sessions_after_first_session_drop": screening_metrics.get("retained_sessions_total", 0),
        "participants_passing_minimum_history": screening_metrics.get("participants_before_density", 0),
        "sessions_passing_minimum_history": screening_metrics.get("sessions_before_density", 0),
        "participants_passing_observation_density": screening_metrics.get("final_eligible_participants", 0),
        "sessions_passing_observation_density": screening_metrics.get("final_eligible_sessions", 0),
        "missing_target_exclusions": missing_target_exclusions,
        "final_model_ready_rows": len(df_model_ready),
        "final_model_ready_participants": int(df_model_ready["ParticipantID"].nunique()),
        "screened_energy_gt_100kwh": int((df_screened["target_energy_kwh"] > 100.0).sum()),
        "screened_energy_gt_150kwh": int((df_screened["target_energy_kwh"] > 150.0).sum()),
        "model_ready_energy_gt_100kwh": int((df_model_ready["target_energy_kwh"] > 100.0).sum()),
        "model_ready_energy_gt_150kwh": int((df_model_ready["target_energy_kwh"] > 150.0).sum()),
        "feature_names": WARM_START_FEATURES,
        "target_names": TARGET_COLUMNS,
        "model_metadata_columns": MODEL_METADATA_COLUMNS,
        "model_ready_columns": MODEL_READY_COLUMNS,
        "model_ready_missingness": {col: int(df_model_ready[col].isna().sum()) for col in MODEL_READY_COLUMNS},
        "outputs": {
            "screened_sessions": {
                "filename": "screened_sessions.parquet",
                "row_count": len(df_screened),
                "participant_count": int(df_screened["ParticipantID"].nunique()),
                "column_count": len(df_screened.columns),
            },
            "model_ready_sessions": {
                "filename": "model_ready_sessions.parquet",
                "row_count": len(df_model_ready),
                "participant_count": int(df_model_ready["ParticipantID"].nunique()),
                "column_count": len(df_model_ready.columns),
            },
        },
    }

    return df_screened, df_model_ready, summary


def save_processed_datasets(
    df_screened: pd.DataFrame,
    df_model_ready: pd.DataFrame,
    summary: Dict[str, Any],
    output_dir: Union[str, Path],
) -> Dict[str, Path]:
    """Save processed datasets as Parquet files and write the processing summary JSON.

    Args:
        df_screened: Intermediate screened DataFrame.
        df_model_ready: Final warm-start modeling DataFrame.
        summary: Processing summary metrics dictionary.
        output_dir: Destination directory.

    Returns:
        Dictionary of created file paths.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    screened_file = out_path / "screened_sessions.parquet"
    model_ready_file = out_path / "model_ready_sessions.parquet"
    summary_file = out_path / "processing_summary.json"

    logger.info("Writing %s (%d rows)...", screened_file.name, len(df_screened))
    df_screened_to_save = df_screened.copy()
    if "Max_Current_Drawn_for_T" in df_screened_to_save.columns:
        df_screened_to_save["Max_Current_Drawn_for_T"] = df_screened_to_save["Max_Current_Drawn_for_T"].astype(str)
    df_screened_to_save.to_parquet(screened_file, engine="pyarrow", index=False)

    logger.info("Writing %s (%d rows)...", model_ready_file.name, len(df_model_ready))
    df_model_ready.to_parquet(model_ready_file, engine="pyarrow", index=False)

    logger.info("Writing %s...", summary_file.name)
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    return {
        "screened_sessions": screened_file,
        "model_ready_sessions": model_ready_file,
        "processing_summary": summary_file,
    }


def run_preprocessing(
    raw_dir: Optional[Union[str, Path]] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Execute complete preprocessing pipeline from raw Excel workbooks to modeling datasets.

    Args:
        raw_dir: Directory containing raw workbooks. Defaults to 'data/raw/electric_nation'.
        output_dir: Output directory. Defaults to 'data/processed/electric_nation'.

    Returns:
        Processing summary dictionary.
    """
    project_root = Path(__file__).resolve().parents[3]
    raw_path = Path(raw_dir) if raw_dir else project_root / "data" / "raw" / "electric_nation"
    out_path = Path(output_dir) if output_dir else project_root / "data" / "processed" / "electric_nation"

    tx_file = raw_path / "CrowdCharge Transactions.xlsx"
    meta_file = raw_path / "ChargerInstall.xlsx"

    if not tx_file.is_file():
        raise FileNotFoundError(f"Transactions workbook missing at: {tx_file}")
    if not meta_file.is_file():
        raise FileNotFoundError(f"Metadata workbook missing at: {meta_file}")

    logger.info("Loading raw transactions sheet 'Transaction Data'...")
    df_tx = load_dataset(tx_file, sheet_name="Transaction Data")

    logger.info("Loading raw metadata sheet '20181126_ChargerInstall_DriveEl'...")
    df_meta = load_dataset(meta_file, sheet_name="20181126_ChargerInstall_DriveEl")

    df_screened, df_model_ready, summary = build_screened_and_model_ready_datasets(df_tx, df_meta)

    save_processed_datasets(df_screened, df_model_ready, summary, out_path)

    logger.info(
        "Preprocessing pipeline completed successfully! Screened sessions: %d, Model-ready sessions: %d (%d participants).",
        len(df_screened),
        len(df_model_ready),
        summary["final_model_ready_participants"],
    )

    return summary
