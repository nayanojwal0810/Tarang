"""Data preparation pipeline orchestration module for TARANG.

Loads raw transactions, executes research screening and feature engineering,
and persists the canonical features.parquet and processing_summary.json.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union
import pandas as pd

from tarang.config import (
    FEATURE_NAMES,
    PROCESSED_FEATURES_FILE,
    PROCESSING_SUMMARY_FILE,
    RAW_TX_FILE,
    RAW_TX_SHEET,
)
from tarang.data.loading import load_dataset
from tarang.data.screening import screen_and_prepare_dataset
from tarang.data.validation import validate_canonical_dataset_schema

logger = logging.getLogger(__name__)


def run_data_preparation(
    raw_tx_path: Optional[Union[str, Path]] = None,
    sheet_name: str = RAW_TX_SHEET,
    output_parquet_path: Optional[Union[str, Path]] = None,
    output_summary_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Execute end-to-end data preparation and persist canonical research artifacts.

    Args:
        raw_tx_path: Path to raw transactions workbook. Defaults to config.RAW_TX_FILE.
        sheet_name: Transactions sheet name. Defaults to config.RAW_TX_SHEET.
        output_parquet_path: Destination path for features.parquet. Defaults to config.PROCESSED_FEATURES_FILE.
        output_summary_path: Destination path for processing_summary.json. Defaults to config.PROCESSING_SUMMARY_FILE.

    Returns:
        Dictionary of processing summary metrics and output paths.
    """
    tx_file = Path(raw_tx_path) if raw_tx_path is not None else RAW_TX_FILE
    parquet_out = Path(output_parquet_path) if output_parquet_path is not None else PROCESSED_FEATURES_FILE
    summary_out = Path(output_summary_path) if output_summary_path is not None else PROCESSING_SUMMARY_FILE

    if not tx_file.is_file():
        raise FileNotFoundError(f"Raw transactions file not found at: {tx_file}")

    logger.info("Loading raw transactions from: %s (sheet: %s)", tx_file, sheet_name)
    df_raw = load_dataset(tx_file, sheet_name=sheet_name)
    logger.info("Loaded %d raw transaction records.", len(df_raw))

    logger.info("Executing screening sequence and feature engineering...")
    df_features, metrics = screen_and_prepare_dataset(df_raw)
    logger.info(
        "Screening complete: retained %d sessions across %d participants.",
        len(df_features),
        metrics["final_eligible_participants"],
    )

    # Validate output schema
    validate_canonical_dataset_schema(df_features)

    # Ensure parent directories exist
    parquet_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)

    # Save canonical parquet dataset
    logger.info("Saving canonical features dataset to: %s", parquet_out)
    df_features.to_parquet(parquet_out, engine="pyarrow", index=False)

    # Build and save processing summary JSON
    summary_data = {
        "raw_records": len(df_raw),
        "raw_participants": metrics["initial_participants"],
        "trial_3_excluded": metrics["trial_3_excluded"],
        "remaining_after_t3": metrics["remaining_after_t3"],
        "short_sessions_excluded": metrics["short_sessions_excluded"],
        "remaining_after_short": metrics["remaining_after_short"],
        "negative_gaps_removed": metrics["negative_gaps_removed"],
        "dropped_first_sessions": metrics["dropped_first_sessions"],
        "retained_sessions_total": metrics["retained_sessions_total"],
        "participants_lt_10_retained": metrics["participants_lt_10_retained"],
        "sessions_in_lt_10_retained": metrics["sessions_in_lt_10_retained"],
        "participants_excluded_density": metrics["participants_excluded_density"],
        "sessions_excluded_density": metrics["sessions_excluded_density"],
        "final_eligible_participants": metrics["final_eligible_participants"],
        "final_eligible_sessions": metrics["final_eligible_sessions"],
        "max_start_hour": metrics["max_start_hour"],
        "feature_names": FEATURE_NAMES,
        "parquet_path": str(parquet_out),
        "total_columns": len(df_features.columns),
    }

    logger.info("Saving processing summary to: %s", summary_out)
    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4)

    return summary_data
