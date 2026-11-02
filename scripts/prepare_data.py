"""Entry-point script to execute the TARANG research data preparation pipeline.

Loads raw transactions, applies the multi-step screening sequence, constructs
canonical features, and saves features.parquet and processing_summary.json.
"""

from __future__ import annotations

import logging
import sys

from tarang.config import (
    PROCESSED_FEATURES_FILE,
    PROCESSING_SUMMARY_FILE,
    RAW_TX_FILE,
    RAW_TX_SHEET,
)
from tarang.data.preparation import run_data_preparation

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("scripts.prepare_data")


def main() -> int:
    """Run data preparation pipeline."""
    logger.info("Starting TARANG Data Preparation Pipeline...")
    try:
        summary = run_data_preparation(
            raw_tx_path=RAW_TX_FILE,
            sheet_name=RAW_TX_SHEET,
            output_parquet_path=PROCESSED_FEATURES_FILE,
            output_summary_path=PROCESSING_SUMMARY_FILE,
        )
        logger.info(
            "Data preparation completed: %d sessions across %d participants saved to %s",
            summary["final_eligible_sessions"],
            summary["final_eligible_participants"],
            PROCESSED_FEATURES_FILE,
        )
        return 0
    except Exception as exc:
        logger.error("Data preparation failed with error: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

