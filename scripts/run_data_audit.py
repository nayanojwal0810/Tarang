"""Entry-point script to execute the complete TARANG data quality audit.

Runs non-destructive quality inspections on raw workbooks and generates summary CSVs
and the Markdown audit report.
"""

from __future__ import annotations

import logging
import sys

from tarang.config import AUDIT_REPORTS_DIR, DATA_RAW_DIR
from tarang.data.audit_reporting import run_full_audit

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("scripts.run_data_audit")


def main() -> int:
    """Run full dataset audit."""
    logger.info("Starting TARANG Data Quality Audit...")
    try:
        results = run_full_audit(raw_dir=DATA_RAW_DIR, output_dir=AUDIT_REPORTS_DIR)
        logger.info("Data audit completed successfully.")
        logger.info("Generated %d report artifacts in %s", len(results["output_files"]), AUDIT_REPORTS_DIR)
        return 0
    except Exception as exc:
        logger.error("Data audit failed with error: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
