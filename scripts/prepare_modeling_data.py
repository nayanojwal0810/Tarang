#!/usr/bin/env python3
"""CLI script to run the TARANG preprocessing and modeling dataset pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to sys.path to support execution without pip install -e .
project_root = Path(__file__).resolve().parents[1]
src_dir = project_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from tarang.data.preprocessing import run_preprocessing

logger = logging.getLogger("prepare_modeling_data")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run TARANG preprocessing pipeline to create leakage-safe modeling datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=str(project_root / "data" / "raw" / "electric_nation"),
        help="Path to directory containing raw Electric Nation Excel workbooks.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(project_root / "data" / "processed" / "electric_nation"),
        help="Path to directory where processed Parquet files and summary JSON will be stored.",
    )
    return parser.parse_args()


def main() -> int:
    """Execute preprocessing pipeline from CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    args = parse_args()

    logger.info("Starting TARANG Preprocessing Pipeline...")
    logger.info("Raw directory: %s", args.raw_dir)
    logger.info("Output directory: %s", args.output_dir)

    try:
        summary = run_preprocessing(raw_dir=args.raw_dir, output_dir=args.output_dir)
        print("\n" + "=" * 60)
        print("TARANG PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"Input Transactions:               {summary['input_transactions_count']:,}")
        print(f"Screened Sessions (intermediate): {summary['outputs']['screened_sessions']['row_count']:,} "
              f"({summary['outputs']['screened_sessions']['participant_count']} participants)")
        print(f"Model-Ready Sessions (final):     {summary['outputs']['model_ready_sessions']['row_count']:,} "
              f"({summary['outputs']['model_ready_sessions']['participant_count']} participants)")
        print(f"Features Generated:               {len(summary['feature_names'])} {summary['feature_names']}")
        print(f"Targets Generated:                {len(summary['target_names'])} {summary['target_names']}")
        print(f"Output Directory:                 {args.output_dir}")
        print("=" * 60 + "\n")
        return 0
    except Exception as e:
        logger.error("Preprocessing pipeline failed: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
