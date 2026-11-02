"""CLI entry point for TARANG Stage 5 Decision Support pipeline.

Executes batch decision support advisory generation, flexibility analysis, and operational reporting.
Generates:
- Session advisories dataset with duration, energy, validity flags, and projected unplug timestamps.
- Prediction distribution summary table (reports/decision_support/prediction_summary.csv).
- Smart-charging flexibility summary table (reports/decision_support/flexibility_summary.csv).
- Comprehensive operational report (reports/decision_support/decision_support_summary.md).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import List, Optional, Set
import pandas as pd

from tarang.config import (
    DECISION_SUPPORT_REPORTS_DIR,
    DEFAULT_SESSION_ADVISORIES_FILE,
    MODEL_NAMES,
    PROCESSED_FEATURES_FILE,
)
from tarang.decision_support.engine import DecisionSupportEngine
from tarang.decision_support.flexibility import analyze_charging_flexibility
from tarang.decision_support.reporting import save_decision_support_artifacts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("scripts.run_decision_support")

SUPPORTED_EXTENSIONS: Set[str] = {".parquet", ".csv"}


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments for the decision support CLI."""
    parser = argparse.ArgumentParser(
        description="Run TARANG Stage 5 Decision Support advisory pipeline."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=str(PROCESSED_FEATURES_FILE),
        help=f"Path to input dataset (.parquet or .csv). Defaults to '{PROCESSED_FEATURES_FILE}'.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=str(DEFAULT_SESSION_ADVISORIES_FILE),
        help=f"Path to output advisories file (.parquet or .csv). Defaults to '{DEFAULT_SESSION_ADVISORIES_FILE}'.",
    )
    parser.add_argument(
        "--report-dir",
        "-r",
        type=str,
        default=str(DECISION_SUPPORT_REPORTS_DIR),
        help=f"Directory for generated reports. Defaults to '{DECISION_SUPPORT_REPORTS_DIR}'.",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="HistGradientBoostingRegressor",
        help=f"Benchmark model architecture. Defaults to 'HistGradientBoostingRegressor'. Supported: {MODEL_NAMES}.",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Optional directory containing pre-trained model artifacts.",
    )
    return parser.parse_args(args)


def main(argv: Optional[List[str]] = None) -> int:
    """Execute decision support advisory and reporting workflow."""
    args = parse_args(argv)

    input_path = Path(args.input)
    output_path = Path(args.output)
    report_dir = Path(args.report_dir)
    models_dir = Path(args.models_dir) if args.models_dir else None

    # Enforce supported input extension
    ext_in = input_path.suffix.lower()
    if ext_in not in SUPPORTED_EXTENSIONS:
        logger.error(
            "Unsupported input file format '%s'. Must be one of: %s.",
            input_path.suffix,
            sorted(SUPPORTED_EXTENSIONS),
        )
        return 1

    # Enforce supported output extension
    ext_out = output_path.suffix.lower()
    if ext_out not in SUPPORTED_EXTENSIONS:
        logger.error(
            "Unsupported output file format '%s'. Must be one of: %s.",
            output_path.suffix,
            sorted(SUPPORTED_EXTENSIONS),
        )
        return 1

    if not input_path.is_file():
        logger.error("Input file does not exist: %s", input_path)
        return 1

    logger.info("Loading input dataset from: %s", input_path)
    try:
        if ext_in == ".parquet":
            input_df = pd.read_parquet(input_path)
        else:
            input_df = pd.read_csv(input_path)
    except Exception as exc:
        logger.error("Failed to read input file: %s", exc)
        return 1

    logger.info("Loaded %d session records. Initializing DecisionSupportEngine...", len(input_df))

    try:
        engine = DecisionSupportEngine(
            model_name=args.model,
            models_dir=models_dir,
        )
    except Exception as exc:
        logger.error("Failed to initialize DecisionSupportEngine: %s", exc)
        return 1

    logger.info("Generating batch session advisories using model '%s'...", args.model)
    try:
        advisories_df = engine.predict_batch_advisories(features_df=input_df)
    except Exception as exc:
        logger.error("Failed to generate batch advisories: %s", exc)
        return 1

    logger.info("Performing smart-charging flexibility analysis...")
    try:
        flex_stats = analyze_charging_flexibility(input_df)
    except Exception as exc:
        logger.warning("Flexibility analysis encountered an error: %s. Continuing...", exc)
        flex_stats = {"available": False, "reason": str(exc), "missing_columns": None}

    logger.info("Persisting advisories and generating operational reports...")
    try:
        artifacts = save_decision_support_artifacts(
            advisories_df=advisories_df,
            flexibility_stats=flex_stats,
            output_advisories_path=output_path,
            report_dir=report_dir,
            model_name=args.model,
            source_file=input_path,
        )
    except Exception as exc:
        logger.error("Failed to save decision support artifacts: %s", exc)
        return 1

    logger.info("Decision support workflow completed successfully:")
    for key, path in artifacts.items():
        logger.info("  - %s: %s", key, path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
