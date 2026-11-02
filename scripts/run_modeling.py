"""Script to execute TARANG Stage 2 predictive modeling benchmark.

Loads the canonical features dataset and runs benchmark training and evaluation
for both prediction targets (plugged-in duration and energy consumption) in one execution.
"""

from __future__ import annotations

import argparse
import logging
import sys
import pandas as pd

from tarang.config import (
    DURATION_MODELING_DIR,
    ENERGY_MODELING_DIR,
    PROCESSED_FEATURES_FILE,
    TARGET_DURATION,
    TARGET_ENERGY,
)
from tarang.modeling.experiment import run_experiment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("scripts.run_modeling")


def main() -> int:
    """Run dual-target benchmark modeling experiments."""
    parser = argparse.ArgumentParser(
        description="Execute TARANG Stage 2 predictive modeling benchmark."
    )
    parser.add_argument(
        "--features-path",
        type=str,
        default=str(PROCESSED_FEATURES_FILE),
        help=f"Path to canonical features.parquet (default: {PROCESSED_FEATURES_FILE})",
    )
    parser.add_argument(
        "--saved",
        action="store_true",
        default=False,
        help="Use existing saved models if present instead of refitting.",
    )
    parser.add_argument(
        "--target",
        choices=["all", TARGET_DURATION, TARGET_ENERGY],
        default="all",
        help="Target to model ('all', 'Duration', or 'ConsumedkWh', default: 'all').",
    )

    args = parser.parse_args()

    features_path = pd.io.common.Path(args.features_path)
    if not features_path.exists():
        logger.error(
            "Canonical features file not found at: %s. "
            "Please run 'python scripts/prepare_data.py' first.",
            features_path,
        )
        return 1

    logger.info("Loading canonical dataset from: %s", features_path)
    sessions = pd.read_parquet(features_path)
    logger.info(
        "Loaded %d sessions across %d unique participants.",
        len(sessions),
        sessions["ParticipantID"].nunique(),
    )

    targets_to_run = (
        [TARGET_DURATION, TARGET_ENERGY]
        if args.target == "all"
        else [args.target]
    )

    for target in targets_to_run:
        target_dir = (
            DURATION_MODELING_DIR
            if target == TARGET_DURATION
            else ENERGY_MODELING_DIR
        )
        logger.info("=" * 70)
        logger.info("STARTING EXPERIMENT: Target='%s'", target)
        logger.info("Artifact destination: %s", target_dir)
        logger.info("=" * 70)

        perf_df, pred_df = run_experiment(
            sessions=sessions,
            target=target,
            output_dir=target_dir,
            saved=args.saved,
        )

        logger.info(
            "Target '%s' completed successfully. Performance shape: %s, Predictions shape: %s",
            target,
            perf_df.shape,
            pred_df.shape,
        )

    logger.info("=" * 70)
    logger.info("ALL BENCHMARK EXPERIMENTS COMPLETED SUCCESSFULLY.")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())

