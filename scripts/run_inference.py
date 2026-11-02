"""CLI for TARANG batch inference pipeline.

Executes batch predictions for charging duration or consumed energy on already-engineered
feature datasets using pre-trained Stage 2 benchmark models.
Preserves all input columns and row alignment, appending the model's raw numeric predictions.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
import pandas as pd

from tarang.config import MODEL_NAMES, TARGET_DURATION, TARGET_ENERGY
from tarang.inference.engine import predict_batch
from tarang.inference.loading import normalize_target_name

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("scripts.run_inference")


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run batch inference with pre-trained TARANG models."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to input dataset (.parquet or .csv) containing the 13 canonical predictors.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Path to output predictions file (.parquet or .csv).",
    )
    parser.add_argument(
        "--target",
        "-t",
        type=str,
        required=True,
        help=f"Prediction target: '{TARGET_DURATION}' or '{TARGET_ENERGY}' (case-insensitive).",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="HistGradientBoostingRegressor",
        help=(
            "Model architecture to use for prediction. "
            f"Defaults to 'HistGradientBoostingRegressor'. Supported: {MODEL_NAMES}."
        ),
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Optional directory containing pre-trained model artifacts.",
    )
    return parser.parse_args(args)


def main(argv: list[str] | None = None) -> int:
    """Execute batch inference workflow."""
    args = parse_args(argv)

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.is_file():
        logger.error("Input file does not exist: %s", input_path)
        return 1

    try:
        canonical_target = normalize_target_name(args.target)
    except ValueError as exc:
        logger.error("Invalid target specified: %s", exc)
        return 1

    logger.info("Loading input data from: %s", input_path)
    try:
        if input_path.suffix.lower() == ".parquet":
            input_df = pd.read_parquet(input_path)
        elif input_path.suffix.lower() in [".csv", ".txt"]:
            input_df = pd.read_csv(input_path)
        else:
            logger.error(
                "Unsupported input file format '%s'. Must be .parquet or .csv.",
                input_path.suffix,
            )
            return 1
    except Exception as exc:
        logger.error("Failed to read input file: %s", exc)
        return 1

    logger.info(
        "Running batch inference for target '%s' with model '%s' on %d samples...",
        canonical_target,
        args.model,
        len(input_df),
    )

    models_dir = Path(args.models_dir) if args.models_dir else None

    try:
        preds = predict_batch(
            features_df=input_df,
            target=canonical_target,
            model_name=args.model,
            models_dir=models_dir,
        )
    except Exception as exc:
        logger.error("Inference execution failed: %s", exc)
        return 1

    # Preserve input DataFrame and append prediction column
    output_df = input_df.copy()
    pred_col_name = f"predicted_{canonical_target}"
    output_df[pred_col_name] = preds

    logger.info("Writing %d predictions to: %s", len(output_df), output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if output_path.suffix.lower() == ".parquet":
            output_df.to_parquet(output_path, index=False)
        elif output_path.suffix.lower() in [".csv", ".txt"]:
            output_df.to_csv(output_path, index=False)
        else:
            # Default to CSV if unknown extension
            output_df.to_csv(output_path, index=False)
    except Exception as exc:
        logger.error("Failed to write output file: %s", exc)
        return 1

    logger.info("Batch inference completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

