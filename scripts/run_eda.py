"""Entry-point script to execute Exploratory Data Analysis (EDA) on the processed dataset.

Calculates participant availability and charging flexibility statistics,
generates publication-grade figures, and produces reports/eda/eda_summary.md.
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys
import pandas as pd

from tarang.analysis.exploratory_data_analysis import (
    calculate_charging_flexibility,
    calculate_data_availability,
    generate_eda_markdown_summary,
)
from tarang.config import EDA_REPORTS_DIR, PROCESSED_FEATURES_FILE
from tarang.data.preparation import run_data_preparation
from tarang.visualization.eda_plots import run_all_eda_visualizations

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("scripts.run_eda")


def main() -> int:
    """Run full EDA pipeline."""
    logger.info("Starting TARANG Exploratory Data Analysis...")
    try:
        if not PROCESSED_FEATURES_FILE.is_file():
            logger.info("Canonical features file not found. Running data preparation first...")
            run_data_preparation()

        logger.info("Loading processed dataset from: %s", PROCESSED_FEATURES_FILE)
        df = pd.read_parquet(PROCESSED_FEATURES_FILE)
        logger.info("Loaded %d sessions with %d columns.", len(df), len(df.columns))

        EDA_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        logger.info("Calculating participant data availability metrics...")
        availability_df, availability_stats = calculate_data_availability(df)

        logger.info("Calculating smart-charging flexibility metrics...")
        flexibility_stats = calculate_charging_flexibility(df)

        logger.info("Generating EDA figures in: %s", EDA_REPORTS_DIR)
        run_all_eda_visualizations(df, output_dir=EDA_REPORTS_DIR)

        logger.info("Generating EDA Markdown summary...")
        report_md = generate_eda_markdown_summary(availability_stats, flexibility_stats, df)
        report_file = EDA_REPORTS_DIR / "eda_summary.md"
        report_file.write_text(report_md, encoding="utf-8")
        logger.info("EDA summary saved to: %s", report_file)

        logger.info("Exploratory Data Analysis completed successfully.")
        return 0
    except Exception as exc:
        logger.error("EDA failed with error: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

