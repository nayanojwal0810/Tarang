"""Data schema and structural validation module for TARANG.

Provides functions to validate input workbooks, raw transaction tables, and
modeling dataset schemas against authoritative data contracts.
"""

from __future__ import annotations

from typing import List
import pandas as pd

from tarang.config import (
    CANONICAL_COLUMNS,
    RAW_SELECTED_COLUMNS,
    REQUIRED_META_COLUMNS,
    REQUIRED_TX_COLUMNS,
)


def validate_workbook_schema(
    df: pd.DataFrame,
    required_columns: List[str],
    workbook_name: str,
) -> None:
    """Validate that required columns exist in the DataFrame.

    Args:
        df: Input DataFrame to inspect.
        required_columns: List of column names that must be present.
        workbook_name: Descriptive name of the workbook or sheet for error messages.

    Raises:
        ValueError: If any required columns are missing, detailing the missing and present columns.
    """
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"Schema validation failed for {workbook_name}: Missing required column(s): {missing}. "
            f"Present columns: {list(df.columns)}."
        )


def validate_raw_selected_columns(df: pd.DataFrame) -> None:
    """Validate that the 9 required raw fields are present before research screening.

    Args:
        df: Transactions DataFrame.

    Raises:
        ValueError: If any required raw fields are missing.
    """
    validate_workbook_schema(df, RAW_SELECTED_COLUMNS, "Selected Transaction Fields")


def validate_canonical_dataset_schema(df: pd.DataFrame) -> None:
    """Validate that the canonical processed research dataset contains all expected columns.

    Args:
        df: Processed DataFrame.

    Raises:
        ValueError: If any canonical columns are missing.
    """
    validate_workbook_schema(df, CANONICAL_COLUMNS, "Canonical Processed Features Dataset")

