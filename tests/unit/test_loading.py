"""Unit tests for data loading and schema validation in TARANG.

Covers:
1. Correct workbook and sheet loading.
34. Empty input handling.
35. Clear invalid-input errors.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import pytest

from tarang.data.loading import inspect_workbook_metadata, load_dataset
from tarang.data.validation import (
    validate_canonical_dataset_schema,
    validate_raw_selected_columns,
    validate_workbook_schema,
)


def test_01_load_dataset_missing_file_raises_filenotfound() -> None:
    """Loading a non-existent workbook must raise FileNotFoundError with a clear path."""
    with pytest.raises(FileNotFoundError) as exc_info:
        load_dataset("non_existent_file_path_12345.xlsx")
    assert "Dataset file not found at" in str(exc_info.value)


def test_02_inspect_metadata_missing_file_raises_filenotfound() -> None:
    """Inspecting non-existent workbook metadata must raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        inspect_workbook_metadata("missing_file.xlsx")


def test_03_validate_workbook_schema_success_and_failure() -> None:
    """validate_workbook_schema must succeed when columns exist and raise ValueError with details when missing."""
    df = pd.DataFrame({"A": [1], "B": [2], "C": [3]})
    # Passes
    validate_workbook_schema(df, ["A", "B"], "TestWorkbook")

    # Fails with clear error message
    with pytest.raises(ValueError) as exc_info:
        validate_workbook_schema(df, ["A", "D", "E"], "TestWorkbook")
    msg = str(exc_info.value)
    assert "Schema validation failed for TestWorkbook" in msg
    assert "Missing required column(s): ['D', 'E']" in msg
    assert "Present columns: ['A', 'B', 'C']" in msg


def test_04_validate_raw_selected_columns_missing_raises_error() -> None:
    """validate_raw_selected_columns raises descriptive error if raw fields are missing."""
    df_incomplete = pd.DataFrame({"StartTime": ["2017-05-01"], "StopTime": ["2017-05-01"]})
    with pytest.raises(ValueError) as exc_info:
        validate_raw_selected_columns(df_incomplete)
    assert "Missing required column(s)" in str(exc_info.value)


def test_05_validate_canonical_dataset_schema_missing_raises_error() -> None:
    """validate_canonical_dataset_schema raises descriptive error if canonical fields are missing."""
    df_incomplete = pd.DataFrame({"Duration": [2.5]})
    with pytest.raises(ValueError) as exc_info:
        validate_canonical_dataset_schema(df_incomplete)
    assert "Canonical Processed Features Dataset" in str(exc_info.value)
