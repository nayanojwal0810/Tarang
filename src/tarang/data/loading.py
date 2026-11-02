"""Data loading module for TARANG.

Provides safe utilities to load raw Excel workbooks, parse sheets into DataFrames,
and inspect workbook file metadata.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union
import openpyxl
import pandas as pd

from tarang.config import (
    RAW_META_FILE,
    RAW_META_SHEET,
    RAW_TX_FILE,
    RAW_TX_SHEET,
)

logger = logging.getLogger(__name__)


def load_dataset(
    filepath: Union[str, Path],
    sheet_name: Optional[Union[str, int]] = 0,
) -> pd.DataFrame:
    """Load an Excel workbook sheet into a pandas DataFrame safely.

    Args:
        filepath: Path to the Excel file.
        sheet_name: Sheet name or index to read. Defaults to 0.

    Returns:
        Loaded DataFrame.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If reading fails or sheet does not exist.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found at: {path}")

    try:
        df = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
        return df
    except Exception as exc:
        raise ValueError(f"Failed to load Excel sheet '{sheet_name}' from {path}: {exc}") from exc


def load_raw_transactions(
    filepath: Optional[Union[str, Path]] = None,
    sheet_name: str = RAW_TX_SHEET,
) -> pd.DataFrame:
    """Load the raw Electric Nation CrowdCharge Transactions sheet.

    Args:
        filepath: Optional path to CrowdCharge Transactions.xlsx. Defaults to config.RAW_TX_FILE.
        sheet_name: Sheet name. Defaults to config.RAW_TX_SHEET ('Transaction Data').

    Returns:
        DataFrame containing raw transactions.
    """
    target = Path(filepath) if filepath is not None else RAW_TX_FILE
    logger.info("Loading raw transactions from: %s (sheet: %s)", target, sheet_name)
    return load_dataset(target, sheet_name=sheet_name)


def load_raw_metadata(
    filepath: Optional[Union[str, Path]] = None,
    sheet_name: str = RAW_META_SHEET,
) -> pd.DataFrame:
    """Load the raw Electric Nation ChargerInstall metadata sheet.

    Args:
        filepath: Optional path to ChargerInstall.xlsx. Defaults to config.RAW_META_FILE.
        sheet_name: Sheet name. Defaults to config.RAW_META_SHEET ('20181126_ChargerInstall_DriveEl').

    Returns:
        DataFrame containing raw charger and vehicle installation metadata.
    """
    target = Path(filepath) if filepath is not None else RAW_META_FILE
    logger.info("Loading raw metadata from: %s (sheet: %s)", target, sheet_name)
    return load_dataset(target, sheet_name=sheet_name)


def inspect_workbook_metadata(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Inspect file metadata and sheet dimensions of an Excel workbook.

    Args:
        filepath: Path to the Excel workbook.

    Returns:
        Dictionary containing file size, sheet names, and dimensions for each sheet.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Workbook not found at: {path}")

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    sheet_info = {}
    for sheet_name in wb.sheetnames:
        sheet = wb[sheet_name]
        sheet_info[sheet_name] = {
            "max_row": sheet.max_row,
            "max_column": sheet.max_column,
        }
    wb.close()

    return {
        "file_path": str(path),
        "file_name": path.name,
        "size_bytes": path.stat().st_size,
        "sheet_names": list(sheet_info.keys()),
        "sheet_dimensions": sheet_info,
    }
