"""Data audit facade for TARANG — EV Charging Intelligence System.

This module re-exports loading, validation, quality inspection, screening,
and reporting routines from specialized submodules for backwards compatibility.
"""

from __future__ import annotations

from tarang.config import (
    REQUIRED_META_COLUMNS,
    REQUIRED_TX_COLUMNS,
    UK_BANK_HOLIDAYS_REFERENCE as UK_BANK_HOLIDAYS_2017_2018,
)
from tarang.data.audit_reporting import (
    generate_markdown_report,
    generate_metadata_quality_summary,
    generate_participant_reconciliation,
    generate_transaction_quality_summary,
    run_full_audit,
)
from tarang.data.loading import (
    inspect_workbook_metadata,
    load_dataset,
    load_raw_metadata,
    load_raw_transactions,
)
from tarang.data.quality_audit import (
    audit_energy_quality,
    audit_feature_feasibility,
    audit_metadata_row_level,
    calculate_missingness,
    compute_simple_shift_overlaps,
    detect_duplicates,
    detect_session_overlaps,
    find_charging_duration_discrepancies,
    investigate_unattributed_sessions,
    parse_utc_timestamps,
)
from tarang.data.screening import (
    screen_participant_history,
    screen_participant_sessions,
)
from tarang.data.validation import (
    validate_raw_selected_columns,
    validate_workbook_schema,
)

__all__ = [
    "UK_BANK_HOLIDAYS_2017_2018",
    "REQUIRED_TX_COLUMNS",
    "REQUIRED_META_COLUMNS",
    "validate_workbook_schema",
    "validate_raw_selected_columns",
    "load_dataset",
    "load_raw_transactions",
    "load_raw_metadata",
    "inspect_workbook_metadata",
    "calculate_missingness",
    "detect_duplicates",
    "parse_utc_timestamps",
    "detect_session_overlaps",
    "compute_simple_shift_overlaps",
    "investigate_unattributed_sessions",
    "audit_metadata_row_level",
    "audit_energy_quality",
    "find_charging_duration_discrepancies",
    "screen_participant_sessions",
    "screen_participant_history",
    "audit_feature_feasibility",
    "generate_transaction_quality_summary",
    "generate_metadata_quality_summary",
    "generate_participant_reconciliation",
    "generate_markdown_report",
    "run_full_audit",
]

if __name__ == "__main__":
    run_full_audit()
