"""TARANG data module.

Provides data loading, schema validation, quality auditing, participant screening,
and pipeline orchestration functions.
"""

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
from tarang.data.preparation import run_data_preparation
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
    screen_and_prepare_dataset,
    screen_participant_history,
    screen_participant_sessions,
)
from tarang.data.validation import (
    validate_canonical_dataset_schema,
    validate_raw_selected_columns,
    validate_workbook_schema,
)

__all__ = [
    "load_dataset",
    "load_raw_transactions",
    "load_raw_metadata",
    "inspect_workbook_metadata",
    "validate_workbook_schema",
    "validate_raw_selected_columns",
    "validate_canonical_dataset_schema",
    "calculate_missingness",
    "detect_duplicates",
    "parse_utc_timestamps",
    "detect_session_overlaps",
    "compute_simple_shift_overlaps",
    "investigate_unattributed_sessions",
    "audit_metadata_row_level",
    "audit_energy_quality",
    "find_charging_duration_discrepancies",
    "audit_feature_feasibility",
    "screen_and_prepare_dataset",
    "screen_participant_sessions",
    "screen_participant_history",
    "run_data_preparation",
    "generate_transaction_quality_summary",
    "generate_metadata_quality_summary",
    "generate_participant_reconciliation",
    "generate_markdown_report",
    "run_full_audit",
]
