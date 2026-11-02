"""Decision support module for TARANG — EV Charging Intelligence System.

Provides high-level session advisory generation, smart-charging flexibility analysis,
occupancy timing projections, and operational reporting.
"""

from __future__ import annotations

from tarang.decision_support.engine import (
    DecisionSupportEngine,
    SessionAdvisory,
    calculate_projected_unplug_time,
    generate_batch_advisories,
    generate_session_advisory,
    is_prediction_valid,
)
from tarang.decision_support.flexibility import (
    REQUIRED_FLEXIBILITY_COLUMNS,
    analyze_charging_flexibility,
)
from tarang.decision_support.reporting import (
    generate_decision_support_markdown_report,
    generate_flexibility_summary_table,
    generate_prediction_summary_table,
    save_decision_support_artifacts,
)

__all__ = [
    "DecisionSupportEngine",
    "SessionAdvisory",
    "calculate_projected_unplug_time",
    "generate_batch_advisories",
    "generate_session_advisory",
    "is_prediction_valid",
    "REQUIRED_FLEXIBILITY_COLUMNS",
    "analyze_charging_flexibility",
    "generate_decision_support_markdown_report",
    "generate_flexibility_summary_table",
    "generate_prediction_summary_table",
    "save_decision_support_artifacts",
]
