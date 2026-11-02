"""TARANG analysis package.

Provides exploratory data analysis calculations and statistics.
"""

from tarang.analysis.exploratory_data_analysis import (
    calculate_charging_flexibility,
    calculate_data_availability,
    generate_eda_markdown_summary,
)

__all__ = [
    "calculate_data_availability",
    "calculate_charging_flexibility",
    "generate_eda_markdown_summary",
]

