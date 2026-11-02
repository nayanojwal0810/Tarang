"""TARANG visualization package.

Provides plotting routines for exploratory data analysis and reporting.
"""

from tarang.visualization.eda_plots import (
    plot_charging_flexibility,
    plot_data_availability,
    plot_duration_by_start_hour,
    plot_energy_by_start_hour,
    run_all_eda_visualizations,
)

__all__ = [
    "plot_data_availability",
    "plot_duration_by_start_hour",
    "plot_energy_by_start_hour",
    "plot_charging_flexibility",
    "run_all_eda_visualizations",
]

