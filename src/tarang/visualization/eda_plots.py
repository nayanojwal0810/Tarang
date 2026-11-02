"""Visualization module for Exploratory Data Analysis (EDA) in TARANG.

Provides clean, publication-grade plotting functions for:
1. Participant data availability (4-panel distribution).
2. 2D arrival hour vs duration joint histogram with marginals (0.5h bins, Europe/London).
3. 2D arrival hour vs energy joint histogram with marginals (0.5h, 0.5 kWh bins, Europe/London).
4. Smart-charging flexibility ratio (same-day vs overnight).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tarang.analysis.exploratory_data_analysis import (
    calculate_charging_flexibility,
    calculate_data_availability,
)

logger = logging.getLogger(__name__)

# TARANG Visual Styling Palette
COLOR_PRIMARY = "#1E3A8A"      # Deep Navy
COLOR_SECONDARY = "#0D9488"    # Teal
COLOR_ACCENT = "#D97706"       # Amber
COLOR_MUTED = "#64748B"        # Slate Gray
COLOR_LIGHT_BG = "#F8FAFC"     # Off-white / light slate


def set_tarang_style() -> None:
    """Set standard plotting aesthetics."""
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    plt.rcParams["axes.edgecolor"] = "#CBD5E1"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["grid.color"] = "#E2E8F0"
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.alpha"] = 0.6


def plot_data_availability(
    availability_df: pd.DataFrame,
    output_path: Union[str, Path],
) -> Path:
    """Generate 4-panel participant data availability figure with 26-bin distributions.

    Subplots:
    1. Observation span in days (26 bins)
    2. Sessions per participant (26 bins)
    3. Weekly session frequency (26 bins)
    4. Mean plugged-in session duration (26 bins over [0, 72] hours)

    Args:
        availability_df: DataFrame from calculate_data_availability.
        output_path: File path to save the generated PNG.

    Returns:
        Path to saved PNG file.
    """
    set_tarang_style()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.patch.set_facecolor("white")

    # Panel 1: Observation Span (26 bins)
    ax = axes[0, 0]
    ax.hist(availability_df["span_days"], bins=26, color=COLOR_PRIMARY, edgecolor="white", alpha=0.85)
    ax.set_title("(a) Observation Span per Participant", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Data Availability Span (Days)", fontsize=10)
    ax.set_ylabel("Vehicles", fontsize=10)
    ax.set_xlim(0, None)
    ax.grid(True)

    # Panel 2: Sessions per Participant (26 bins)
    ax = axes[0, 1]
    ax.hist(availability_df["session_count"], bins=26, color=COLOR_SECONDARY, edgecolor="white", alpha=0.85)
    ax.set_title("(b) Total Sessions per Participant", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Sessions", fontsize=10)
    ax.set_ylabel("Vehicles", fontsize=10)
    ax.set_xlim(0, None)
    ax.grid(True)

    # Panel 3: Weekly Frequency (26 bins)
    ax = axes[1, 0]
    ax.hist(availability_df["weekly_frequency"], bins=26, color=COLOR_ACCENT, edgecolor="white", alpha=0.85)
    ax.set_title("(c) Mean Weekly Sessions", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Mean Weekly Sessions", fontsize=10)
    ax.set_ylabel("Vehicles", fontsize=10)
    ax.set_xlim(0, None)
    ax.grid(True)

    # Panel 4: Mean Session Duration (26 bins on [0, 72] hours)
    ax = axes[1, 1]
    duration_bins = np.linspace(0.0, 72.0, 26)
    ax.hist(availability_df["mean_duration_hours"], bins=duration_bins, color="#6366F1", edgecolor="white", alpha=0.85)
    ax.set_title("(d) Mean Session Duration", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Mean Session Duration (Hours)", fontsize=10)
    ax.set_ylabel("Vehicles", fontsize=10)
    ax.set_xlim(0, None)
    ax.grid(True)

    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved data availability plot to: %s", path)
    return path


def plot_duration_by_start_hour(
    df: pd.DataFrame,
    output_path: Union[str, Path],
    max_duration: float = 72.0,
) -> Path:
    """Generate 2D duration vs local arrival hour joint histogram with marginals.

    Converts timestamps to Europe/London local time for arrival hour calculation.
    Uses half-hour bins for arrival hour [0, 24] and duration [0, 72].

    Args:
        df: Canonical features DataFrame.
        output_path: File path to save the generated PNG.
        max_duration: Display cutoff in hours. Defaults to 72.0.

    Returns:
        Path to saved PNG file.
    """
    set_tarang_style()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Local UK time conversion (Europe/London) for EDA arrival hour
    start_local = pd.to_datetime(df["StartTime"]).dt.tz_convert("Europe/London")
    stop_local = pd.to_datetime(df["StopTime"]).dt.tz_convert("Europe/London")

    start_hours = start_local.dt.hour + start_local.dt.minute / 60.0 + start_local.dt.second / 3600.0
    durations = (stop_local - start_local).dt.total_seconds() / 3600.0

    mask = (durations >= 0) & (durations <= max_duration) & (start_hours >= 0) & (start_hours <= 24)
    x = start_hours[mask]
    y = durations[mask]

    # Setup bins: 0.5h for start hour, 0.5h for duration up to cutoff
    x_bins = np.arange(0.0, 24.5, 0.5)
    y_bins = np.arange(0.0, max_duration + 0.5, 0.5)

    fig = plt.figure(figsize=(10, 8))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(4, 4, hspace=0.08, wspace=0.08)

    ax_main = fig.add_subplot(gs[1:4, 0:3])
    ax_top = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1:4, 3], sharey=ax_main)

    ax_main.hist2d(
        x, y,
        bins=[x_bins, y_bins],
        cmap="viridis",
        norm=mcolors.LogNorm(clip=True),
    )
    ax_main.set_xlabel("Start Hour (Local Time, Europe/London)", fontsize=11)
    ax_main.set_ylabel("Duration (Hours)", fontsize=11)
    ax_main.set_xlim(0, 24)
    ax_main.set_ylim(0, max_duration)
    ax_main.set_xticks(range(0, 25, 4))
    ax_main.grid(True, alpha=0.3)

    # Top marginal (Start Hour)
    ax_top.hist(x, bins=x_bins, color=COLOR_PRIMARY, edgecolor="white", alpha=0.85)
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.set_ylabel("Sessions", fontsize=9)
    ax_top.grid(True, alpha=0.3)
    ax_top.set_title("Plugged-In Duration by Local Arrival Hour", fontsize=12, fontweight="bold", pad=12)

    # Right marginal (Duration)
    ax_right.hist(y, bins=y_bins, orientation="horizontal", color=COLOR_SECONDARY, edgecolor="white", alpha=0.85)
    ax_right.tick_params(axis="y", labelleft=False)
    ax_right.set_xlabel("Sessions", fontsize=9)
    ax_right.set_xscale("log")
    ax_right.grid(True, alpha=0.3)

    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved duration by start hour plot to: %s", path)
    return path


def plot_energy_by_start_hour(
    df: pd.DataFrame,
    output_path: Union[str, Path],
    max_energy: float = 75.0,
) -> Path:
    """Generate 2D consumed energy vs local arrival hour joint histogram with marginals.

    Converts timestamps to Europe/London local time for arrival hour calculation.
    Uses half-hour bins for arrival hour [0, 24] and 0.5-kWh bins for energy [0, 75].

    Args:
        df: Canonical features DataFrame.
        output_path: File path to save the generated PNG.
        max_energy: Display cutoff in kWh. Defaults to 75.0.

    Returns:
        Path to saved PNG file.
    """
    set_tarang_style()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    start_local = pd.to_datetime(df["StartTime"]).dt.tz_convert("Europe/London")
    start_hours = start_local.dt.hour + start_local.dt.minute / 60.0 + start_local.dt.second / 3600.0
    energy = pd.to_numeric(df["ConsumedkWh"], errors="coerce")

    mask = energy.notnull() & (energy >= 0) & (energy <= max_energy) & (start_hours >= 0) & (start_hours <= 24)
    x = start_hours[mask]
    y = energy[mask]

    # Half-hour bins for arrival time, 0.5-kWh bins for energy
    x_bins = np.arange(0.0, 24.5, 0.5)
    y_bins = np.arange(0.0, max_energy + 0.5, 0.5)

    fig = plt.figure(figsize=(10, 8))
    fig.patch.set_facecolor("white")
    gs = gridspec.GridSpec(4, 4, hspace=0.08, wspace=0.08)

    ax_main = fig.add_subplot(gs[1:4, 0:3])
    ax_top = fig.add_subplot(gs[0, 0:3], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1:4, 3], sharey=ax_main)

    ax_main.hist2d(
        x, y,
        bins=[x_bins, y_bins],
        cmap="plasma",
        norm=mcolors.LogNorm(clip=True),
    )
    ax_main.set_xlabel("Start Hour (Local Time, Europe/London)", fontsize=11)
    ax_main.set_ylabel("Energy (kWh)", fontsize=11)
    ax_main.set_xlim(0, 24)
    ax_main.set_ylim(0, max_energy)
    ax_main.set_xticks(range(0, 25, 4))
    ax_main.grid(True, alpha=0.3)

    # Top marginal
    ax_top.hist(x, bins=x_bins, color=COLOR_PRIMARY, edgecolor="white", alpha=0.85)
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.set_ylabel("Sessions", fontsize=9)
    ax_top.grid(True, alpha=0.3)
    ax_top.set_title("Energy Consumption by Local Arrival Hour", fontsize=12, fontweight="bold", pad=12)

    # Right marginal
    ax_right.hist(y, bins=y_bins, orientation="horizontal", color=COLOR_ACCENT, edgecolor="white", alpha=0.85)
    ax_right.tick_params(axis="y", labelleft=False)
    ax_right.set_xlabel("Sessions", fontsize=9)
    ax_right.set_xscale("log")
    ax_right.grid(True, alpha=0.3)

    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved energy by start hour plot to: %s", path)
    return path


def plot_charging_flexibility(
    flexibility_data: Dict[str, Any],
    output_path: Union[str, Path],
) -> Path:
    """Generate smart-charging flexibility ratio distribution comparing same-day vs overnight sessions.

    Uses 25-bin histogram across [0.0, 1.0] normalized to percentage PMF.

    Args:
        flexibility_data: Dictionary output from calculate_charging_flexibility.
        output_path: File path to save the generated PNG.

    Returns:
        Path to saved PNG file.
    """
    set_tarang_style()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    bin_edges = np.array(flexibility_data["bin_edges"])
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1] - bin_edges[0]

    same_pct = flexibility_data["same_day_pct"]
    over_pct = flexibility_data["overnight_pct"]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("white")

    w = bin_width * 0.42
    n_same = flexibility_data.get("same_day_in_range", flexibility_data.get("same_day_count", 0))
    n_over = flexibility_data.get("overnight_in_range", flexibility_data.get("overnight_count", 0))

    ax.bar(
        bin_centers - w / 2, same_pct,
        width=w,
        label=f"Same-Day Sessions (n={n_same:,})",
        color=COLOR_PRIMARY,
        alpha=0.85,
        edgecolor="white",
    )
    ax.bar(
        bin_centers + w / 2, over_pct,
        width=w,
        label=f"Overnight Sessions (n={n_over:,})",
        color=COLOR_SECONDARY,
        alpha=0.85,
        edgecolor="white",
    )

    ax.set_title("Smart-Charging Flexibility Ratio: Active Charging Duration / Plugged-In Duration", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Flexibility Ratio (Charging Duration / Plugged-In Duration)", fontsize=10)
    ax.set_ylabel("Probability Mass Function (%)", fontsize=10)
    ax.set_xlim(-0.02, 1.02)
    ax.legend(frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=10)
    ax.grid(True, alpha=0.4)

    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved charging flexibility plot to: %s", path)
    return path


def run_all_eda_visualizations(
    df: pd.DataFrame,
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Path]:
    """Execute all EDA visualizations and save standard output files.

    Args:
        df: Canonical features DataFrame.
        output_dir: Directory to save generated PNGs. Defaults to 'reports/eda'.

    Returns:
        Dictionary mapping figure names to saved file paths.
    """
    out_dir = Path(output_dir) if output_dir is not None else Path("reports/eda")
    out_dir.mkdir(parents=True, exist_ok=True)

    availability_df, _ = calculate_data_availability(df)
    flexibility_data = calculate_charging_flexibility(df)

    p1 = plot_data_availability(availability_df, out_dir / "data_availability.png")
    p2 = plot_duration_by_start_hour(df, out_dir / "duration_by_start_hour.png")
    p3 = plot_energy_by_start_hour(df, out_dir / "energy_by_start_hour.png")
    p4 = plot_charging_flexibility(flexibility_data, out_dir / "charging_flexibility.png")

    return {
        "data_availability": p1,
        "duration_by_start_hour": p2,
        "energy_by_start_hour": p3,
        "charging_flexibility": p4,
    }
