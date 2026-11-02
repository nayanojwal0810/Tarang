"""Exploratory Data Analysis (EDA) module for TARANG.

Provides computational routines for participant availability, bivariate arrival-duration
and arrival-energy distributions, and smart-charging flexibility ratios.
Calculations are strictly separated from chart rendering routines.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_data_availability(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Compute per-participant data availability metrics and aggregate distribution statistics.

    Participant-level calculations:
    - span_days: (StopTime.max() - StartTime.min()).days
    - session_count: Total retained sessions
    - weekly_frequency: session_count / span_days * 7.0
    - mean_duration_hours: Mean of raw plugged-in duration (StopTime - StartTime in hours)

    Aggregate statistics use population standard deviation (ddof=0) without pre-rounding.

    Args:
        df: Canonical features DataFrame.

    Returns:
        Tuple of:
        - participant_summary_df: DataFrame indexed by participant with unrounded metrics.
        - aggregate_stats: Dictionary of mean, median, std (ddof=0), min, max, p25, p75, p90.
    """
    group_col = "id" if "id" in df.columns else "ParticipantID"
    if df.empty or group_col not in df.columns or "StartTime" not in df.columns or "StopTime" not in df.columns:
        empty_df = pd.DataFrame(columns=[group_col, "span_days", "session_count", "weekly_frequency", "mean_duration_hours"])
        empty_stats = {
            "participant_count": 0,
            "span_days": {},
            "session_count": {},
            "weekly_frequency": {},
            "mean_duration_hours": {},
        }
        return empty_df, empty_stats

    records = []
    for pid, group in df.groupby(group_col, sort=True):
        start_min = group["StartTime"].min()
        stop_max = group["StopTime"].max()
        span_days = (stop_max - start_min).days
        n_sessions = len(group.index)
        weekly_freq = n_sessions / span_days * 7.0
        durations = (group["StopTime"] - group["StartTime"]).dt.total_seconds() / 3600.0
        mean_dur = float(durations.mean())

        records.append({
            group_col: pid,
            "span_days": span_days,
            "session_count": n_sessions,
            "weekly_frequency": weekly_freq,
            "mean_duration_hours": mean_dur,
        })

    summary_df = pd.DataFrame(records)

    aggregate_stats: Dict[str, Any] = {
        "participant_count": len(summary_df),
        "total_sessions": int(summary_df["session_count"].sum()),
    }

    for col in ["span_days", "session_count", "weekly_frequency", "mean_duration_hours"]:
        vals = summary_df[col].to_numpy(dtype=float)
        aggregate_stats[col] = {
            "mean": round(float(np.mean(vals)), 4),
            "std": round(float(np.std(vals, ddof=0)), 4),
            "median": round(float(np.median(vals)), 4),
            "min": round(float(np.min(vals)), 4),
            "max": round(float(np.max(vals)), 4),
            "p25": round(float(np.quantile(vals, 0.25)), 4),
            "p75": round(float(np.quantile(vals, 0.75)), 4),
            "p90": round(float(np.quantile(vals, 0.90)), 4),
        }

    return summary_df, aggregate_stats


def calculate_charging_flexibility(
    df: pd.DataFrame,
    n_bins: int = 25,
) -> Dict[str, Any]:
    """Calculate charging-to-plugged-in duration ratios classified by same-day vs overnight sessions.

    Session classification:
    - Same-day: Session starts and ends on the same local date in Europe/London.
    - Overnight: Session spans across different local dates in Europe/London.

    Ratio formula:
    (ChargingDuration / 60.0) / ((StopTime - StartTime) in hours)

    Distribution uses n_bins evenly spaced across [0.0, 1.0], normalized to percentage PMF.

    Args:
        df: Canonical features DataFrame.
        n_bins: Number of histogram bins over the [0.0, 1.0] interval. Defaults to 25.

    Returns:
        Dictionary containing distributions, counts, and summary statistics.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    if df.empty or "ChargingDuration" not in df.columns or "StartTime" not in df.columns or "StopTime" not in df.columns:
        return {
            "bin_edges": bins.tolist(),
            "same_day_pct": [0.0] * n_bins,
            "overnight_pct": [0.0] * n_bins,
            "same_day_total": 0,
            "overnight_total": 0,
            "same_day_valid": 0,
            "overnight_valid": 0,
            "same_day_in_range": 0,
            "overnight_in_range": 0,
            "same_day_count": 0,
            "overnight_count": 0,
            "same_day_median_ratio": np.nan,
            "overnight_median_ratio": np.nan,
            "same_day_mean_ratio": np.nan,
            "overnight_mean_ratio": np.nan,
        }

    # Convert to Europe/London local time for calendar date classification
    start_local = pd.to_datetime(df["StartTime"]).dt.tz_convert("Europe/London")
    stop_local = pd.to_datetime(df["StopTime"]).dt.tz_convert("Europe/London")
    is_same_day = start_local.dt.date == stop_local.dt.date

    same_day_total = int(is_same_day.sum())
    overnight_total = int((~is_same_day).sum())

    dur_plug = (stop_local - start_local).dt.total_seconds() / 3600.0
    dur_ch = pd.to_numeric(df["ChargingDuration"], errors="coerce") / 60.0

    valid_mask = dur_ch.notna() & np.isfinite(dur_ch) & (dur_ch >= 0.0) & np.isfinite(dur_plug) & (dur_plug > 0)
    ratio = dur_ch[valid_mask] / dur_plug[valid_mask]

    is_same_day_valid = is_same_day[valid_mask]

    same_ratios = ratio[is_same_day_valid]
    over_ratios = ratio[~is_same_day_valid]

    same_valid_count = len(same_ratios)
    over_valid_count = len(over_ratios)

    same_hist, _ = np.histogram(same_ratios[(same_ratios >= 0.0) & (same_ratios <= 1.0)], bins=bins)
    over_hist, _ = np.histogram(over_ratios[(over_ratios >= 0.0) & (over_ratios <= 1.0)], bins=bins)

    same_in_range = int(same_hist.sum())
    over_in_range = int(over_hist.sum())

    same_pct = ((same_hist / same_in_range) * 100.0).tolist() if same_in_range > 0 else [0.0] * n_bins
    over_pct = ((over_hist / over_in_range) * 100.0).tolist() if over_in_range > 0 else [0.0] * n_bins

    return {
        "bin_edges": bins.tolist(),
        "same_day_pct": same_pct,
        "overnight_pct": over_pct,
        "same_day_total": same_day_total,
        "overnight_total": overnight_total,
        "same_day_valid": same_valid_count,
        "overnight_valid": over_valid_count,
        "same_day_in_range": same_in_range,
        "overnight_in_range": over_in_range,
        "same_day_count": same_in_range,  # Evaluated in-range ratio count
        "overnight_count": over_in_range,
        "same_day_median_ratio": round(float(same_ratios.median()), 4) if not same_ratios.empty else np.nan,
        "overnight_median_ratio": round(float(over_ratios.median()), 4) if not over_ratios.empty else np.nan,
        "same_day_mean_ratio": round(float(same_ratios.mean()), 4) if not same_ratios.empty else np.nan,
        "overnight_mean_ratio": round(float(over_ratios.mean()), 4) if not over_ratios.empty else np.nan,
    }


def generate_eda_markdown_summary(
    availability_stats: Dict[str, Any],
    flexibility_stats: Dict[str, Any],
    df: pd.DataFrame,
) -> str:
    """Generate Markdown text summarizing key EDA findings.

    Args:
        availability_stats: Output from calculate_data_availability.
        flexibility_stats: Output from calculate_charging_flexibility.
        df: Canonical features DataFrame.

    Returns:
        Structured Markdown summary string.
    """
    n_participants = availability_stats.get("participant_count", 0)
    n_sessions = len(df)

    raw_dur = (df["StopTime"] - df["StartTime"]).dt.total_seconds() / 3600.0 if not df.empty else pd.Series(dtype=float)
    energy = df["ConsumedkWh"].dropna() if "ConsumedkWh" in df.columns else pd.Series(dtype=float)

    start_local = pd.to_datetime(df["StartTime"]).dt.tz_convert("Europe/London") if not df.empty else pd.Series(dtype=float)
    arrival_hours = start_local.dt.hour if not df.empty else pd.Series(dtype=int)
    if not arrival_hours.empty and len(arrival_hours.mode()) > 0:
        top_arrival_hour_str = f"{int(arrival_hours.mode().iloc[0]):02d}:00 Local UK Time (`Europe/London`)"
    else:
        top_arrival_hour_str = "Unavailable"

    formula_str = "$$\\text{Flexibility Ratio} = \\frac{\\text{ChargingDuration} / 60}{(\\text{StopTime} - \\text{StartTime})_{\\text{hours}}}$$"

    report = f"""# TARANG — Exploratory Data Analysis (EDA) Summary

**Project**: TARANG — EV Charging Intelligence System  
**Dataset**: Processed Electric Nation Modeling Dataset (`features.parquet`)  
**Scope**: {n_participants} screened participants, {n_sessions:,} charging sessions  
**Generated Outputs**:
- `reports/eda/data_availability.png`: 4-panel participant availability distributions
- `reports/eda/duration_by_start_hour.png`: 2D histogram with marginal distributions (72h cutoff)
- `reports/eda/energy_by_start_hour.png`: 2D histogram with marginal distributions (75 kWh cutoff)
- `reports/eda/charging_flexibility.png`: Charging flexibility ratio (same-day vs overnight)

---

## 1. Participant Data Availability and Frequency

The screened cohort comprises **{n_participants} participants** monitored across the trial window.

| Availability Dimension | Mean | Median | Std Dev (pop) | Min | 25% | 75% | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Observation Span (Days) | {availability_stats['span_days']['mean']:.2f} | {availability_stats['span_days']['median']:.1f} | {availability_stats['span_days']['std']:.2f} | {availability_stats['span_days']['min']:.0f} | {availability_stats['span_days']['p25']:.1f} | {availability_stats['span_days']['p75']:.1f} | {availability_stats['span_days']['max']:.0f} |
| Sessions per Participant | {availability_stats['session_count']['mean']:.2f} | {availability_stats['session_count']['median']:.1f} | {availability_stats['session_count']['std']:.2f} | {availability_stats['session_count']['min']:.0f} | {availability_stats['session_count']['p25']:.1f} | {availability_stats['session_count']['p75']:.1f} | {availability_stats['session_count']['max']:.0f} |
| Weekly Frequency (Sessions/Week) | {availability_stats['weekly_frequency']['mean']:.2f} | {availability_stats['weekly_frequency']['median']:.2f} | {availability_stats['weekly_frequency']['std']:.2f} | {availability_stats['weekly_frequency']['min']:.2f} | {availability_stats['weekly_frequency']['p25']:.2f} | {availability_stats['weekly_frequency']['p75']:.2f} | {availability_stats['weekly_frequency']['max']:.2f} |
| Mean Plugged-In Duration (Hours) | {availability_stats['mean_duration_hours']['mean']:.2f} | {availability_stats['mean_duration_hours']['median']:.2f} | {availability_stats['mean_duration_hours']['std']:.2f} | {availability_stats['mean_duration_hours']['min']:.2f} | {availability_stats['mean_duration_hours']['p25']:.2f} | {availability_stats['mean_duration_hours']['p75']:.2f} | {availability_stats['mean_duration_hours']['max']:.2f} |

**Key Findings:**
- Average observation span per vehicle: {availability_stats['span_days']['mean']:.2f} days (median: {availability_stats['span_days']['median']:.1f} days).
- Average session frequency: {availability_stats['weekly_frequency']['mean']:.2f} sessions/week (population std: {availability_stats['weekly_frequency']['std']:.2f}).
- Average session duration across participants: {availability_stats['mean_duration_hours']['mean']:.2f} hours (90th percentile: {availability_stats['mean_duration_hours']['p90']:.2f} hours).

---

## 2. Arrival Timing and Connection Duration Distributions

- **Modal Local Arrival Hour**: {top_arrival_hour_str}
- **Overall Raw Duration**:
  - Median: {raw_dur.median():.2f} hours
  - Mean: {raw_dur.mean():.2f} hours
  - 75th Percentile: {raw_dur.quantile(0.75):.2f} hours
  - 95th Percentile: {raw_dur.quantile(0.95):.2f} hours
- **Distribution Output**: The joint distribution of connection durations across local UK arrival hours is visualized in `duration_by_start_hour.png` (binned in 0.5-hour increments up to the 72-hour cutoff).

---

## 3. Energy Consumption Profiles

- **Overall Energy Consumption**:
  - Median: {energy.median():.2f} kWh
  - Mean: {energy.mean():.2f} kWh
  - 75th Percentile: {energy.quantile(0.75):.2f} kWh
  - 95th Percentile: {energy.quantile(0.95):.2f} kWh
- **Distribution Output**: The joint distribution of energy consumption across local UK arrival hours is visualized in `energy_by_start_hour.png` (binned in 0.5-hour arrival and 0.5-kWh energy increments up to the 75-kWh cutoff).

---

## 4. Smart-Charging Flexibility Ratio

The charging flexibility ratio is defined as:

{formula_str}

Evaluated using local UK dates (`Europe/London`):

| Session Category | Total Sessions | Evaluated in [0, 1] | Mean Ratio | Median Ratio |
| :--- | :---: | :---: | :---: | :---: |
| **Same-Day Sessions** | {flexibility_stats['same_day_total']:,} | {flexibility_stats['same_day_in_range']:,} | {flexibility_stats['same_day_mean_ratio']:.3f} | {flexibility_stats['same_day_median_ratio']:.3f} |
| **Overnight Sessions** | {flexibility_stats['overnight_total']:,} | {flexibility_stats['overnight_in_range']:,} | {flexibility_stats['overnight_mean_ratio']:.3f} | {flexibility_stats['overnight_median_ratio']:.3f} |

**Key Findings:**
- Same-day sessions exhibit a median flexibility ratio of **{flexibility_stats['same_day_median_ratio']:.3f}** (mean: **{flexibility_stats['same_day_mean_ratio']:.3f}**).
- Overnight sessions exhibit a median flexibility ratio of **{flexibility_stats['overnight_median_ratio']:.3f}** (mean: **{flexibility_stats['overnight_mean_ratio']:.3f}**).
- The full ratio distributions across 25 uniform bins in [0, 1] are rendered in `charging_flexibility.png`.
"""
    return report
