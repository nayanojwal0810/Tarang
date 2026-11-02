"""Overview and landing view for TARANG Streamlit Application."""

from __future__ import annotations

import streamlit as st

from tarang.app.components import render_disclaimer, render_hero


def render_overview_view() -> None:
    """Render the main executive overview and product landing page."""
    render_hero(
        title="TARANG — EV Charging Intelligence System",
        subtitle="Empirical predictive modeling, session occupancy forecasting, and smart-charging decision support.",
        tag="INTELLIGENCE PLATFORM",
    )

    st.markdown(
        """
        ### Executive Summary

        **TARANG** is an advanced EV charging intelligence system engineered to address user-behavior
        heterogeneity in residential and destination charging environments. Built on empirical trial data
        from the **Electric Nation** project—comprising **59,520 charging sessions** across monitored UK EV drivers—TARANG
        forecasts two fundamental operational parameters as soon as a vehicle plugs into a charging station:

        1. **Connection Duration (hours)** — Total duration the charge point remains occupied by the vehicle.
        2. **Energy Consumption (kWh)** — Total electrical energy transferred during the charging session.

        By combining these dual predictions with arrival timestamps, TARANG calculates projected vehicle departure times
        and evaluates descriptive charging flexibility ratios to inform facility managers, fleet dispatchers, and smart-charging planners.
        """
    )

    st.markdown("---")

    # Workflow Navigation Cards
    st.markdown("### Operational Workflows")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            """
            <div class="tarang-card">
                <div class="tarang-card-header">
                    <span>⚡ Single-Session Advisory</span>
                    <span class="badge-neutral">Real-Time</span>
                </div>
                <p style="color:#475569; font-size:0.92rem; min-height: 4rem;">
                    Generate an instant predictive advisory for an individual arriving vehicle.
                    Specify arrival date/time, vehicle charging specifications (kW & kWh), and supply
                    the driver's prior completed charging session history.
                </p>
                <div style="margin-top: 1rem; font-size: 0.85rem; color:#64748B;">
                    &bull; Real-time duration & energy forecasts<br>
                    &bull; Projected unplug departure timestamp<br>
                    &bull; Cold-start requirement detection
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="tarang-card">
                <div class="tarang-card-header">
                    <span>📊 Batch Network Advisory</span>
                    <span class="badge-neutral">High-Throughput</span>
                </div>
                <p style="color:#475569; font-size:0.92rem; min-height: 4rem;">
                    Upload a dataset of charging sessions via CSV or Parquet to run high-throughput batch forecasting.
                    Includes automated schema validation across the 13 canonical predictors and CSV export of results.
                </p>
                <div style="margin-top: 1rem; font-size: 0.85rem; color:#64748B;">
                    &bull; High-throughput batch inference<br>
                    &bull; Summary distribution metrics & validity checks<br>
                    &bull; Descriptive smart-charging flexibility analysis
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Methodology & Feature Foundation
    st.markdown("### Predictive Modeling Foundation")

    col_feat1, col_feat2 = st.columns([1.2, 1])

    with col_feat1:
        st.markdown(
            """
            The predictive core relies on an empirical feature engineering architecture that translates session arrival conditions
            into **13 canonical predictors** across four engineering domains:

            - **Calendar Dynamics**: `weekday`, `weekend`, `holiday`, `month`.
            - **Diurnal Cyclicality**: `StartCos`, `StartSin`, `StartHourNorm` (harmonic representations of arrival hour).
            - **Driver Habit History**: `TimeSinceLastStop`, `sessionsToday`, `LastDuration`, `LastConsumedkWh`.
            - **Vehicle Hardware**: `CarKW` (onboard charger rating), `CarKWh` (battery pack capacity).

            Estimations are powered by pre-trained **Histogram-Based Gradient Boosting Regressors (`HistGradientBoostingRegressor`)**
            with absolute-error loss and early stopping, optimized to handle the non-Gaussian variance observed in EV driver behavior.
            """
        )

    with col_feat2:
        st.markdown(
            """
            <div class="metric-container" style="background:#F8FAFC; border-color:#CBD5E1;">
                <div class="metric-label">Benchmark Model Performance</div>
                <div style="margin-top:0.5rem; font-size:0.88rem; color:#334155; line-height: 1.6;">
                    <strong>Connection Duration:</strong><br>
                    &bull; HGBR Test MAE: <strong>2.96 hrs</strong> (R²: 0.320)<br>
                    &bull; Baseline Mean MAE: 4.09 hrs<br>
                    <hr style="margin: 0.5rem 0; border: none; border-top: 1px solid #E2E8F0;">
                    <strong>Energy Consumption:</strong><br>
                    &bull; HGBR Test MAE: <strong>4.24 kWh</strong> (R²: 0.354)<br>
                    &bull; Baseline Mean MAE: 5.86 kWh
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Operational Boundaries Callout
    render_disclaimer()

    st.markdown("---")

    # References
    st.markdown(
        """
        ### References & Citations

        - **Research Paper**: Kreft, C., et al. (2024). *Predictability of electric vehicle charging: Explaining extensive user behavior-specific heterogeneity*. **Applied Energy**, 370, 123544. [DOI: 10.1016/j.apenergy.2024.123544](https://doi.org/10.1016/j.apenergy.2024.123544).
        - **Benchmark Dataset**: National Grid Electricity Distribution (formerly Western Power Distribution), *Electric Nation Trial Dataset* (CrowdCharge Transactions and Charger Install Registry).
        """
    )
