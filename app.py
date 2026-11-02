"""TARANG — EV Charging Intelligence System

Main Streamlit Application Entry Point.
"""

from __future__ import annotations

import streamlit as st

# Configure page metadata first before any other Streamlit calls
st.set_page_config(
    page_title="TARANG — EV Charging Intelligence System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

from tarang.app.batch_prediction import render_batch_prediction_view
from tarang.app.components import inject_custom_css, render_footer
from tarang.app.overview import render_overview_view
from tarang.app.single_prediction import render_single_prediction_view


def main() -> None:
    """Execute main application loop with styled navigation."""
    # Inject enterprise custom CSS styling
    inject_custom_css()

    # Sidebar Navigation & System Meta
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 0.5rem 0 1rem 0;">
                <h2 style="font-size: 1.6rem; font-weight: 700; color: #0F172A; margin: 0; letter-spacing: -0.02em;">
                    ⚡ TARANG
                </h2>
                <div style="font-size: 0.85rem; font-weight: 500; color: #0D9488; text-transform: uppercase; letter-spacing: 0.05em;">
                    EV Charging Intelligence
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        page_choice = st.radio(
            "Navigation",
            options=["Overview", "Single Prediction", "Batch Prediction"],
            index=0,
            label_visibility="collapsed",
        )

        st.markdown("---")

        st.markdown(
            """
            <div style="font-size: 0.82rem; color: #64748B; line-height: 1.6;">
                <strong>System Configuration</strong><br>
                &bull; Model: <code>HGBR (Absolute Error)</code><br>
                &bull; Targets: <code>Duration (h)</code> & <code>Energy (kWh)</code><br>
                &bull; Benchmark: <code>Electric Nation Trial</code><br>
                &bull; Cohort: <code>59,520 Sessions</code><br>
                &bull; Release: <code>v0.1.0-prod</code>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Route content
    if page_choice == "Overview":
        render_overview_view()
    elif page_choice == "Single Prediction":
        render_single_prediction_view()
    elif page_choice == "Batch Prediction":
        render_batch_prediction_view()

    # Global enterprise footer
    render_footer()


if __name__ == "__main__":
    main()
