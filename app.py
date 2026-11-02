"""TARANG — EV Charging Intelligence System

Main Streamlit Application Entry Point.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure `src` is on sys.path when running from repository root (e.g., Streamlit Cloud)
_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import streamlit as st

# Must be the first Streamlit call.
st.set_page_config(
    page_title="TARANG — EV Charging Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "**TARANG** — EV Charging Intelligence System\n\n"
            "Predictive advisory for EV charging session duration and energy consumption.\n\n"
            "Built on the Electric Nation trial dataset (NGED Open Data)."
        ),
    },
)

from tarang.app.batch_prediction import render_batch_prediction_view
from tarang.app.components import inject_custom_css, render_footer
from tarang.app.overview import render_overview_view
from tarang.app.single_prediction import render_single_prediction_view


def main() -> None:
    """Execute main application loop with navigation."""
    inject_custom_css()

    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 0.75rem 0 1.25rem 0;">
                <div style="font-size: 1.5rem; font-weight: 700; color: #142D43;
                            letter-spacing: -0.02em; line-height: 1.1;">
                    ⚡ TARANG
                </div>
                <div style="font-size: 0.78rem; font-weight: 500; color: #0D8585;
                            text-transform: uppercase; letter-spacing: 0.08em;
                            margin-top: 0.2rem;">
                    EV Charging Intelligence
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        page_choice = st.radio(
            "Navigation",
            options=["Overview", "Single Prediction", "Batch Prediction"],
            index=0,
            label_visibility="collapsed",
        )

        st.markdown("---")

        st.markdown(
            """
            <div style="font-size: 0.78rem; color: #64748B; line-height: 1.65;">
                <div style="font-weight: 600; color: #475569; margin-bottom: 0.35rem;">
                    Model
                </div>
                HistGradientBoostingRegressor<br>
                Absolute error · Early stopping<br><br>
                <div style="font-weight: 600; color: #475569; margin-bottom: 0.35rem;">
                    Dataset
                </div>
                Electric Nation Trial<br>
                59,520 sessions · 267 participants<br><br>
                <div style="font-weight: 600; color: #475569; margin-bottom: 0.35rem;">
                    Release
                </div>
                v0.1.0-preview
            </div>
            """,
            unsafe_allow_html=True,
        )

    if page_choice == "Overview":
        render_overview_view()
    elif page_choice == "Single Prediction":
        render_single_prediction_view()
    elif page_choice == "Batch Prediction":
        render_batch_prediction_view()

    render_footer()


if __name__ == "__main__":
    main()
