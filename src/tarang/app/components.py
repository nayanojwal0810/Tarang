"""UI Components and Enterprise Styling for TARANG Streamlit Application.

Provides reusable presentation components, metric cards, status badges,
and custom CSS adhering to enterprise energy technology aesthetic standards.
"""

from __future__ import annotations

from typing import Optional
import pandas as pd
import streamlit as st


def inject_custom_css() -> None:
    """Inject premium, enterprise-grade custom CSS styling into Streamlit DOM."""
    st.markdown(
        """
        <style>
        /* Base typography & layout smoothing */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            color: #0F172A;
        }

        /* Container & Card Surfaces */
        .tarang-hero {
            background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
            color: #FFFFFF;
            padding: 2.2rem 2.5rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
            border: 1px solid #334155;
        }
        .tarang-hero h1 {
            color: #FFFFFF;
            font-size: 2.1rem;
            font-weight: 700;
            letter-spacing: -0.025em;
            margin-bottom: 0.4rem;
        }
        .tarang-hero p.subtitle {
            color: #94A3B8;
            font-size: 1.05rem;
            font-weight: 400;
            margin-bottom: 0;
            line-height: 1.5;
        }

        .tarang-card {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 1.5rem;
            margin-bottom: 1.25rem;
            box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.03);
            transition: border-color 0.15s ease-in-out;
        }
        .tarang-card:hover {
            border-color: #CBD5E1;
        }

        .tarang-card-header {
            font-size: 1.1rem;
            font-weight: 600;
            color: #0F172A;
            margin-bottom: 0.5rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        /* Metrics Display */
        .metric-container {
            background-color: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 1.25rem 1.4rem;
            margin-bottom: 1rem;
        }
        .metric-label {
            font-size: 0.85rem;
            font-weight: 500;
            color: #64748B;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.35rem;
        }
        .metric-value {
            font-size: 1.85rem;
            font-weight: 700;
            color: #0F172A;
            line-height: 1.1;
        }
        .metric-unit {
            font-size: 0.95rem;
            font-weight: 500;
            color: #64748B;
            margin-left: 0.25rem;
        }

        /* Status Badges */
        .badge-valid {
            display: inline-block;
            background-color: #ECFDF5;
            color: #047857;
            font-size: 0.75rem;
            font-weight: 600;
            padding: 0.2rem 0.55rem;
            border-radius: 9999px;
            border: 1px solid #A7F3D0;
        }
        .badge-invalid {
            display: inline-block;
            background-color: #FEF2F2;
            color: #B91C1C;
            font-size: 0.75rem;
            font-weight: 600;
            padding: 0.2rem 0.55rem;
            border-radius: 9999px;
            border: 1px solid #FECACA;
        }
        .badge-neutral {
            display: inline-block;
            background-color: #F1F5F9;
            color: #475569;
            font-size: 0.75rem;
            font-weight: 600;
            padding: 0.2rem 0.55rem;
            border-radius: 9999px;
            border: 1px solid #CBD5E1;
        }

        /* Operational Notice Callout */
        .notice-box {
            background-color: #F8FAFC;
            border-left: 4px solid #0D9488;
            padding: 1rem 1.25rem;
            border-radius: 0 8px 8px 0;
            margin: 1.25rem 0;
            font-size: 0.92rem;
            color: #334155;
            line-height: 1.5;
        }
        .notice-box strong {
            color: #0F172A;
        }

        /* Streamlit Button Tweaks */
        div.stButton > button:first-child {
            font-weight: 500;
            border-radius: 8px;
            padding: 0.55rem 1.25rem;
            transition: all 0.15s ease-in-out;
        }

        /* Footer styling */
        .tarang-footer {
            margin-top: 3.5rem;
            padding-top: 1.5rem;
            border-top: 1px solid #E2E8F0;
            font-size: 0.85rem;
            color: #64748B;
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(title: str, subtitle: str, tag: Optional[str] = "INTELLIGENCE LAYER") -> None:
    """Render high-contrast hero banner at top of views."""
    tag_html = f'<span class="badge-neutral" style="background:#334155; color:#F8FAFC; border:none; margin-bottom:0.75rem;">{tag}</span>' if tag else ""
    st.markdown(
        f"""
        <div class="tarang-hero">
            {tag_html}
            <h1>{title}</h1>
            <p class="subtitle">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(
    label: str,
    value: str,
    unit: str = "",
    badge_text: Optional[str] = None,
    is_valid: Optional[bool] = None,
) -> None:
    """Render polished statistical metric card with optional validity badge."""
    badge_html = ""
    if badge_text is not None and is_valid is not None:
        cls = "badge-valid" if is_valid else "badge-invalid"
        badge_html = f'<div style="margin-top:0.5rem;"><span class="{cls}">{badge_text}</span></div>'
    elif badge_text is not None:
        badge_html = f'<div style="margin-top:0.5rem;"><span class="badge-neutral">{badge_text}</span></div>'

    unit_html = f'<span class="metric-unit">{unit}</span>' if unit else ""

    st.markdown(
        f"""
        <div class="metric-container">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}{unit_html}</div>
            {badge_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_advisory_card(
    duration_hours: float,
    energy_kwh: float,
    arrival_utc: pd.Timestamp,
    unplug_utc: Optional[pd.Timestamp],
    duration_valid: bool,
    energy_valid: bool,
) -> None:
    """Render comprehensive single-session advisory summary cards."""
    c1, c2, c3 = st.columns(3)

    dur_val_str = f"{duration_hours:.2f}"
    energy_val_str = f"{energy_kwh:.2f}"
    unplug_str = (
        unplug_utc.strftime("%Y-%m-%d %H:%M UTC")
        if (unplug_utc is not None and pd.notna(unplug_utc))
        else "Unavailable (Invalid Duration)"
    )

    with c1:
        render_metric_card(
            label="Connection Duration",
            value=dur_val_str,
            unit="hours",
            badge_text="Valid Estimate" if duration_valid else "Flagged Non-Negative",
            is_valid=duration_valid,
        )

    with c2:
        render_metric_card(
            label="Energy Demand",
            value=energy_val_str,
            unit="kWh",
            badge_text="Valid Estimate" if energy_valid else "Flagged Non-Negative",
            is_valid=energy_valid,
        )

    with c3:
        render_metric_card(
            label="Projected Departure",
            value=unplug_str,
            unit="",
            badge_text="Arrival + Predicted Duration" if duration_valid else "Requires Valid Duration",
            is_valid=duration_valid,
        )


def render_disclaimer() -> None:
    """Render concise advisory boundaries notice."""
    st.markdown(
        """
        <div class="notice-box">
            <strong>Advisory Decision-Support Notice:</strong> TARANG generates statistical forecasts
            of EV connection duration and energy demand to assist facility planning and bay utilization.
            It does not perform real-time hardware actuation, power throttling, dynamic tariff scheduling,
            or grid dispatch. Continuous model predictions are preserved in their raw continuous form
            without artificial clipping.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    """Render professional application footer."""
    st.markdown(
        """
        <div class="tarang-footer">
            <strong>TARANG — EV Charging Intelligence System</strong> &bull;
            Lead Author: <em>A. Nayanojwal Sri Tej</em><br>
            Empirical Benchmark Dataset: <em>Electric Nation (National Grid Electricity Distribution)</em> &bull;
            Reference: Kreft et al. (2024), <em>Applied Energy</em>.
        </div>
        """,
        unsafe_allow_html=True,
    )
