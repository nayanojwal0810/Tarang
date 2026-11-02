"""Shared UI components, CSS injection, and rendering utilities for TARANG Streamlit App."""

from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st


# ──────────────────────────────────────────────────────────────────────────────
# CSS Injection
# ──────────────────────────────────────────────────────────────────────────────

def inject_custom_css() -> None:
    """Inject minimal, scoped custom CSS for TARANG enterprise design system."""
    st.markdown(
        """
        <style>
        /* ── Typography ── */
        html, body, [class*="st-"] {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                         "Helvetica Neue", Arial, sans-serif;
        }

        /* ── Page hero ── */
        .tarang-hero {
            background: linear-gradient(135deg, #142D43 0%, #1a3d5a 100%);
            border-radius: 12px;
            padding: 2rem 2.25rem 1.75rem;
            margin-bottom: 1.75rem;
            color: #F0F5F9;
        }
        .tarang-hero .hero-tag {
            display: inline-block;
            font-size: 0.7rem;
            font-weight: 600;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: #0D8585;
            background: rgba(13, 133, 133, 0.12);
            border: 1px solid rgba(13, 133, 133, 0.3);
            border-radius: 4px;
            padding: 0.15rem 0.55rem;
            margin-bottom: 0.6rem;
        }
        .tarang-hero h1 {
            font-size: 1.75rem;
            font-weight: 700;
            color: #F0F5F9;
            margin: 0.3rem 0 0.45rem;
            letter-spacing: -0.02em;
            line-height: 1.2;
        }
        .tarang-hero .hero-subtitle {
            font-size: 0.95rem;
            color: #94A3B8;
            line-height: 1.5;
            margin: 0;
        }

        /* ── Metric cards ── */
        .tarang-metric {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 1.1rem 1.3rem;
            margin-bottom: 0.85rem;
        }
        .tarang-metric .metric-label {
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            color: #64748B;
            margin-bottom: 0.3rem;
        }
        .tarang-metric .metric-value {
            font-size: 1.7rem;
            font-weight: 700;
            color: #142D43;
            line-height: 1.1;
        }
        .tarang-metric .metric-unit {
            font-size: 0.9rem;
            font-weight: 500;
            color: #64748B;
            margin-left: 0.2rem;
        }
        .tarang-metric .metric-badge {
            display: inline-block;
            margin-top: 0.4rem;
            font-size: 0.7rem;
            font-weight: 600;
            padding: 0.15rem 0.5rem;
            border-radius: 20px;
        }
        .badge-ok, .badge-valid   { background: #ECFDF5; color: #047857; border: 1px solid #A7F3D0; }
        .badge-warn               { background: #FEF3C7; color: #92400E; border: 1px solid #FDE68A; }
        .badge-err, .badge-invalid { background: #FEF2F2; color: #B91C1C; border: 1px solid #FECACA; }
        .badge-info, .badge-neutral { background: #F1F5F9; color: #475569; border: 1px solid #CBD5E1; }

        /* ── Content card ── */
        .tarang-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 1.25rem 1.4rem;
            margin-bottom: 1rem;
        }
        .tarang-card-title {
            font-size: 0.95rem;
            font-weight: 600;
            color: #142D43;
            margin-bottom: 0.5rem;
        }

        /* ── Section divider ── */
        .section-label {
            font-size: 0.7rem;
            font-weight: 700;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: #94A3B8;
            margin-bottom: 0.6rem;
        }

        /* ── Notice box ── */
        .notice-box {
            background: #F8FAFC;
            border-left: 3px solid #0D8585;
            padding: 0.85rem 1.1rem;
            border-radius: 0 8px 8px 0;
            font-size: 0.88rem;
            color: #334155;
            line-height: 1.55;
            margin: 1rem 0;
        }
        .notice-box strong { color: #142D43; }

        /* ── Footer ── */
        .tarang-footer {
            margin-top: 3rem;
            padding-top: 1.25rem;
            border-top: 1px solid #E2E8F0;
            font-size: 0.78rem;
            color: #94A3B8;
            text-align: center;
            line-height: 1.6;
        }

        /* ── Privacy notice ── */
        .privacy-notice {
            background: #FFFBEB;
            border: 1px solid #FDE68A;
            border-radius: 8px;
            padding: 0.7rem 1rem;
            font-size: 0.8rem;
            color: #78350F;
            line-height: 1.5;
            margin-bottom: 0.75rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Page-level Primitives
# ──────────────────────────────────────────────────────────────────────────────

def render_hero(
    title: str,
    subtitle: str,
    tag: Optional[str] = None,
) -> None:
    """Render a dark navy hero banner at the top of a page."""
    tag_html = (
        f'<div class="hero-tag">{tag}</div>' if tag else ""
    )
    st.markdown(
        f"""
        <div class="tarang-hero">
            {tag_html}
            <h1>{title}</h1>
            <p class="hero-subtitle">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(
    label: str,
    value: str,
    unit: str = "",
    badge_text: Optional[str] = None,
    badge_style: str = "info",
    is_valid: Optional[bool] = None,
) -> None:
    """Render a compact metric display card.

    Args:
        label: Short uppercase label.
        value: Formatted numeric or string value.
        unit: Optional unit string appended after value.
        badge_text: Optional status badge text.
        badge_style: One of 'ok', 'warn', 'err', 'info'.
        is_valid: Optional boolean flag to set badge style ('ok' if True, 'err' if False).
    """
    if is_valid is not None:
        badge_style = "ok" if is_valid else "err"

    unit_html = f'<span class="metric-unit">{unit}</span>' if unit else ""
    badge_html = ""
    if badge_text:
        badge_html = f'<div><span class="metric-badge badge-{badge_style}">{badge_text}</span></div>'

    st.markdown(
        f"""
        <div class="tarang-metric">
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
    """Render three-column advisory result cards."""
    c1, c2, c3 = st.columns(3)

    dur_str = f"{duration_hours:.2f}"
    energy_str = f"{energy_kwh:.2f}"
    unplug_str = (
        unplug_utc.strftime("%Y-%m-%d %H:%M UTC")
        if (unplug_utc is not None and pd.notna(unplug_utc))
        else "Unavailable"
    )

    with c1:
        render_metric_card(
            label="Connection Duration",
            value=dur_str,
            unit="h",
            badge_text="Valid estimate" if duration_valid else "Flagged — non-positive",
            badge_style="ok" if duration_valid else "err",
        )
    with c2:
        render_metric_card(
            label="Energy Demand",
            value=energy_str,
            unit="kWh",
            badge_text="Valid estimate" if energy_valid else "Flagged — non-positive",
            badge_style="ok" if energy_valid else "err",
        )
    with c3:
        render_metric_card(
            label="Projected Departure",
            value=unplug_str,
            unit="",
            badge_text="Arrival + predicted duration" if duration_valid else "Requires valid duration",
            badge_style="ok" if duration_valid else "warn",
        )


def render_disclaimer() -> None:
    """Render operational boundary notice."""
    st.markdown(
        """
        <div class="notice-box">
            <strong>Advisory Scope.</strong> TARANG generates statistical session forecasts based on
            historical driver behaviour patterns. Predictions are informational estimates and do not
            control charging hardware, execute grid dispatch commands, or guarantee charging
            availability. Connection duration differs from active charging time — a vehicle may
            finish charging before departure. Forecasts should be interpreted as probabilistic
            advisories, not operational guarantees.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_privacy_notice() -> None:
    """Render a short upload-privacy notice for workflows that accept user files."""
    st.markdown(
        """
        <div class="privacy-notice">
            <strong>Privacy notice.</strong> Uploaded files are processed in your active browser
            session and are not stored, logged, or shared. Do not upload files containing personally
            identifiable, confidential, or commercially sensitive data.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    """Render a compact global footer."""
    st.markdown(
        """
        <div class="tarang-footer">
            TARANG — EV Charging Intelligence System &nbsp;·&nbsp;
            Built on the <a href="https://commercial.nationalgrid.co.uk/electric-nation-data"
            target="_blank" style="color: #64748B;">Electric Nation Trial</a> dataset
            (NGED Open Data) &nbsp;·&nbsp;
            A. Nayanojwal Sri Tej
        </div>
        """,
        unsafe_allow_html=True,
    )
