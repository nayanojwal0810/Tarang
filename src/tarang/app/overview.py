"""Overview / landing page for TARANG Streamlit Application."""

from __future__ import annotations

import streamlit as st

from tarang.app.components import render_disclaimer, render_hero


def render_overview_view() -> None:
    """Render the TARANG product overview and landing page."""
    render_hero(
        title="TARANG — EV Charging Intelligence",
        subtitle=(
            "Predict EV charging session duration and energy consumption from "
            "arrival conditions and driver history."
        ),
        tag="Overview",
    )

    # ── What TARANG predicts ──────────────────────────────────────────────────
    st.markdown("### What TARANG predicts")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            """
            <div class="tarang-card">
                <div class="tarang-card-title">Connection Duration</div>
                <p style="font-size: 0.9rem; color: #475569; margin: 0.25rem 0 0.5rem;">
                    Total time the vehicle remains plugged into the charge point, in <strong>hours</strong>.
                </p>
                <p style="font-size: 0.85rem; color: #64748B; margin: 0;">
                    Connection duration includes both active battery charging and any idle time
                    the vehicle occupies the station after charging completes.
                    The charge point cannot be used by another driver until the vehicle unplugs.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="tarang-card">
                <div class="tarang-card-title">Energy Consumption</div>
                <p style="font-size: 0.9rem; color: #475569; margin: 0.25rem 0 0.5rem;">
                    Total electrical energy transferred during the session, in <strong>kWh</strong>.
                </p>
                <p style="font-size: 0.85rem; color: #64748B; margin: 0;">
                    Energy consumed depends on the vehicle's state of charge on arrival,
                    battery capacity, and the onboard charger's maximum acceptance rate.
                    It is independent of how long the vehicle stays plugged in.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div style="font-size: 0.88rem; color: #475569; margin-bottom: 1.25rem;">
        Combining these two forecasts with the arrival timestamp, TARANG also calculates a
        <strong>projected unplug time</strong> and produces descriptive
        <strong>charging flexibility</strong> statistics from historical session records.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Two workflows ─────────────────────────────────────────────────────────
    st.markdown("### How to use TARANG")
    w1, w2 = st.columns(2)

    with w1:
        st.markdown(
            """
            <div class="tarang-card">
                <div class="tarang-card-title">Single Prediction</div>
                <p style="font-size: 0.88rem; color: #475569; margin: 0.25rem 0 0.5rem;">
                    Generate a real-time advisory for one arriving vehicle.
                </p>
                <ol style="font-size: 0.85rem; color: #64748B; padding-left: 1.1rem; margin: 0;">
                    <li>Enter the expected arrival date and time (UTC).</li>
                    <li>Enter the vehicle's charger power (kW) and battery capacity (kWh).</li>
                    <li>Upload the driver's prior session history as a CSV file.</li>
                    <li>Click <strong>Generate Session Advisory</strong>.</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with w2:
        st.markdown(
            """
            <div class="tarang-card">
                <div class="tarang-card-title">Batch Prediction</div>
                <p style="font-size: 0.88rem; color: #475569; margin: 0.25rem 0 0.5rem;">
                    Process a dataset of pre-engineered session records.
                </p>
                <ol style="font-size: 0.85rem; color: #64748B; padding-left: 1.1rem; margin: 0;">
                    <li>Upload a CSV or Parquet file with the 13 canonical feature columns.</li>
                    <li>Review the validation summary.</li>
                    <li>Click <strong>Generate Batch Advisories</strong>.</li>
                    <li>Download the advisory output as CSV.</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── How it works ──────────────────────────────────────────────────────────
    st.markdown("### How the models work")

    col_text, col_perf = st.columns([1.3, 1])

    with col_text:
        st.markdown(
            """
            For each session, TARANG computes **13 canonical predictors** from the arrival
            conditions and driver history:

            | Domain | Features |
            |---|---|
            | Calendar | `weekday`, `weekend`, `holiday`, `month` |
            | Time of day | `StartCos`, `StartSin`, `StartHourNorm` |
            | Driver habit | `TimeSinceLastStop`, `sessionsToday`, `LastDuration`, `LastConsumedkWh` |
            | Vehicle | `CarKW`, `CarKWh` |

            These features are passed to two independent
            **HistGradientBoostingRegressor** models trained with absolute-error loss and
            early stopping. One model predicts connection duration; the other predicts
            energy consumption.
            """
        )

    with col_perf:
        st.markdown(
            """
            <div class="tarang-card" style="margin-top: 0;">
                <div class="tarang-card-title">Benchmark Performance</div>
                <table style="width: 100%; font-size: 0.83rem; color: #475569; border-collapse: collapse;">
                    <tr style="border-bottom: 1px solid #E2E8F0;">
                        <th style="text-align: left; padding: 0.3rem 0; font-weight: 600; color: #142D43;">Target</th>
                        <th style="text-align: right; padding: 0.3rem 0; font-weight: 600; color: #142D43;">HGBR MAE</th>
                        <th style="text-align: right; padding: 0.3rem 0; font-weight: 600; color: #142D43;">HGBR MdAPE</th>
                        <th style="text-align: right; padding: 0.3rem 0; font-weight: 600; color: #142D43;">Mean Baseline</th>
                    </tr>
                    <tr style="border-bottom: 1px solid #F1F5F9;">
                        <td style="padding: 0.35rem 0;">Duration</td>
                        <td style="text-align: right; font-weight: 600; color: #142D43;">5.53 h</td>
                        <td style="text-align: right; color: #047857;">23.4%</td>
                        <td style="text-align: right;">7.08 h</td>
                    </tr>
                    <tr>
                        <td style="padding: 0.35rem 0;">Energy</td>
                        <td style="text-align: right; font-weight: 600; color: #142D43;">3.83 kWh</td>
                        <td style="text-align: right; color: #047857;">25.0%</td>
                        <td style="text-align: right;">6.91 kWh</td>
                    </tr>
                </table>
                <p style="font-size: 0.75rem; color: #94A3B8; margin: 0.5rem 0 0;">
                    Held-out test split: 11,904 sessions (20%). Baseline: mean predictor.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Dataset provenance ────────────────────────────────────────────────────
    st.markdown("### Dataset and attribution")
    st.markdown(
        """
        TARANG is trained and evaluated on the **Electric Nation** trial dataset,
        one of the largest monitored real-world home EV charging trials conducted in the UK
        (2017–2018, South West England).

        | Detail | Value |
        |---|---|
        | Provider | National Grid Electricity Distribution (NGED), formerly Western Power Distribution |
        | Sessions (canonical, after quality filtering) | 59,520 |
        | Participants | 267 |
        | Licence | NGED Open Data Licence |
        | Attribution | *Supported by NGED Open Data* |
        | Dataset URL | https://commercial.nationalgrid.co.uk/electric-nation-data |

        **Research reference:** Kreft, C., et al. (2024). *Predictability of electric vehicle
        charging: Explaining extensive user behavior-specific heterogeneity.* Applied Energy, 370,
        123544. [doi:10.1016/j.apenergy.2024.123544](https://doi.org/10.1016/j.apenergy.2024.123544)
        """
    )

    st.markdown("---")
    render_disclaimer()
