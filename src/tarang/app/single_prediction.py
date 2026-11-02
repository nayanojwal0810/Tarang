"""Single-Session Predictive Advisory view for TARANG Streamlit Application."""

from __future__ import annotations

from datetime import datetime, timezone
import io
import pandas as pd
import streamlit as st

from tarang.app.components import render_advisory_card, render_disclaimer, render_hero
from tarang.app.helpers import (
    get_decision_support_engine,
    get_sample_history_csv,
    validate_history_dataframe,
)
from tarang.inference.feature_builder import build_inference_features


def render_single_prediction_view() -> None:
    """Render the single-session prediction interface and advisory output."""
    render_hero(
        title="Single-Session Predictive Advisory",
        subtitle="Forecast connection duration, energy demand, and projected unplug time for an arriving EV.",
        tag="REAL-TIME INFERENCE",
    )

    engine = get_decision_support_engine()

    st.markdown(
        """
        Configure the vehicle arrival conditions and provide completed historical charging sessions
        for the driver. The TARANG feature engine will calculate diurnal harmonics, habit lag metrics,
        and vehicle hardware attributes to generate real-time operational advisories.
        """
    )

    col_left, col_right = st.columns([1, 1], gap="large")

    # LEFT COLUMN: Inputs
    with col_left:
        st.markdown("#### 1. Vehicle Arrival & Hardware Specs")

        # Arrival Date and Time
        c_date, c_time = st.columns(2)
        with c_date:
            arr_date = st.date_input("Arrival Date (UTC)", value=datetime.now(timezone.utc).date())
        with c_time:
            arr_time = st.time_input("Arrival Time (UTC)", value=datetime.now(timezone.utc).time().replace(second=0, microsecond=0))

        arrival_str = f"{arr_date} {arr_time.strftime('%H:%M:%S')}"
        st.caption(f"Configured arrival timestamp: `{arrival_str} UTC`")

        # Vehicle Specs
        c_kw, c_kwh = st.columns(2)
        with c_kw:
            car_kw = st.number_input(
                "Charger Power (kW)",
                min_value=1.0,
                max_value=350.0,
                value=7.0,
                step=0.5,
                help="Rated maximum charging power of the vehicle onboard charger or station (e.g., 3.6, 7.0, 11.0, 22.0 kW).",
            )
        with c_kwh:
            car_kwh = st.number_input(
                "Battery Capacity (kWh)",
                min_value=5.0,
                max_value=250.0,
                value=40.0,
                step=1.0,
                help="Usable traction battery pack capacity of the electric vehicle (e.g., 24, 40, 64, 75, 100 kWh).",
            )

        st.markdown("---")
        st.markdown("#### 2. Participant Charging History")
        st.markdown(
            """
            TARANG requires prior completed charging sessions to calculate behavioral habit features
            (`TimeSinceLastStop`, `LastDuration`, `LastConsumedkWh`).
            The uploaded CSV must contain: `StartTime`, `StopTime`, and `ConsumedkWh`.
            """
        )

        # Quick sample loader
        sample_col1, sample_col2 = st.columns([1, 1])
        use_sample = False
        with sample_col1:
            if st.button("Use Sample History Data", help="Populate with pre-formatted demonstration history matching arrival timestamp"):
                st.session_state["single_history_data"] = get_sample_history_csv(reference_time=arrival_str)
                st.session_state["history_source"] = "sample"

        with sample_col2:
            st.download_button(
                label="Download Sample CSV",
                data=get_sample_history_csv(reference_time=arrival_str),
                file_name="sample_charging_history.csv",
                mime="text/csv",
                help="Download sample CSV template for driver history",
            )

        uploaded_file = st.file_uploader(
            "Upload Driver Charging History (CSV)",
            type=["csv"],
            help="CSV file containing driver's completed charging transactions.",
        )

        history_df: Optional[pd.DataFrame] = None

        if uploaded_file is not None:
            try:
                history_df = pd.read_csv(uploaded_file)
                st.session_state["history_source"] = "uploaded"
            except Exception as exc:
                st.error(f"Failed to parse uploaded CSV: {exc}")
        elif "single_history_data" in st.session_state and st.session_state.get("history_source") == "sample":
            history_df = pd.read_csv(io.StringIO(st.session_state["single_history_data"]))
            st.info("Loaded demonstration driver history.")

        # Validate history DataFrame
        is_valid_history, error_msg, multiple_participants = validate_history_dataframe(history_df)

        selected_participant: Optional[str] = None
        if history_df is not None:
            if not is_valid_history:
                st.error(f"Invalid history data: {error_msg}")
            else:
                if multiple_participants:
                    st.warning(
                        "Uploaded file contains multiple participants. "
                        "TARANG requires history for a single participant to prevent driver mixing."
                    )
                    selected_participant = st.selectbox(
                        "Select Target Participant:",
                        options=multiple_participants,
                        index=0,
                        help="Filter historical records to this individual driver before feature extraction.",
                    )
                    history_df = history_df[
                        history_df["ParticipantID"].notna()
                        & (history_df["ParticipantID"].astype(str) == str(selected_participant))
                    ].copy()
                    st.caption(f"Filtered to **{len(history_df)} sessions** for participant `{selected_participant}`.")
                elif "ParticipantID" in history_df.columns:
                    # Single participant with explicit identifier: ensure no null IDs leak
                    history_df = history_df[history_df["ParticipantID"].notna()].copy()
                    st.success(f"History validated successfully ({len(history_df)} sessions loaded).")
                else:
                    st.success(f"History validated successfully ({len(history_df)} sessions loaded).")

        # Predict Button
        predict_clicked = st.button("Generate Session Advisory", type="primary", use_container_width=True)

    # RIGHT COLUMN: Results & Feature Inspection
    with col_right:
        st.markdown("#### 3. Operational Advisory Forecast")

        if not predict_clicked:
            st.info("Provide arrival inputs and historical sessions, then click **Generate Session Advisory**.")
        else:
            if history_df is None or not is_valid_history:
                st.error(
                    "Cannot generate advisory: Completed driver charging history is missing or invalid. "
                    "Please upload a valid CSV or click 'Use Sample History Data'."
                )
            else:
                try:
                    with st.spinner("Extracting features and generating predictions..."):
                        # Execute advisory prediction via DecisionSupportEngine
                        advisory = engine.predict_single_advisory(
                            start_time=arrival_str,
                            car_kw=float(car_kw),
                            car_kwh=float(car_kwh),
                            history=history_df,
                        )

                        # Extract computed features for transparency
                        feat_df = build_inference_features(
                            start_time=arrival_str,
                            car_kw=float(car_kw),
                            car_kwh=float(car_kwh),
                            history=history_df,
                            max_start_hour=engine.max_start_hour,
                            summary_file=engine.summary_file,
                        )

                    # Display KPI cards
                    render_advisory_card(
                        duration_hours=advisory.predicted_duration_hours,
                        energy_kwh=advisory.predicted_energy_kwh,
                        arrival_utc=advisory.arrival_time_utc,
                        unplug_utc=advisory.projected_unplug_time_utc,
                        duration_valid=advisory.duration_prediction_valid,
                        energy_valid=advisory.energy_prediction_valid,
                    )

                    st.markdown(
                        """
                        > **Connection vs. Charging Duration**: Connection duration reflects the total bay occupancy
                        > time during which the vehicle remains plugged in. Active battery charging completes earlier,
                        > after which the vehicle occupies the charging station in an idle state until unplugged.
                        """
                    )

                    # Feature transparency expander
                    with st.expander("Engineered Model Predictors (13 Canonical Features)", expanded=False):
                        st.markdown(
                            "These 13 engineered features were computed from the arrival timestamp, vehicle hardware, "
                            "and completed charging history:"
                        )
                        feat_display = feat_df.T.reset_index()
                        feat_display.columns = ["Predictor", "Engineered Value"]
                        st.dataframe(feat_display, hide_index=True, use_container_width=True)

                except ValueError as exc:
                    st.error(f"Inference error: {exc}")
                    if "cold-start" in str(exc).lower() or "empty" in str(exc).lower() or "prior" in str(exc).lower():
                        st.warning(
                            "**Cold-Start Limitation**: TARANG requires at least one prior completed session "
                            "before the arrival timestamp to compute lag habit features (LastDuration, LastConsumedkWh)."
                        )
                except Exception as exc:
                    st.error(f"Unexpected operational error: {exc}")

    render_disclaimer()
