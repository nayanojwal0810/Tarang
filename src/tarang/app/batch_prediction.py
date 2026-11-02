"""Batch Network Advisory and Flexibility view for TARANG Streamlit Application."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import streamlit as st

from tarang.app.components import render_disclaimer, render_hero, render_metric_card
from tarang.app.helpers import (
    convert_dataframe_to_csv_bytes,
    get_decision_support_engine,
    get_sample_batch_dataframe,
    load_file_into_dataframe,
    validate_batch_dataframe,
)
from tarang.config import FEATURE_NAMES, TARGET_DURATION, TARGET_ENERGY
from tarang.decision_support.flexibility import analyze_charging_flexibility
from tarang.decision_support.reporting import (
    generate_flexibility_summary_table,
    generate_prediction_summary_table,
)


def _compute_dataset_id(df: pd.DataFrame) -> str:
    """Compute a deterministic hash fingerprint for an input DataFrame to prevent stale results."""
    shape_sig = f"{df.shape[0]}x{df.shape[1]}"
    cols_sig = ",".join(sorted(df.columns))
    head_sig = str(df.head(2).to_dict()) if not df.empty else "empty"
    tail_sig = str(df.tail(2).to_dict()) if not df.empty else "empty"
    raw = f"{shape_sig}|{cols_sig}|{head_sig}|{tail_sig}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def render_batch_prediction_view() -> None:
    """Render the batch prediction workflow, dataset preview, results table, and flexibility analysis."""
    render_hero(
        title="Batch Network Advisory & Forecasting",
        subtitle="Process multi-session fleets, generate occupancy timelines, and evaluate charging flexibility.",
        tag="BATCH INFERENCE",
    )

    engine = get_decision_support_engine()

    st.markdown(
        """
        Upload a dataset of charging sessions formatted with an arrival timestamp (`StartTime`) and the
        **13 canonical model predictors**. TARANG will validate the schema, compute raw continuous duration
        and energy forecasts, project unplug timestamps, and assess descriptive smart-charging flexibility.
        """
    )

    # UPLOAD & SAMPLE SECTION
    col_upload, col_sample = st.columns([2, 1], gap="medium")

    with col_sample:
        st.markdown("#### Sample Dataset")
        st.markdown("Test the batch pipeline instantly using 50 pre-validated session records:")
        if st.button("Load Sample Batch Data (50 sessions)", use_container_width=True):
            sample_df, is_synth = get_sample_batch_dataframe(n_rows=50)
            st.session_state["batch_data_source"] = "sample"
            st.session_state["sample_batch_df"] = sample_df
            st.session_state["sample_batch_is_synthetic"] = is_synth

        # Download sample template
        template_df, _ = get_sample_batch_dataframe(n_rows=10)
        sample_csv = convert_dataframe_to_csv_bytes(template_df)
        st.download_button(
            label="Download Batch Template (CSV)",
            data=sample_csv,
            file_name="tarang_batch_template.csv",
            mime="text/csv",
            use_container_width=True,
            help="Download 10-row CSV template with required column schema.",
        )

    with col_upload:
        st.markdown("#### Upload Dataset")
        uploaded_file = st.file_uploader(
            "Upload Batch Session Data (.csv or .parquet)",
            type=["csv", "parquet"],
            help="File containing StartTime and the 13 canonical features.",
        )

    batch_df: Optional[pd.DataFrame] = None

    if uploaded_file is not None:
        try:
            batch_df = load_file_into_dataframe(uploaded_file)
            st.session_state["batch_data_source"] = "uploaded"
            st.session_state["uploaded_filename"] = uploaded_file.name
        except Exception as exc:
            st.error(f"Error reading uploaded file: {exc}")
    elif st.session_state.get("batch_data_source") == "sample":
        batch_df = st.session_state.get("sample_batch_df")
        is_synth = st.session_state.get("sample_batch_is_synthetic", False)
        if is_synth:
            st.info(
                "Loaded synthetic demonstration dataset (50 simulated sessions). "
                "Note: Synthetic data is generated for demonstration and testing purposes; "
                "it does not represent actual historical observations."
            )
        else:
            st.info("Loaded sample batch records (50 sessions) from the Electric Nation empirical trial dataset.")

    if batch_df is None:
        st.info("Upload a `.csv` or `.parquet` file or click **Load Sample Batch Data** to begin.")
        render_disclaimer()
        return

    # Compute dataset fingerprint for cache binding
    dataset_id = _compute_dataset_id(batch_df)

    # VALIDATION
    is_valid, error_msg, missing_cols = validate_batch_dataframe(batch_df)

    st.markdown("---")
    st.markdown("### Dataset Verification & Preview")

    c_meta1, c_meta2, c_meta3 = st.columns(3)
    with c_meta1:
        st.metric("Total Sessions", f"{len(batch_df):,}")
    with c_meta2:
        st.metric("Feature Columns", f"{len(batch_df.columns)}")
    with c_meta3:
        status_text = "Schema Valid" if is_valid else "Schema Invalid"
        st.metric("Verification Status", status_text)

    if not is_valid:
        st.error(f"Validation Error: {error_msg}")
        if missing_cols:
            st.markdown(f"**Missing Required Predictors:** `{missing_cols}`")
            st.markdown(f"**Expected 13 Canonical Features:** `{FEATURE_NAMES}`")
        return

    # Preview table
    with st.expander("Preview Uploaded Sessions (First 5 Rows)", expanded=False):
        st.dataframe(batch_df.head(5), use_container_width=True)

    # EXECUTE PREDICTIONS
    st.markdown("---")
    st.markdown("### Operational Inference Execution")

    if st.button("Generate Batch Advisories", type="primary"):
        with st.spinner(f"Evaluating {len(batch_df):,} sessions across dual HGBR estimators..."):
            try:
                advisories_df = engine.predict_batch_advisories(batch_df)
                flex_stats = analyze_charging_flexibility(batch_df)
                st.session_state["batch_advisories_cache"] = {
                    "dataset_id": dataset_id,
                    "advisories": advisories_df,
                    "flex_stats": flex_stats,
                }
            except Exception as exc:
                st.error(f"Batch advisory generation failed: {exc}")
                return

    # DISPLAY RESULTS ONLY IF BOUND TO CURRENT DATASET FINGERPRINT
    cached_payload = st.session_state.get("batch_advisories_cache")

    if cached_payload and cached_payload.get("dataset_id") == dataset_id:
        results_df: pd.DataFrame = cached_payload["advisories"]
        flex_stats: dict = cached_payload["flex_stats"]

        st.success(f"Successfully generated advisories for {len(results_df):,} sessions.")

        # Summary Metrics
        st.markdown("#### Advisory Distribution Summary")
        pred_summary = generate_prediction_summary_table(results_df)

        col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)

        dur_valid_count = int(results_df["duration_prediction_valid"].sum())
        dur_valid_pct = (dur_valid_count / len(results_df)) * 100.0
        dur_mean = results_df.loc[results_df["duration_prediction_valid"], f"predicted_{TARGET_DURATION}"].mean()

        ene_valid_count = int(results_df["energy_prediction_valid"].sum())
        ene_valid_pct = (ene_valid_count / len(results_df)) * 100.0
        ene_mean = results_df.loc[results_df["energy_prediction_valid"], f"predicted_{TARGET_ENERGY}"].mean()

        with col_m1:
            render_metric_card("Total Sessions", f"{len(results_df):,}")
        with col_m2:
            render_metric_card(
                "Valid Duration",
                f"{dur_valid_pct:.1f}%",
                badge_text=f"{dur_valid_count:,}/{len(results_df):,}",
                is_valid=(dur_valid_pct == 100.0),
            )
        with col_m3:
            render_metric_card(
                "Mean Duration",
                f"{dur_mean:.2f}",
                unit="hrs",
                badge_text="Valid only",
            )
        with col_m4:
            render_metric_card(
                "Valid Energy",
                f"{ene_valid_pct:.1f}%",
                badge_text=f"{ene_valid_count:,}/{len(results_df):,}",
                is_valid=(ene_valid_pct == 100.0),
            )
        with col_m5:
            render_metric_card(
                "Mean Energy",
                f"{ene_mean:.2f}",
                unit="kWh",
                badge_text="Valid only",
            )

        # Tabular Display
        st.markdown("#### Session Advisory Results")

        show_all = st.checkbox("Show all input features alongside predictions", value=False)

        key_cols = [
            "StartTime",
            f"predicted_{TARGET_DURATION}",
            f"predicted_{TARGET_ENERGY}",
            "projected_unplug_time_utc",
            "duration_prediction_valid",
            "energy_prediction_valid",
        ]

        display_cols = results_df.columns.tolist() if show_all else key_cols
        st.dataframe(results_df[display_cols], use_container_width=True, height=350)

        # CSV Download Button
        csv_bytes = convert_dataframe_to_csv_bytes(results_df)
        st.download_button(
            label="Download Full Advisories (CSV)",
            data=csv_bytes,
            file_name="tarang_session_advisories.csv",
            mime="text/csv",
            type="primary",
        )

        # FLEXIBILITY ANALYSIS SECTION (referring to the EXACT same dataset)
        st.markdown("---")
        st.markdown("### Smart-Charging Flexibility Analysis")

        if flex_stats.get("available", False):
            st.markdown(
                """
                Descriptive flexibility evaluates the ratio of active charging duration to total connection duration:
                $$\\text{Flexibility Ratio} = \\frac{\\text{ChargingDuration} / 60.0}{(\\text{StopTime} - \\text{StartTime})_{\\text{hours}}}$$
                Sessions are segmented by UK local date (`Europe/London`) into same-day versus overnight stays.
                """
            )

            flex_summary_df = generate_flexibility_summary_table(flex_stats)
            if flex_summary_df is not None:
                st.dataframe(flex_summary_df, hide_index=True, use_container_width=True)

            # Interactive Binned Histogram Visualization
            st.markdown("#### Binned Ratio Distribution (25 Uniform Intervals in [0, 1])")
            bin_edges = flex_stats.get("bin_edges", [])
            if len(bin_edges) > 1:
                bin_labels = [f"{bin_edges[i]:.2f}–{bin_edges[i+1]:.2f}" for i in range(len(bin_edges) - 1)]
                chart_df = pd.DataFrame(
                    {
                        "Ratio Bin": bin_labels,
                        "Same-Day (%)": flex_stats.get("same_day_pct", []),
                        "Overnight (%)": flex_stats.get("overnight_pct", []),
                    }
                ).set_index("Ratio Bin")
                st.bar_chart(chart_df)

            st.markdown(
                """
                > **Flexibility Disclaimer**: This ratio characterizes historical empirical duration relationships.
                > It reflects that overnight vehicles frequently remained connected long after charging completed.
                > This descriptive measure does **not** prove that the remaining idle connection time or energy can be
                > shifted under operational constraints, nor does it guarantee schedulable flexibility.
                """
            )
        else:
            reason = flex_stats.get("reason", "Historical connection columns omitted.")
            st.info(
                f"**Flexibility Analysis Unavailable**: {reason} "
                "To evaluate smart-charging flexibility, include `StartTime`, `StopTime`, and `ChargingDuration` in the dataset."
            )
    else:
        st.info("Click **Generate Batch Advisories** to evaluate the currently loaded dataset.")

    render_disclaimer()

