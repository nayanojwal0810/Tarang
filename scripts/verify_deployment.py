"""Deployment smoke verification script for TARANG.

Tests package loading, artifact checksum verification, and inference execution
under a clean environment (e.g., Streamlit Community Cloud runner) without dev dependencies.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure root directory is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# 1. Verify app import and sys.path resolution
import app

# 2. Verify Streamlit page modules
from tarang.app.batch_prediction import render_batch_prediction_view
from tarang.app.single_prediction import render_single_prediction_view
from tarang.app.overview import render_overview_view
from tarang.inference.loading import verify_deployment_artifacts
from tarang.decision_support.engine import DecisionSupportEngine

import pandas as pd


def verify_deployment() -> None:
    print("[1/5] Verifying deployment artifact checksums against manifest.json...")
    artifacts_ok = verify_deployment_artifacts()
    if not artifacts_ok:
        raise RuntimeError("Deployment artifact verification failed! Checksum mismatch or missing files.")
    print("      Checksum verification passed.")

    print("[2/5] Initializing DecisionSupportEngine from deployment artifacts...")
    engine = DecisionSupportEngine()
    assert engine.duration_model is not None, "Duration model failed to load"
    assert engine.energy_model is not None, "Energy model failed to load"
    print(f"      Duration model: {type(engine.duration_model).__name__}")
    print(f"      Energy model:   {type(engine.energy_model).__name__}")

    print("[3/5] Running single-session advisory inference...")
    history = pd.DataFrame([
        {
            "StartTime": "2024-01-10 10:00:00",
            "StopTime": "2024-01-10 14:00:00",
            "ConsumedkWh": 15.0,
        }
    ])
    single_adv = engine.predict_single_advisory(
        start_time="2024-01-15T18:00:00Z",
        car_kw=7.4,
        car_kwh=50.0,
        history=history,
    )
    assert single_adv.duration_prediction_valid, "Single duration prediction was invalid"
    assert single_adv.energy_prediction_valid, "Single energy prediction was invalid"
    assert single_adv.predicted_duration_hours > 0, "Single duration must be positive"
    assert single_adv.predicted_energy_kwh > 0, "Single energy must be positive"
    print(
        f"      Single advisory result: {single_adv.predicted_duration_hours:.2f} h, "
        f"{single_adv.predicted_energy_kwh:.2f} kWh, "
        f"Projected Unplug: {single_adv.projected_unplug_time_utc}"
    )

    print("[4/5] Running batch-session advisory inference...")
    batch_df = pd.DataFrame([
        {
            "StartTime": "2024-01-15 18:00:00",
            "weekday": 0,
            "weekend": 0,
            "holiday": 0,
            "month": 1,
            "StartCos": 0.0,
            "StartSin": -1.0,
            "StartHourNorm": 0.75,
            "TimeSinceLastStop": 24.0,
            "sessionsToday": 0,
            "LastDuration": 8.0,
            "LastConsumedkWh": 12.0,
            "CarKW": 7.4,
            "CarKWh": 50.0,
        }
    ])
    batch_res = engine.predict_batch_advisories(batch_df)
    assert len(batch_res) == 1, "Batch result length mismatch"
    assert "predicted_Duration" in batch_res.columns, "predicted_Duration missing from batch results"
    assert "predicted_ConsumedkWh" in batch_res.columns, "predicted_ConsumedkWh missing from batch results"
    assert "projected_unplug_time_utc" in batch_res.columns, "projected_unplug_time_utc missing"
    print(
        f"      Batch advisory result: {batch_res['predicted_Duration'].iloc[0]:.2f} h, "
        f"{batch_res['predicted_ConsumedkWh'].iloc[0]:.2f} kWh"
    )

    print("[5/5] All deployment smoke checks PASSED successfully.")


if __name__ == "__main__":
    verify_deployment()
