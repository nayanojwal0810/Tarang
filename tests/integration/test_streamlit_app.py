"""Integration tests for TARANG Streamlit application using Streamlit AppTest."""

from __future__ import annotations

import io
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
from streamlit.testing.v1 import AppTest

from tarang.config import FEATURE_NAMES, PROJECT_ROOT, TARGET_DURATION, TARGET_ENERGY

APP_ENTRYPOINT = str((PROJECT_ROOT / "app.py").resolve())


@pytest.fixture(autouse=True)
def isolated_test_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Create lightweight test models and metadata in tmp_path so tests run cleanly without local files."""
    duration_dir = tmp_path / "duration" / "models"
    energy_dir = tmp_path / "energy" / "models"
    duration_dir.mkdir(parents=True, exist_ok=True)
    energy_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(42)
    n_samples = 25
    X_train = pd.DataFrame(np.random.rand(n_samples, 13) * 10.0, columns=FEATURE_NAMES)
    y_duration = X_train["LastDuration"] * 0.8 + 2.0
    y_energy = X_train["CarKWh"] * 0.5 + 5.0

    hgbr_dur = HistGradientBoostingRegressor(random_state=0).fit(X_train, y_duration)
    hgbr_ene = HistGradientBoostingRegressor(random_state=0).fit(X_train, y_energy)

    joblib.dump(hgbr_dur, duration_dir / "HistGradientBoostingRegressor.joblib")
    joblib.dump(hgbr_ene, energy_dir / "HistGradientBoostingRegressor.joblib")

    summary_file = tmp_path / "processing_summary.json"
    summary_file.write_text(json.dumps({"max_start_hour": 23.95}), encoding="utf-8")

    monkeypatch.setenv("TARANG_MODELS_DIR", str(tmp_path))
    monkeypatch.setenv("TARANG_SUMMARY_FILE", str(summary_file))

    from tarang.app.helpers import get_decision_support_engine
    get_decision_support_engine.clear()


def test_app_loads_overview_page() -> None:
    """Verify application initializes without exceptions and renders the Overview hero."""
    at = AppTest.from_file(APP_ENTRYPOINT, default_timeout=30)
    at.run()
    assert not at.exception
    assert len(at.markdown) > 0


def test_app_navigate_to_single_prediction() -> None:
    """Verify navigation to Single Prediction view and execution with sample history."""
    at = AppTest.from_file(APP_ENTRYPOINT, default_timeout=30)
    at.run()

    # Navigate to Single Prediction via sidebar radio (index 1)
    at.sidebar.radio[0].set_value("Single Prediction").run()
    assert not at.exception

    # Click "Use Sample History Data" button
    sample_btn = [b for b in at.button if "Use Sample History Data" in b.label]
    assert len(sample_btn) == 1
    sample_btn[0].click().run()
    assert not at.exception

    # Click "Generate Session Advisory" button
    predict_btn = [b for b in at.button if "Generate Session Advisory" in b.label]
    assert len(predict_btn) == 1
    predict_btn[0].click().run()
    assert not at.exception


def test_single_prediction_numeric_participant_selection() -> None:
    """Verify single prediction correctly handles and filters numeric participant IDs."""
    at = AppTest.from_file(APP_ENTRYPOINT, default_timeout=30)
    at.run()

    at.sidebar.radio[0].set_value("Single Prediction").run()

    # Set history with numeric IDs (101 and 102)
    numeric_history = pd.DataFrame(
        [
            {"ParticipantID": 101, "StartTime": "2026-11-01 10:00:00+00:00", "StopTime": "2026-11-01 18:00:00+00:00", "ConsumedkWh": 15.0},
            {"ParticipantID": 102, "StartTime": "2026-11-02 10:00:00+00:00", "StopTime": "2026-11-02 18:00:00+00:00", "ConsumedkWh": 20.0},
        ]
    ).to_csv(index=False)

    at.session_state["single_history_data"] = numeric_history
    at.session_state["history_source"] = "sample"
    at.run()
    assert not at.exception

    # Ensure selectbox for participants appeared with string options
    assert len(at.selectbox) > 0
    at.selectbox[0].set_value("101").run()
    assert not at.exception

    # Click generate
    predict_btn = [b for b in at.button if "Generate Session Advisory" in b.label]
    assert len(predict_btn) == 1
    predict_btn[0].click().run()
    assert not at.exception


def test_single_prediction_missing_participant_id_rejection() -> None:
    """Verify single prediction explicitly rejects mixed-null participant histories."""
    at = AppTest.from_file(APP_ENTRYPOINT, default_timeout=30)
    at.run()

    at.sidebar.radio[0].set_value("Single Prediction").run()

    # Set history with mixed null IDs
    mixed_history = pd.DataFrame(
        [
            {"ParticipantID": "WP1-001", "StartTime": "2026-11-01 10:00:00+00:00", "StopTime": "2026-11-01 18:00:00+00:00", "ConsumedkWh": 15.0},
            {"ParticipantID": None, "StartTime": "2026-11-02 10:00:00+00:00", "StopTime": "2026-11-02 18:00:00+00:00", "ConsumedkWh": 20.0},
        ]
    ).to_csv(index=False)

    at.session_state["single_history_data"] = mixed_history
    at.session_state["history_source"] = "sample"
    at.run()

    # An error banner must appear indicating mixed/missing participant identifiers
    assert len(at.error) > 0
    assert any("missing/null participant identifiers" in e.value for e in at.error)


def test_app_navigate_to_batch_prediction() -> None:
    """Verify navigation to Batch Prediction view and execution with sample batch data."""
    at = AppTest.from_file(APP_ENTRYPOINT, default_timeout=30)
    at.run()

    # Navigate to Batch Prediction via sidebar radio (index 2)
    at.sidebar.radio[0].set_value("Batch Prediction").run()
    assert not at.exception

    # Click "Load Sample Batch Data (50 sessions)" button
    load_sample_btn = [b for b in at.button if "Load Sample Batch Data" in b.label]
    assert len(load_sample_btn) == 1
    load_sample_btn[0].click().run()
    assert not at.exception

    # Click "Generate Batch Advisories" button
    gen_btn = [b for b in at.button if "Generate Batch Advisories" in b.label]
    assert len(gen_btn) == 1
    gen_btn[0].click().run()
    assert not at.exception


def test_batch_prediction_stale_results_not_displayed() -> None:
    """Verify modifying or switching the batch dataset immediately clears old results until re-evaluated."""
    from tarang.app.helpers import get_sample_batch_dataframe

    at = AppTest.from_file(APP_ENTRYPOINT, default_timeout=30)
    at.run()

    at.sidebar.radio[0].set_value("Batch Prediction").run()

    # 1. Load dataset A (50 sessions)
    load_sample_btn = [b for b in at.button if "Load Sample Batch Data" in b.label]
    load_sample_btn[0].click().run()

    # Generate advisories for dataset A
    gen_btn = [b for b in at.button if "Generate Batch Advisories" in b.label]
    gen_btn[0].click().run()
    assert not at.exception

    # Verify advisories are displayed
    assert len(at.success) > 0
    assert any("Successfully generated advisories for 50 sessions" in s.value for s in at.success)

    # 2. Switch dataset to dataset B (20 sessions)
    df_b, is_synth = get_sample_batch_dataframe(n_rows=20)
    at.session_state["sample_batch_df"] = df_b
    at.session_state["batch_data_source"] = "sample"
    at.run()

    # Verify old dataset A results are NOT displayed
    assert not any("Successfully generated advisories for 50 sessions" in s.value for s in at.success)
    # Prompt to click generate must be visible
    assert any("Click **Generate Batch Advisories**" in i.value for i in at.info)

    # 3. Generate advisories for dataset B
    gen_btn = [b for b in at.button if "Generate Batch Advisories" in b.label]
    gen_btn[0].click().run()
    assert not at.exception
    # Now dataset B results (20 sessions) are displayed
    assert any("Successfully generated advisories for 20 sessions" in s.value for s in at.success)
