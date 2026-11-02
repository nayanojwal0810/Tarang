"""Integration tests for TARANG Streamlit application using Streamlit AppTest."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from tarang.config import FEATURE_NAMES, PROJECT_ROOT, TARGET_DURATION, TARGET_ENERGY

APP_ENTRYPOINT = str((PROJECT_ROOT / "app.py").resolve())


def test_app_loads_overview_page() -> None:
    """Verify application initializes without exceptions and renders the Overview hero."""
    at = AppTest.from_file(APP_ENTRYPOINT, default_timeout=30)
    at.run()
    assert not at.exception
    # Check that Overview contents are present
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
