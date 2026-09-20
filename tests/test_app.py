"""Smoke tests that the Streamlit entry point renders."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
APP_TITLE = "Volatility Analytics & Market Risk Engine"


def _run_app() -> AppTest:
    return AppTest.from_file(str(APP_PATH), default_timeout=30).run()


def test_app_renders_without_errors() -> None:
    at = _run_app()

    assert not at.exception
    assert [title.value for title in at.title] == [APP_TITLE]
    assert not at.error


def test_app_shows_the_active_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VRE_DEFAULT_TICKER", "spy")
    monkeypatch.setenv("VRE_CONFIDENCE_LEVELS", "0.9,0.975")

    at = _run_app()

    assert not at.exception
    shown = json.loads(at.json[0].value)
    assert shown["data"]["ticker"] == "SPY"
    assert shown["risk"]["confidence_levels"] == [0.9, 0.975]


def test_app_reports_invalid_configuration_instead_of_crashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VRE_LOG_LEVEL", "VERBOSE")

    at = _run_app()

    assert not at.exception
    assert len(at.error) == 1
    assert "Invalid configuration" in at.error[0].value
    assert not at.json
