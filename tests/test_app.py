"""Smoke tests that the Streamlit entry point renders."""

from __future__ import annotations

from pathlib import Path
from datetime import date

import pytest
from streamlit.testing.v1 import AppTest

from app import _price_unit, load_synopsis_data, resolve_ticker
from tests.data_fakes import make_market_data, make_prices

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"
APP_TITLE = "Volatility Analytics"


def _run_app() -> AppTest:
    return AppTest.from_file(str(APP_PATH), default_timeout=30).run()


def test_app_renders_without_errors() -> None:
    at = _run_app()

    assert not at.exception
    assert [title.value for title in at.title] == [APP_TITLE]
    assert not at.error


def test_app_does_not_expose_configuration_or_internal_details() -> None:
    at = _run_app()

    assert not at.exception
    assert not at.json
    visible = " ".join(
        [element.value for group in (at.title, at.header, at.subheader, at.markdown, at.caption, at.info)
         for element in group]
    )
    assert "Active default configuration" not in visible
    assert "Proposed methodology" not in visible
    assert "C:\\Users\\" not in visible
    assert "Volatility Analytics" in visible


def test_app_reports_invalid_configuration_instead_of_crashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VRE_LOG_LEVEL", "VERBOSE")

    at = _run_app()

    assert not at.exception
    assert len(at.error) == 1
    assert "configuration" in at.error[0].value.lower()
    assert not at.json


def test_synopsis_orchestration_uses_loader_and_existing_eda() -> None:
    class StubLoader:
        def __init__(self) -> None:
            self.request = None

        def load(self, request):
            self.request = request
            return make_market_data(
                make_prices(periods=520), ticker=request.ticker, source="test_provider"
            )

    loader = StubLoader()
    result = load_synopsis_data(loader, "^NSEI", date(2022, 1, 1), date(2024, 1, 1))

    assert loader.request.ticker == "^NSEI"
    assert loader.request.start == date(2022, 1, 1)
    assert loader.request.end == date(2024, 1, 1)
    assert result.profile.source == "test_provider"
    assert result.profile.price_observations == 520
    assert result.profile.return_observations == 518
    assert result.profile.quality.provisional_bar is not None
    assert result.profile.quality.provisional_bar_excluded
    assert result.stats.count == result.returns.return_observations


def test_asset_presets_and_custom_ticker_resolution() -> None:
    assert resolve_ticker("NIFTY 50 (^NSEI)") == "^NSEI"
    assert resolve_ticker("Apple (AAPL)") == "AAPL"
    assert resolve_ticker("Custom ticker", custom_ticker="  msft ") == "MSFT"


def test_asset_price_units_are_correct_for_indices_and_equities() -> None:
    assert _price_unit("^NSEI") == "index points"
    assert _price_unit("RELIANCE.NS") == "₹"
    assert _price_unit("AAPL") == "$"
