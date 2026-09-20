"""Tests for the data-layer composition root and its use of Settings."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
import yfinance as yf

from src.config import DataConfig, Settings, get_settings
from src.data.factory import create_market_data_loader
from src.data.market_data import MarketDataRequest
from src.exceptions import InsufficientHistoryError
from tests.data_fakes import FakeYahoo, StaticProvider, make_prices, make_request


def settings_for(data_dir: Path, **overrides: object) -> Settings:
    return Settings(data=DataConfig(data_dir=data_dir, **overrides))  # type: ignore[arg-type]


def test_cache_lives_under_the_configured_data_directory(tmp_path: Path) -> None:
    loader = create_market_data_loader(
        settings_for(tmp_path), provider=StaticProvider(frame=make_prices())
    )

    loader.load(make_request())

    expected = tmp_path / "cache" / "market" / "TEST__1d__2023-01-02__latest__adjusted.json"
    assert expected.is_file()


def test_disabling_the_cache_in_settings_bypasses_it_completely(tmp_path: Path) -> None:
    provider = StaticProvider(frame=make_prices())
    loader = create_market_data_loader(settings_for(tmp_path, use_cache=False), provider=provider)

    loader.load(make_request())
    loader.load(make_request())

    assert len(provider.requests) == 2
    assert list(tmp_path.iterdir()) == []


def test_min_observations_comes_from_settings(tmp_path: Path) -> None:
    loader = create_market_data_loader(
        settings_for(tmp_path, min_observations=301), provider=StaticProvider(frame=make_prices())
    )

    with pytest.raises(InsufficientHistoryError, match="at least 301"):
        loader.load(make_request())


def test_cache_max_age_comes_from_settings(tmp_path: Path) -> None:
    provider = StaticProvider(frame=make_prices())
    # 1e-9 hours is a few microseconds: any real delay between two loads exceeds it.
    loader = create_market_data_loader(
        settings_for(tmp_path, cache_max_age_hours=1e-9), provider=provider
    )

    loader.load(make_request(end=None))
    second = loader.load(make_request(end=None))

    assert len(provider.requests) == 2 and second.from_cache is False


def test_yahoo_finance_is_the_default_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeYahoo({"TEST": make_prices()})
    monkeypatch.setattr(yf, "Ticker", fake)

    data = create_market_data_loader(settings_for(tmp_path)).load(make_request())

    assert data.source == "yahoo_finance"
    assert len(fake.calls) == 1


def test_environment_configuration_flows_through_to_the_provider_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeYahoo({"TEST": make_prices()})
    monkeypatch.setattr(yf, "Ticker", fake)
    for name, value in {
        "VRE_DEFAULT_TICKER": "test",
        "VRE_DEFAULT_START_DATE": "2023-01-02",
        "VRE_ADJUST_PRICES": "false",
        "VRE_DATA_DIR": str(tmp_path),
        "VRE_MIN_OBSERVATIONS": "10",
    }.items():
        monkeypatch.setenv(name, value)

    request = MarketDataRequest.from_config(get_settings().data)
    data = create_market_data_loader().load(request)

    assert request.end is None and request.adjust_prices is False
    assert fake.calls[0]["symbol"] == "TEST" and fake.calls[0]["auto_adjust"] is False
    assert fake.calls[0]["start"] == date(2023, 1, 2).isoformat()
    assert data.observations == 300
    cache_file = tmp_path / "cache" / "market" / "TEST__1d__2023-01-02__latest__unadjusted.json"
    assert cache_file.exists()
