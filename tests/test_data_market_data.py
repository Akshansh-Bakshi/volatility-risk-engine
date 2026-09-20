"""Tests for the market data domain types."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any

import pandas as pd
import pytest

from src.config import DataConfig
from src.data.market_data import (
    INDEX_NAME,
    PRICE_COLUMN,
    STANDARD_COLUMNS,
    SUPPORTED_INTERVALS,
    MarketData,
    MarketDataRequest,
)
from src.exceptions import DataError, DataValidationError
from tests.data_fakes import make_prices

UTC_NOW = datetime(2024, 3, 1, 12, 0, tzinfo=timezone.utc)


def test_schema_constants_define_one_authoritative_price_column() -> None:
    assert PRICE_COLUMN == "close"
    assert PRICE_COLUMN in STANDARD_COLUMNS
    assert STANDARD_COLUMNS == ("open", "high", "low", "close", "volume")
    assert INDEX_NAME == "date"
    assert SUPPORTED_INTERVALS == ("1d",)


def test_request_normalises_ticker_and_applies_defaults() -> None:
    request = MarketDataRequest(ticker="  spy ", start=date(2020, 1, 1))

    assert request.ticker == "SPY"
    assert request.end is None
    assert request.interval == "1d"
    assert request.adjust_prices is True


def test_request_range_label_names_open_ended_and_explicit_ranges() -> None:
    open_ended = MarketDataRequest("SPY", date(2020, 1, 1))
    bounded = MarketDataRequest("SPY", date(2020, 1, 1), date(2021, 1, 1))

    assert open_ended.range_label == "2020-01-01..latest"
    assert bounded.range_label == "2020-01-01..2021-01-01"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ticker": ""},
        {"ticker": "   "},
        {"ticker": None},
        {"start": datetime(2020, 1, 1)},
        {"start": "2020-01-01"},
        {"end": datetime(2021, 1, 1)},
        {"end": "2021-01-01"},
        {"end": date(2020, 1, 1)},  # equal to start
        {"end": date(2019, 12, 31)},  # before start
        {"interval": "1h"},
        {"interval": "1wk"},
        {"adjust_prices": "yes"},
    ],
)
def test_invalid_requests_raise_data_validation_error(kwargs: dict[str, Any]) -> None:
    fields: dict[str, Any] = {"ticker": "SPY", "start": date(2020, 1, 1), "end": date(2021, 1, 1)}
    fields.update(kwargs)

    with pytest.raises(DataValidationError) as error:
        MarketDataRequest(**fields)

    assert isinstance(error.value, DataError)


def test_impossible_range_message_names_ticker_and_both_dates() -> None:
    with pytest.raises(DataValidationError, match=r"SPY.*2021-06-01.*2020-01-01"):
        MarketDataRequest("SPY", date(2021, 6, 1), date(2020, 1, 1))


def test_request_from_config_carries_the_configured_defaults() -> None:
    config = DataConfig(
        ticker="qqq", start_date=date(2010, 1, 4), end_date=None, adjust_prices=False
    )

    request = MarketDataRequest.from_config(config)

    assert request == MarketDataRequest("QQQ", date(2010, 1, 4), None, "1d", False)


def test_request_from_config_keeps_an_explicit_end_date() -> None:
    config = DataConfig(start_date=date(2010, 1, 4), end_date=date(2012, 1, 4))

    assert MarketDataRequest.from_config(config).end == date(2012, 1, 4)


def test_market_data_exposes_the_authoritative_series_and_date_bounds() -> None:
    prices = make_prices(periods=10)
    data = MarketData(
        request=MarketDataRequest("TEST", date(2023, 1, 2)),
        prices=prices,
        source="unit_test",
        fetched_at=UTC_NOW,
    )

    assert data.ticker == "TEST"
    assert data.observations == 10
    assert data.first_date == date(2023, 1, 2)
    assert data.last_date == date(2023, 1, 13)
    assert isinstance(data.price, pd.Series)
    assert data.price.name == "close"
    assert data.price.equals(prices["close"])
    assert data.rows_dropped == 0
    assert data.from_cache is False


@pytest.mark.parametrize(
    "fetched_at",
    [
        datetime(2024, 3, 1, 12, 0),  # naive
        datetime(2024, 3, 1, 12, 0, tzinfo=timezone(timedelta(hours=5, minutes=30))),  # not UTC
    ],
)
def test_fetched_at_must_be_timezone_aware_utc(fetched_at: datetime) -> None:
    with pytest.raises(DataValidationError, match="fetched_at"):
        MarketData(
            request=MarketDataRequest("TEST", date(2023, 1, 2)),
            prices=make_prices(periods=3),
            source="unit_test",
            fetched_at=fetched_at,
        )
