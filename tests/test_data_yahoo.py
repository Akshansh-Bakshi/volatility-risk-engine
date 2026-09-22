"""Tests for the Yahoo Finance provider, with the yfinance boundary replaced by a fake."""

from __future__ import annotations

import json
import warnings
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yfinance as yf
from yfinance import exceptions as yf_exceptions

from src.data.market_data import INDEX_NAME, STANDARD_COLUMNS
from src.data.provider import MarketDataProvider
from src.data.validation import validate_prices
from src.data.yahoo import YahooFinanceProvider
from src.exceptions import (
    DataFetchError,
    DataValidationError,
    EmptyDataError,
    InvalidTickerError,
)
from tests.data_fakes import (
    UNADJUSTED_FACTOR,
    FakeYahoo,
    datetime_index,
    make_prices,
    make_request,
)

STORED = make_prices()  # business days 2023-01-02 .. 2024-02-23


def provider_with(fake: FakeYahoo) -> YahooFinanceProvider:
    return YahooFinanceProvider(ticker_factory=fake)


@pytest.fixture
def fake() -> FakeYahoo:
    return FakeYahoo({"TEST": STORED})


# --- A. successful fetch -------------------------------------------------------------------


def test_successful_fetch_returns_the_project_schema(fake: FakeYahoo) -> None:
    frame = provider_with(fake).fetch(make_request())

    assert list(frame.columns) == list(STANDARD_COLUMNS)
    assert isinstance(frame.index, pd.DatetimeIndex)
    assert frame.index.tz is None
    assert frame.index.name == INDEX_NAME
    assert list(frame.index) == list(STORED.index)
    assert np.array_equal(frame["close"].to_numpy(), STORED["close"].to_numpy())
    assert np.array_equal(frame["high"].to_numpy(), STORED["high"].to_numpy())
    # ... and the shared validation accepts it unchanged.
    cleaned, dropped = validate_prices(frame, make_request(), min_observations=250)
    assert len(dropped) == 0 and len(cleaned) == len(STORED)


def test_history_is_requested_with_every_value_affecting_option_pinned(fake: FakeYahoo) -> None:
    provider_with(fake).fetch(make_request())

    (call,) = fake.calls
    assert call == {
        "symbol": "TEST",
        "start": "2023-01-02",
        "end": None,
        "interval": "1d",
        "auto_adjust": True,
        "actions": False,
        "repair": False,
        "keepna": True,
        "raise_errors": True,
    }


def test_vendor_only_columns_are_dropped(fake: FakeYahoo) -> None:
    noisy = FakeYahoo({"TEST": STORED}, always_include_actions=True)

    frame = provider_with(noisy).fetch(make_request())

    assert list(frame.columns) == list(STANDARD_COLUMNS)


def test_vendor_column_names_are_matched_case_insensitively(fake: FakeYahoo) -> None:
    fake.override = pd.DataFrame(
        {"OPEN": [1.0], "high": [2.0], " Low ": [0.5], "cLoSe": [1.5], "Volume": [10]},
        index=pd.DatetimeIndex(["2023-01-02"]),
    )

    frame = provider_with(fake).fetch(make_request())

    assert frame.loc["2023-01-02", "close"] == 1.5
    assert frame.loc["2023-01-02", "low"] == 0.5


# --- price basis ------------------------------------------------------------------------------


def test_adjusted_request_uses_the_fully_adjusted_close(fake: FakeYahoo) -> None:
    frame = provider_with(fake).fetch(make_request(adjust_prices=True))

    assert fake.calls[0]["auto_adjust"] is True
    assert np.allclose(frame["close"], STORED["close"])


def test_unadjusted_request_uses_the_vendor_close_not_adj_close(fake: FakeYahoo) -> None:
    frame = provider_with(fake).fetch(make_request(adjust_prices=False))

    assert fake.calls[0]["auto_adjust"] is False
    assert np.allclose(frame["close"], STORED["close"] * UNADJUSTED_FACTOR)
    assert not np.allclose(frame["close"], STORED["close"])
    assert list(frame.columns) == list(STANDARD_COLUMNS), "Adj Close must not leak through"


# --- K, L, M. date semantics ---------------------------------------------------------------------


def test_start_date_is_inclusive(fake: FakeYahoo) -> None:
    start = date(2023, 3, 1)  # a Wednesday with data

    frame = provider_with(fake).fetch(make_request(start=start))

    assert fake.calls[0]["start"] == "2023-03-01"
    assert frame.index[0].date() == start
    assert pd.Timestamp("2023-02-28") not in frame.index


def test_end_date_is_inclusive_although_yfinance_end_is_exclusive(fake: FakeYahoo) -> None:
    end = date(2023, 3, 15)  # a Wednesday with data

    frame = provider_with(fake).fetch(make_request(end=end))

    assert fake.calls[0]["end"] == "2023-03-16", "inclusive end must be shifted by one day"
    assert frame.index[-1].date() == end, "the final requested session must not be lost"
    assert pd.Timestamp("2023-03-16") not in frame.index


def test_end_date_on_a_non_trading_day_returns_the_last_session_before_it(
    fake: FakeYahoo,
) -> None:
    frame = provider_with(fake).fetch(make_request(end=date(2023, 3, 18)))  # a Saturday

    assert frame.index[-1].date() == date(2023, 3, 17)


def test_end_date_none_requests_the_latest_session_without_an_end_bound(fake: FakeYahoo) -> None:
    frame = provider_with(fake).fetch(make_request(end=None))

    assert fake.calls[0]["end"] is None
    assert frame.index[-1] == STORED.index[-1], "the final available session must be included"


def test_end_date_none_is_resolved_on_every_fetch_not_once(fake: FakeYahoo) -> None:
    provider = provider_with(fake)
    request = make_request(end=None)

    first = provider.fetch(request)
    fake.frames["TEST"] = make_prices(periods=305)  # five more sessions appear at Yahoo
    second = provider.fetch(request)

    assert request.end is None, "the request itself never bakes in a date"
    assert len(second) - len(first) == 5
    assert second.index[-1] == fake.frames["TEST"].index[-1]


# --- H. timezone normalisation --------------------------------------------------------------------


@pytest.mark.parametrize(
    "tz",
    ["America/New_York", "Asia/Tokyo", "Asia/Kolkata", "Australia/Sydney", "Europe/London", "UTC"],
)
def test_exchange_local_timestamps_become_naive_session_dates(tz: str) -> None:
    fake = FakeYahoo({"TEST": STORED}, tz=tz)

    frame = provider_with(fake).fetch(make_request())

    assert datetime_index(frame).tz is None
    assert list(frame.index) == list(STORED.index), (
        "labels must be the exchange-local dates; converting to UTC would shift Asian "
        "sessions back by one day"
    )


def test_timestamps_shifted_off_midnight_by_dst_are_normalised_to_the_date() -> None:
    fake = FakeYahoo()
    index = pd.DatetimeIndex(["2023-03-13 01:00", "2023-03-14 00:00"], tz="America/Sao_Paulo")
    fake.override = pd.DataFrame(
        {"Open": [1.0, 1.0], "High": [2.0, 2.0], "Low": [0.5, 0.5], "Close": [1.5, 1.6],
         "Volume": [1, 2]},
        index=index,
    )

    frame = provider_with(fake).fetch(make_request())

    assert list(frame.index) == [pd.Timestamp("2023-03-13"), pd.Timestamp("2023-03-14")]


# --- B. empty result ------------------------------------------------------------------------------


def test_an_empty_vendor_frame_becomes_an_empty_frame_in_the_standard_schema(
    fake: FakeYahoo,
) -> None:
    fake.override = pd.DataFrame()

    frame = provider_with(fake).fetch(make_request())

    assert len(frame) == 0
    assert list(frame.columns) == list(STANDARD_COLUMNS)
    assert isinstance(frame.index, pd.DatetimeIndex) and frame.index.name == INDEX_NAME


def test_vendor_report_of_missing_prices_becomes_empty_data_error(fake: FakeYahoo) -> None:
    request = make_request(start=date(2030, 1, 2))  # beyond the stored history

    with pytest.raises(EmptyDataError, match=r"TEST.*2030-01-02\.\.latest") as error:
        provider_with(fake).fetch(request)

    assert isinstance(error.value.__cause__, yf_exceptions.YFPricesMissingError)


# --- C. malformed provider output -----------------------------------------------------------------


def _frame(**overrides: Any) -> pd.DataFrame:
    columns: dict[str, Any] = {
        "Open": [1.0], "High": [2.0], "Low": [0.5], "Close": [1.5], "Volume": [10],
    }
    columns.update(overrides)
    return pd.DataFrame(columns, index=pd.DatetimeIndex(["2023-01-02"]))


def _multiindex(frame: pd.DataFrame) -> pd.DataFrame:
    frame.columns = pd.MultiIndex.from_product([frame.columns, ["TEST"]])
    return frame


@pytest.mark.parametrize(
    "malformed",
    [
        None,
        [1, 2, 3],
        "not a frame",
        _multiindex(_frame()),
        _frame().drop(columns=["Close"]),
        _frame().set_axis(pd.RangeIndex(1), axis=0),
        pd.concat([_frame(), _frame()[["Close"]]], axis=1),
    ],
    ids=["none", "list", "string", "multiindex", "no-close", "range-index", "duplicate-close"],
)
def test_malformed_vendor_output_raises_data_validation_error(
    fake: FakeYahoo, malformed: object
) -> None:
    fake.override = malformed

    with pytest.raises(DataValidationError, match="Malformed Yahoo Finance output"):
        provider_with(fake).fetch(make_request())


# --- D. invalid ticker / provider failure ---------------------------------------------------------


def test_unknown_symbol_becomes_invalid_ticker_error(fake: FakeYahoo) -> None:
    with pytest.raises(InvalidTickerError, match=r"'NOPE'.*cannot be reached|'NOPE'") as error:
        provider_with(fake).fetch(make_request(ticker="NOPE"))

    assert isinstance(error.value.__cause__, yf_exceptions.YFTzMissingError)
    assert "your connection" in str(error.value), "the message must not overclaim certainty"


@pytest.mark.parametrize(
    "failure",
    [
        ConnectionError("boom"),
        TimeoutError("slow"),
        json.JSONDecodeError("Expecting value", "", 0),  # what yfinance raised when blocked
        KeyError("chart"),
        yf_exceptions.YFDataException("*** YAHOO! FINANCE IS CURRENTLY DOWN! ***"),
    ],
    ids=lambda failure: type(failure).__name__,
)
def test_vendor_failures_become_data_fetch_error_with_cause(
    fake: FakeYahoo, failure: Exception
) -> None:
    fake.error = failure

    with pytest.raises(DataFetchError, match=rf"TEST.*{type(failure).__name__}") as error:
        provider_with(fake).fetch(make_request())

    assert error.value.__cause__ is failure


def test_json_parse_failures_carry_a_connectivity_hint(fake: FakeYahoo) -> None:
    fake.error = json.JSONDecodeError("Expecting value", "", 0)

    with pytest.raises(DataFetchError, match="check network access to Yahoo Finance"):
        provider_with(fake).fetch(make_request())


def test_other_failures_carry_no_connectivity_hint(fake: FakeYahoo) -> None:
    fake.error = KeyError("chart")

    with pytest.raises(DataFetchError) as error:
        provider_with(fake).fetch(make_request())

    assert "network access" not in str(error.value)


def test_rate_limiting_is_reported_as_a_fetch_error(fake: FakeYahoo) -> None:
    fake.error = yf_exceptions.YFRateLimitError()

    with pytest.raises(DataFetchError, match="rate-limited"):
        provider_with(fake).fetch(make_request())


def test_interrupts_are_not_swallowed_by_the_vendor_boundary(fake: FakeYahoo) -> None:
    fake.error = KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        provider_with(fake).fetch(make_request())


# --- compatibility and wiring ---------------------------------------------------------------------


def test_the_raise_errors_deprecation_warning_is_silenced_locally_only(fake: FakeYahoo) -> None:
    filters_before = list(warnings.filters)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any escaping warning would raise
        provider_with(fake).fetch(make_request())
        with pytest.raises(DeprecationWarning, match="something else"):
            warnings.warn("something else", DeprecationWarning, stacklevel=1)

    assert list(warnings.filters) == filters_before


def test_default_ticker_factory_is_yfinance_ticker_resolved_at_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeYahoo({"TEST": STORED})
    monkeypatch.setattr(yf, "Ticker", fake)

    frame = YahooFinanceProvider().fetch(make_request())

    assert len(fake.calls) == 1 and len(frame) == len(STORED)


# --- R. provider abstraction ----------------------------------------------------------------------


def test_yahoo_provider_satisfies_the_provider_contract(fake: FakeYahoo) -> None:
    provider = provider_with(fake)

    assert isinstance(provider, MarketDataProvider)
    assert provider.name == "yahoo_finance"
