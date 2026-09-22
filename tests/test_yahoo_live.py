"""Opt-in integration checks against the real Yahoo Finance.

Not part of the deterministic suite: deselected by default (see ``pytest.ini``).
Run explicitly, with network access to Yahoo Finance, using::

    python -m pytest -m live

These tests fail (rather than skip) when Yahoo cannot be reached, so a broken
connection or a changed yfinance API is never mistaken for a pass.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.data.market_data import STANDARD_COLUMNS, MarketDataRequest
from src.data.validation import validate_prices
from src.data.yahoo import YahooFinanceProvider
from src.exceptions import EmptyDataError, InvalidTickerError
from tests.data_fakes import datetime_index

pytestmark = pytest.mark.live


def fetch_validated(request: MarketDataRequest, min_observations: int) -> pd.DataFrame:
    frame = YahooFinanceProvider().fetch(request)
    cleaned, dropped = validate_prices(frame, request, min_observations=min_observations)
    assert len(dropped) == 0
    return cleaned


def test_spy_january_2024_has_the_expected_sessions_and_schema() -> None:
    request = MarketDataRequest("SPY", date(2024, 1, 2), date(2024, 1, 31))

    prices = fetch_validated(request, min_observations=15)

    assert list(prices.columns) == list(STANDARD_COLUMNS)
    assert datetime_index(prices).tz is None and prices.index.name == "date"
    assert prices.index[0] == pd.Timestamp("2024-01-02"), "start is inclusive"
    assert prices.index[-1] == pd.Timestamp("2024-01-31"), "the inclusive end session is kept"
    assert len(prices) == 21, "January 2024 had 21 NYSE sessions from the 2nd (New Year, MLK off)"
    assert (prices["close"] > 0).all()


def test_adjusted_close_differs_from_the_unadjusted_close_for_a_dividend_payer() -> None:
    def close(adjust_prices: bool) -> pd.Series:
        request = MarketDataRequest(
            "SPY", date(2024, 1, 2), date(2024, 1, 31), adjust_prices=adjust_prices
        )
        return fetch_validated(request, 15)["close"]

    adjusted, unadjusted = close(True), close(False)

    assert not np.allclose(adjusted, unadjusted)
    assert (adjusted <= unadjusted).all(), "later dividends can only lower the adjusted history"


def test_unadjusted_close_is_still_split_adjusted() -> None:
    """Apple's 4-for-1 split took effect on 2020-08-31; Yahoo's plain Close must not jump 4x."""
    request = MarketDataRequest(
        "AAPL", date(2020, 8, 24), date(2020, 9, 4), adjust_prices=False
    )

    close = fetch_validated(request, min_observations=5)["close"]

    assert close.max() / close.min() < 1.5, "an unadjusted split would show as a ~4x jump"


def test_tokyo_sessions_keep_their_local_dates() -> None:
    request = MarketDataRequest("^N225", date(2024, 1, 3), date(2024, 1, 12))

    prices = fetch_validated(request, min_observations=4)

    assert prices.index[0] == pd.Timestamp("2024-01-04"), "first session after the holiday"
    weekdays = datetime_index(prices).dayofweek
    assert not (weekdays >= 5).any(), "a UTC shift would produce weekend labels"


def test_an_unknown_symbol_is_reported_as_a_domain_error() -> None:
    request = MarketDataRequest("ZZZZ-NOT-A-REAL-SYMBOL", date(2024, 1, 2), date(2024, 1, 31))

    with pytest.raises((InvalidTickerError, EmptyDataError)):
        YahooFinanceProvider().fetch(request)
