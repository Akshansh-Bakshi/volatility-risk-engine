"""Tests for the market data loader: orchestration, caching, logging, error propagation."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.data.cache import MarketDataCache
from src.data.loader import MarketDataLoader
from src.data.market_data import STANDARD_COLUMNS, MarketData
from src.data.provider import MarketDataProvider
from src.data.yahoo import YahooFinanceProvider
from src.exceptions import (
    DataError,
    DataFetchError,
    DataValidationError,
    EmptyDataError,
    InsufficientHistoryError,
    InvalidTickerError,
)
from tests.data_fakes import FakeYahoo, StaticProvider, datetime_index, make_prices, make_request

STORED = make_prices()  # 300 sessions: 2023-01-02 .. 2024-02-23
T0 = datetime(2024, 3, 1, 12, 0, tzinfo=timezone.utc)
LOADER_LOGGER = "src.data.loader"


class Clock:
    """A settable UTC clock."""

    def __init__(self, now: datetime = T0) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **delta: float) -> None:
        self.now += timedelta(**delta)


@dataclass
class Rig:
    loader: MarketDataLoader
    fake: FakeYahoo
    clock: Clock
    cache_dir: Path

    @property
    def cache(self) -> MarketDataCache:
        return MarketDataCache(self.cache_dir)


def build_rig(tmp_path: Path, *, cached: bool = True, min_observations: int = 250) -> Rig:
    fake = FakeYahoo({"TEST": STORED})
    clock = Clock()
    cache_dir = tmp_path / "cache"
    loader = MarketDataLoader(
        YahooFinanceProvider(ticker_factory=fake),
        cache=MarketDataCache(cache_dir) if cached else None,
        min_observations=min_observations,
        cache_max_age=timedelta(hours=6),
        clock=clock,
    )
    return Rig(loader, fake, clock, cache_dir)


@pytest.fixture
def rig(tmp_path: Path) -> Rig:
    return build_rig(tmp_path)


def loader_records(records: list[logging.LogRecord], level: int) -> list[logging.LogRecord]:
    return [r for r in records if r.name == LOADER_LOGGER and r.levelno == level]


# --- A. successful load ------------------------------------------------------------------------


def test_successful_load_returns_validated_market_data(rig: Rig) -> None:
    data = rig.loader.load(make_request())

    assert isinstance(data, MarketData)
    assert data.ticker == "TEST"
    assert data.source == "yahoo_finance"
    assert data.fetched_at == T0 and data.from_cache is False
    assert data.observations == 300 and data.rows_dropped == 0
    assert (data.first_date, data.last_date) == (date(2023, 1, 2), date(2024, 2, 23))
    assert list(data.prices.columns) == list(STANDARD_COLUMNS)
    assert datetime_index(data.prices).tz is None
    assert np.array_equal(data.price.to_numpy(), STORED["close"].to_numpy())


def test_loading_is_deterministic(tmp_path: Path) -> None:
    first = build_rig(tmp_path / "a", cached=False).loader.load(make_request())
    second = build_rig(tmp_path / "b", cached=False).loader.load(make_request())

    pd.testing.assert_frame_equal(first.prices, second.prices, check_exact=True)


# --- R. provider abstraction ----------------------------------------------------------------------


def test_loader_works_with_any_object_honouring_the_provider_contract() -> None:
    provider = StaticProvider(frame=STORED, name="custom_source")
    request = make_request()

    data = MarketDataLoader(provider, clock=Clock()).load(request)

    assert isinstance(provider, MarketDataProvider)
    assert provider.requests == [request], "the provider receives the request unchanged"
    assert data.source == "custom_source"
    assert data.observations == 300


# --- B, C, D, E, F, G, I, J. failures propagate as domain errors ----------------------------------


def _frame(edit: Callable[[pd.DataFrame], pd.DataFrame]) -> StaticProvider:
    return StaticProvider(frame=edit(make_prices()))


def _all_missing(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.assign(close=np.nan)


FAILURES: list[tuple[str, StaticProvider, type[DataError]]] = [
    ("B-empty", _frame(lambda f: f.iloc[0:0]), EmptyDataError),
    ("J-all-close-missing", _frame(_all_missing), EmptyDataError),
    ("C-missing-column", _frame(lambda f: f.drop(columns=["close"])), DataValidationError),
    ("F-duplicate-dates", _frame(lambda f: pd.concat([f, f.iloc[[3]]]).sort_index()),
     DataValidationError),
    ("G-unsorted", _frame(lambda f: f.iloc[::-1]), DataValidationError),
    ("I-non-positive", _frame(lambda f: f.assign(close=f["close"] - 200.0)), DataValidationError),
    ("E-insufficient", _frame(lambda f: f.iloc[:100]), InsufficientHistoryError),
    ("D-invalid-ticker", StaticProvider(error=InvalidTickerError("unknown symbol")),
     InvalidTickerError),
    ("D-fetch-failure", StaticProvider(error=DataFetchError("provider down")), DataFetchError),
]


@pytest.mark.parametrize(
    ("provider", "expected"), [(p, e) for _, p, e in FAILURES], ids=[n for n, _, _ in FAILURES]
)
def test_failures_surface_as_the_matching_domain_error_and_cache_nothing(
    tmp_path: Path, provider: StaticProvider, expected: type[DataError]
) -> None:
    cache = MarketDataCache(tmp_path / "cache")
    loader = MarketDataLoader(provider, cache=cache, clock=Clock())

    with pytest.raises(expected):
        loader.load(make_request())

    assert not cache.directory.exists() or not any(cache.directory.iterdir())


def test_provider_errors_are_not_wrapped_or_replaced() -> None:
    original = InvalidTickerError("unknown symbol XYZ")
    loader = MarketDataLoader(StaticProvider(error=original), clock=Clock())

    with pytest.raises(InvalidTickerError) as error:
        loader.load(make_request())

    assert error.value is original


def test_rows_without_a_close_are_removed_counted_and_survive_the_cache(tmp_path: Path) -> None:
    frame = make_prices()
    missing = frame.index[[10, 11]]
    frame.loc[missing, "close"] = np.nan
    loader = MarketDataLoader(
        StaticProvider(frame=frame),
        cache=MarketDataCache(tmp_path / "cache"),
        clock=Clock(),
    )

    fresh = loader.load(make_request())
    cached = loader.load(make_request())

    assert fresh.rows_dropped == cached.rows_dropped == 2
    assert fresh.observations == 298 and not fresh.prices.index.isin(missing).any()
    assert cached.from_cache is True
    pd.testing.assert_frame_equal(fresh.prices, cached.prices, check_exact=True)


def test_min_observations_is_enforced_by_the_loader(tmp_path: Path) -> None:
    strict = build_rig(tmp_path, min_observations=301)

    with pytest.raises(InsufficientHistoryError, match=r"300 usable.*at least 301"):
        strict.loader.load(make_request())


# --- N. cache hit ---------------------------------------------------------------------------------


def test_second_load_is_served_from_the_cache_without_calling_the_provider(rig: Rig) -> None:
    first = rig.loader.load(make_request())
    rig.clock.advance(minutes=30)

    second = rig.loader.load(make_request())

    assert len(rig.fake.calls) == 1
    assert first.from_cache is False and second.from_cache is True
    assert second.fetched_at == first.fetched_at == T0, "provenance keeps the download time"
    pd.testing.assert_frame_equal(first.prices, second.prices, check_exact=True)
    assert second.source == "yahoo_finance"


def test_cached_data_is_revalidated_under_the_current_minimum(tmp_path: Path) -> None:
    lenient = build_rig(tmp_path, min_observations=250)
    lenient.loader.load(make_request())
    provider = StaticProvider(frame=STORED)
    strict = MarketDataLoader(
        provider,
        cache=MarketDataCache(lenient.cache_dir),
        min_observations=301,
        clock=Clock(),
    )

    with pytest.raises(InsufficientHistoryError, match="at least 301"):
        strict.load(make_request())

    assert len(provider.requests) == 1, "the entry was rejected, so the provider was consulted"


# --- O. refresh / bypass --------------------------------------------------------------------------


def test_refresh_skips_the_cache_read_and_overwrites_the_entry(rig: Rig) -> None:
    rig.loader.load(make_request())
    rig.clock.advance(hours=1)

    refreshed = rig.loader.load(make_request(), refresh=True)
    after = rig.loader.load(make_request())

    assert len(rig.fake.calls) == 2
    assert refreshed.from_cache is False and refreshed.fetched_at == T0 + timedelta(hours=1)
    assert after.from_cache is True and after.fetched_at == refreshed.fetched_at


def test_a_loader_without_a_cache_never_reads_or_writes_disk(tmp_path: Path) -> None:
    uncached = build_rig(tmp_path, cached=False)

    uncached.loader.load(make_request())
    again = uncached.loader.load(make_request())
    refreshed = uncached.loader.load(make_request(), refresh=True)

    assert len(uncached.fake.calls) == 3
    assert not again.from_cache and not refreshed.from_cache
    assert not uncached.cache_dir.exists()


# --- staleness ------------------------------------------------------------------------------------


def test_latest_data_is_reused_only_while_it_is_current(rig: Rig) -> None:
    rig.loader.load(make_request(end=None))

    rig.clock.advance(hours=5)
    within = rig.loader.load(make_request(end=None))
    rig.clock.advance(hours=2)  # now 7h after the download
    expired = rig.loader.load(make_request(end=None))
    rig.clock.advance(minutes=10)
    renewed = rig.loader.load(make_request(end=None))

    assert (within.from_cache, expired.from_cache, renewed.from_cache) == (True, False, True)
    assert len(rig.fake.calls) == 2
    assert expired.fetched_at == T0 + timedelta(hours=7)


def test_settled_history_is_reused_indefinitely(tmp_path: Path) -> None:
    provider = StaticProvider(frame=STORED)
    clock = Clock(datetime(2025, 6, 1, 8, 0, tzinfo=timezone.utc))  # well after the window
    loader = MarketDataLoader(provider, cache=MarketDataCache(tmp_path), clock=clock)
    request = make_request(end=date(2024, 12, 31))

    loader.load(request)
    clock.advance(days=5 * 365)
    later = loader.load(request)

    assert len(provider.requests) == 1 and later.from_cache is True


def test_a_window_still_open_at_download_time_is_treated_as_live(tmp_path: Path) -> None:
    provider = StaticProvider(frame=STORED)
    clock = Clock()  # 2024-03-01, long before the requested end
    loader = MarketDataLoader(provider, cache=MarketDataCache(tmp_path), clock=clock)
    request = make_request(end=date(2024, 12, 31))

    loader.load(request)
    clock.advance(hours=7)
    loader.load(request)

    assert len(provider.requests) == 2


# --- cache identity: no cross-contamination -------------------------------------------------------


def test_latest_and_explicit_end_requests_do_not_share_entries(rig: Rig) -> None:
    latest = rig.loader.load(make_request(end=None))
    bounded = rig.loader.load(make_request(end=date(2024, 2, 9)))

    assert len(rig.fake.calls) == 2
    assert bounded.last_date == date(2024, 2, 9) and latest.last_date == date(2024, 2, 23)


def test_adjusted_and_unadjusted_requests_do_not_share_entries(rig: Rig) -> None:
    adjusted = rig.loader.load(make_request(adjust_prices=True))
    unadjusted = rig.loader.load(make_request(adjust_prices=False))
    adjusted_again = rig.loader.load(make_request(adjust_prices=True))

    assert len(rig.fake.calls) == 2
    assert not np.allclose(adjusted.price, unadjusted.price)
    assert adjusted_again.from_cache and np.array_equal(adjusted_again.price, adjusted.price)


# --- damaged cache entries ------------------------------------------------------------------------


def test_a_corrupt_cache_file_triggers_a_fresh_download_and_is_repaired(
    rig: Rig, log_records: list[logging.LogRecord]
) -> None:
    request = make_request()
    rig.loader.load(request)
    rig.cache.path_for(request).write_text("{truncated", encoding="utf-8")

    recovered = rig.loader.load(request)
    afterwards = rig.loader.load(request)

    assert recovered.from_cache is False and afterwards.from_cache is True
    assert len(rig.fake.calls) == 2
    assert any("unusable cache entry" in r.getMessage() for r in log_records)


def test_a_cache_entry_that_no_longer_validates_is_discarded(
    rig: Rig, log_records: list[logging.LogRecord]
) -> None:
    request = make_request()
    rig.loader.load(request)
    path = rig.cache.path_for(request)
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    payload["columns"]["close"][7] = -1.0
    path.write_text(json.dumps(payload), encoding="utf-8")

    recovered = rig.loader.load(request)

    assert recovered.from_cache is False and (recovered.price > 0).all()
    assert len(rig.fake.calls) == 2
    warnings = loader_records(log_records, logging.WARNING)
    assert len(warnings) == 1 and "no longer validates" in warnings[0].getMessage()
    assert "non-positive" in warnings[0].getMessage()


def test_a_cache_write_failure_does_not_lose_the_data(
    tmp_path: Path, log_records: list[logging.LogRecord]
) -> None:
    blocker = tmp_path / "not_a_directory"
    blocker.write_text("occupied")
    loader = MarketDataLoader(
        StaticProvider(frame=STORED), cache=MarketDataCache(blocker / "cache"), clock=Clock()
    )

    data = loader.load(make_request())

    assert data.observations == 300 and data.from_cache is False
    warnings = loader_records(log_records, logging.WARNING)
    assert len(warnings) == 1 and "Could not write cache entry" in warnings[0].getMessage()


# --- P. logging on success ------------------------------------------------------------------------


def test_success_is_logged_with_full_context(
    rig: Rig, log_records: list[logging.LogRecord]
) -> None:
    rig.loader.load(make_request())

    (record,) = loader_records(log_records, logging.INFO)
    message = record.getMessage()
    for expected in (
        "ticker=TEST",
        "interval=1d",
        "adjust_prices=True",
        "requested=2023-01-02..latest",
        "actual=2023-01-02..2024-02-23",
        "rows=300",
        "rows_dropped=0",
        "source=yahoo_finance",
        f"fetched_at={T0.isoformat()}",
        "cache=miss",
    ):
        assert expected in message, f"missing {expected!r} in: {message}"


def test_log_reports_the_cache_status_of_each_path(
    tmp_path: Path, log_records: list[logging.LogRecord]
) -> None:
    rig = build_rig(tmp_path)
    request = make_request()
    rig.loader.load(request)
    rig.loader.load(request)
    rig.loader.load(request, refresh=True)
    rig.clock.advance(hours=7)
    rig.loader.load(request)
    build_rig(tmp_path / "off", cached=False).loader.load(request)

    statuses = [
        r.getMessage().rsplit("cache=", 1)[1] for r in loader_records(log_records, logging.INFO)
        if r.getMessage().startswith("Loaded")
    ]
    assert statuses == ["miss", "hit", "refresh", "stale", "disabled"]


def test_dropped_rows_are_reported_in_the_success_log(
    tmp_path: Path, log_records: list[logging.LogRecord]
) -> None:
    frame = make_prices()
    frame.loc[frame.index[[3, 4, 5]], "close"] = np.nan

    MarketDataLoader(StaticProvider(frame=frame), clock=Clock()).load(make_request())

    (record,) = loader_records(log_records, logging.INFO)
    assert "rows=297" in record.getMessage() and "rows_dropped=3" in record.getMessage()


# --- Q. logging on failure ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("provider", "expected"), [(p, e) for _, p, e in FAILURES], ids=[n for n, _, _ in FAILURES]
)
def test_every_failure_is_logged_once_as_an_error_with_context(
    provider: StaticProvider, expected: type[DataError], log_records: list[logging.LogRecord]
) -> None:
    with pytest.raises(expected):
        MarketDataLoader(provider, clock=Clock()).load(make_request(ticker="tst"))

    (record,) = loader_records(log_records, logging.ERROR)
    message = record.getMessage()
    for expected_text in ("ticker=TST", "interval=1d", "requested=2023-01-02..latest",
                          f"error={expected.__name__}"):
        assert expected_text in message
    assert not loader_records(log_records, logging.INFO), "a failed load must not log success"


def test_fetch_failures_are_logged_with_a_traceback_and_validation_failures_without(
    log_records: list[logging.LogRecord],
) -> None:
    for provider in (
        StaticProvider(error=DataFetchError("provider down")),
        StaticProvider(frame=make_prices().iloc[::-1]),
    ):
        with pytest.raises(DataError):
            MarketDataLoader(provider, clock=Clock()).load(make_request())

    fetch_record, validation_record = loader_records(log_records, logging.ERROR)
    assert fetch_record.exc_info is not None
    assert not validation_record.exc_info
