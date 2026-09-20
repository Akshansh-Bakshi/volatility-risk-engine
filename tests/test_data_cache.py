"""Tests for the local cache: naming, exact round-trips, corruption handling, freshness."""

from __future__ import annotations

import json
import logging
import shutil
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.data.cache import (
    CACHE_SUBDIRECTORY,
    SETTLEMENT_MARGIN,
    MarketDataCache,
    is_entry_fresh,
)
from src.data.market_data import MarketData, MarketDataRequest
from src.data.validation import validate_prices
from tests.data_fakes import make_prices, make_request

FETCHED_AT = datetime(2024, 3, 1, 9, 30, 15, 123456, tzinfo=timezone.utc)


def make_entry(request: MarketDataRequest | None = None, **frame_edits: Any) -> MarketData:
    request = request or make_request()
    frame = make_prices(periods=30)
    frame.loc[frame.index[2], "open"] = np.nan
    frame.loc[frame.index[3], "close"] = 0.1 + 0.2  # not exactly representable in decimal
    frame.loc[frame.index[4], "low"] = 1e-9
    frame.loc[frame.index[5], "volume"] = 9_007_199_254_740_991.0  # 2**53 - 1
    frame.loc[frame.index[6], "high"] = 123456789.123456789
    prices, _ = validate_prices(frame, request, min_observations=2)
    return MarketData(
        request=request, prices=prices, source="unit_test", fetched_at=FETCHED_AT,
        rows_dropped=2,
    )


@pytest.fixture
def cache(tmp_path: Path) -> MarketDataCache:
    return MarketDataCache(tmp_path / CACHE_SUBDIRECTORY)


# --- naming ---------------------------------------------------------------------------------------


def test_path_is_deterministic_and_encodes_the_full_identity(cache: MarketDataCache) -> None:
    latest = make_request(ticker="TEST")
    bounded = make_request(ticker="TEST", end=date(2023, 12, 29))

    assert cache.path_for(latest) == cache.path_for(make_request(ticker="test"))
    assert cache.path_for(latest).name == "TEST__1d__2023-01-02__latest__adjusted.json"
    assert cache.path_for(bounded).name == "TEST__1d__2023-01-02__2023-12-29__adjusted.json"
    assert cache.path_for(make_request(adjust_prices=False)).name.endswith("__unadjusted.json")
    assert cache.path_for(latest).parent == cache.directory


@pytest.mark.parametrize(
    ("ticker", "expected_prefix"),
    [("^GSPC", "%5EGSPC"), ("EURUSD=X", "EURUSD%3DX"), ("BRK-B", "BRK-B"), ("A/B", "A%2FB")],
)
def test_symbols_become_safe_file_names(
    cache: MarketDataCache, ticker: str, expected_prefix: str
) -> None:
    path = cache.path_for(make_request(ticker=ticker))

    assert path.name.startswith(expected_prefix + "__")
    assert path.parent == cache.directory, "a symbol must never introduce a path separator"


def test_every_identity_component_yields_a_distinct_file(cache: MarketDataCache) -> None:
    base = make_request()
    variants = [
        base,
        make_request(ticker="OTHER"),
        make_request(start=date(2023, 1, 3)),
        make_request(end=date(2024, 6, 28)),
        make_request(adjust_prices=False),
    ]

    assert len({cache.path_for(request) for request in variants}) == len(variants)


# --- round trips ----------------------------------------------------------------------------------


def test_write_then_read_round_trips_exactly(cache: MarketDataCache) -> None:
    entry = make_entry()

    path = cache.write(entry)
    restored = cache.read(entry.request)

    assert path == cache.path_for(entry.request) and path.is_file()
    assert restored is not None
    pd.testing.assert_frame_equal(restored.prices, entry.prices, check_exact=True)
    assert restored.prices.index.name == "date"
    assert (restored.prices.dtypes == "float64").all()
    assert restored.request == entry.request
    assert restored.source == "unit_test"
    assert restored.fetched_at == FETCHED_AT
    assert restored.rows_dropped == 2
    assert restored.from_cache is False


def test_the_cache_file_is_plain_standard_json(cache: MarketDataCache) -> None:
    entry = make_entry()

    payload = json.loads(cache.write(entry).read_text(encoding="utf-8"))  # allow_nan is off

    assert payload["ticker"] == "TEST" and payload["end"] is None
    assert payload["adjust_prices"] is True
    assert payload["columns"]["open"][2] is None, "NaN is stored as JSON null"


def test_reading_an_absent_entry_returns_none(cache: MarketDataCache) -> None:
    assert cache.read(make_request()) is None


def test_rewriting_replaces_the_entry_and_leaves_no_temporary_files(
    cache: MarketDataCache,
) -> None:
    request = make_request()
    cache.write(make_entry(request))
    newer = MarketData(
        request=request, prices=make_entry(request).prices, source="second",
        fetched_at=FETCHED_AT + timedelta(hours=1),
    )

    cache.write(newer)

    restored = cache.read(request)
    assert restored is not None and restored.source == "second"
    assert [p.name for p in cache.directory.iterdir()] == [cache.path_for(request).name]


def test_write_failure_raises_oserror_and_leaves_no_debris(tmp_path: Path) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("a file where the cache directory should be")
    broken = MarketDataCache(blocker / "cache")

    with pytest.raises(OSError):
        broken.write(make_entry())

    assert blocker.read_text() == "a file where the cache directory should be"


# --- unusable entries are misses, loudly ----------------------------------------------------------


def _corrupt(cache: MarketDataCache, request: MarketDataRequest, text: str) -> None:
    cache.directory.mkdir(parents=True, exist_ok=True)
    cache.path_for(request).write_text(text, encoding="utf-8")


def _tampered(cache: MarketDataCache, entry: MarketData, **changes: Any) -> str:
    payload = json.loads(cache.write(entry).read_text(encoding="utf-8"))
    payload.update(changes)
    return json.dumps(payload)


@pytest.mark.parametrize(
    "text",
    ["", "{not json", "[1, 2, 3]", '{"schema_version": 1}', "null"],
    ids=["empty", "invalid-json", "wrong-type", "missing-keys", "null"],
)
def test_unparseable_entries_are_ignored_with_a_warning(
    cache: MarketDataCache, log_records: list[logging.LogRecord], text: str
) -> None:
    request = make_request()
    _corrupt(cache, request, text)

    assert cache.read(request) is None

    warnings = [r for r in log_records if r.levelno == logging.WARNING]
    assert len(warnings) == 1 and "unusable cache entry" in warnings[0].getMessage()


@pytest.mark.parametrize(
    "changes",
    [
        {"schema_version": 999},
        {"ticker": "OTHER"},
        {"start": "2020-01-01"},
        {"end": "2024-01-31"},
        {"interval": "1wk"},
        {"adjust_prices": False},
        {"fetched_at": "2024-03-01T09:30:15"},  # naive timestamp
        {"columns": {"open": [1.0]}},  # missing columns
    ],
    ids=["schema", "ticker", "start", "end", "interval", "adjust", "naive-time", "columns"],
)
def test_entries_that_do_not_match_the_request_are_never_served(
    cache: MarketDataCache, log_records: list[logging.LogRecord], changes: dict[str, Any]
) -> None:
    entry = make_entry()
    text = _tampered(cache, entry, **changes)
    cache.path_for(entry.request).write_text(text, encoding="utf-8")

    assert cache.read(entry.request) is None
    assert any(r.levelno == logging.WARNING for r in log_records)


def test_a_file_copied_from_another_request_cannot_answer_this_one(
    cache: MarketDataCache,
) -> None:
    other = make_entry(make_request(ticker="OTHER"))
    wanted = make_request(ticker="TEST")
    cache.directory.mkdir(parents=True, exist_ok=True)
    shutil.copy(cache.write(other), cache.path_for(wanted))

    assert cache.read(wanted) is None


# --- freshness ------------------------------------------------------------------------------------

NOW = datetime(2024, 6, 3, 12, 0, tzinfo=timezone.utc)
TTL = timedelta(hours=6)


def fresh(request: MarketDataRequest, fetched_at: datetime, now: datetime = NOW) -> bool:
    return is_entry_fresh(request, fetched_at, now=now, max_age=TTL)


def test_latest_requests_expire_after_the_maximum_age() -> None:
    request = make_request(end=None)

    assert fresh(request, NOW - timedelta(hours=5, minutes=59))
    assert not fresh(request, NOW - TTL), "an entry exactly max_age old is already stale"
    assert not fresh(request, NOW - timedelta(days=30))


def test_entries_downloaded_long_after_their_window_closed_never_expire() -> None:
    request = make_request(end=date(2023, 12, 29))
    fetched_at = datetime(2024, 1, 5, 8, 0, tzinfo=timezone.utc)

    assert fresh(request, fetched_at, now=fetched_at + timedelta(days=2000))


def test_windows_that_had_not_clearly_closed_at_download_time_still_expire() -> None:
    request = make_request(end=date(2024, 6, 3))
    same_day = datetime(2024, 6, 3, 15, 0, tzinfo=timezone.utc)
    next_day = datetime(2024, 6, 4, 1, 0, tzinfo=timezone.utc)  # inside the settlement margin

    for fetched_at in (same_day, next_day):
        assert fresh(request, fetched_at, now=fetched_at + timedelta(hours=1))
        assert not fresh(request, fetched_at, now=fetched_at + TTL + timedelta(minutes=1))


def test_the_settlement_boundary_is_strictly_after_end_plus_margin() -> None:
    request = make_request(end=date(2024, 6, 3))
    boundary_day = date(2024, 6, 3) + SETTLEMENT_MARGIN
    late = datetime.combine(boundary_day, datetime.min.time(), tzinfo=timezone.utc)
    settled = late + timedelta(days=1)
    far_future = NOW + timedelta(days=400)

    assert not fresh(request, late + timedelta(hours=23), now=far_future)
    assert fresh(request, settled, now=far_future)


def test_entries_timestamped_in_the_future_are_never_fresh() -> None:
    assert not fresh(make_request(end=None), NOW + timedelta(minutes=1))
    settled_window = make_request(start=date(2019, 1, 2), end=date(2020, 1, 31))
    assert not fresh(settled_window, NOW + timedelta(days=5))
