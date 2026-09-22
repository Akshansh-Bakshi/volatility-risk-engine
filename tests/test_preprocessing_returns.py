"""Tests for return construction: mathematics, scale, gaps, provisional bar, integrity."""

from __future__ import annotations

import logging
import math
import warnings
from collections.abc import Sequence
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import PreprocessingConfig, get_settings
from src.data.cache import MarketDataCache
from src.data.loader import MarketDataLoader
from src.data.market_data import MarketData
from src.data.yahoo import YahooFinanceProvider
from src.exceptions import (
    DataValidationError,
    InsufficientHistoryError,
    InvalidReturnSeriesError,
)
from src.preprocessing import MODELING_SCALE, PriceBasis, ReturnScale, build_return_series
from tests.data_fakes import (
    DEFAULT_FETCHED_AT,
    FakeYahoo,
    datetime_index,
    make_market_data,
    make_prices,
    make_request,
    prices_from_closes,
)

SMALL = PreprocessingConfig(min_returns=2)  # for hand-sized examples
THREE_DAYS = ["2024-01-02", "2024-01-03", "2024-01-04"]  # Tue, Wed, Thu
RETURNS_LOGGER = "src.preprocessing.returns"


def small_market_data(
    closes: list[float],
    dates: list[str] | None = None,
    *,
    dropped_dates: Sequence[str] = (),
    fetched_at: datetime = DEFAULT_FETCHED_AT,
    adjust_prices: bool = True,
    source: str = "unit_test",
) -> MarketData:
    return make_market_data(
        prices_from_closes(closes, dates or THREE_DAYS),
        dropped_dates=dropped_dates,
        fetched_at=fetched_at,
        adjust_prices=adjust_prices,
        source=source,
    )


def returns_records(records: list[logging.LogRecord], level: int) -> list[logging.LogRecord]:
    return [r for r in records if r.name == RETURNS_LOGGER and r.levelno == level]


# --- A. mathematics ------------------------------------------------------------------------------


def test_log_returns_match_the_hand_calculation() -> None:
    # ln(110/100) = ln(1.1) and ln(99/110) = ln(0.9), worked out independently.
    returns = build_return_series(small_market_data([100.0, 110.0, 99.0]), SMALL)

    assert returns.decimal.iloc[0] == pytest.approx(0.0953101798043248600, abs=1e-14)
    assert returns.decimal.iloc[1] == pytest.approx(-0.105360515657826301, abs=1e-14)
    assert returns.decimal.iloc[0] == pytest.approx(math.log(110 / 100), rel=1e-12)
    assert returns.decimal.iloc[1] == pytest.approx(math.log(99 / 110), rel=1e-12)


def test_log_returns_of_the_default_series_match_an_independent_reference() -> None:
    market_data = make_market_data()
    closes = market_data.prices["close"].tolist()

    returns = build_return_series(market_data)

    reference = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
    assert np.max(np.abs(returns.decimal.to_numpy() - np.array(reference))) < 1e-13


def test_log_returns_telescope_to_the_total_log_price_change() -> None:
    market_data = make_market_data()
    closes = market_data.prices["close"]

    returns = build_return_series(market_data)

    assert returns.decimal.sum() == pytest.approx(math.log(closes.iloc[-1] / closes.iloc[0]))


def test_log_return_of_an_unchanged_price_is_exactly_zero() -> None:
    returns = build_return_series(small_market_data([50.0, 50.0, 50.0]), SMALL)

    assert (returns.decimal == 0.0).all()


def test_returns_are_additive_and_use_the_log_not_the_simple_return() -> None:
    returns = build_return_series(small_market_data([100.0, 200.0, 100.0]), SMALL)

    assert returns.decimal.iloc[0] == pytest.approx(math.log(2.0))  # 69.3%, not the simple 100%
    assert returns.decimal.iloc[1] == pytest.approx(-math.log(2.0))
    assert returns.decimal.sum() == pytest.approx(0.0, abs=1e-15)


def test_a_large_series_is_processed_vectorised_and_matches_numpy() -> None:
    frame = make_prices(periods=50_000)  # runs to the year 2214, so pretend to download later
    market_data = make_market_data(frame, fetched_at=datetime(2300, 1, 1, tzinfo=timezone.utc))

    returns = build_return_series(market_data, PreprocessingConfig())

    closes = frame["close"].to_numpy()
    expected = np.log(closes[1:] / closes[:-1])
    assert returns.return_observations == 49_999
    assert np.allclose(returns.decimal.to_numpy(), expected, rtol=0, atol=1e-15)


# --- B. first observation -------------------------------------------------------------------------


def test_the_first_price_has_no_return_and_is_recorded_as_the_base() -> None:
    market_data = small_market_data([100.0, 110.0, 99.0])

    returns = build_return_series(market_data, SMALL)

    assert returns.return_observations == 2 and returns.price_observations == 3
    assert list(returns.decimal.index) == [pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-04")]
    assert returns.first_price_date == date(2024, 1, 2)
    assert not returns.decimal.isna().any(), "no NaN placeholder for the first observation"


# --- C, D. scaling --------------------------------------------------------------------------------


def test_percentage_scaling_of_the_hand_example() -> None:
    returns = build_return_series(small_market_data([100.0, 110.0, 99.0]), SMALL)

    assert returns.percent.iloc[0] == pytest.approx(9.53101798043248600, abs=1e-12)
    assert returns.percent.iloc[1] == pytest.approx(-10.5360515657826301, abs=1e-12)
    assert np.array_equal(returns.percent.to_numpy(), returns.decimal.to_numpy() * 100.0)


def test_scale_metadata_and_series_names_state_the_units() -> None:
    returns = build_return_series(small_market_data([100.0, 110.0, 99.0]), SMALL)

    assert returns.scale is ReturnScale.DECIMAL
    assert MODELING_SCALE is ReturnScale.PERCENT
    assert returns.decimal.name == "log_return_decimal"
    assert returns.percent.name == "log_return_percent"
    assert returns.metadata()["stored_scale"] == "decimal"
    assert returns.metadata()["modeling_scale"] == "percent"
    assert returns.metadata()["return_definition"] == "log: R_t = ln(P_t / P_(t-1))"


def test_a_realistic_series_has_percent_dispersion_one_hundred_times_decimal() -> None:
    returns = build_return_series(make_market_data())

    assert returns.decimal.std() < 0.05, "stored returns are decimals"
    assert returns.percent.std() == pytest.approx(returns.decimal.std() * 100.0, rel=1e-12)


# --- E. chronology --------------------------------------------------------------------------------


def test_chronological_order_and_dates_are_preserved() -> None:
    market_data = make_market_data()

    returns = build_return_series(market_data)

    assert returns.decimal.index.equals(market_data.prices.index[1:])
    assert returns.decimal.index.is_monotonic_increasing and returns.decimal.index.is_unique
    assert returns.spans_gap.index.equals(returns.decimal.index)


# --- F, G, H, I, J. invalid input is rejected, never repaired -------------------------------------


@pytest.mark.parametrize("bad_price", [0.0, -5.0])
def test_non_positive_prices_are_rejected_before_any_logarithm(bad_price: float) -> None:
    with pytest.raises(DataValidationError, match="non-positive close on 2024-01-03"):
        build_return_series(small_market_data([100.0, bad_price, 99.0]), SMALL)


def test_a_missing_close_is_rejected_not_filled() -> None:
    with pytest.raises(DataValidationError, match=r"1 row\(s\) without a close.*never filled"):
        build_return_series(small_market_data([100.0, np.nan, 99.0]), SMALL)


def test_missing_values_outside_close_do_not_affect_the_returns() -> None:
    frame = prices_from_closes([100.0, 110.0, 99.0], THREE_DAYS)
    frame.loc[frame.index[1], ["open", "high", "volume"]] = np.nan

    returns = build_return_series(make_market_data(frame), SMALL)

    assert returns.decimal.iloc[0] == pytest.approx(math.log(1.1))


@pytest.mark.parametrize("bad_price", [np.inf, -np.inf])
def test_infinite_prices_are_rejected(bad_price: float) -> None:
    with pytest.raises(DataValidationError, match="infinite"):
        build_return_series(small_market_data([100.0, bad_price, 99.0]), SMALL)


@pytest.mark.parametrize(
    "closes",
    [[1.0, 1e-300, 1e300], [1.0, 1e300, 1e-300]],
    ids=["ratio-overflows", "ratio-underflows-to-zero"],
)
def test_finite_prices_whose_ratio_overflows_never_yield_infinite_returns(
    closes: list[float],
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a leaked RuntimeWarning would mean a silent overflow
        with pytest.raises(InvalidReturnSeriesError, match="overflows or underflows"):
            build_return_series(small_market_data(closes), SMALL)


def test_extreme_but_representable_moves_are_finite_and_accepted() -> None:
    returns = build_return_series(small_market_data([1e-150, 1e150, 1.0]), SMALL)

    assert np.isfinite(returns.decimal).all()
    assert returns.decimal.iloc[0] == pytest.approx(math.log(1e300))


def test_duplicate_observations_are_rejected_not_dropped() -> None:
    frame = make_prices(periods=300)
    duplicated = pd.concat([frame, frame.iloc[[5]]]).sort_index()

    with pytest.raises(DataValidationError, match="duplicate timestamps"):
        build_return_series(make_market_data(duplicated))


def test_unsorted_observations_are_rejected_not_sorted() -> None:
    with pytest.raises(DataValidationError, match="not sorted in ascending order"):
        build_return_series(make_market_data(make_prices().iloc[::-1]))


def test_a_timezone_aware_index_is_rejected() -> None:
    frame = make_prices()
    frame.index = datetime_index(frame).tz_localize("UTC")

    with pytest.raises(DataValidationError, match="timezone-aware"):
        build_return_series(make_market_data(frame))


# --- K. insufficient observations -----------------------------------------------------------------


def test_fewer_returns_than_the_configured_minimum_is_insufficient_history() -> None:
    market_data = make_market_data(make_prices(periods=250))  # 249 returns

    with pytest.raises(InsufficientHistoryError, match=r"TEST.*249 return\(s\).*at least 250"):
        build_return_series(market_data)


def test_exactly_the_minimum_number_of_returns_is_accepted() -> None:
    returns = build_return_series(make_market_data(make_prices(periods=251)))

    assert returns.return_observations == 250


def test_a_single_price_cannot_produce_a_return_series() -> None:
    market_data = make_market_data(prices_from_closes([100.0], ["2024-01-02"]))

    with pytest.raises(InsufficientHistoryError, match="at least 2"):
        build_return_series(market_data, SMALL)


def test_the_minimum_comes_from_the_configuration() -> None:
    market_data = small_market_data([100.0, 110.0, 99.0])

    built = build_return_series(market_data, PreprocessingConfig(min_returns=2))
    assert built.return_observations == 2
    with pytest.raises(InsufficientHistoryError, match="at least 3 required"):
        build_return_series(market_data, PreprocessingConfig(min_returns=3))


def test_default_configuration_is_read_from_the_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    market_data = small_market_data([100.0, 110.0, 99.0])
    with pytest.raises(InsufficientHistoryError, match="at least 250"):
        build_return_series(market_data)

    monkeypatch.setenv("VRE_MIN_RETURNS", "2")
    get_settings.cache_clear()  # settings are read once per process

    assert build_return_series(market_data).return_observations == 2


# --- L. gap detection -----------------------------------------------------------------------------


def flags(
    dates: list[str],
    config: PreprocessingConfig = SMALL,
    *,
    dropped_dates: Sequence[str] = (),
) -> list[bool]:
    closes = [100.0 + i for i in range(len(dates))]
    market_data = small_market_data(closes, dates, dropped_dates=dropped_dates)
    return build_return_series(market_data, config).spans_gap.tolist()


def test_a_return_across_missing_weekdays_is_flagged_as_spanning_a_gap() -> None:
    # Monday, Tuesday, Friday: Tuesday -> Friday skips Wednesday and Thursday.
    assert flags(["2024-01-08", "2024-01-09", "2024-01-12"]) == [False, True]


def test_ordinary_weekends_are_not_gaps() -> None:
    dates = ["2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09"]  # Thu, Fri, Mon, Tue

    assert flags(dates, PreprocessingConfig(min_returns=2, max_gap_weekdays=0)) == [False] * 3


def test_a_single_missing_weekday_is_tolerated_by_default_but_flagged_when_strict() -> None:
    dates = ["2024-01-04", "2024-01-08", "2024-01-09"]  # Thu -> Mon skips the Friday

    assert flags(dates) == [False, False]
    assert flags(dates, PreprocessingConfig(min_returns=2, max_gap_weekdays=0)) == [True, False]


def test_the_tolerance_is_the_number_of_missing_weekdays_allowed() -> None:
    dates = ["2024-01-08", "2024-01-09", "2024-01-12"]  # Tuesday -> Friday: 2 weekdays missing

    assert flags(dates, PreprocessingConfig(min_returns=2, max_gap_weekdays=2)) == [False, False]
    assert flags(dates, PreprocessingConfig(min_returns=2, max_gap_weekdays=1)) == [False, True]


def test_a_long_closure_is_flagged() -> None:
    assert flags(["2024-01-02", "2024-01-03", "2024-01-17"]) == [False, True]


def test_daily_series_including_weekends_have_no_gaps() -> None:
    dates = [d.strftime("%Y-%m-%d") for d in pd.date_range("2024-01-01", periods=14)]

    assert not any(flags(dates, PreprocessingConfig(min_returns=2, max_gap_weekdays=0)))


def test_rows_removed_at_ingestion_flag_exactly_the_returns_that_span_them() -> None:
    # Monday, Tuesday, Thursday; Wednesday was delivered without a close and removed.  By
    # weekday arithmetic alone Tue -> Thu skips one weekday (tolerated), so only the exact
    # record of the removed row can flag it.
    dates = ["2024-01-08", "2024-01-09", "2024-01-11"]

    assert flags(dates) == [False, False]
    assert flags(dates, dropped_dates=["2024-01-10"]) == [False, True]


def test_removed_rows_outside_the_series_or_inside_one_interval_flag_correctly() -> None:
    dates = ["2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12"]

    before_and_after = flags(dates, dropped_dates=["2024-01-08", "2024-01-15"])
    two_in_one_interval = flags(
        ["2024-01-08", "2024-01-09", "2024-01-15"], dropped_dates=["2024-01-10", "2024-01-11"]
    )
    separate_intervals = flags(
        ["2024-01-08", "2024-01-10", "2024-01-12"], dropped_dates=["2024-01-09", "2024-01-11"]
    )

    assert before_and_after == [False, False, False]
    assert two_in_one_interval == [False, True], "one return, flagged once"
    assert separate_intervals == [True, True]


def test_removed_row_dates_in_any_datetime_resolution_are_handled() -> None:
    dates = ["2024-01-08", "2024-01-09", "2024-01-11"]
    market_data = small_market_data([100.0, 101.0, 102.0], dates)
    coarse = pd.DatetimeIndex(["2024-01-10"], name="date").as_unit("s")
    object.__setattr__(market_data, "dropped_dates", coarse)

    assert build_return_series(market_data, SMALL).spans_gap.tolist() == [False, True]


def test_flagged_returns_are_kept_and_counted() -> None:
    market_data = small_market_data(
        [100.0, 101.0, 102.0, 103.0], ["2024-01-08", "2024-01-09", "2024-01-16", "2024-01-17"]
    )

    returns = build_return_series(market_data, SMALL)

    assert returns.return_observations == 3, "a gap-spanning return is computed, not removed"
    assert returns.spans_gap.tolist() == [False, True, False]
    assert returns.flagged_gaps == 1
    assert returns.decimal.iloc[1] == pytest.approx(math.log(102.0 / 101.0))


# --- M. metadata ----------------------------------------------------------------------------------


def test_metadata_describes_how_the_series_was_produced() -> None:
    dates = ["2024-01-08", "2024-01-09", "2024-01-12", "2024-01-15", "2024-01-16"]
    market_data = small_market_data(
        [100.0, 101.0, 103.0, 102.0, 104.0],
        dates,
        dropped_dates=["2024-01-13", "2024-01-14"],
        fetched_at=datetime(2024, 1, 16, 9, 0, tzinfo=timezone.utc),
        adjust_prices=False,
        source="some_provider",
    )

    returns = build_return_series(market_data, PreprocessingConfig(min_returns=2))

    assert returns.metadata() == {
        "ticker": "TEST",
        "frequency": "1d",
        "price_basis": "split_adjusted",
        "return_definition": "log: R_t = ln(P_t / P_(t-1))",
        "stored_scale": "decimal",
        "modeling_scale": "percent",
        "price_observations": 4,  # the provisional 2024-01-16 bar was excluded
        "return_observations": 3,
        "first_price_date": "2024-01-08",
        "first_return_date": "2024-01-09",
        "last_return_date": "2024-01-15",
        "flagged_gaps": 2,  # Tue -> Fri (two weekdays) and Fri -> Mon (two dropped rows between)
        "gap_tolerance_weekdays": 1,
        "ingestion_rows_dropped": 2,
        "excluded_observations": 1,
        "provisional_bar": "2024-01-16",
        "provisional_bar_excluded": True,
        "last_return_is_provisional": False,
        "source": "some_provider",
        "fetched_at": "2024-01-16T09:00:00+00:00",
    }


def test_price_basis_follows_the_request() -> None:
    adjusted = build_return_series(small_market_data([1.0, 2.0, 3.0], adjust_prices=True), SMALL)
    split_only = build_return_series(small_market_data([1.0, 2.0, 3.0], adjust_prices=False), SMALL)

    assert adjusted.price_basis is PriceBasis.ADJUSTED
    assert split_only.price_basis is PriceBasis.SPLIT_ADJUSTED


# --- N, O. purity and determinism -----------------------------------------------------------------


def test_the_original_market_data_is_never_modified() -> None:
    market_data = make_market_data(dropped_dates=["2023-05-02"])
    prices_before = market_data.prices.copy(deep=True)
    dropped_before = market_data.dropped_dates.copy()

    returns = build_return_series(market_data)
    returns.decimal.iloc[0] = 123.0  # even scribbling on the result must not reach the input

    pd.testing.assert_frame_equal(market_data.prices, prices_before)
    assert market_data.dropped_dates.equals(dropped_before)
    assert not np.shares_memory(returns.decimal.to_numpy(), market_data.prices["close"].to_numpy())


def test_repeated_construction_is_deterministic() -> None:
    market_data = make_market_data(dropped_dates=["2023-05-02"])

    first = build_return_series(market_data)
    second = build_return_series(market_data)
    third = build_return_series(make_market_data(dropped_dates=["2023-05-02"]))

    for other in (second, third):
        pd.testing.assert_series_equal(first.decimal, other.decimal, check_exact=True)
        pd.testing.assert_series_equal(first.spans_gap, other.spans_gap, check_exact=True)
        assert first.metadata() == other.metadata()


# --- P. provisional final bar ---------------------------------------------------------------------


def at(year: int, month: int, day: int, hour: int = 0, minute: int = 0, sec: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, sec, tzinfo=timezone.utc)


LAST_BAR = date(2024, 2, 23)  # last session of make_prices()
PREVIOUS_BAR = date(2024, 2, 22)


@pytest.mark.parametrize(
    "fetched_at",
    [at(2024, 2, 23, 15), at(2024, 2, 23, 23, 59, 59), at(2024, 2, 23), at(2024, 2, 22, 10)],
    ids=["mid-day", "one-second-before-midnight", "midnight-of-the-bar", "bar-after-download"],
)
def test_a_bar_dated_on_or_after_the_download_date_is_excluded_by_default(
    fetched_at: datetime,
) -> None:
    market_data = make_market_data(fetched_at=fetched_at)

    returns = build_return_series(market_data)

    assert returns.provisional_bar == LAST_BAR and returns.provisional_bar_excluded
    assert returns.last_return_date == PREVIOUS_BAR
    assert returns.return_observations == 298, "one fewer than the 299 available"
    assert returns.excluded_observations == 1
    assert returns.last_return_is_provisional is False
    assert market_data.last_date == LAST_BAR, "the input still holds the bar"


@pytest.mark.parametrize(
    "fetched_at", [at(2024, 2, 24), at(2024, 2, 24, 0, 0, 1), at(2024, 6, 3, 12)],
    ids=["utc-midnight-after", "just-after", "months-later"],
)
def test_a_bar_from_an_earlier_utc_day_is_final_and_kept(fetched_at: datetime) -> None:
    returns = build_return_series(make_market_data(fetched_at=fetched_at))

    assert returns.provisional_bar is None and not returns.provisional_bar_excluded
    assert returns.last_return_date == LAST_BAR
    assert returns.return_observations == 299 and returns.excluded_observations == 0


def test_including_the_provisional_bar_is_an_explicit_choice_that_is_recorded_and_warned_about(
    log_records: list[logging.LogRecord],
) -> None:
    market_data = make_market_data(fetched_at=at(2024, 2, 23, 15))
    config = PreprocessingConfig(include_provisional_bar=True)

    returns = build_return_series(market_data, config)

    assert returns.last_return_date == LAST_BAR and returns.return_observations == 299
    assert returns.provisional_bar == LAST_BAR and not returns.provisional_bar_excluded
    assert returns.last_return_is_provisional is True
    assert returns.excluded_observations == 0
    (warning,) = returns_records(log_records, logging.WARNING)
    assert "may still have been forming" in warning.getMessage()
    assert "2024-02-23" in warning.getMessage()


def test_finalised_data_never_triggers_the_provisional_warning(
    log_records: list[logging.LogRecord],
) -> None:
    build_return_series(make_market_data(), PreprocessingConfig(include_provisional_bar=True))

    assert not returns_records(log_records, logging.WARNING)


def test_data_pinned_to_a_past_end_date_is_final_even_when_downloaded_today() -> None:
    market_data = make_market_data(
        make_prices(periods=300), end=date(2024, 2, 23), fetched_at=at(2024, 6, 3, 12)
    )

    assert build_return_series(market_data).provisional_bar is None


def test_a_cached_download_is_judged_by_when_it_was_originally_fetched() -> None:
    stale_but_final = make_market_data(fetched_at=at(2024, 2, 26, 8))  # Monday morning

    assert build_return_series(stale_but_final).provisional_bar is None


def test_excluding_the_provisional_bar_counts_against_the_minimum_with_an_explanation() -> None:
    frame = make_prices(periods=251)  # 250 returns, but the last bar is provisional
    last = pd.Timestamp(frame.index[-1])
    downloaded = last.to_pydatetime().replace(tzinfo=timezone.utc)  # midnight UTC of the last bar
    market_data = make_market_data(frame, fetched_at=downloaded)

    with pytest.raises(InsufficientHistoryError, match=r"249 return\(s\).*provisional"):
        build_return_series(market_data)
    included = build_return_series(market_data, PreprocessingConfig(include_provisional_bar=True))
    assert included.return_observations == 250


def test_the_exclusion_uses_only_metadata_and_never_looks_at_prices() -> None:
    # An extreme final price is neither hidden nor altered: it is simply the excluded bar.
    frame = make_prices()
    frame.loc[frame.index[-1], "close"] = frame["close"].iloc[-2] * 1000.0
    provisional = make_market_data(frame, fetched_at=at(2024, 2, 23, 15))
    final = make_market_data(frame, fetched_at=at(2024, 6, 3))

    excluded = build_return_series(provisional)
    included = build_return_series(final)

    assert included.decimal.iloc[-1] == pytest.approx(math.log(1000.0))
    pd.testing.assert_series_equal(excluded.decimal, included.decimal.iloc[:-1])


# --- Q. compatibility with Stage 2 MarketData -----------------------------------------------------


def stage2_loader(frame: pd.DataFrame, tmp_path: Path, now: datetime) -> MarketDataLoader:
    provider = YahooFinanceProvider(ticker_factory=FakeYahoo({"TEST": frame}))
    return MarketDataLoader(provider, cache=MarketDataCache(tmp_path), clock=lambda: now)


def test_loader_output_flows_into_returns_with_removed_rows_flagged_exactly(tmp_path: Path) -> None:
    frame = make_prices()
    removed = frame.index[[10, 11, 100]]
    frame.loc[removed, "close"] = np.nan
    loader = stage2_loader(frame, tmp_path, at(2024, 6, 3, 12))

    fresh = build_return_series(loader.load(make_request()))
    cached_data = loader.load(make_request())
    cached = build_return_series(cached_data)

    assert cached_data.from_cache is True
    flagged = fresh.spans_gap[fresh.spans_gap].index
    # rows 10 and 11 vanish together (two weekdays skipped); row 100 alone skips one weekday,
    # which the weekday rule tolerates, so only the recorded dates can flag it.
    assert list(flagged) == [frame.index[12], frame.index[101]]
    assert fresh.ingestion_rows_dropped == 3
    assert fresh.price_observations == 297
    pd.testing.assert_series_equal(fresh.decimal, cached.decimal, check_exact=True)
    pd.testing.assert_series_equal(fresh.spans_gap, cached.spans_gap, check_exact=True)
    assert fresh.metadata()["fetched_at"] == cached.metadata()["fetched_at"]


def test_a_latest_request_downloaded_during_the_last_session_excludes_that_bar(
    tmp_path: Path,
) -> None:
    frame = make_prices()
    loader = stage2_loader(frame, tmp_path, at(2024, 2, 23, 15))

    data = loader.load(make_request(end=None))
    returns = build_return_series(data)

    assert data.last_date == LAST_BAR
    assert returns.provisional_bar == LAST_BAR and returns.provisional_bar_excluded
    assert returns.last_return_date == PREVIOUS_BAR


def test_the_unadjusted_price_basis_survives_the_loader(tmp_path: Path) -> None:
    loader = stage2_loader(make_prices(), tmp_path, at(2024, 6, 3, 12))

    returns = build_return_series(loader.load(make_request(adjust_prices=False)))

    assert returns.price_basis is PriceBasis.SPLIT_ADJUSTED
    assert returns.source == "yahoo_finance"


# --- logging --------------------------------------------------------------------------------------


def test_construction_is_logged_with_an_audit_summary(log_records: list[logging.LogRecord]) -> None:
    build_return_series(make_market_data(dropped_dates=["2023-05-02"]))

    (record,) = returns_records(log_records, logging.INFO)
    message = record.getMessage()
    for expected in (
        "ticker=TEST",
        "frequency=1d",
        "price_basis=adjusted",
        "definition=log",
        "stored_scale=decimal",
        "prices=300",
        "returns=299",
        "flagged_gaps=1",
        "ingestion_rows_dropped=1",
        "provisional_bar=None",
    ):
        assert expected in message, f"missing {expected!r} in: {message}"


def test_failure_is_logged_once_with_context(log_records: list[logging.LogRecord]) -> None:
    with pytest.raises(InsufficientHistoryError):
        build_return_series(make_market_data(make_prices(periods=100)))

    (record,) = returns_records(log_records, logging.ERROR)
    message = record.getMessage()
    assert "ticker=TEST" in message and "InsufficientHistoryError" in message
    assert not returns_records(log_records, logging.INFO)


def test_the_result_can_be_consumed_without_any_market_data_object() -> None:
    def downstream_model_input(returns_percent: pd.Series) -> float:
        """Stands in for a later modelling layer: it receives only the scaled Series."""
        return float(returns_percent.std())

    returns = build_return_series(make_market_data())

    assert downstream_model_input(returns.scaled(MODELING_SCALE)) == pytest.approx(
        returns.decimal.std() * 100.0, rel=1e-12
    )
