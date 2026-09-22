"""Tests for the provider-independent validation policy."""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.data.market_data import STANDARD_COLUMNS
from src.data.validation import validate_prices
from src.exceptions import (
    DataValidationError,
    EmptyDataError,
    InsufficientHistoryError,
)
from tests.data_fakes import datetime_index, make_prices, make_request

MIN_OBS = 250


def validate(
    frame: pd.DataFrame, **request_overrides: object
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    return validate_prices(frame, make_request(**request_overrides), min_observations=MIN_OBS)


# --- accepted input ----------------------------------------------------------------------


def test_clean_frame_is_returned_in_the_project_schema() -> None:
    frame = make_prices()

    cleaned, dropped = validate(frame)

    assert isinstance(dropped, pd.DatetimeIndex) and len(dropped) == 0
    assert list(cleaned.columns) == list(STANDARD_COLUMNS)
    assert isinstance(cleaned.index, pd.DatetimeIndex)
    assert cleaned.index.tz is None
    assert cleaned.index.name == "date"
    assert str(cleaned.index.dtype) == "datetime64[ns]"
    assert cleaned.index.is_unique and cleaned.index.is_monotonic_increasing
    assert cleaned.index.freq is None, "irregular session calendars must not carry a fixed freq"
    assert (cleaned.dtypes == "float64").all()
    assert list(cleaned.index) == list(frame.index)
    assert np.array_equal(cleaned.to_numpy(), frame.to_numpy())


def test_columns_are_reordered_to_the_standard_order() -> None:
    frame = make_prices()[["volume", "close", "low", "high", "open"]]

    cleaned, _ = validate(frame)

    assert list(cleaned.columns) == list(STANDARD_COLUMNS)


def test_integer_volume_is_converted_to_float() -> None:
    frame = make_prices()
    frame["volume"] = frame["volume"].astype("int64")

    cleaned, _ = validate(frame)

    assert cleaned["volume"].dtype == "float64"


def test_index_resolution_is_normalised_to_nanoseconds() -> None:
    frame = make_prices()
    frame.index = datetime_index(frame).as_unit("s")

    cleaned, _ = validate(frame)

    assert str(cleaned.index.dtype) == "datetime64[ns]"


def test_input_frame_is_never_modified() -> None:
    frame = make_prices()
    frame.loc[frame.index[3], "close"] = np.nan
    snapshot = frame.copy()

    validate(frame)

    pd.testing.assert_frame_equal(frame, snapshot)


# --- emptiness / malformed output ----------------------------------------------------------


def test_empty_frame_raises_empty_data_error_with_context() -> None:
    empty = pd.DataFrame(
        {name: pd.Series(dtype="float64") for name in STANDARD_COLUMNS},
        index=pd.DatetimeIndex([], name="date"),
    )

    with pytest.raises(EmptyDataError, match=r"TEST.*2023-01-02\.\.latest"):
        validate(empty)


@pytest.mark.parametrize("bad", [None, [1, 2, 3], pd.Series([1.0, 2.0]), {"close": [1.0]}])
def test_non_dataframe_output_is_rejected(bad: object) -> None:
    with pytest.raises(DataValidationError, match="expected a DataFrame"):
        validate(bad)  # type: ignore[arg-type]


def _multiindex_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = pd.MultiIndex.from_product([frame.columns, ["TEST"]])
    return frame


def _capitalised(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.rename(columns=str.capitalize)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda f: f.drop(columns=["close"]),
        lambda f: f.drop(columns=["volume"]),
        lambda f: f.assign(**{"adj close": f["close"]}),
        lambda f: pd.concat([f, f[["close"]]], axis=1),
        _multiindex_columns,
        _capitalised,
    ],
    ids=["no-close", "no-volume", "extra-column", "duplicate-column", "multiindex", "vendor-names"],
)
def test_wrong_column_layout_is_rejected(mutate: Callable[[pd.DataFrame], pd.DataFrame]) -> None:
    with pytest.raises(DataValidationError, match="Malformed provider output"):
        validate(mutate(make_prices()))


def test_non_numeric_columns_are_rejected() -> None:
    text = make_prices().astype({"volume": "object"})
    text["volume"] = "n/a"
    flags = make_prices().assign(volume=True)

    for bad in (text, flags):
        with pytest.raises(DataValidationError, match=r"non-numeric column\(s\) \['volume'\]"):
            validate(bad)


# --- index rules ----------------------------------------------------------------------------


def _with_index(frame: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    frame = frame.copy()
    frame.index = index
    return frame


@pytest.mark.parametrize(
    ("make_index", "message"),
    [
        (lambda idx: pd.RangeIndex(len(idx)), "expected a DatetimeIndex"),
        (lambda idx: pd.Index([d.isoformat() for d in idx]), "expected a DatetimeIndex"),
        (lambda idx: idx.tz_localize("UTC"), "timezone-aware"),
        (lambda idx: idx.tz_localize("Asia/Tokyo"), "timezone-aware"),
        (lambda idx: idx + pd.Timedelta(hours=1), "midnight"),
        (lambda idx: idx.insert(5, pd.NaT)[:-1], "missing timestamps"),
    ],
    ids=["range", "strings", "utc", "tokyo", "time-of-day", "nat"],
)
def test_invalid_index_is_rejected(
    make_index: Callable[[pd.DatetimeIndex], pd.Index], message: str
) -> None:
    frame = make_prices()

    with pytest.raises(DataValidationError, match=message):
        validate(_with_index(frame, make_index(datetime_index(frame))))


def test_duplicate_timestamps_are_rejected_not_dropped() -> None:
    frame = make_prices()
    duplicated = pd.concat([frame, frame.iloc[[10]]]).sort_index()

    with pytest.raises(DataValidationError, match=r"duplicate timestamps at 2023-01-16"):
        validate(duplicated)


def test_unsorted_timestamps_are_rejected_not_sorted() -> None:
    frame = make_prices()
    swapped = frame.iloc[[*range(5), 6, 5, *range(7, len(frame))]]

    with pytest.raises(DataValidationError, match="not sorted in ascending order"):
        validate(swapped)
    with pytest.raises(DataValidationError, match="not sorted in ascending order"):
        validate(frame.iloc[::-1])


# --- date window ------------------------------------------------------------------------------


def test_rows_before_the_requested_start_are_rejected() -> None:
    with pytest.raises(DataValidationError, match="outside the requested range"):
        validate(make_prices(), start=date(2023, 1, 3))


def test_rows_after_the_requested_end_are_rejected() -> None:
    frame = make_prices()  # ends 2024-02-23

    with pytest.raises(DataValidationError, match="outside the requested range"):
        validate(frame, end=date(2024, 2, 22))


def test_start_and_end_dates_are_inclusive() -> None:
    frame = make_prices()

    cleaned, _ = validate(frame, start=date(2023, 1, 2), end=date(2024, 2, 23))

    assert cleaned.index[0].date() == date(2023, 1, 2)
    assert cleaned.index[-1].date() == date(2024, 2, 23)


# --- price rules --------------------------------------------------------------------------------


@pytest.mark.parametrize("bad_price", [0.0, -5.0])
def test_non_positive_close_is_rejected_with_its_date(bad_price: float) -> None:
    frame = make_prices()
    frame.loc[frame.index[10], "close"] = bad_price

    with pytest.raises(DataValidationError, match=r"non-positive close on 2023-01-16"):
        validate(frame)


@pytest.mark.parametrize(("column", "value"), [("close", np.inf), ("volume", -np.inf)])
def test_infinite_values_are_rejected(column: str, value: float) -> None:
    frame = make_prices()
    frame.loc[frame.index[4], column] = value

    expected = rf"infinite\) entries in column\(s\) \['{column}'\]"
    with pytest.raises(DataValidationError, match=expected):
        validate(frame)


def test_rows_without_a_close_are_removed_never_filled(
    log_records: list[logging.LogRecord],
) -> None:
    frame = make_prices()
    missing = [frame.index[5], frame.index[6], frame.index[100]]
    frame.loc[missing, "close"] = np.nan

    cleaned, dropped = validate(frame)

    assert list(dropped) == missing, "the exact dates removed are reported, not just a count"
    assert dropped.name == "date" and str(dropped.dtype) == "datetime64[ns]"
    assert len(cleaned) == len(frame) - 3
    assert not cleaned.index.isin(missing).any()
    assert set(cleaned.index) <= set(frame.index), "no observation may be invented"
    kept = frame.index.difference(pd.DatetimeIndex(missing))
    assert np.array_equal(cleaned.loc[kept].to_numpy(), frame.loc[kept].to_numpy())
    assert not cleaned["close"].isna().any()
    warnings = [r for r in log_records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    text = warnings[0].getMessage()
    assert "TEST" in text and "2023-01-09" in text and "not filled" in text


def test_missing_values_outside_close_are_preserved_as_delivered() -> None:
    frame = make_prices()
    frame.loc[frame.index[7], ["open", "high", "volume"]] = np.nan

    cleaned, dropped = validate(frame)

    assert len(dropped) == 0
    assert len(cleaned) == len(frame)
    row = cleaned.loc[frame.index[7]]
    assert row[["open", "high", "volume"]].isna().all()
    assert row["close"] == frame.loc[frame.index[7], "close"]


def test_a_frame_whose_closes_are_all_missing_has_no_usable_observations() -> None:
    frame = make_prices()
    frame["close"] = np.nan

    with pytest.raises(EmptyDataError, match=r"No usable observations for TEST.*all 300 rows"):
        validate(frame)


# --- history length -------------------------------------------------------------------------------


def test_fewer_than_the_minimum_observations_is_insufficient_history() -> None:
    with pytest.raises(InsufficientHistoryError, match=r"TEST.*249 usable.*at least 250"):
        validate(make_prices(periods=249))


def test_exactly_the_minimum_is_accepted() -> None:
    cleaned, _ = validate(make_prices(periods=250))

    assert len(cleaned) == 250


def test_removed_rows_count_against_the_minimum() -> None:
    frame = make_prices(periods=252)
    frame.loc[frame.index[[1, 2, 3]], "close"] = np.nan

    with pytest.raises(InsufficientHistoryError, match="249 usable"):
        validate(frame)
