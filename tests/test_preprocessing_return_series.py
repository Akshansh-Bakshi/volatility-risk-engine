"""Tests for the ReturnSeries domain object: explicit scale, validation, audit metadata."""

from __future__ import annotations

import dataclasses
import json
from datetime import date, datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.exceptions import InvalidReturnSeriesError, PreprocessingError
from src.preprocessing import (
    MODELING_SCALE,
    STORED_SCALE,
    PriceBasis,
    ReturnScale,
    ReturnSeries,
)

FETCHED_AT = datetime(2024, 6, 3, 12, 0, tzinfo=timezone.utc)
RETURNS = [0.0123, -0.0207, 0.0, 0.0311, -0.0045]


def make_series(**overrides: Any) -> ReturnSeries:
    index = pd.DatetimeIndex(
        ["2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09"], name="date"
    )
    fields: dict[str, Any] = {
        "ticker": "TEST",
        "frequency": "1d",
        "price_basis": PriceBasis.ADJUSTED,
        "decimal": pd.Series(RETURNS, index=index, dtype="float64"),
        "spans_gap": pd.Series([False, False, True, False, False], index=index),
        "first_price_date": date(2024, 1, 2),
        "gap_tolerance_weekdays": 1,
        "provisional_bar": None,
        "provisional_bar_excluded": False,
        "ingestion_rows_dropped": 0,
        "source": "unit_test",
        "fetched_at": FETCHED_AT,
    }
    fields.update(overrides)
    return ReturnSeries(**fields)


# --- scale: explicit, single-stored, impossible to mix silently ----------------------------------


def test_scales_are_named_and_carry_their_multiplier() -> None:
    assert (ReturnScale.DECIMAL.factor, ReturnScale.PERCENT.factor) == (1.0, 100.0)
    assert ReturnScale.DECIMAL.series_name == "log_return_decimal"
    assert ReturnScale.PERCENT.series_name == "log_return_percent"
    assert STORED_SCALE is ReturnScale.DECIMAL
    assert MODELING_SCALE is ReturnScale.PERCENT


def test_the_stored_series_is_decimal_and_named_for_its_unit() -> None:
    series = make_series()

    assert series.scale is ReturnScale.DECIMAL
    assert series.decimal.name == "log_return_decimal"
    assert np.array_equal(series.decimal.to_numpy(), RETURNS)


def test_percent_is_exactly_one_hundred_times_decimal_and_named_for_its_unit() -> None:
    series = make_series()

    assert series.percent.name == "log_return_percent"
    assert np.array_equal(series.percent.to_numpy(), series.decimal.to_numpy() * 100.0)
    assert series.percent.iloc[0] == pytest.approx(1.23, rel=1e-12)
    assert series.percent.index.equals(series.decimal.index)


def test_a_factor_of_one_hundred_is_visible_in_the_dispersion() -> None:
    series = make_series()

    ratio = series.percent.std() / series.decimal.std()

    assert ratio == pytest.approx(100.0, rel=1e-12)


def test_scaled_dispatches_on_the_requested_scale() -> None:
    series = make_series()

    assert series.scaled(ReturnScale.DECIMAL) is series.decimal
    pd.testing.assert_series_equal(series.scaled(ReturnScale.PERCENT), series.percent)


def test_percent_is_computed_on_access_and_never_stored() -> None:
    series = make_series()
    stored_fields = {field.name for field in dataclasses.fields(series)}

    first, second = series.percent, series.percent
    first.iloc[0] = 999.0  # scribbling on a view must not touch the stored returns

    assert "percent" not in stored_fields
    assert first is not second
    assert series.decimal.iloc[0] == RETURNS[0]
    assert series.percent.iloc[0] == pytest.approx(1.23, rel=1e-12)


def test_there_is_no_scale_less_accessor() -> None:
    series = make_series()

    for ambiguous in ("values", "returns", "series", "log_returns"):
        assert not hasattr(series, ambiguous), f"ReturnSeries.{ambiguous} would hide the scale"


def test_return_series_is_frozen() -> None:
    series = make_series()

    with pytest.raises(dataclasses.FrozenInstanceError):
        series.decimal = series.percent  # type: ignore[misc]


def test_the_return_series_carries_no_reference_to_market_data() -> None:
    names = {field.name for field in dataclasses.fields(ReturnSeries)}

    assert not names & {"market_data", "prices", "request", "provider"}


# --- audit metadata ------------------------------------------------------------------------


def test_derived_counts_and_dates() -> None:
    series = make_series()

    assert series.return_observations == 5
    assert series.price_observations == 6
    assert series.first_return_date == date(2024, 1, 3)
    assert series.last_return_date == date(2024, 1, 9)
    assert series.flagged_gaps == 1
    assert series.excluded_observations == 0
    assert series.definition == "log: R_t = ln(P_t / P_(t-1))"


def test_metadata_is_a_complete_json_friendly_audit_record() -> None:
    series = make_series(ingestion_rows_dropped=3)

    record = series.metadata()

    assert json.loads(json.dumps(record)) == record
    assert record == {
        "ticker": "TEST",
        "frequency": "1d",
        "price_basis": "adjusted",
        "return_definition": "log: R_t = ln(P_t / P_(t-1))",
        "stored_scale": "decimal",
        "modeling_scale": "percent",
        "price_observations": 6,
        "return_observations": 5,
        "first_price_date": "2024-01-02",
        "first_return_date": "2024-01-03",
        "last_return_date": "2024-01-09",
        "flagged_gaps": 1,
        "gap_tolerance_weekdays": 1,
        "ingestion_rows_dropped": 3,
        "excluded_observations": 0,
        "provisional_bar": None,
        "provisional_bar_excluded": False,
        "last_return_is_provisional": False,
        "source": "unit_test",
        "fetched_at": "2024-06-03T12:00:00+00:00",
    }


def test_an_excluded_provisional_bar_is_recorded_but_not_part_of_the_series() -> None:
    series = make_series(provisional_bar=date(2024, 1, 10), provisional_bar_excluded=True)

    assert series.excluded_observations == 1
    assert series.last_return_is_provisional is False
    assert series.metadata()["provisional_bar"] == "2024-01-10"


def test_an_included_provisional_bar_marks_the_last_return_provisional() -> None:
    series = make_series(provisional_bar=date(2024, 1, 9), provisional_bar_excluded=False)

    assert series.excluded_observations == 0
    assert series.last_return_is_provisional is True


# --- construction validates everything ---------------------------------------------------------


def _index(n: int = 5) -> pd.DatetimeIndex:
    return pd.bdate_range("2024-01-03", periods=n, name="date")


def _returns(values: list[float], index: pd.Index | None = None) -> pd.Series:
    return pd.Series(values, index=_index(len(values)) if index is None else index, dtype="float64")


BAD_CONSTRUCTIONS: list[tuple[str, dict[str, Any]]] = [
    ("nan-return", {"decimal": _returns([0.01, np.nan, 0.0, 0.01, 0.02])}),
    ("inf-return", {"decimal": _returns([0.01, np.inf, 0.0, 0.01, 0.02])}),
    ("int-dtype", {"decimal": pd.Series([1, 2, 3, 4, 5], index=_index())}),
    ("not-a-series", {"decimal": [0.01, 0.02]}),
    ("empty", {"decimal": _returns([]), "spans_gap": pd.Series([], index=_index(0), dtype=bool)}),
    ("range-index", {"decimal": _returns(RETURNS, pd.RangeIndex(5))}),
    ("tz-aware-index", {"decimal": _returns(RETURNS, _index().tz_localize("UTC"))}),
    (
        "duplicate-dates",
        {"decimal": _returns(RETURNS, pd.DatetimeIndex(["2024-01-03"] * 2 + list(_index()[2:])))},
    ),
    ("unsorted-dates", {"decimal": _returns(RETURNS, _index()[::-1])}),
    ("flags-not-bool", {"spans_gap": pd.Series([0, 0, 1, 0, 0], index=_index())}),
    (
        "flags-misaligned",
        {"spans_gap": pd.Series([False] * 5, index=_index() + pd.Timedelta(days=7))},
    ),
    ("flags-wrong-length", {"spans_gap": pd.Series([False] * 4, index=_index(4))}),
    ("first-price-not-before-first-return", {"first_price_date": date(2024, 1, 3)}),
    ("excluded-without-a-bar", {"provisional_bar_excluded": True}),
    ("included-bar-not-last", {"provisional_bar": date(2024, 1, 5)}),
    ("naive-fetched-at", {"fetched_at": datetime(2024, 6, 3, 12, 0)}),
    (
        "non-utc-fetched-at",
        {"fetched_at": datetime(2024, 6, 3, 12, 0, tzinfo=timezone(timedelta(hours=2)))},
    ),
    ("negative-tolerance", {"gap_tolerance_weekdays": -1}),
    ("negative-rows-dropped", {"ingestion_rows_dropped": -1}),
    ("bool-rows-dropped", {"ingestion_rows_dropped": True}),
    ("empty-ticker", {"ticker": ""}),
    ("empty-frequency", {"frequency": ""}),
    ("string-price-basis", {"price_basis": "adjusted"}),
]


@pytest.mark.parametrize(
    "overrides", [overrides for _, overrides in BAD_CONSTRUCTIONS],
    ids=[name for name, _ in BAD_CONSTRUCTIONS],
)
def test_invalid_construction_raises_invalid_return_series_error(overrides: dict[str, Any]) -> None:
    with pytest.raises(InvalidReturnSeriesError) as error:
        make_series(**overrides)

    assert isinstance(error.value, PreprocessingError)


def test_a_valid_alternative_construction_is_accepted() -> None:
    series = make_series(price_basis=PriceBasis.SPLIT_ADJUSTED, frequency="1d")

    assert series.price_basis is PriceBasis.SPLIT_ADJUSTED


