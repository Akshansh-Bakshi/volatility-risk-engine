"""Provider-independent validation of price data.

Every frame that enters the system, whether freshly downloaded or read back from
the cache, passes through :func:`validate_prices`.  The policy is deliberately
strict and has exactly one repair:

* **Rejected, never repaired:** duplicate or unsorted timestamps, tz-aware or
  non-midnight timestamps, malformed schema, non-numeric data, non-finite values,
  non-positive closes, and rows outside the requested date range.  Each of these
  signals a provider or pipeline fault that repairing would only hide.
* **Removed:** rows whose ``close`` is missing.  A row without the authoritative
  price carries no price information, and removing it invents nothing.  The
  alternative, forward-filling, would fabricate observations and later show up as
  artificial zero returns that bias volatility downwards.  The consequence, which
  return construction must respect, is that the return across a removed row spans
  more than one session.  Removals are counted, logged with their dates, and
  reported on :class:`~src.data.market_data.MarketData`.
* **Preserved as delivered:** missing values in ``open``, ``high``, ``low`` and
  ``volume`` on rows that do have a close.

Deliberately out of scope: cross-field plausibility (for example ``high >= low``)
and outlier detection.  Those judgements belong to later, explicitly configured
stages rather than to a hard ingestion gate.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.data.market_data import INDEX_NAME, PRICE_COLUMN, STANDARD_COLUMNS, MarketDataRequest
from src.exceptions import DataValidationError, EmptyDataError, InsufficientHistoryError
from src.logging_config import PACKAGE_LOGGER_NAME

logger = logging.getLogger(f"{PACKAGE_LOGGER_NAME}.data.validation")

_MAX_DATES_LISTED = 5


def _context(request: MarketDataRequest) -> str:
    return f"{request.ticker} ({request.interval}, {request.range_label})"


def _dates(index: pd.DatetimeIndex) -> str:
    """Render up to a handful of dates for an error message."""
    shown = [ts.date().isoformat() for ts in index[:_MAX_DATES_LISTED]]
    more = f" (+{len(index) - _MAX_DATES_LISTED} more)" if len(index) > _MAX_DATES_LISTED else ""
    return ", ".join(shown) + more


def _check_columns(frame: pd.DataFrame, context: str) -> None:
    if isinstance(frame.columns, pd.MultiIndex):
        raise DataValidationError(
            f"Malformed provider output for {context}: MultiIndex columns are not allowed; "
            "providers must return flat, standardised columns."
        )
    columns = list(frame.columns)
    missing = [name for name in STANDARD_COLUMNS if name not in columns]
    unexpected = [name for name in columns if name not in STANDARD_COLUMNS]
    duplicated = sorted({name for name in columns if columns.count(name) > 1}, key=str)
    if missing or unexpected or duplicated:
        raise DataValidationError(
            f"Malformed provider output for {context}: expected columns "
            f"{list(STANDARD_COLUMNS)}, got {columns} "
            f"(missing: {missing}, unexpected: {unexpected}, duplicated: {duplicated})."
        )


def _check_index(index: pd.Index, context: str) -> pd.DatetimeIndex:
    if not isinstance(index, pd.DatetimeIndex):
        raise DataValidationError(
            f"Invalid index for {context}: expected a DatetimeIndex, got {type(index).__name__}."
        )
    if index.tz is not None:
        raise DataValidationError(
            f"Invalid index for {context}: timestamps are timezone-aware ({index.tz}); the "
            "project convention is timezone-naive exchange-local session dates."
        )
    if index.hasnans:
        raise DataValidationError(f"Invalid index for {context}: contains missing timestamps.")
    off_midnight = index[index != index.normalize()]
    if len(off_midnight):
        raise DataValidationError(
            f"Invalid index for {context}: daily data must be normalised to midnight; "
            f"found time-of-day components at {_dates(off_midnight)}."
        )
    duplicated = index[index.duplicated()]
    if len(duplicated):
        raise DataValidationError(
            f"Invalid index for {context}: duplicate timestamps at {_dates(duplicated)}."
        )
    if not index.is_monotonic_increasing:
        raise DataValidationError(
            f"Invalid index for {context}: timestamps are not sorted in ascending order."
        )
    return index


def _numeric_values(frame: pd.DataFrame, context: str) -> pd.DataFrame:
    bad = [
        name
        for name in STANDARD_COLUMNS
        if not pd.api.types.is_numeric_dtype(frame[name]) or pd.api.types.is_bool_dtype(frame[name])
    ]
    if bad:
        raise DataValidationError(
            f"Malformed provider output for {context}: non-numeric column(s) {bad}."
        )
    values = frame.loc[:, list(STANDARD_COLUMNS)].astype("float64")
    infinite = [name for name in STANDARD_COLUMNS if np.isinf(values[name]).any()]
    if infinite:
        raise DataValidationError(
            f"Invalid values for {context}: non-finite (infinite) entries in column(s) {infinite}."
        )
    return values


def validate_prices(
    frame: pd.DataFrame, request: MarketDataRequest, *, min_observations: int
) -> tuple[pd.DataFrame, int]:
    """Validate ``frame`` against the market data contract and return it cleaned.

    Args:
        frame: Candidate price frame, normally a provider's output or a cache entry.
        request: The request the frame is supposed to answer.
        min_observations: Minimum number of usable rows required after cleaning.

    Returns:
        ``(prices, rows_dropped)``: a new float64 frame in standard column order
        with a ``datetime64[ns]`` session-date index, and the number of rows
        removed for lacking a close.  The input is never modified.

    Raises:
        EmptyDataError: No rows, or no row with a close.
        DataValidationError: The frame violates any rule listed in the module docs.
        InsufficientHistoryError: Fewer than ``min_observations`` usable rows remain.
    """
    context = _context(request)
    if not isinstance(frame, pd.DataFrame):
        raise DataValidationError(
            f"Malformed provider output for {context}: expected a DataFrame, "
            f"got {type(frame).__name__}."
        )
    if len(frame) == 0:
        raise EmptyDataError(
            f"No data returned for {context}. The symbol may be unknown or delisted, or the "
            "provider has no observations in this date range."
        )

    _check_columns(frame, context)
    index = _check_index(frame.index, context)
    values = _numeric_values(frame, context)

    first, last = index[0].date(), index[-1].date()
    if first < request.start or (request.end is not None and last > request.end):
        raise DataValidationError(
            f"Provider returned observations outside the requested range for {context}: "
            f"received {first.isoformat()}..{last.isoformat()}."
        )

    missing_close = values[PRICE_COLUMN].isna().to_numpy()
    rows_dropped = int(missing_close.sum())
    if rows_dropped == len(values):
        raise EmptyDataError(
            f"No usable observations for {context}: all {rows_dropped} rows lack a close price."
        )
    usable = values.loc[~missing_close].copy()
    usable_index = index[~missing_close]

    non_positive = usable_index[(usable[PRICE_COLUMN] <= 0).to_numpy()]
    if len(non_positive):
        raise DataValidationError(
            f"Invalid prices for {context}: non-positive close on {_dates(non_positive)}."
        )
    if rows_dropped:
        logger.warning(
            "Dropped %d row(s) without a close price for %s (not filled): %s",
            rows_dropped,
            context,
            _dates(index[missing_close]),
        )

    if len(usable) < min_observations:
        raise InsufficientHistoryError(
            f"Insufficient history for {context}: {len(usable)} usable observation(s) "
            f"({usable_index[0].date().isoformat()}..{usable_index[-1].date().isoformat()}), "
            f"at least {min_observations} required."
        )

    # Rebuilding the index from raw values drops any ``freq`` the provider attached, so the
    # result does not depend on whether rows happened to be removed.
    usable.index = pd.DatetimeIndex(usable_index.as_unit("ns").to_numpy(), name=INDEX_NAME)
    return usable, rows_dropped
