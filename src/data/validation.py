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
  recorded on :class:`~src.data.market_data.MarketData` (``dropped_dates``).
* **Preserved as delivered:** missing values in ``open``, ``high``, ``low`` and
  ``volume`` on rows that do have a close.

Deliberately out of scope: cross-field plausibility (for example ``high >= low``)
and outlier detection.  Those judgements belong to later, explicitly configured
stages rather than to a hard ingestion gate.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

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
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Validate ``frame`` against the market data contract and return it cleaned.

    Args:
        frame: Candidate price frame, normally a provider's output or a cache entry.
        request: The request the frame is supposed to answer.
        min_observations: Minimum number of usable rows required after cleaning.

    Returns:
        ``(prices, dropped_dates)``: a new float64 frame in standard column order
        with a ``datetime64[ns]`` session-date index, and the dates of the rows
        removed for lacking a close (empty when none).  Later stages need the
        dates, not just a count, to tell which returns span a removed row.  The
        input is never modified.

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
    n_missing = int(missing_close.sum())
    if n_missing == len(values):
        raise EmptyDataError(
            f"No usable observations for {context}: all {n_missing} rows lack a close price."
        )
    usable = values.loc[~missing_close].copy()
    usable_index = index[~missing_close]

    non_positive = usable_index[(usable[PRICE_COLUMN] <= 0).to_numpy()]
    if len(non_positive):
        raise DataValidationError(
            f"Invalid prices for {context}: non-positive close on {_dates(non_positive)}."
        )
    for column in ("open", "high", "low"):
        invalid = usable_index[(usable[column].notna() & usable[column].le(0)).to_numpy()]
        if len(invalid):
            raise DataValidationError(
                f"Invalid prices for {context}: non-positive {column} on {_dates(invalid)}."
            )
    negative_volume = usable_index[(usable["volume"].notna() & usable["volume"].lt(0)).to_numpy()]
    if len(negative_volume):
        raise DataValidationError(
            f"Invalid volume for {context}: negative value on {_dates(negative_volume)}."
        )
    dropped = index[missing_close]
    if n_missing:
        logger.warning(
            "Dropped %d row(s) without a close price for %s (not filled): %s",
            n_missing,
            context,
            _dates(dropped),
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
    return usable, pd.DatetimeIndex(dropped.as_unit("ns").to_numpy(), name=INDEX_NAME)


def audit_price_quality(
    prices: pd.DataFrame,
    request: MarketDataRequest,
    *,
    now: datetime | None = None,
    large_move_warning_pct: float = 20.0,
    stale_after_days: int = 7,
) -> tuple[str, ...]:
    """Return warning-only diagnostics for an already validated daily price frame.

    Weekday gaps are candidates because exchange holidays cannot be distinguished
    without an exchange calendar. Large close changes are reported, never removed:
    a genuine crash must remain available to risk models. A stale-date warning is
    only emitted for requests asking for the latest data.
    """
    if not isinstance(prices, pd.DataFrame) or PRICE_COLUMN not in prices:
        raise DataValidationError("Quality audit requires a validated frame with a close column.")
    if isinstance(large_move_warning_pct, bool) or not np.isfinite(large_move_warning_pct) or large_move_warning_pct <= 0:
        raise ValueError("large_move_warning_pct must be positive and finite.")
    if isinstance(stale_after_days, bool) or not isinstance(stale_after_days, int) or stale_after_days < 1:
        raise ValueError("stale_after_days must be a positive integer.")
    warnings: list[str] = []
    close = prices[PRICE_COLUMN].astype("float64")

    changes = close.pct_change(fill_method=None) * 100.0
    large = changes.abs() >= large_move_warning_pct
    if large.any():
        dates = [pd.Timestamp(value).date().isoformat() for value in changes.index[large][:_MAX_DATES_LISTED]]
        warnings.append(
            f"{int(large.sum())} daily close change(s) were at least {large_move_warning_pct:g}%; "
            f"review without excluding valid market moves (dates: {', '.join(dates)})."
        )

    missing_weekdays: list[tuple[str, int]] = []
    index = pd.DatetimeIndex(prices.index)
    for previous, current in zip(index[:-1], index[1:], strict=True):
        start = np.datetime64((previous + pd.Timedelta(days=1)).date())
        end = np.datetime64(current.date())
        count = int(np.busday_count(start, end))
        if count > 1:
            missing_weekdays.append((previous.date().isoformat(), current.date().isoformat()))
    if missing_weekdays:
        examples = ", ".join(f"{left}..{right}" for left, right in missing_weekdays[:_MAX_DATES_LISTED])
        warnings.append(
            f"{len(missing_weekdays)} interval(s) contain more than one missing weekday; "
            f"exchange holidays cannot be distinguished from missing sessions without a calendar ({examples})."
        )

    if request.end is None and not prices.empty:
        current_time = now or datetime.now(timezone.utc)
        if current_time.tzinfo is None:
            raise ValueError("now must be timezone-aware.")
        age = (current_time.astimezone(timezone.utc).date() - index[-1].date()).days
        if age > stale_after_days:
            warnings.append(
                f"Latest observation is {age} calendar days old (warning threshold {stale_after_days}); "
                "exchange holidays and provider delays may affect freshness."
            )
    return tuple(warnings)
