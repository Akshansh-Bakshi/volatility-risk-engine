"""Return construction: validated prices -> log returns -> :class:`ReturnSeries`.

:func:`build_return_series` is the only entry point.  It reads the authoritative
``close`` series of a :class:`~src.data.market_data.MarketData`, never modifies it,
and produces the project's canonical return

.. math:: R_t = \\ln(P_t / P_{t-1})

vectorised over the whole series.  Return ``t`` uses only prices ``t`` and
``t-1``, so there is no look-ahead.  The first price has no return and is not
represented.  Scale conventions are defined by
:mod:`src.preprocessing.return_series`.

Input checks
------------
The prices are re-validated with the same rules the market data layer applies
(:func:`src.data.validation.validate_prices`), so there is a single definition of
"valid prices": finite, positive, unique, strictly increasing session dates.  Unlike
at ingestion, a missing close is not repaired here: ``MarketData`` promises none, so
one is an error.  Nothing is filled or converted silently.  Before returning, every
computed return must be finite; an overflowing price ratio raises
:class:`~src.exceptions.InvalidReturnSeriesError` instead of producing ``inf``.

Gaps
----
A return *spans a gap* (``ReturnSeries.spans_gap``) when either

* a row removed at ingestion for lacking a close lies between the two prices
  (exact: ``MarketData.dropped_dates``), or
* more than ``max_gap_weekdays`` consecutive weekdays (Monday to Friday) with no
  observation lie between the two prices.

The second rule is plain weekday arithmetic, **not a trading calendar**.  Weekends
never count; a weekday without an observation is either missing data or an exchange
holiday, and the two cannot be told apart without a calendar.  The default
tolerance of one weekday therefore treats a single closure as ordinary and flags two
or more.  Consequently a single session missing from the provider's output (as
opposed to removed at ingestion) is not detected, and neither is a missing weekend
day for assets that trade every day.  Prices are never invented to close a gap and
flagged returns are never removed.

Provisional latest bar
----------------------
A daily bar delivered while its session is still open is not final.  Without an
exchange calendar the session state is unknowable, so the rule is deliberately
conservative: the latest observation is *potentially provisional* when its date is
on or after the UTC calendar date of the download (``MarketData.fetched_at``).  A
session dated earlier than that UTC date has closed on every major exchange.  By
default (``include_provisional_bar=False``) such a bar is excluded, so only
finalised observations reach the modelling layers; the decision is recorded on the
result.  The cost is that the latest session may be missing until the next UTC day,
even if its market has already closed; set ``include_provisional_bar`` to keep it,
in which case the result says so (``last_return_is_provisional``).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import PreprocessingConfig, get_settings
from src.data.market_data import PRICE_COLUMN, MarketData
from src.data.validation import validate_prices
from src.exceptions import (
    DataError,
    DataValidationError,
    InsufficientHistoryError,
    InvalidReturnSeriesError,
    PreprocessingError,
)
from src.logging_config import PACKAGE_LOGGER_NAME
from src.preprocessing.return_series import STORED_SCALE, PriceBasis, ReturnSeries

logger = logging.getLogger(f"{PACKAGE_LOGGER_NAME}.preprocessing.returns")


def build_return_series(
    market_data: MarketData, config: PreprocessingConfig | None = None
) -> ReturnSeries:
    """Turn validated market data into a :class:`ReturnSeries` of daily log returns.

    Args:
        market_data: Output of the market data loader.  It is not modified.
        config: Preprocessing settings; defaults to ``get_settings().preprocessing``.

    Raises:
        DataValidationError: The prices violate the market data contract (including a
            missing close, which is never filled).
        InsufficientHistoryError: Fewer than ``config.min_returns`` returns are available.
        InvalidReturnSeriesError: The prices cannot produce finite log returns.
    """
    config = config if config is not None else get_settings().preprocessing
    try:
        returns = _build(market_data, config)
    except (DataError, PreprocessingError) as exc:
        logger.error(
            "Return construction failed: ticker=%s interval=%s error=%s: %s",
            market_data.ticker,
            market_data.request.interval,
            type(exc).__name__,
            exc,
        )
        raise
    logger.info(
        "Built return series: ticker=%s frequency=%s price_basis=%s definition=log "
        "stored_scale=%s prices=%d returns=%d span=%s..%s flagged_gaps=%d "
        "ingestion_rows_dropped=%d provisional_bar=%s excluded=%s",
        returns.ticker,
        returns.frequency,
        returns.price_basis.value,
        STORED_SCALE.value,
        returns.price_observations,
        returns.return_observations,
        returns.first_price_date.isoformat(),
        returns.last_return_date.isoformat(),
        returns.flagged_gaps,
        returns.ingestion_rows_dropped,
        returns.provisional_bar,
        returns.provisional_bar_excluded,
    )
    if returns.last_return_is_provisional:
        logger.warning(
            "The last return of %s (%s) rests on a bar that may still have been forming when "
            "the prices were downloaded (%s); it is included because "
            "include_provisional_bar is set.",
            returns.ticker,
            returns.last_return_date.isoformat(),
            returns.fetched_at.isoformat(),
        )
    return returns


def _build(market_data: MarketData, config: PreprocessingConfig) -> ReturnSeries:
    close = _validated_close(market_data)

    provisional = _potentially_provisional(close, market_data)
    provisional_bar = pd.Timestamp(close.index[-1]).date() if provisional else None
    exclude = provisional and not config.include_provisional_bar
    if exclude:
        close = close.iloc[:-1]
    _require_enough_returns(close, market_data, config, provisional_excluded=exclude)

    request = market_data.request
    return ReturnSeries(
        ticker=market_data.ticker,
        frequency=request.interval,
        price_basis=PriceBasis.ADJUSTED if request.adjust_prices else PriceBasis.SPLIT_ADJUSTED,
        decimal=_log_returns(close, market_data.ticker),
        spans_gap=_spans_gap(
            pd.DatetimeIndex(close.index), market_data.dropped_dates, config.max_gap_weekdays
        ),
        first_price_date=pd.Timestamp(close.index[0]).date(),
        gap_tolerance_weekdays=config.max_gap_weekdays,
        provisional_bar=provisional_bar,
        provisional_bar_excluded=exclude,
        ingestion_rows_dropped=market_data.rows_dropped,
        source=market_data.source,
        fetched_at=market_data.fetched_at,
    )


def _validated_close(market_data: MarketData) -> pd.Series:
    """Return a validated copy of the authoritative price series."""
    prices, unexpected = validate_prices(
        market_data.prices, market_data.request, min_observations=2
    )
    if len(unexpected):
        raise DataValidationError(
            f"MarketData for {market_data.ticker} contains {len(unexpected)} row(s) without a "
            "close price; expected the validated output of the market data loader. Prices are "
            "never filled."
        )
    return prices[PRICE_COLUMN]


def _potentially_provisional(close: pd.Series, market_data: MarketData) -> bool:
    """Whether the latest bar is dated on or after the UTC date of the download."""
    return pd.Timestamp(close.index[-1]).date() >= market_data.fetched_at.date()


def _require_enough_returns(
    close: pd.Series,
    market_data: MarketData,
    config: PreprocessingConfig,
    *,
    provisional_excluded: bool,
) -> None:
    available = len(close) - 1
    if available >= config.min_returns:
        return
    note = (
        " after excluding the potentially provisional latest bar "
        "(see include_provisional_bar)"
        if provisional_excluded
        else ""
    )
    raise InsufficientHistoryError(
        f"Insufficient history for {market_data.ticker}: {max(available, 0)} return(s) "
        f"available{note}, at least {config.min_returns} required."
    )


def _log_returns(close: pd.Series, ticker: str) -> pd.Series:
    """Compute ``ln(P_t / P_(t-1))`` for every price after the first, guarding overflow."""
    values = close.to_numpy(dtype="float64")
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            log_returns = np.log(values[1:] / values[:-1])
    except FloatingPointError as exc:
        raise InvalidReturnSeriesError(
            f"Cannot compute finite log returns for {ticker}: the price ratio overflows or "
            f"underflows to zero ({exc})."
        ) from exc
    if not np.isfinite(log_returns).all():
        raise InvalidReturnSeriesError(f"Log returns for {ticker} contain non-finite values.")
    return pd.Series(log_returns, index=close.index[1:], dtype="float64")


def _spans_gap(
    index: pd.DatetimeIndex, dropped_dates: pd.DatetimeIndex, max_gap_weekdays: int
) -> pd.Series:
    """Flag, for each return, whether it spans a removed row or a run of missing weekdays."""
    flags = _spans_removed_rows(index, dropped_dates) | (
        _missing_weekdays(index) > max_gap_weekdays
    )
    return pd.Series(flags, index=index[1:], dtype=bool)


def _spans_removed_rows(index: pd.DatetimeIndex, dropped_dates: pd.DatetimeIndex) -> np.ndarray:
    """Flag returns with a removed row strictly between their two prices (exact)."""
    flags = np.zeros(len(index) - 1, dtype=bool)
    if len(dropped_dates) == 0:
        return flags
    position = index.searchsorted(dropped_dates.as_unit("ns"))  # first price after each drop
    # A drop before the first or after the last price lies inside no return interval.
    inside = (position > 0) & (position < len(index))
    flags[position[inside] - 1] = True
    return flags


def _missing_weekdays(index: pd.DatetimeIndex) -> np.ndarray:
    """Count weekdays strictly between consecutive observations (weekends never count)."""
    days = index.to_numpy().astype("datetime64[D]")
    return np.busday_count(days[:-1] + np.timedelta64(1, "D"), days[1:])
