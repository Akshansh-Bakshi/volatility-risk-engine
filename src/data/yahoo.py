"""Yahoo Finance implementation of :class:`~src.data.provider.MarketDataProvider`.

This is the only module in the project that imports ``yfinance``.  It

* translates the project's request semantics (inclusive ``end``; ``end=None`` for
  "latest") into yfinance's (exclusive ``end``; ``None`` resolved by yfinance to
  the current time *at call time*),
* pins every yfinance option that affects the returned values instead of relying
  on library defaults that have changed between releases,
* normalises the vendor frame (capitalised columns, tz-aware exchange-local
  index) to the project schema, and
* translates vendor failures into the project's exception hierarchy.

Price basis: ``request.adjust_prices=True`` passes ``auto_adjust=True``, so *all*
of open/high/low/close are adjusted for splits and dividends and ``close`` is the
total-return-basis series.  ``False`` passes ``auto_adjust=False``, which yields
Yahoo's ``Close``: adjusted for splits but not for dividends, per Yahoo's column
definitions.  Either way the vendor's ``Adj Close`` column is not used; the
adjustment is applied by yfinance consistently to every price column instead.

Compatibility: written against the ``Ticker.history`` API shared by yfinance
0.2.60 through 1.x.  yfinance 1.x deprecates ``raise_errors=True`` in favour of a
process-global config flag; the flag is still honoured, so it is passed and the
one deprecation warning it emits is silenced locally rather than mutating global
library state.
"""

from __future__ import annotations

import json
import logging
import math
import time
import threading
import warnings
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
import yfinance as yf
from yfinance import exceptions as yf_exceptions

from src.config import DEFAULT_DATA_DIR
from src.data.market_data import INDEX_NAME, STANDARD_COLUMNS, MarketDataRequest
from src.exceptions import (
    DataFetchError,
    DataValidationError,
    EmptyDataError,
    InvalidTickerError,
)
from src.logging_config import PACKAGE_LOGGER_NAME

logger = logging.getLogger(f"{PACKAGE_LOGGER_NAME}.data.yahoo")

_RAISE_ERRORS_DEPRECATION = r"'raise_errors' deprecated"
_YFINANCE_CACHE_LOCK = threading.RLock()


class _HistoryClient(Protocol):
    """The slice of ``yfinance.Ticker`` this provider uses."""

    def history(self, **kwargs: Any) -> pd.DataFrame: ...


TickerFactory = Callable[[str], _HistoryClient]


def configure_yfinance_cache(directory: str | Path) -> None:
    """Place yfinance's internal SQLite caches under the configured data directory.

    yfinance otherwise chooses a user-profile cache location, which can be
    unavailable to service accounts or restricted environments. Call before the
    first provider request; yfinance applies this location to its SQLite caches.
    """
    location = Path(directory).expanduser()
    with _YFINANCE_CACHE_LOCK:
        location.mkdir(parents=True, exist_ok=True)
        yf.set_tz_cache_location(str(location))


class YahooFinanceProvider:
    """Fetch daily price history for one ticker from Yahoo Finance via ``yfinance``.

    Args:
        ticker_factory: Callable creating a yfinance-like object for a symbol.
            Defaults to ``yfinance.Ticker``; tests inject a fake here so the
            network boundary can be replaced without patching.
    """

    name = "yahoo_finance"

    def __init__(
        self,
        ticker_factory: TickerFactory | None = None,
        *,
        cache_directory: str | Path | None = None,
        timeout_seconds: float = 15.0,
        retries: int = 1,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)) or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        if isinstance(retries, bool) or not isinstance(retries, int) or not 0 <= retries <= 5:
            raise ValueError("retries must be an integer between 0 and 5.")
        self._ticker_factory: TickerFactory = ticker_factory or yf.Ticker
        self._timeout_seconds = float(timeout_seconds)
        self._retries = retries
        self._sleep = sleep
        self._cache_directory = Path(
            cache_directory if cache_directory is not None else DEFAULT_DATA_DIR / "yfinance-cache"
        )
        configure_yfinance_cache(self._cache_directory)

    def fetch(self, request: MarketDataRequest) -> pd.DataFrame:
        """Download and structurally normalise the history described by ``request``."""
        kwargs = _history_kwargs(request, timeout_seconds=self._timeout_seconds)
        context = f"{request.ticker} ({request.interval}, {request.range_label})"
        logger.debug("Requesting %s from Yahoo Finance with %s", request.ticker, kwargs)
        with _YFINANCE_CACHE_LOCK:
            # yfinance stores its SQLite cache manager globally. Re-select the
            # provider's directory under a lock so concurrent project loaders
            # cannot switch paths while another request has an open connection.
            configure_yfinance_cache(self._cache_directory)
            raw = self._fetch_with_retries(kwargs, context, request.ticker)
        return _normalize(raw, context)

    def _fetch_with_retries(self, kwargs: dict[str, Any], context: str, ticker: str) -> object:
        last_error: Exception | None = None
        for attempt in range(self._retries + 1):
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", message=_RAISE_ERRORS_DEPRECATION, category=DeprecationWarning
                    )
                    raw = self._ticker_factory(ticker).history(**kwargs)
                break
            except Exception as exc:  # noqa: BLE001 - vendor boundary.
                last_error = exc
                if attempt >= self._retries or not _is_retryable(exc):
                    raise _translate_vendor_error(exc, context, ticker, attempts=attempt + 1) from exc
                delay = min(0.25 * (2**attempt), 1.0)
                logger.warning(
                    "Transient Yahoo Finance failure for %s (%s: %s); retry %d/%d in %.2fs.",
                    context, type(exc).__name__, exc, attempt + 1, self._retries, delay,
                )
                self._sleep(delay)
        else:  # pragma: no cover - the loop always exits by break or exception.
            assert last_error is not None
            raise _translate_vendor_error(last_error, context, ticker, attempts=self._retries + 1)
        return raw


def _translate_vendor_error(exc: Exception, context: str, ticker: str, *, attempts: int) -> Exception:
    if isinstance(exc, yf_exceptions.YFPricesMissingError):
        return EmptyDataError(f"Yahoo Finance has no price data for {context}: {exc}")
    if isinstance(exc, yf_exceptions.YFTickerMissingError):
        return InvalidTickerError(
            f"Yahoo Finance does not recognise symbol {ticker!r} ({exc}). "
            "The provider may also report this when Yahoo cannot be reached; check the symbol and your connection."
        )
    if isinstance(exc, yf_exceptions.YFRateLimitError):
        return DataFetchError(f"Yahoo Finance rate-limited the request for {context}; retry later.")
    suffix = f" after {attempts} attempt(s)" if attempts > 1 else ""
    return DataFetchError(
        f"Yahoo Finance request failed for {context}{suffix}: {type(exc).__name__}: {exc}"
        f"{_connectivity_hint(exc)}"
    )


def _is_retryable(exc: Exception) -> bool:
    """Retry only transient connectivity/time-out/response-decoding failures."""
    retryable_names = {"ConnectionError", "ConnectTimeout", "ReadTimeout", "Timeout", "ConnectionTimeout"}
    names = {cls.__name__ for cls in type(exc).__mro__}
    return isinstance(exc, (ConnectionError, TimeoutError, json.JSONDecodeError)) or bool(names & retryable_names)


def _connectivity_hint(exc: Exception) -> str:
    """Explain yfinance's most misleading failure mode.

    When a request is blocked (firewall, proxy allow-list) or answered with an HTML/text
    error page, yfinance fails while parsing the body, so the visible error is a bare
    ``JSONDecodeError`` rather than anything about connectivity.
    """
    if isinstance(exc, json.JSONDecodeError):
        return (
            " (yfinance reports blocked or failed HTTP requests as JSON parsing errors; "
            "check network access to Yahoo Finance)"
        )
    return ""


def _history_kwargs(request: MarketDataRequest, *, timeout_seconds: float = 15.0) -> dict[str, Any]:
    """Translate a request into explicit ``Ticker.history`` arguments."""
    # yfinance's end is exclusive, so an inclusive request end needs +1 day.  None is
    # passed through: yfinance then resolves "now" when the call is made.
    end = None if request.end is None else (request.end + timedelta(days=1)).isoformat()
    return {
        "start": request.start.isoformat(),  # inclusive in yfinance
        "end": end,
        "interval": request.interval,
        "auto_adjust": request.adjust_prices,
        "actions": False,
        "repair": False,  # no silent vendor-side price "repairs"
        "keepna": True,  # deliver rows without prices so validation can count them
        "raise_errors": True,
        "timeout": timeout_seconds,
    }


def _normalize(raw: object, context: str) -> pd.DataFrame:
    """Map a yfinance frame onto the project schema without repairing its rows."""
    if not isinstance(raw, pd.DataFrame):
        raise DataValidationError(
            f"Malformed Yahoo Finance output for {context}: expected a DataFrame, "
            f"got {type(raw).__name__}."
        )
    if len(raw) == 0:
        return pd.DataFrame(
            {name: pd.Series(dtype="float64") for name in STANDARD_COLUMNS},
            index=pd.DatetimeIndex([], name=INDEX_NAME),
        )
    if isinstance(raw.columns, pd.MultiIndex) or not raw.columns.is_unique:
        raise DataValidationError(
            f"Malformed Yahoo Finance output for {context}: expected flat, unique columns, "
            f"got {list(raw.columns)}."
        )
    if not isinstance(raw.index, pd.DatetimeIndex):
        raise DataValidationError(
            f"Malformed Yahoo Finance output for {context}: expected a DatetimeIndex, "
            f"got {type(raw.index).__name__}."
        )
    by_name = {str(column).strip().lower(): column for column in raw.columns}
    missing = [name for name in STANDARD_COLUMNS if name not in by_name]
    if missing:
        raise DataValidationError(
            f"Malformed Yahoo Finance output for {context}: missing column(s) {missing}; "
            f"received {list(raw.columns)}."
        )

    frame = raw.loc[:, [by_name[name] for name in STANDARD_COLUMNS]].copy()
    frame.columns = pd.Index(STANDARD_COLUMNS)
    frame.index = _session_dates(raw.index)
    return frame


def _session_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Convert yfinance's exchange-local timestamps to naive midnight session dates.

    The timezone is *dropped*, never converted: ``tz_convert("UTC")`` would move the
    label of an Asian session (local midnight) to the previous calendar day.
    ``normalize`` then removes any time-of-day yfinance introduced when localising a
    midnight that does not exist in the exchange timezone (DST transitions).
    """
    naive = index.tz_localize(None) if index.tz is not None else index
    return naive.normalize().rename(INDEX_NAME)
