"""The market data entry point: provider + validation + cache, with logging.

:class:`MarketDataLoader` is what the rest of the application calls.  It knows
the provider only through the :class:`~src.data.provider.MarketDataProvider`
contract.  Order of operations for one request:

1. Unless ``refresh`` is set, look for a cache entry.  A hit is used only if it is
   fresh (see :func:`~src.data.cache.is_entry_fresh`) and still passes
   validation; anything else is a miss.
2. Otherwise ask the provider, validate the result, stamp it with the download
   time and the provider name, and store it (cache write failures are logged, not
   fatal).

Cache controls: build the loader without a cache to bypass it entirely (nothing is
read or written); pass ``refresh=True`` to skip the read and overwrite the entry.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from src.config import DEFAULT_CACHE_MAX_AGE_HOURS, DEFAULT_MIN_OBSERVATIONS
from src.data.cache import MarketDataCache, is_entry_fresh
from src.data.market_data import MarketData, MarketDataRequest
from src.data.provider import MarketDataProvider
from src.data.validation import validate_prices
from src.exceptions import DataError, DataFetchError
from src.logging_config import PACKAGE_LOGGER_NAME

logger = logging.getLogger(f"{PACKAGE_LOGGER_NAME}.data.loader")

Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class MarketDataLoader:
    """Load validated market data through a provider, with an optional local cache.

    Args:
        provider: Source of raw price history.
        cache: Where to keep validated downloads; ``None`` disables caching.
        min_observations: Minimum usable observations for a result to be accepted.
        cache_max_age: How long a "live" cache entry (one whose window had not
            clearly closed when it was downloaded) counts as current.
        clock: Returns the current time as a timezone-aware UTC datetime.
            Injected so cache freshness is testable; defaults to the system clock.
    """

    def __init__(
        self,
        provider: MarketDataProvider,
        *,
        cache: MarketDataCache | None = None,
        min_observations: int = DEFAULT_MIN_OBSERVATIONS,
        cache_max_age: timedelta = timedelta(hours=DEFAULT_CACHE_MAX_AGE_HOURS),
        clock: Clock = _utc_now,
    ) -> None:
        self._provider = provider
        self._cache = cache
        self._min_observations = min_observations
        self._cache_max_age = cache_max_age
        self._clock = clock

    def load(self, request: MarketDataRequest, *, refresh: bool = False) -> MarketData:
        """Return validated market data for ``request``.

        Args:
            request: What to load.
            refresh: Ignore any cached entry, download again and overwrite it.

        Raises:
            DataError: Any of its subclasses; see :mod:`src.exceptions`.
        """
        try:
            data, cache_status = self._load(request, refresh)
        except DataError as exc:
            logger.error(
                "Market data load failed: ticker=%s interval=%s requested=%s error=%s: %s",
                request.ticker,
                request.interval,
                request.range_label,
                type(exc).__name__,
                exc,
                exc_info=isinstance(exc, DataFetchError),
            )
            raise
        logger.info(
            "Loaded market data: ticker=%s interval=%s adjust_prices=%s requested=%s "
            "actual=%s..%s rows=%d rows_dropped=%d source=%s fetched_at=%s cache=%s",
            data.ticker,
            request.interval,
            request.adjust_prices,
            request.range_label,
            data.first_date.isoformat(),
            data.last_date.isoformat(),
            data.observations,
            data.rows_dropped,
            data.source,
            data.fetched_at.isoformat(),
            cache_status,
        )
        return data

    def _load(self, request: MarketDataRequest, refresh: bool) -> tuple[MarketData, str]:
        if self._cache is None:
            status = "disabled"
        elif refresh:
            status = "refresh"
        else:
            cached, status = self._read_cache(self._cache, request)
            if cached is not None:
                return cached, status

        data = self._download(request)
        if self._cache is not None:
            self._write_cache(self._cache, data)
        return data, status

    def _read_cache(
        self, cache: MarketDataCache, request: MarketDataRequest
    ) -> tuple[MarketData | None, str]:
        entry = cache.read(request)
        if entry is None:
            return None, "miss"
        if not is_entry_fresh(
            request, entry.fetched_at, now=self._clock(), max_age=self._cache_max_age
        ):
            logger.info(
                "Cache entry for %s [%s] is stale (downloaded %s); downloading again.",
                request.ticker,
                request.range_label,
                entry.fetched_at.isoformat(),
            )
            return None, "stale"
        try:
            prices, _ = validate_prices(
                entry.prices, request, min_observations=self._min_observations
            )
        except DataError as exc:
            logger.warning(
                "Discarding cache entry for %s [%s] that no longer validates: %s",
                request.ticker,
                request.range_label,
                exc,
            )
            return None, "invalid"
        return replace(entry, prices=prices, from_cache=True), "hit"

    def _download(self, request: MarketDataRequest) -> MarketData:
        raw = self._provider.fetch(request)
        prices, rows_dropped = validate_prices(
            raw, request, min_observations=self._min_observations
        )
        return MarketData(
            request=request,
            prices=prices,
            source=self._provider.name,
            fetched_at=self._clock(),
            rows_dropped=rows_dropped,
        )

    @staticmethod
    def _write_cache(cache: MarketDataCache, data: MarketData) -> None:
        try:
            cache.write(data)
        except OSError as exc:
            logger.warning(
                "Could not write cache entry for %s [%s] (%s); continuing without caching.",
                data.ticker,
                data.request.range_label,
                exc,
            )
