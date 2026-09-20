"""A small local cache for validated market data.

Design, in short:

* **One file per request identity.**  The name encodes ticker, interval, start,
  end (or ``latest``) and price basis, e.g.
  ``%5EGSPC__1d__2005-01-01__latest__adjusted.json``.  The ticker is
  percent-encoded so any symbol yields a safe, collision-free name.  The same
  identity is stored inside the file and checked on read, so a renamed or copied
  file can never answer the wrong request.
* **Only validated data is stored,** after cleaning, together with its provenance
  (source, download time, rows dropped).  Entries are re-validated on every read.
* **JSON, written atomically.**  Human-readable, no extra dependencies, and
  Python's float ``repr`` round-trips float64 values exactly.  Files are written
  to a temporary name and moved into place, so readers never see partial writes.
* **Unreadable entries are misses, not crashes,** and are logged.  The cache is an
  optimisation: nothing depends on it for correctness.

Freshness (see :func:`is_entry_fresh`) is what keeps stale data from passing as
current: an entry is reusable indefinitely only if it was downloaded well after
the requested window closed; every other entry, including all ``end=None``
("latest") requests, expires after a configurable maximum age.
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
import pandas as pd

from src.data.market_data import INDEX_NAME, STANDARD_COLUMNS, MarketData, MarketDataRequest
from src.exceptions import DataError
from src.logging_config import PACKAGE_LOGGER_NAME

logger = logging.getLogger(f"{PACKAGE_LOGGER_NAME}.data.cache")

CACHE_SUBDIRECTORY = Path("cache") / "market"
SCHEMA_VERSION = 1

# A window is treated as settled once the download happened more than this many
# days after its last requested date.  The margin covers exchanges in timezones
# far from UTC and short provider publication delays.
SETTLEMENT_MARGIN = timedelta(days=1)


def is_entry_fresh(
    request: MarketDataRequest,
    fetched_at: datetime,
    *,
    now: datetime,
    max_age: timedelta,
) -> bool:
    """Decide whether a cached download may still be served for ``request``.

    * A request with an explicit ``end`` that was fully in the past (beyond the
      settlement margin) when downloaded is *settled*: new sessions cannot change
      it, and later dividend adjustments rescale the window uniformly, leaving
      returns untouched.  Such entries never expire.
    * Everything else (``end=None``, or an ``end`` that had not clearly passed at
      download time) is *live* and expires after ``max_age``.
    * An entry timestamped in the future (clock skew) is never fresh.
    """
    age = now - fetched_at
    if age < timedelta(0):
        return False
    if request.end is not None and fetched_at.date() > request.end + SETTLEMENT_MARGIN:
        return True
    return age < max_age


def _identity(request: MarketDataRequest) -> dict[str, Any]:
    return {
        "ticker": request.ticker,
        "interval": request.interval,
        "start": request.start.isoformat(),
        "end": None if request.end is None else request.end.isoformat(),
        "adjust_prices": request.adjust_prices,
    }


class MarketDataCache:
    """Directory-backed cache of :class:`~src.data.market_data.MarketData` objects.

    Args:
        directory: Where entries live.  Created on first write.
    """

    def __init__(self, directory: Path) -> None:
        self._directory = Path(directory)

    @property
    def directory(self) -> Path:
        return self._directory

    def path_for(self, request: MarketDataRequest) -> Path:
        """Return the deterministic file path for ``request``."""
        parts = (
            quote(request.ticker, safe=""),
            request.interval,
            request.start.isoformat(),
            "latest" if request.end is None else request.end.isoformat(),
            "adjusted" if request.adjust_prices else "unadjusted",
        )
        return self._directory / ("__".join(parts) + ".json")

    def read(self, request: MarketDataRequest) -> MarketData | None:
        """Return the stored entry for ``request``, or ``None`` if absent or unusable.

        Freshness is *not* judged here; see :func:`is_entry_fresh`.
        """
        path = self.path_for(request)
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError as exc:
            logger.warning("Cannot read cache entry %s (%s); treating as a miss.", path, exc)
            return None
        try:
            return _deserialize(json.loads(text), request)
        except (ValueError, KeyError, TypeError, DataError) as exc:
            logger.warning(
                "Ignoring unusable cache entry %s (%s: %s); it will be re-downloaded.",
                path,
                type(exc).__name__,
                exc,
            )
            return None

    def write(self, market_data: MarketData) -> Path:
        """Atomically store ``market_data``; return the file written.

        Raises:
            OSError: The entry could not be written.  Callers treat this as
                non-fatal because the cache is only an optimisation.
        """
        path = self.path_for(market_data.request)
        payload = json.dumps(_serialize(market_data), allow_nan=False, separators=(",", ":"))
        self._directory.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(dir=self._directory, prefix=".write-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
            os.replace(temp_name, path)
        finally:
            Path(temp_name).unlink(missing_ok=True)
        return path


def _serialize(market_data: MarketData) -> dict[str, Any]:
    prices = market_data.prices
    return {
        "schema_version": SCHEMA_VERSION,
        **_identity(market_data.request),
        "source": market_data.source,
        "fetched_at": market_data.fetched_at.astimezone(timezone.utc).isoformat(),
        "rows_dropped": market_data.rows_dropped,
        "index": pd.DatetimeIndex(prices.index).strftime("%Y-%m-%d").tolist(),
        "columns": {
            name: [None if math.isnan(value) else value for value in prices[name].tolist()]
            for name in STANDARD_COLUMNS
        },
    }


def _deserialize(payload: dict[str, Any], request: MarketDataRequest) -> MarketData:
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"unsupported cache schema version {payload['schema_version']!r}")
    stored = {key: payload[key] for key in _identity(request)}
    if stored != _identity(request):
        raise ValueError(f"entry identity {stored} does not match the request")

    index = pd.DatetimeIndex(pd.to_datetime(payload["index"]), name=INDEX_NAME).as_unit("ns")
    prices = pd.DataFrame(
        {
            name: np.array([np.nan if v is None else v for v in payload["columns"][name]], float)
            for name in STANDARD_COLUMNS
        },
        index=index,
    )
    return MarketData(
        request=request,
        prices=prices,
        source=str(payload["source"]),
        fetched_at=datetime.fromisoformat(payload["fetched_at"]),
        rows_dropped=int(payload["rows_dropped"]),
    )

