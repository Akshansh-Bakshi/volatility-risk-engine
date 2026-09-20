"""Domain types for market data: what is asked for and what is delivered.

Conventions every consumer of this package can rely on:

* **Schema.** Prices are a :class:`pandas.DataFrame` with the float64 columns in
  :data:`STANDARD_COLUMNS`.  ``close`` (:data:`PRICE_COLUMN`) is the single
  authoritative price series for all later return calculations; the other
  columns are preserved for range-based analysis and are informational.
* **Index.** A :class:`pandas.DatetimeIndex` named ``date`` that is
  timezone-naive, normalised to midnight, strictly increasing and unique.  Each
  label is the *exchange-local session date* of a daily bar, not an instant.
  Naive session dates (rather than tz-aware timestamps) keep series from
  exchanges in different timezones alignable by calendar date and avoid
  DST artefacts.
* **Dates.** Requests use an *inclusive* ``start`` and an *inclusive* ``end``;
  ``end=None`` means "the latest available session" and is resolved each time
  data is fetched.  Provider-specific end-exclusive conventions are translated
  inside the provider.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta

import pandas as pd

from src.config import DataConfig
from src.exceptions import DataValidationError

INDEX_NAME = "date"
PRICE_COLUMN = "close"
STANDARD_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")

# Daily bars only: the rest of the pipeline (return construction, annualisation,
# daily VaR) assumes one observation per session.  Widen this only together with
# the downstream code that depends on the frequency.
SUPPORTED_INTERVALS: tuple[str, ...] = ("1d",)


@dataclass(frozen=True)
class MarketDataRequest:
    """A provider-independent description of the price history to obtain.

    Attributes:
        ticker: Provider symbol, normalised to upper case.
        start: First session date wanted (inclusive).
        end: Last session date wanted (inclusive), or ``None`` for the latest available.
        interval: Bar size; see :data:`SUPPORTED_INTERVALS`.
        adjust_prices: ``True`` for prices adjusted for splits and dividends
            (total-return basis); ``False`` for split-adjusted prices only.
    """

    ticker: str
    start: date
    end: date | None = None
    interval: str = "1d"
    adjust_prices: bool = True

    def __post_init__(self) -> None:
        ticker = self.ticker.strip().upper() if isinstance(self.ticker, str) else ""
        if not ticker:
            raise DataValidationError("Invalid request: ticker must be a non-empty string.")
        object.__setattr__(self, "ticker", ticker)

        for name in ("start", "end"):
            value = getattr(self, name)
            if value is None and name == "end":
                continue
            if not isinstance(value, date) or isinstance(value, datetime):
                raise DataValidationError(
                    f"Invalid request for {ticker}: {name} must be a calendar date "
                    f"(datetime.date), got {value!r}."
                )
        if self.end is not None and self.start >= self.end:
            raise DataValidationError(
                f"Invalid request for {ticker}: start ({self.start.isoformat()}) must be "
                f"strictly earlier than end ({self.end.isoformat()})."
            )
        if self.interval not in SUPPORTED_INTERVALS:
            raise DataValidationError(
                f"Invalid request for {ticker}: unsupported interval {self.interval!r}; "
                f"supported: {', '.join(SUPPORTED_INTERVALS)}."
            )
        if not isinstance(self.adjust_prices, bool):
            raise DataValidationError(
                f"Invalid request for {ticker}: adjust_prices must be a boolean."
            )

    @classmethod
    def from_config(cls, config: DataConfig) -> MarketDataRequest:
        """Build the default request described by ``config`` (``end_date=None`` is kept as-is)."""
        return cls(
            ticker=config.ticker,
            start=config.start_date,
            end=config.end_date,
            adjust_prices=config.adjust_prices,
        )

    @property
    def range_label(self) -> str:
        """Human-readable requested range, e.g. ``2005-01-01..latest``."""
        return f"{self.start.isoformat()}..{'latest' if self.end is None else self.end.isoformat()}"


@dataclass(frozen=True, eq=False)
class MarketData:
    """Validated market data for one ticker, plus the provenance needed to trust it.

    Attributes:
        request: The request this data answers.
        prices: Validated price frame following the module conventions.
        source: Name of the provider that supplied the data.
        fetched_at: When the data was downloaded (timezone-aware, UTC).  For data
            served from the cache this is the time of the original download.
        rows_dropped: Rows the provider delivered without a close price, removed
            during validation.
        from_cache: Whether this object was served from the local cache.
    """

    request: MarketDataRequest
    prices: pd.DataFrame
    source: str
    fetched_at: datetime
    rows_dropped: int = 0
    from_cache: bool = False

    def __post_init__(self) -> None:
        if self.fetched_at.tzinfo is None or self.fetched_at.utcoffset() != timedelta(0):
            raise DataValidationError("MarketData.fetched_at must be timezone-aware and in UTC.")

    @property
    def ticker(self) -> str:
        return self.request.ticker

    @property
    def price(self) -> pd.Series:
        """The authoritative price series (the ``close`` column)."""
        return self.prices[PRICE_COLUMN]

    @property
    def observations(self) -> int:
        return len(self.prices)

    @property
    def first_date(self) -> date:
        return pd.Timestamp(self.prices.index[0]).date()

    @property
    def last_date(self) -> date:
        return pd.Timestamp(self.prices.index[-1]).date()
