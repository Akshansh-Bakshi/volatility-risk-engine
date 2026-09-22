"""Domain type for a return series: the hand-over from preprocessing to modelling.

A :class:`ReturnSeries` is what the statistical and volatility layers consume.  It
knows nothing about where the prices came from (no provider, no ``MarketData``),
only what the numbers *are*.

Conventions
-----------

* **Definition.** Log returns, :math:`R_t = \\ln(P_t / P_{t-1})`, of the
  authoritative price series.  The return dated ``t`` is earned over the interval
  ending at observation ``t``; the first price therefore has no return and is not
  represented (its date is :attr:`ReturnSeries.first_price_date`).
* **Scale.** The series is *stored once*, as a **decimal** log return
  (``0.0123`` is 1.23%), which is the canonical economic quantity.  There is
  deliberately no scale-less accessor.  Every view names its unit:

  - :attr:`ReturnSeries.decimal` (stored; series name ``log_return_decimal``),
  - :attr:`ReturnSeries.percent` (computed on access, never stored; series name
    ``log_return_percent``), the representation volatility models built on the
    ``arch`` package should be fitted on, because they are tuned for data whose
    standard deviation is of order 1, not 0.01.

  Series names carry the unit into any frame or plot built from them, and
  :attr:`ReturnSeries.percent` is exactly ``decimal * 100``, so a factor-100 mix-up
  shows up as a wrong name or a wrong number, never as a silent default.
* **Gaps.** Returns are never dropped for spanning a gap; they are flagged in
  :attr:`ReturnSeries.spans_gap` so downstream diagnostics can decide.
* **Provisional bar.** Whether the latest observation may still have been forming
  when the prices were downloaded is recorded, together with whether it was
  excluded.  See :mod:`src.preprocessing.returns` for the rule.

Treat instances as immutable: the dataclass is frozen and validated on
construction, but the pandas objects it holds are not copied on access.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from src.exceptions import InvalidReturnSeriesError

RETURN_DEFINITION = "log: R_t = ln(P_t / P_(t-1))"
GAP_FLAG_NAME = "spans_gap"


class ReturnScale(str, Enum):
    """Unit in which a return series is expressed."""

    DECIMAL = "decimal"  # 0.0123 == 1.23 %
    PERCENT = "percent"  # 1.23 == 1.23 %

    @property
    def factor(self) -> float:
        """Multiplier that converts a decimal return into this scale."""
        return 100.0 if self is ReturnScale.PERCENT else 1.0

    @property
    def series_name(self) -> str:
        """Name given to series expressed in this scale, so the unit travels with the data."""
        return f"log_return_{self.value}"


class PriceBasis(str, Enum):
    """Which corporate-action adjustment the underlying prices carry."""

    ADJUSTED = "adjusted"  # adjusted for splits and dividends (total-return basis)
    SPLIT_ADJUSTED = "split_adjusted"  # adjusted for splits only


STORED_SCALE = ReturnScale.DECIMAL
MODELING_SCALE = ReturnScale.PERCENT


@dataclass(frozen=True, eq=False)
class ReturnSeries:
    """Validated log returns plus everything needed to audit how they were produced.

    Attributes:
        ticker: Symbol of the asset the returns belong to.
        frequency: Observation frequency of the underlying prices (for example ``1d``).
        price_basis: Corporate-action adjustment of the underlying prices.
        decimal: Log returns as decimals, indexed by the date each return is realised.
            Strictly increasing, unique, finite, float64.
        spans_gap: Boolean flags aligned with ``decimal``: ``True`` where the return
            spans one or more missing observations (see the returns module for the
            exact rule).  Flagged returns are kept, not removed.
        first_price_date: Date of the first price, which has no return.
        gap_tolerance_weekdays: The gap rule's tolerance the flags were computed with.
        provisional_bar: Date of the latest price bar if it was potentially still
            forming when downloaded, else ``None``.
        provisional_bar_excluded: Whether that bar was excluded from the series.
        ingestion_rows_dropped: Rows removed for lacking a close before preprocessing.
        source: Name of the provider that supplied the prices (informational).
        fetched_at: When the prices were downloaded (timezone-aware, UTC).
    """

    ticker: str
    frequency: str
    price_basis: PriceBasis
    decimal: pd.Series
    spans_gap: pd.Series
    first_price_date: date
    gap_tolerance_weekdays: int
    provisional_bar: date | None
    provisional_bar_excluded: bool
    ingestion_rows_dropped: int
    source: str
    fetched_at: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.ticker, str) or not self.ticker:
            raise InvalidReturnSeriesError("ReturnSeries.ticker must be a non-empty string.")
        if not isinstance(self.frequency, str) or not self.frequency:
            raise InvalidReturnSeriesError("ReturnSeries.frequency must be a non-empty string.")
        if not isinstance(self.price_basis, PriceBasis):
            raise InvalidReturnSeriesError("ReturnSeries.price_basis must be a PriceBasis.")
        if self.fetched_at.tzinfo is None or self.fetched_at.utcoffset() != timedelta(0):
            raise InvalidReturnSeriesError("ReturnSeries.fetched_at must be timezone-aware UTC.")
        for name in ("gap_tolerance_weekdays", "ingestion_rows_dropped"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise InvalidReturnSeriesError(f"ReturnSeries.{name} must be an integer >= 0.")

        decimal = self._checked_returns()
        flags = self._checked_flags(decimal)
        object.__setattr__(self, "decimal", decimal.rename(STORED_SCALE.series_name))
        object.__setattr__(self, "spans_gap", flags.rename(GAP_FLAG_NAME))

        first_return = pd.Timestamp(decimal.index[0]).date()
        if self.first_price_date >= first_return:
            raise InvalidReturnSeriesError(
                f"ReturnSeries.first_price_date ({self.first_price_date}) must precede the "
                f"first return date ({first_return})."
            )
        self._check_provisional_consistency()

    def _checked_returns(self) -> pd.Series:
        returns, ticker = self.decimal, self.ticker
        if not isinstance(returns, pd.Series) or returns.dtype != np.float64:
            raise InvalidReturnSeriesError(f"Returns for {ticker} must be a float64 Series.")
        index = returns.index
        if not isinstance(index, pd.DatetimeIndex) or index.tz is not None or index.hasnans:
            raise InvalidReturnSeriesError(
                f"Returns for {ticker} need a timezone-naive DatetimeIndex without NaT."
            )
        if len(returns) == 0:
            raise InvalidReturnSeriesError(f"Returns for {ticker} are empty.")
        if not (index.is_unique and index.is_monotonic_increasing):
            raise InvalidReturnSeriesError(
                f"Return dates for {ticker} must be unique and strictly increasing."
            )
        if not np.isfinite(returns.to_numpy()).all():
            raise InvalidReturnSeriesError(f"Returns for {ticker} contain non-finite values.")
        return returns

    def _checked_flags(self, returns: pd.Series) -> pd.Series:
        flags = self.spans_gap
        if not isinstance(flags, pd.Series) or flags.dtype != np.bool_:
            raise InvalidReturnSeriesError(f"Gap flags for {self.ticker} must be a bool Series.")
        if not flags.index.equals(returns.index):
            raise InvalidReturnSeriesError(
                f"Gap flags for {self.ticker} must be aligned with the return dates."
            )
        return flags

    def _check_provisional_consistency(self) -> None:
        if self.provisional_bar is None:
            if self.provisional_bar_excluded:
                raise InvalidReturnSeriesError(
                    "provisional_bar_excluded is set but no provisional bar was recorded."
                )
        elif not self.provisional_bar_excluded and self.provisional_bar != self.last_return_date:
            raise InvalidReturnSeriesError(
                f"An included provisional bar ({self.provisional_bar}) must be the last "
                f"observation ({self.last_return_date})."
            )

    # --- scale-explicit views ---------------------------------------------------------------

    @property
    def scale(self) -> ReturnScale:
        """Scale of the stored series (always :data:`STORED_SCALE`, decimal)."""
        return STORED_SCALE

    def scaled(self, scale: ReturnScale) -> pd.Series:
        """Return the log returns expressed in ``scale``, named after that scale.

        The decimal scale returns the stored series itself; any other scale is
        computed on the fly from it and is not retained.
        """
        if scale is STORED_SCALE:
            return self.decimal
        return (self.decimal * scale.factor).rename(scale.series_name)

    @property
    def percent(self) -> pd.Series:
        """Log returns in percent (``decimal * 100``); the input for ``arch`` models."""
        return self.scaled(ReturnScale.PERCENT)

    # --- audit metadata -----------------------------------------------------------------------

    @property
    def definition(self) -> str:
        return RETURN_DEFINITION

    @property
    def return_observations(self) -> int:
        return len(self.decimal)

    @property
    def price_observations(self) -> int:
        """Prices used: one more than the number of returns (the first has none)."""
        return len(self.decimal) + 1

    @property
    def first_return_date(self) -> date:
        return pd.Timestamp(self.decimal.index[0]).date()

    @property
    def last_return_date(self) -> date:
        return pd.Timestamp(self.decimal.index[-1]).date()

    @property
    def flagged_gaps(self) -> int:
        """Number of returns that span a gap."""
        return int(self.spans_gap.sum())

    @property
    def excluded_observations(self) -> int:
        """Price observations removed by preprocessing (the provisional bar, if excluded)."""
        return int(self.provisional_bar_excluded)

    @property
    def last_return_is_provisional(self) -> bool:
        """Whether the final return rests on a bar that may still have been forming."""
        return self.provisional_bar is not None and not self.provisional_bar_excluded

    def metadata(self) -> dict[str, Any]:
        """Return a plain, JSON-friendly audit record of how this series was produced."""
        provisional = None if self.provisional_bar is None else self.provisional_bar.isoformat()
        return {
            "ticker": self.ticker,
            "frequency": self.frequency,
            "price_basis": self.price_basis.value,
            "return_definition": self.definition,
            "stored_scale": STORED_SCALE.value,
            "modeling_scale": MODELING_SCALE.value,
            "price_observations": self.price_observations,
            "return_observations": self.return_observations,
            "first_price_date": self.first_price_date.isoformat(),
            "first_return_date": self.first_return_date.isoformat(),
            "last_return_date": self.last_return_date.isoformat(),
            "flagged_gaps": self.flagged_gaps,
            "gap_tolerance_weekdays": self.gap_tolerance_weekdays,
            "ingestion_rows_dropped": self.ingestion_rows_dropped,
            "excluded_observations": self.excluded_observations,
            "provisional_bar": provisional,
            "provisional_bar_excluded": self.provisional_bar_excluded,
            "last_return_is_provisional": self.last_return_is_provisional,
            "source": self.source,
            "fetched_at": self.fetched_at.isoformat(),
        }
