"""Dataset profile: a structured, JSON-serialisable summary of a return series.

:class:`DatasetProfile` is built from a :class:`~src.data.market_data.MarketData`
and the :class:`~src.preprocessing.return_series.ReturnSeries` derived from it.
It answers the EDA question *"what exactly did we load, and is it healthy?"* in
one reproducible, auditable object.

The profile is deliberately a pure data object — it contains no Streamlit, no
plotting, and no side effects.  The dashboard layer can serialise it straight to
JSON for the synopsis snapshot or display its fields without any further logic.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

from src.data.market_data import STANDARD_COLUMNS, MarketData
from src.exceptions import InvalidProfileInputError
from src.logging_config import PACKAGE_LOGGER_NAME
from src.preprocessing.return_series import ReturnSeries

logger = logging.getLogger(f"{PACKAGE_LOGGER_NAME}.statistics.profile")


@dataclass(frozen=True)
class DataQualityFindings:
    """Concise data-quality observations extracted from the price frame.

    Attributes:
        missing_close_rows: Number of rows that had no close price (already
            removed from the series by the loader before preprocessing).
        duplicate_timestamps: Number of duplicate index entries in the
            *raw* price frame (zero once the loader has validated it, but
            recorded for completeness).
        gap_flagged_returns: Number of return observations that span a
            gap (missing session or dropped row).
        gap_tolerance_weekdays: The weekday gap tolerance used when the
            flags were computed.
        provisional_bar: ISO-8601 date string of the last bar if it was
            potentially still forming when downloaded, else ``None``.
        provisional_bar_excluded: Whether that bar was excluded from the
            return series.
        columns_present: Columns found in the validated price frame.
        quality_summary: Brief human-readable verdict.
    """

    missing_close_rows: int
    duplicate_timestamps: int
    gap_flagged_returns: int
    gap_tolerance_weekdays: int
    provisional_bar: str | None
    provisional_bar_excluded: bool
    columns_present: tuple[str, ...]
    quality_summary: str


@dataclass(frozen=True)
class DatasetProfile:
    """A reproducible snapshot of one asset's dataset at the point of EDA.

    Build with :func:`build_dataset_profile`; never construct directly.

    Attributes:
        ticker: Asset symbol.
        source: Name of the data provider.
        frequency: Bar frequency of the underlying prices (e.g. ``"1d"``).
        price_basis: Price adjustment applied (``"adjusted"`` or
            ``"split_adjusted"``).
        first_price_date: Calendar date of the earliest price.
        last_price_date: Calendar date of the most recent price used.
        first_return_date: Calendar date of the first return observation.
        last_return_date: Calendar date of the last return observation.
        price_observations: Total rows in the validated price frame.
        return_observations: Total log-return observations.
        fetched_at: UTC timestamp of when prices were downloaded.
        from_cache: Whether the prices came from the local cache.
        return_definition: Human-readable formula for the returns.
        quality: Data-quality findings sub-record.
    """

    ticker: str
    source: str
    frequency: str
    price_basis: str
    first_price_date: date
    last_price_date: date
    first_return_date: date
    last_return_date: date
    price_observations: int
    return_observations: int
    fetched_at: datetime
    from_cache: bool
    return_definition: str
    quality: DataQualityFindings

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of this profile.

        All ``date`` and ``datetime`` fields are converted to ISO-8601 strings.
        """
        raw = asdict(self)
        return _to_jsonable(raw)

    def to_json(self, *, indent: int = 2) -> str:
        """Serialise to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


def build_dataset_profile(
    market_data: MarketData,
    returns: ReturnSeries,
) -> DatasetProfile:
    """Build a :class:`DatasetProfile` from a loaded dataset and its return series.

    Args:
        market_data: The validated market data delivered by the loader.
        returns: The return series built from ``market_data`` by the
            preprocessing layer.

    Returns:
        A frozen, JSON-serialisable profile of the dataset.

    Raises:
        InvalidProfileInputError: If ``market_data`` and ``returns`` refer to
            different tickers or are otherwise inconsistent.
    """
    _validate_consistency(market_data, returns)

    prices = market_data.prices
    duplicate_count = int(prices.index.duplicated().sum())
    cols_present = tuple(c for c in STANDARD_COLUMNS if c in prices.columns)

    provisional_str = (
        returns.provisional_bar.isoformat() if returns.provisional_bar is not None else None
    )
    quality = DataQualityFindings(
        missing_close_rows=market_data.rows_dropped,
        duplicate_timestamps=duplicate_count,
        gap_flagged_returns=returns.flagged_gaps,
        gap_tolerance_weekdays=returns.gap_tolerance_weekdays,
        provisional_bar=provisional_str,
        provisional_bar_excluded=returns.provisional_bar_excluded,
        columns_present=cols_present,
        quality_summary=_quality_summary(
            missing_close_rows=market_data.rows_dropped,
            duplicate_timestamps=duplicate_count,
            gap_flagged_returns=returns.flagged_gaps,
        ),
    )

    profile = DatasetProfile(
        ticker=returns.ticker,
        source=returns.source,
        frequency=returns.frequency,
        price_basis=returns.price_basis.value,
        first_price_date=returns.first_price_date,
        last_price_date=market_data.last_date,
        first_return_date=returns.first_return_date,
        last_return_date=returns.last_return_date,
        price_observations=market_data.observations,
        return_observations=returns.return_observations,
        fetched_at=returns.fetched_at,
        from_cache=market_data.from_cache,
        return_definition=returns.definition,
        quality=quality,
    )
    logger.info(
        "Built dataset profile: ticker=%s source=%s prices=%d returns=%d "
        "gaps=%d missing_close=%d",
        profile.ticker,
        profile.source,
        profile.price_observations,
        profile.return_observations,
        quality.gap_flagged_returns,
        quality.missing_close_rows,
    )
    return profile


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_consistency(market_data: MarketData, returns: ReturnSeries) -> None:
    """Raise :exc:`~src.exceptions.InvalidProfileInputError` if the pair is inconsistent."""
    if market_data.ticker != returns.ticker:
        raise InvalidProfileInputError(
            f"market_data.ticker ({market_data.ticker!r}) does not match "
            f"returns.ticker ({returns.ticker!r})."
        )
    if market_data.observations < 2:
        raise InvalidProfileInputError(
            f"market_data for {market_data.ticker} must contain at least 2 price rows; "
            f"got {market_data.observations}."
        )


def _quality_summary(
    *,
    missing_close_rows: int,
    duplicate_timestamps: int,
    gap_flagged_returns: int,
) -> str:
    issues: list[str] = []
    if missing_close_rows:
        issues.append(f"{missing_close_rows} row(s) dropped for missing close")
    if duplicate_timestamps:
        issues.append(f"{duplicate_timestamps} duplicate timestamp(s)")
    if gap_flagged_returns:
        issues.append(f"{gap_flagged_returns} return(s) span a gap")
    return "No quality issues detected." if not issues else "Issues: " + "; ".join(issues) + "."


def _to_jsonable(value: Any) -> Any:
    """Recursively convert a nested structure to JSON-safe types."""
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value
