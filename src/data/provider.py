"""The contract between the data layer and any market data source.

Everything above the provider (validation, caching, the loader, and later the
statistical and modelling layers) depends on this protocol, never on a concrete
vendor library.  A new source is added by implementing :class:`MarketDataProvider`;
nothing else changes.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd

from src.data.market_data import MarketDataRequest


@runtime_checkable
class MarketDataProvider(Protocol):
    """A source of daily price history for a single ticker.

    Attributes:
        name: Short, stable identifier recorded as the ``source`` of the data.
    """

    name: str

    def fetch(self, request: MarketDataRequest) -> pd.DataFrame:
        """Download the history described by ``request`` and normalise its *structure*.

        The returned frame must already use the project schema (see
        :mod:`src.data.market_data`): the :data:`~src.data.market_data.STANDARD_COLUMNS`
        columns, a timezone-naive session-date ``DatetimeIndex``, and no
        vendor-specific artefacts such as MultiIndex columns.  A provider must not
        sort, de-duplicate, fill or otherwise repair the rows it received; the
        shared validation step judges them.

        A provider that received no rows returns an empty frame.

        Raises:
            InvalidTickerError: The source does not recognise the symbol.
            EmptyDataError: The source reports no data for the requested range.
            DataFetchError: The source could not be reached or failed.
            DataValidationError: The source's response cannot be interpreted.
        """
        ...
