"""Market data ingestion.

Responsibility: obtain historical price data from an external source, validate
it, and hand it to the rest of the system in one well-defined form.  Contains no
statistical or modelling logic.

Public surface (see the module docstrings for the full contracts):

* :class:`MarketDataRequest` / :class:`MarketData`: what is asked for and what is
  delivered (:mod:`src.data.market_data`).
* :class:`MarketDataProvider`: the contract every data source implements
  (:mod:`src.data.provider`); :mod:`src.data.yahoo` is the only module that
  touches ``yfinance``.
* :class:`MarketDataLoader`: provider + validation + cache (:mod:`src.data.loader`).
* :func:`src.data.factory.create_market_data_loader`: wires it all from settings.
"""

from src.data.cache import MarketDataCache
from src.data.loader import MarketDataLoader
from src.data.market_data import (
    INDEX_NAME,
    PRICE_COLUMN,
    STANDARD_COLUMNS,
    MarketData,
    MarketDataRequest,
)
from src.data.provider import MarketDataProvider

__all__ = [
    "INDEX_NAME",
    "PRICE_COLUMN",
    "STANDARD_COLUMNS",
    "MarketData",
    "MarketDataCache",
    "MarketDataLoader",
    "MarketDataProvider",
    "MarketDataRequest",
]
