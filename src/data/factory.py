"""Composition root of the data layer: wires the concrete provider and cache.

This is the one place that combines :class:`~src.data.yahoo.YahooFinanceProvider`
with the settings-driven cache.  Application code calls
:func:`create_market_data_loader` and otherwise depends only on the provider
contract.
"""

from __future__ import annotations

from datetime import timedelta

from src.config import Settings, get_settings
from src.data.cache import CACHE_SUBDIRECTORY, MarketDataCache
from src.data.loader import MarketDataLoader
from src.data.provider import MarketDataProvider
from src.data.yahoo import YahooFinanceProvider


def create_market_data_loader(
    settings: Settings | None = None, *, provider: MarketDataProvider | None = None
) -> MarketDataLoader:
    """Build a :class:`MarketDataLoader` from configuration.

    Args:
        settings: Configuration to use; defaults to :func:`~src.config.get_settings`.
        provider: Data source; defaults to Yahoo Finance.
    """
    config = (settings if settings is not None else get_settings()).data
    cache = MarketDataCache(config.data_dir / CACHE_SUBDIRECTORY) if config.use_cache else None
    return MarketDataLoader(
        provider if provider is not None else YahooFinanceProvider(),
        cache=cache,
        min_observations=config.min_observations,
        cache_max_age=timedelta(hours=config.cache_max_age_hours),
    )
