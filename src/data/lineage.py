"""Shared, JSON-safe market-data provenance and fingerprint helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.cache import MarketDataCache
from src.data.market_data import MarketData
from src.data.validation import audit_price_quality


def build_data_lineage(
    market_data: MarketData,
    *,
    cache_directory: Path | None = None,
    source_state: str | None = None,
    fallback_reason: str | None = None,
    provisional_handling: dict[str, Any] | None = None,
    ingestion_configuration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe the exact price input, its source, validation notes and stable cache key."""
    if fallback_reason is None:
        fallback_reason = market_data.fallback_reason
    if source_state is None:
        source_state = "fallback" if fallback_reason else ("cached" if market_data.from_cache else "fresh_provider")
    if source_state not in {"fresh_provider", "cached", "fallback"}:
        raise ValueError("source_state must be fresh_provider, cached, or fallback.")

    digest = hashlib.sha256()
    digest.update(repr((
        market_data.request.ticker,
        market_data.request.interval,
        market_data.request.start.isoformat(),
        market_data.request.end.isoformat() if market_data.request.end else None,
        market_data.request.adjust_prices,
        tuple(market_data.prices.columns),
    )).encode("utf-8"))
    hashed = pd.util.hash_pandas_object(market_data.prices, index=True).to_numpy(dtype="uint64")
    digest.update(hashed.tobytes())

    cache_id = None
    if cache_directory is not None:
        cache_id = MarketDataCache(cache_directory).path_for(market_data.request).name
    warnings = list(audit_price_quality(market_data.prices, market_data.request))
    if market_data.rows_dropped:
        warnings.append(
            f"{market_data.rows_dropped} provider row(s) without a close were removed and not filled."
        )
    return {
        "schema_version": 1,
        "ticker": market_data.ticker,
        "provider": market_data.source,
        "source_state": source_state,
        "retrieved_at_utc": market_data.fetched_at.isoformat(),
        "requested_range": market_data.request.range_label,
        "actual_start_date": market_data.first_date.isoformat(),
        "actual_last_observation_date": market_data.last_date.isoformat(),
        "observation_count": market_data.observations,
        "adjusted_prices": market_data.request.adjust_prices,
        "cache_identifier": cache_id,
        "data_sha256": digest.hexdigest(),
        "validation_warnings": warnings,
        "dropped_close_dates": market_data.dropped_dates.strftime("%Y-%m-%d").tolist(),
        "provisional_bar_handling": provisional_handling,
        "ingestion_configuration": ingestion_configuration,
        "fallback_reason": fallback_reason,
    }
