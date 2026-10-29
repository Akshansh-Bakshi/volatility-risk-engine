from __future__ import annotations

from src.data.lineage import build_data_lineage
from src.data.market_data import MarketData
from tests.data_fakes import make_market_data


def test_lineage_fingerprint_is_deterministic_and_cache_key_hides_absolute_path(tmp_path) -> None:
    data = make_market_data()
    first = build_data_lineage(data, cache_directory=tmp_path / "cache" / "market")
    second = build_data_lineage(data, cache_directory=tmp_path / "cache" / "market")

    assert first == second
    assert len(first["data_sha256"]) == 64
    assert first["source_state"] == "fresh_provider"
    assert first["cache_identifier"].endswith(".json")
    assert str(tmp_path) not in first["cache_identifier"]
    assert first["actual_last_observation_date"] == data.last_date.isoformat()


def test_lineage_marks_cached_fallback_and_provisional_policy(tmp_path) -> None:
    data = make_market_data()
    cached = MarketData(
        request=data.request,
        prices=data.prices,
        source=data.source,
        fetched_at=data.fetched_at,
        dropped_dates=data.dropped_dates,
        from_cache=True,
        fallback_reason="DataFetchError: connection refused",
    )
    lineage = build_data_lineage(
        cached,
        cache_directory=tmp_path,
        provisional_handling={"excluded": True},
    )
    assert lineage["source_state"] == "fallback"
    assert lineage["fallback_reason"] == cached.fallback_reason
    assert lineage["provisional_bar_handling"] == {"excluded": True}
