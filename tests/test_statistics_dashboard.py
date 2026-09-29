"""Focused tests for presentation-only historical market summaries."""

from __future__ import annotations

from src.config import PreprocessingConfig
from src.preprocessing.returns import build_return_series
from src.statistics.dashboard import calculate_dashboard_metrics, factual_findings
from src.statistics.descriptive import compute_return_stats
from tests.data_fakes import make_market_data, make_prices


def _dataset(periods: int):
    market_data = make_market_data(make_prices(periods=periods))
    returns = build_return_series(
        market_data,
        PreprocessingConfig(min_returns=2),
    )
    return market_data, returns


def test_metrics_report_historical_price_return_volatility_and_drawdown() -> None:
    market_data, returns = _dataset(80)
    metrics = calculate_dashboard_metrics(market_data, returns)

    assert metrics.latest_price == float(market_data.price.iloc[-1])
    assert metrics.latest_completed_return_pct is not None
    assert metrics.rolling_volatility_20_pct is not None
    assert metrics.trailing_20_session_return_pct is not None
    assert metrics.maximum_drawdown_pct <= 0
    assert metrics.peak_volatility_date is not None
    assert len(metrics.drawdown_series_pct) == len(market_data.prices)


def test_metrics_leave_insufficient_trailing_and_rolling_windows_unavailable() -> None:
    market_data, returns = _dataset(12)
    metrics = calculate_dashboard_metrics(market_data, returns)

    assert metrics.latest_price is not None
    assert metrics.latest_completed_return_pct is not None
    assert metrics.rolling_volatility_20_pct is None
    assert metrics.trailing_20_session_return_pct is None


def test_factual_findings_are_deterministic_for_a_selected_sample() -> None:
    market_data, returns = _dataset(80)
    stats = compute_return_stats(returns)
    metrics = calculate_dashboard_metrics(market_data, returns)

    first = factual_findings(stats, metrics)
    second = factual_findings(stats, metrics)

    assert first == second
    assert 3 <= len(first) <= 5
    assert all(isinstance(finding, str) and finding for finding in first)
    assert "forecast" not in " ".join(first).lower()
