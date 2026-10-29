"""Generate saved Stage 13 rolling risk and stress-scenario artifacts."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import get_settings
from src.data.cache import CACHE_SUBDIRECTORY, MarketDataCache
from src.data.factory import create_market_data_loader
from src.data.market_data import MarketData, MarketDataRequest
from src.data.lineage import build_data_lineage
from src.data.validation import validate_prices
from src.exceptions import DataError, DataFetchError, InvalidTickerError
from src.preprocessing.returns import build_return_series
from src.risk.monitoring import RiskMonitoringConfig, calculate_risk_monitoring
from src.risk.stress import Portfolio, historical_scenarios, hypothetical_scenarios
from src.utils.artifacts import write_artifact_bundles

RISK_OUTPUT_DIR = Path("results/risk_monitoring/nifty50")
STRESS_OUTPUT_DIR = Path("results/stress_testing/nifty50")
DEFAULT_PORTFOLIO_VALUE = 1_000_000.0
logger = logging.getLogger(__name__)


def load_market_data_with_cache_fallback(
    settings: Any,
    request: MarketDataRequest,
) -> tuple[MarketData, bool]:
    """Try the normal freshness-aware loader, then use a validated stale cache on fetch failure."""
    try:
        return create_market_data_loader(settings).load(request), False
    except (DataFetchError, InvalidTickerError) as provider_error:
        if not settings.data.use_cache:
            raise
        cache = MarketDataCache(settings.data.data_dir / CACHE_SUBDIRECTORY)
        entry = cache.read(request)
        if entry is None:
            raise
        try:
            prices, unexpected = validate_prices(
                entry.prices, request, min_observations=settings.data.min_observations
            )
        except DataError as cache_error:
            raise DataFetchError(
                f"Yahoo retrieval failed ({type(provider_error).__name__}: {provider_error}); "
                f"the identity-matched cached entry also failed validation ({type(cache_error).__name__}: {cache_error})."
            ) from provider_error
        if len(unexpected):
            raise DataFetchError(
                f"Yahoo retrieval failed ({type(provider_error).__name__}: {provider_error}); "
                f"cached data for {request.ticker} contains rows without valid close prices."
            ) from provider_error
        logger.warning(
            "Yahoo retrieval failed for %s (%s: %s); using the validated local cache entry fetched at %s.",
            request.ticker, type(provider_error).__name__, provider_error, entry.fetched_at.isoformat(),
        )
        return replace(
            entry,
            prices=prices,
            from_cache=True,
            fallback_reason=f"{type(provider_error).__name__}: {provider_error}",
        ), True


def run_stage13_experiment(
    market_data: MarketData,
    risk_output_dir: Path = RISK_OUTPUT_DIR,
    stress_output_dir: Path = STRESS_OUTPUT_DIR,
    *,
    risk_config: RiskMonitoringConfig | None = None,
    portfolio_value: float = DEFAULT_PORTFOLIO_VALUE,
    portfolio_currency: str = "INR",
    historical_worst_n: int = 5,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, dict[str, Any]]:
    """Calculate and persist Stage 13 monitoring and stress results."""
    started = time.perf_counter()
    settings = get_settings()
    config = risk_config or RiskMonitoringConfig(
        confidence_levels=settings.risk.confidence_levels
    )
    returns = build_return_series(market_data, settings.preprocessing)
    risk_frame, risk_summary = calculate_risk_monitoring(returns, market_data.price, config)

    ticker = market_data.ticker
    portfolio = Portfolio(
        name=f"{ticker} single-asset index proxy",
        starting_value=portfolio_value,
        weights={ticker: 1.0},
        currency=portfolio_currency,
    )
    log_return_frame = returns.decimal.to_frame(name=ticker)
    historical_1d = historical_scenarios(
        portfolio,
        log_return_frame,
        horizon=1,
        worst_n=historical_worst_n,
        gap_flags=returns.spans_gap,
    )
    historical_5d = historical_scenarios(
        portfolio,
        log_return_frame,
        horizon=5,
        worst_n=historical_worst_n,
        gap_flags=returns.spans_gap,
    )
    hypothetical = hypothetical_scenarios(
        portfolio, shocks=(-0.05, -0.10, -0.20), scope="portfolio_wide"
    )
    stress_frame = pd.concat([historical_1d, historical_5d, hypothetical], ignore_index=True)

    data_metadata = {
        "source": market_data.source,
        "ticker": ticker,
        "requested_start": market_data.request.start.isoformat(),
        "requested_end": market_data.request.end.isoformat() if market_data.request.end else "latest available",
        "adjusted_prices": market_data.request.adjust_prices,
        "retrieved_at_utc": market_data.fetched_at.isoformat(),
        "cache_entry_used": market_data.from_cache,
        "lineage": build_data_lineage(
            market_data,
            cache_directory=settings.data.data_dir / CACHE_SUBDIRECTORY,
            provisional_handling={
                "date": returns.provisional_bar.isoformat() if returns.provisional_bar else None,
                "excluded": returns.provisional_bar_excluded,
                "include_provisional_bar_setting": settings.preprocessing.include_provisional_bar,
            },
            ingestion_configuration={
                "cache_enabled": settings.data.use_cache,
                "cache_max_age_hours": settings.data.cache_max_age_hours,
                "provider_timeout_seconds": settings.data.provider_timeout_seconds,
                "provider_retries": settings.data.provider_retries,
                "minimum_observations": settings.data.min_observations,
            },
        ),
        "actual_price_start": market_data.first_date.isoformat(),
        "actual_price_end": market_data.last_date.isoformat(),
        "price_observations": market_data.observations,
        "return_metadata": returns.metadata(),
    }
    run_at = datetime.now(timezone.utc).isoformat()
    risk_metadata: dict[str, Any] = {
        "artifact_schema_version": 1,
        "experiment": "Stage 13A trailing historical risk monitoring",
        "created_at_utc": run_at,
        "data": data_metadata,
        "risk_configuration": {
            "realised_volatility_window": config.realised_volatility_window,
            "historical_var_es_window": config.historical_var_window,
            "breach_summary_window": config.breach_summary_window,
            "confidence_levels": list(config.confidence_levels),
            "annualisation_factor": config.annualisation_factor,
            "quantile_method": config.quantile_method,
            "return_definition": returns.definition,
            "return_input_scale": "decimal log returns",
            "risk_output_scale": "percent log-return points",
            "historical_gaps_retained": True,
            "provisional_bar_excluded": returns.provisional_bar_excluded,
        },
        "summary": risk_summary,
        "duration_seconds": time.perf_counter() - started,
    }
    stress_metadata: dict[str, Any] = {
        "artifact_schema_version": 1,
        "experiment": "Stage 13B historical and hypothetical portfolio stress scenarios",
        "created_at_utc": run_at,
        "data": data_metadata,
        "portfolio": {
            "name": portfolio.name,
            "starting_value": portfolio.starting_value,
            "currency": portfolio.currency,
            "weights": dict(portfolio.weights),
            "scope_note": (
                "Only a single NIFTY 50 index holding is available; this is an index proxy, not a diversified portfolio."
                if ticker == "^NSEI"
                else f"Only a single {ticker} asset holding is supplied; this is not a diversified portfolio."
            ),
            "assumptions": "Fully invested, long-only, buy-and-hold weights; no fees, cash flows, leverage, or rebalancing.",
            "valuation_formula": "stressed_value = starting_value * sum(weight_i * (1 + simple_return_i)); loss = starting_value - stressed_value; asset contribution_i = -starting_value * weight_i * simple_return_i",
        },
        "historical_scenarios": {
            "input": "decimal log returns from the existing ReturnSeries",
            "horizons_sessions": [1, 5],
            "worst_scenarios_per_horizon": historical_worst_n,
            "window_policy": "all overlapping consecutive observation windows; retain the worst losses",
            "compounding": "asset simple return = exp(sum(log returns)) - 1",
            "gap_policy": "gap-spanning observations are included and flagged in each scenario row",
        },
        "hypothetical_scenarios": {
            "shocks_simple_return": [-0.05, -0.10, -0.20],
            "scope": "portfolio-wide; the same shock is applied to every holding (currently one index proxy)",
            "probability_claim": None,
        },
        "duration_seconds": time.perf_counter() - started,
    }

    write_artifact_bundles({
        risk_output_dir: {
            "daily_risk_monitoring.csv": lambda path: risk_frame.to_csv(path, index_label="date"),
            "risk_summary.json": lambda path: path.write_text(json.dumps(risk_summary, indent=2), encoding="utf-8"),
            "metadata.json": lambda path: path.write_text(json.dumps(risk_metadata, indent=2), encoding="utf-8"),
        },
        stress_output_dir: {
            "scenario_results.csv": lambda path: stress_frame.to_csv(path, index=False),
            "metadata.json": lambda path: path.write_text(json.dumps(stress_metadata, indent=2), encoding="utf-8"),
        },
    })
    return risk_frame, risk_metadata, stress_frame, stress_metadata


def main() -> None:
    settings = get_settings()
    request = MarketDataRequest(
        ticker="^NSEI",
        start=settings.data.start_date,
        end=settings.data.end_date,
        interval="1d",
        adjust_prices=True,
    )
    data, cache_fallback = load_market_data_with_cache_fallback(settings, request)
    if cache_fallback:
        print(f"Fresh retrieval failed; using validated local cache from {data.fetched_at.isoformat()}.")
    risk, _, stress, _ = run_stage13_experiment(data)
    print(f"Saved {len(risk)} risk-monitoring observations and {len(stress)} scenarios.")
    print(f"Risk artifacts: {RISK_OUTPUT_DIR}")
    print(f"Stress artifacts: {STRESS_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
