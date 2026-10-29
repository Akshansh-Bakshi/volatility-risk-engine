"""Run the Stage 12A leakage-safe XGBoost price forecast experiment."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import json
import time
from dataclasses import replace
from datetime import datetime, timezone

import pandas as pd

from src.config import get_settings
from src.data.factory import create_market_data_loader
from src.data.cache import CACHE_SUBDIRECTORY
from src.data.lineage import build_data_lineage
from src.data.market_data import MarketData, MarketDataRequest
from src.forecasting.arima import ARIMAForecaster
from src.forecasting.evaluation import evaluate_forecast
from src.forecasting.holtwinters import HoltWintersForecaster
from src.forecasting.result import ForecastResult
from src.forecasting.sarima import SARIMAForecaster
from src.forecasting.split import make_split
from src.forecasting.xgboost import XGBoostPriceForecaster
from src.utils.artifacts import write_artifact_bundles

OUTPUT_DIR = Path("results/advanced_forecasting/nifty50")
TEST_HORIZON = 60
logger = logging.getLogger(__name__)


def run_experiment(
    market_data: MarketData,
    output_dir: Path = OUTPUT_DIR,
    *,
    test_horizon: int = TEST_HORIZON,
    xgboost_config: dict[str, Any] | None = None,
    include_provisional_bar: bool | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fit all price baselines on one common split, evaluate, and save artifacts."""
    started = time.perf_counter()
    settings = get_settings()
    include_provisional = (
        settings.preprocessing.include_provisional_bar
        if include_provisional_bar is None
        else include_provisional_bar
    )
    if not isinstance(include_provisional, bool):
        raise ValueError("include_provisional_bar must be a boolean or None.")
    provisional_date = market_data.last_date if market_data.last_date >= market_data.fetched_at.date() else None
    provisional_excluded = provisional_date is not None and not include_provisional
    model_data = (
        replace(market_data, prices=market_data.prices.iloc[:-1].copy())
        if provisional_excluded
        else market_data
    )
    if provisional_excluded:
        logger.warning(
            "Excluded potentially provisional latest bar %s for %s before Stage 12A model fitting.",
            provisional_date, market_data.ticker,
        )
    split = make_split(model_data, test_horizon=test_horizon)
    models = [
        ARIMAForecaster(order=(1, 1, 0), ticker=model_data.ticker),
        SARIMAForecaster(order=(1, 1, 0), seasonal_order=(1, 0, 0, 5), ticker=model_data.ticker),
        HoltWintersForecaster(trend="add", seasonal=None, ticker=model_data.ticker),
    ]
    forecasts: dict[str, pd.Series] = {"actual": split.test_series.rename("actual")}
    metric_rows: list[dict[str, Any]] = []
    for model in models:
        result = model.fit(split.train_series).forecast(test_horizon, forecast_index=split.test_series.index)
        series = result.forecast_series().rename(model.model_name)
        forecasts[model.model_name] = series
        metric_rows.append(evaluate_forecast(split.test_series, result).to_dict())

    xgb = XGBoostPriceForecaster(config=xgboost_config)
    xgb.fit(split.train_series)
    xgb_series = xgb.forecast(split.test_series.index).rename("XGBoost recursive")
    forecasts["XGBoost recursive"] = xgb_series
    xgb_result = ForecastResult(
        model_name="XGBoost recursive",
        model_type="xgboost",
        config=dict(xgb.config or {}),
        ticker=model_data.ticker,
        series_description=split.series_description,
        frequency=split.frequency,
        train_start=split.train_start,
        train_end=split.train_end,
        n_train=split.n_train,
        forecast_start=split.test_start,
        forecast_end=split.test_end,
        n_forecast=test_horizon,
        forecast_values={pd.Timestamp(date).date().isoformat(): float(value) for date, value in xgb_series.items()},
        fitted=True,
    )
    metric_rows.append(evaluate_forecast(split.test_series, xgb_result).to_dict())

    forecast_frame = pd.concat(forecasts.values(), axis=1)
    forecast_frame.index.name = "date"
    metrics = pd.DataFrame(metric_rows)
    metadata = {
        "artifact_schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "Stage 12A recursive XGBoost adjusted-close forecasting",
        "target": {
            "name": "adjusted close price",
            "definition": "MarketData.price (close), one-session-ahead supervised target; multi-step test forecast is recursive",
            "units": "index points or asset currency units",
        },
        "data": {
            "source": market_data.source,
            "ticker": market_data.ticker,
            "adjusted_prices": market_data.request.adjust_prices,
            "requested_start": market_data.request.start.isoformat(),
            "requested_end": market_data.request.end.isoformat() if market_data.request.end else "latest available",
            "retrieved_at_utc": market_data.fetched_at.isoformat(),
            "actual_start": market_data.first_date.isoformat(),
            "actual_end": market_data.last_date.isoformat(),
            "observations": market_data.observations,
            "rows_dropped_by_loader": market_data.rows_dropped,
            "lineage": build_data_lineage(
                market_data,
                cache_directory=settings.data.data_dir / CACHE_SUBDIRECTORY,
                provisional_handling={
                    "date": provisional_date.isoformat() if provisional_date else None,
                    "excluded": provisional_excluded,
                    "included": provisional_date is not None and include_provisional,
                    "include_provisional_bar_setting": include_provisional,
                },
                ingestion_configuration={
                    "cache_enabled": settings.data.use_cache,
                    "cache_max_age_hours": settings.data.cache_max_age_hours,
                    "provider_timeout_seconds": settings.data.provider_timeout_seconds,
                    "provider_retries": settings.data.provider_retries,
                    "minimum_observations": settings.data.min_observations,
                },
            ),
        },
        "split": split.info(),
        "evaluation": {"target": "adjusted close price", "horizon_sessions": test_horizon, "forecast_origin": split.train_end.isoformat(), "start": split.test_start.isoformat(), "end": split.test_end.isoformat(), "observations": test_horizon, "metric_units": "price units; MAPE percent"},
        "features": {
            "lags": [1, 2, 5, 10, 20],
            "past_only_rolling_windows": [5, 20],
            "rolling_std_ddof": 1,
            "calendar_features": "none",
            "warmup_rows_dropped": 20,
        },
        "models": {
            "xgboost": {"configuration": xgb.config, "version": xgb.xgboost_version, "forecast_strategy": "recursive; each prediction is appended to lag history"},
            "baselines": [
                {"name": "ARIMA(1,1,0)", "configuration": {"order": [1, 1, 0]}},
                {"name": "SARIMA(1,1,0)(1,0,0,5)", "configuration": {"order": [1, 1, 0], "seasonal_order": [1, 0, 0, 5]}},
                {"name": "HoltWinters(trend=add)", "configuration": {"trend": "add", "seasonal": None, "damped_trend": False}},
            ],
        },
        "evaluation_metrics": {"definitions": {"rmse": "sqrt(mean((actual - forecast)^2))", "mae": "mean(abs(actual - forecast))", "mape": "100 * mean(abs(error / actual)); exact zero actuals excluded by shared evaluator"}, "caveat": "MAPE is scale-sensitive and unsuitable when actuals are zero or near zero; RMSE/MAE are primary."},
        "test_data_policy": "The final holdout is not used for feature fitting, transformations, or hyperparameter selection. XGBoost test predictions are recursive.",
        "duration_seconds": time.perf_counter() - started,
    }
    write_artifact_bundles({output_dir: {
        "daily_forecasts.csv": lambda path: forecast_frame.to_csv(path, index_label="date"),
        "evaluation_metrics.csv": lambda path: metrics.to_csv(path, index=False),
        "metadata.json": lambda path: path.write_text(json.dumps(metadata, indent=2), encoding="utf-8"),
    }})
    return forecast_frame, metrics, metadata


def main() -> None:
    settings = get_settings()
    request = MarketDataRequest(
        ticker="^NSEI", start=settings.data.start_date, end=None,
        interval="1d", adjust_prices=True,
    )
    data = create_market_data_loader(settings).load(request)
    forecasts, metrics, metadata = run_experiment(
        data, include_provisional_bar=settings.preprocessing.include_provisional_bar
    )
    print(f"Saved {len(forecasts)} common-date forecasts to {OUTPUT_DIR}")
    print(metrics.loc[:, ["model_name", "rmse", "mae", "mape"]].to_string(index=False))
    print(f"Evaluation: {metadata['evaluation']['start']} to {metadata['evaluation']['end']}")


if __name__ == "__main__":
    main()
