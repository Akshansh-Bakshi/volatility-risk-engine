"""Focused Stage 12A tests for alignment, leakage, recursion, and saved artifacts."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from app import load_forecast_dashboard_artifacts
from scripts.run_stage12a_xgboost import run_experiment
from src.exceptions import ForecastingDataError
from src.forecasting.evaluation import evaluate_forecast
from src.forecasting.result import ForecastResult
from src.forecasting.split import make_split
from src.forecasting.xgboost import (
    XGBoostPriceForecaster,
    build_price_features,
    build_supervised_price_data,
)
from tests.data_fakes import make_market_data


def _prices(count: int = 240) -> pd.Series:
    index = pd.bdate_range("2023-01-02", periods=count, name="date")
    values = 100 + np.arange(count) * 0.2 + np.sin(np.arange(count) / 6)
    return pd.Series(values, index=index, name="close")


def test_features_and_target_are_aligned_one_step_ahead() -> None:
    prices = _prices()
    features, target = build_supervised_price_data(prices)
    assert features.index.equals(target.index)
    assert features.index[0] == prices.index[20]
    assert target.iloc[0] == prices.iloc[20]
    assert features.iloc[0]["lag_1"] == prices.iloc[19]
    assert features.iloc[0]["rolling_mean_5"] == pytest.approx(prices.iloc[15:20].mean())


def test_feature_engineering_sorts_valid_observations_chronologically() -> None:
    prices = _prices().iloc[::-1]
    features, target = build_supervised_price_data(prices)
    assert features.index.is_monotonic_increasing
    assert target.index.is_monotonic_increasing


def test_features_do_not_change_when_future_prices_change() -> None:
    prices = _prices()
    original = build_price_features(prices)
    changed = prices.copy()
    changed.iloc[130:] += 5000
    revised = build_price_features(changed)
    pd.testing.assert_frame_equal(original.iloc[:130], revised.iloc[:130])


def test_chronological_split_preserves_exact_final_dates() -> None:
    prices = _prices()
    data = make_market_data(pd.DataFrame({"close": prices}, index=prices.index))
    split = make_split(data, test_horizon=60)
    assert split.no_leakage()
    assert split.test_series.index.equals(prices.index[-60:])
    assert split.train_series.index[-1] < split.test_series.index[0]


def test_recursive_forecast_feeds_previous_prediction_into_next_step() -> None:
    model = XGBoostPriceForecaster()
    model._train = _prices(40)

    class SequenceModel:
        calls: list[np.ndarray] = []

        def predict(self, row: np.ndarray) -> np.ndarray:
            self.calls.append(row.copy())
            return np.array([900.0 + len(self.calls)])

    model._model = SequenceModel()
    future = pd.bdate_range(model._train.index[-1] + pd.Timedelta(days=1), periods=3)
    forecast = model.forecast(future)
    assert forecast.tolist() == [901.0, 902.0, 903.0]
    assert model._model.calls[1][0, 0] == 901.0
    assert model._model.calls[2][0, 0] == 902.0


@pytest.mark.parametrize("bad", [
    pd.Series([1.0, np.nan], index=pd.date_range("2020-01-01", periods=2)),
    pd.Series([1.0, 0.0], index=pd.date_range("2020-01-01", periods=2)),
])
def test_missing_or_nonpositive_prices_fail_explicitly(bad: pd.Series) -> None:
    with pytest.raises(ForecastingDataError):
        build_price_features(bad)


def test_insufficient_history_fails_with_actionable_message() -> None:
    with pytest.raises(ForecastingDataError, match="Insufficient history"):
        XGBoostPriceForecaster().fit(_prices(40))


def test_shared_metric_calculation_and_near_zero_mape_behavior() -> None:
    actual = pd.Series([1.0, 2.0], index=pd.date_range("2024-01-01", periods=2))
    result = ForecastResult(
        model_name="test", model_type="xgboost", config={}, ticker="TEST",
        series_description="adjusted close price", frequency="1d",
        train_start=pd.Timestamp("2023-01-01").date(), train_end=pd.Timestamp("2023-12-29").date(),
        n_train=200, forecast_start=actual.index[0].date(), forecast_end=actual.index[-1].date(),
        n_forecast=2, forecast_values={str(actual.index[i].date()): value for i, value in enumerate([2.0, 2.0])},
        fitted=True,
    )
    metrics = evaluate_forecast(actual, result)
    assert metrics.rmse == pytest.approx(np.sqrt(0.5))
    assert metrics.mae == pytest.approx(0.5)
    assert metrics.mape == pytest.approx(50.0)


def test_run_persists_common_date_forecasts_metrics_and_dashboard_schema(tmp_path) -> None:
    prices = _prices(220)
    data = make_market_data(
        pd.DataFrame({"close": prices}, index=prices.index),
        ticker="^NSEI",
        fetched_at=datetime.now(timezone.utc),
    )
    config = {"n_estimators": 8, "max_depth": 2, "learning_rate": 0.1, "objective": "reg:squarederror", "random_state": 42, "n_jobs": 1, "tree_method": "hist", "verbosity": 0}
    forecasts, metrics, metadata = run_experiment(data, tmp_path, test_horizon=60, xgboost_config=config)
    assert forecasts.index.equals(prices.index[-60:])
    assert set(("actual", "XGBoost recursive", "ARIMA(1,1,0)", "SARIMA(1,1,0)(1,0,0,5)", "HoltWinters(trend=add)")).issubset(forecasts.columns)
    assert set(("rmse", "mae", "mape")).issubset(metrics.columns)
    assert metadata["evaluation"]["observations"] == len(forecasts) == 60
    assert metadata["models"]["xgboost"]["configuration"] == config
    assert json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))["target"]["name"] == "adjusted close price"
    loaded = load_forecast_dashboard_artifacts(tmp_path)
    assert loaded.available and loaded.status == "ready"
    assert loaded.forecasts is not None and len(loaded.forecasts) == 60


def test_run_excludes_a_potentially_provisional_final_price_bar(tmp_path) -> None:
    prices = _prices(220)
    data = make_market_data(pd.DataFrame({"close": prices}, index=prices.index), ticker="^NSEI")
    data = replace(
        data,
        fetched_at=pd.Timestamp(data.last_date, tz="UTC").to_pydatetime(),
    )
    config = {"n_estimators": 8, "max_depth": 2, "learning_rate": 0.1, "objective": "reg:squarederror", "random_state": 42, "n_jobs": 1, "tree_method": "hist", "verbosity": 0}

    forecasts, _, metadata = run_experiment(data, tmp_path, test_horizon=60, xgboost_config=config)

    assert data.last_date not in [value.date() for value in forecasts.index]
    assert metadata["data"]["lineage"]["provisional_bar_handling"]["excluded"] is True
    assert metadata["split"]["total_observations"] == len(prices) - 1


def test_dashboard_loader_reports_missing_and_invalid_artifacts(tmp_path) -> None:
    assert load_forecast_dashboard_artifacts(tmp_path).status == "missing"
    (tmp_path / "daily_forecasts.csv").write_text("date,actual\n2024-01-01,1\n", encoding="utf-8")
    (tmp_path / "evaluation_metrics.csv").write_text("model_name,rmse,mae,mape\ntest,1,1,1\n", encoding="utf-8")
    (tmp_path / "metadata.json").write_text("{}", encoding="utf-8")
    assert load_forecast_dashboard_artifacts(tmp_path).status == "incomplete"


def test_xgboost_training_is_reproducible() -> None:
    pytest.importorskip("xgboost")
    config = {"n_estimators": 8, "max_depth": 2, "learning_rate": 0.1, "objective": "reg:squarederror", "random_state": 42, "n_jobs": 1, "tree_method": "hist", "verbosity": 0}
    prices = _prices()
    future = pd.bdate_range(prices.index[-1] + pd.Timedelta(days=1), periods=4)
    first = XGBoostPriceForecaster(config).fit(prices).forecast(future)
    second = XGBoostPriceForecaster(config).fit(prices).forecast(future)
    np.testing.assert_array_equal(first.to_numpy(), second.to_numpy())
