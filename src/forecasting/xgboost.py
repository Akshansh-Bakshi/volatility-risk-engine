"""Leakage-safe recursive XGBoost forecasting for adjusted close prices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.exceptions import ForecastingDataError, ForecastingError

LAGS = (1, 2, 5, 10, 20)
ROLLING_WINDOWS = (5, 20)
RANDOM_SEED = 42
DEFAULT_CONFIG: dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
    "random_state": RANDOM_SEED,
    "n_jobs": 1,
    "tree_method": "hist",
    "verbosity": 0,
}


def _validate_prices(prices: pd.Series) -> pd.Series:
    if not isinstance(prices, pd.Series) or not isinstance(prices.index, pd.DatetimeIndex):
        raise ForecastingDataError("prices must be a Series with a DatetimeIndex.")
    if prices.empty or prices.index.hasnans or prices.index.has_duplicates:
        raise ForecastingDataError("prices must have non-empty, unique, non-missing dates.")
    values = pd.to_numeric(prices, errors="coerce").to_numpy(dtype="float64")
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ForecastingDataError("prices must be finite, non-missing, and strictly positive.")
    return pd.Series(values, index=prices.index, name=prices.name or "close").sort_index()


def build_price_features(prices: pd.Series) -> pd.DataFrame:
    """Build one-step features for each date using observations strictly before it."""
    series = _validate_prices(prices)
    past = series.shift(1)
    frame = pd.DataFrame(index=series.index)
    for lag in LAGS:
        frame[f"lag_{lag}"] = series.shift(lag)
    for window in ROLLING_WINDOWS:
        rolling = past.rolling(window=window, min_periods=window)
        frame[f"rolling_mean_{window}"] = rolling.mean()
        frame[f"rolling_std_{window}"] = rolling.std(ddof=1)
    return frame


def build_supervised_price_data(prices: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Return aligned features and price-level targets, dropping warm-up rows only."""
    series = _validate_prices(prices)
    features = build_price_features(series)
    valid = features.notna().all(axis=1)
    return features.loc[valid], series.loc[valid].rename("target")


def _features_from_history(history: list[float]) -> np.ndarray:
    if len(history) < max(LAGS):
        raise ForecastingDataError(f"At least {max(LAGS)} prior prices are required to forecast.")
    values: dict[str, float] = {}
    for lag in LAGS:
        values[f"lag_{lag}"] = float(history[-lag])
    for window in ROLLING_WINDOWS:
        past = np.asarray(history[-window:], dtype="float64")
        values[f"rolling_mean_{window}"] = float(past.mean())
        values[f"rolling_std_{window}"] = float(past.std(ddof=1))
    return np.asarray([values[name] for name in _feature_columns()], dtype="float64")


def _feature_columns() -> list[str]:
    return [
        *(f"lag_{lag}" for lag in LAGS),
        *(name for window in ROLLING_WINDOWS for name in (f"rolling_mean_{window}", f"rolling_std_{window}")),
    ]


@dataclass
class XGBoostPriceForecaster:
    """Train a reproducible XGBoost model and forecast recursively from train history."""

    config: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        self.config = dict(DEFAULT_CONFIG if self.config is None else self.config)
        self._model: Any = None
        self._train: pd.Series | None = None
        self._xgboost_version: str | None = None

    @property
    def xgboost_version(self) -> str:
        if self._xgboost_version is None:
            raise ForecastingError("XGBoost version is available after fitting.")
        return self._xgboost_version

    def fit(self, train_prices: pd.Series) -> "XGBoostPriceForecaster":
        train = _validate_prices(train_prices)
        features, target = build_supervised_price_data(train)
        if len(target) < 30:
            raise ForecastingDataError(
                f"Insufficient history: need at least {max(LAGS) + 30} prices; got {len(train)}."
            )
        try:
            import xgboost
        except ImportError as exc:
            raise ForecastingError(
                "XGBoost is required for Stage 12A; install dependencies from requirements.txt."
            ) from exc
        self._model = xgboost.XGBRegressor(**self.config)
        self._model.fit(features, target)
        self._train = train
        self._xgboost_version = xgboost.__version__
        return self

    def forecast(self, forecast_index: pd.DatetimeIndex) -> pd.Series:
        if self._model is None or self._train is None:
            raise ForecastingError("Fit the XGBoost forecaster before requesting a forecast.")
        if (
            not isinstance(forecast_index, pd.DatetimeIndex)
            or forecast_index.empty
            or forecast_index.hasnans
            or forecast_index.has_duplicates
            or not forecast_index.is_monotonic_increasing
            or forecast_index[0] <= self._train.index[-1]
        ):
            raise ForecastingDataError("forecast_index must contain unique future dates in order.")
        history = self._train.to_list()
        predictions: list[float] = []
        for _ in forecast_index:
            row = _features_from_history(history).reshape(1, -1)
            prediction = float(self._model.predict(row)[0])
            if not np.isfinite(prediction):
                raise ForecastingError("XGBoost produced a non-finite forecast.")
            predictions.append(prediction)
            history.append(prediction)
        return pd.Series(predictions, index=forecast_index.copy(), name="xgboost_forecast")
