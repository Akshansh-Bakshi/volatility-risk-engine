"""Holt-Winters / Exponential Smoothing forecaster.

**Model**: Exponential Smoothing (Holt-Winters) via statsmodels
:class:`~statsmodels.tsa.holtwinters.ExponentialSmoothing`.

**Target series**: The adjusted close-price level (``MarketData.price``),
*not* log returns.

Trend and seasonality configuration
-------------------------------------
The trend and seasonal components are fully configurable:

- ``trend``: ``"add"`` (additive trend), ``"mul"`` (multiplicative),
  or ``None`` (no trend / simple exponential smoothing).
- ``damped_trend``: Whether to dampen the trend component.
- ``seasonal``: ``"add"``, ``"mul"``, or ``None`` (no seasonality).
- ``seasonal_periods``: Integer period for the seasonal component.  Only
  relevant when ``seasonal`` is not ``None``; requires at least two full
  seasonal cycles in the training data.

**Important**: Do not add a seasonal component blindly.  Daily equity price
series rarely exhibit a strong intra-year seasonal cycle.  Consider using
``trend="add"`` and ``seasonal=None`` as a robust starting configuration.

Usage::

    from src.forecasting.holtwinters import HoltWintersForecaster

    # Simple Holt (linear trend, no seasonality)
    model = HoltWintersForecaster(trend="add", seasonal=None)
    model.fit(split.train_series)
    result = model.forecast(horizon=split.n_test)
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Literal

import pandas as pd

from src.exceptions import (
    ForecastingDataError,
    InvalidModelConfigError,
    ModelNotFittedError,
)
from src.forecasting.arima import _validate_series
from src.forecasting.result import ForecastResult
from src.logging_config import PACKAGE_LOGGER_NAME

logger = logging.getLogger(f"{PACKAGE_LOGGER_NAME}.forecasting.holtwinters")

_MIN_TRAIN = 20
_SERIES_DESCRIPTION = "adjusted close price"

_VALID_TREND = (None, "add", "mul")
_VALID_SEASONAL = (None, "add", "mul")


class HoltWintersForecaster:
    """Fit an Exponential Smoothing / Holt-Winters model and produce forecasts.

    Args:
        trend: Trend component type: ``"add"``, ``"mul"``, or ``None``.
        damped_trend: If ``True``, the trend is damped (only applies when
            ``trend`` is not ``None``).
        seasonal: Seasonal component type: ``"add"``, ``"mul"``, or ``None``.
        seasonal_periods: Length of one seasonal cycle.  Required when
            ``seasonal`` is not ``None``.
        ticker: Asset symbol (informational).
        series_description: Label for the modelled series.

    Raises:
        InvalidModelConfigError: If any combination of parameters is invalid.
    """

    def __init__(
        self,
        trend: Literal["add", "mul"] | None = "add",
        *,
        damped_trend: bool = False,
        seasonal: Literal["add", "mul"] | None = None,
        seasonal_periods: int | None = None,
        ticker: str = "UNKNOWN",
        series_description: str = _SERIES_DESCRIPTION,
    ) -> None:
        _validate_hw_config(trend, damped_trend, seasonal, seasonal_periods)
        self._trend = trend
        self._damped_trend = damped_trend
        self._seasonal = seasonal
        self._seasonal_periods = seasonal_periods
        self._ticker = ticker
        self._series_description = series_description
        self._fitted_model: Any = None
        self._train_series: pd.Series | None = None
        self._aic: float | None = None
        self._bic: float | None = None
        self._notes: list[str] = []

    @property
    def is_fitted(self) -> bool:
        return self._fitted_model is not None

    @property
    def model_name(self) -> str:
        parts = []
        if self._trend is None:
            parts.append("trend=none")
        else:
            parts.append(f"trend={self._trend}")
            if self._damped_trend:
                parts.append("damped")
        if self._seasonal is not None:
            parts.append(f"seasonal={self._seasonal}(s={self._seasonal_periods})")
        return f"HoltWinters({', '.join(parts)})"

    @property
    def config(self) -> dict[str, Any]:
        return {
            "trend": self._trend,
            "damped_trend": self._damped_trend,
            "seasonal": self._seasonal,
            "seasonal_periods": self._seasonal_periods,
        }

    def fit(self, train_series: pd.Series) -> "HoltWintersForecaster":
        """Fit the Holt-Winters model to the training series.

        Args:
            train_series: Price series with a DatetimeIndex, no NaN, ≥ 20 obs.

        Returns:
            ``self`` (fluent interface).

        Raises:
            ForecastingDataError: If the series is invalid or fitting fails.
        """
        _validate_series(train_series, self._ticker, _MIN_TRAIN)
        if self._seasonal is not None and self._seasonal_periods is not None:
            min_for_seasonal = 2 * self._seasonal_periods
            if len(train_series) < min_for_seasonal:
                raise ForecastingDataError(
                    f"Holt-Winters with seasonal_periods={self._seasonal_periods} needs at least "
                    f"{min_for_seasonal} training observations (two full cycles); "
                    f"got {len(train_series)}."
                )

        self._train_series = train_series.copy()
        self._fitted_model = None
        self._aic = None
        self._bic = None
        self._notes = []

        from statsmodels.tsa.holtwinters import ExponentialSmoothing as _ES

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                sm_model = _ES(
                    train_series,
                    trend=self._trend,
                    damped_trend=self._damped_trend,
                    seasonal=self._seasonal,
                    seasonal_periods=self._seasonal_periods,
                )
                fit_result = sm_model.fit(optimized=True)
                self._fitted_model = fit_result
                self._aic = float(fit_result.aic)
                self._bic = float(fit_result.bic)
            except Exception as exc:
                raise ForecastingDataError(
                    f"Holt-Winters fitting failed for {self._ticker}: {exc}"
                ) from exc

        for w in caught:
            msg = str(w.message)
            if "convergence" in msg.lower():
                self._notes.append(f"Convergence warning: {msg}")

        logger.info(
            "%s fitted: ticker=%s n_train=%d aic=%.2f bic=%.2f",
            self.model_name, self._ticker,
            len(train_series),
            self._aic or float("nan"),
            self._bic or float("nan"),
        )
        return self

    def forecast(self, horizon: int) -> ForecastResult:
        """Generate an out-of-sample forecast.

        Args:
            horizon: Number of steps ahead (≥ 1).

        Returns:
            A :class:`~src.forecasting.result.ForecastResult`.

        Raises:
            ModelNotFittedError: If :meth:`fit` has not been called.
            ForecastingDataError: If ``horizon`` is invalid.
        """
        if not self.is_fitted:
            raise ModelNotFittedError(
                f"{self.model_name} has not been fitted yet. Call .fit() first."
            )
        if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
            raise ForecastingDataError(
                f"horizon must be a positive integer, got {horizon!r}."
            )

        assert self._train_series is not None
        # ExponentialSmoothing.forecast returns a pd.Series indexed by steps
        fc = self._fitted_model.forecast(horizon)
        forecast_values = {
            str(pd.Timestamp(idx).date()): float(val)
            for idx, val in fc.items()
        }

        train = self._train_series
        return ForecastResult(
            model_name=self.model_name,
            model_type="holtwinters",
            config=self.config,
            ticker=self._ticker,
            series_description=self._series_description,
            frequency="1d",
            train_start=pd.Timestamp(train.index[0]).date(),
            train_end=pd.Timestamp(train.index[-1]).date(),
            n_train=len(train),
            forecast_start=pd.Timestamp(list(forecast_values.keys())[0]).date(),
            forecast_end=pd.Timestamp(list(forecast_values.keys())[-1]).date(),
            n_forecast=horizon,
            forecast_values=forecast_values,
            fitted=True,
            aic=self._aic,
            bic=self._bic,
            notes=tuple(self._notes),
        )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_hw_config(
    trend: Any,
    damped_trend: bool,
    seasonal: Any,
    seasonal_periods: Any,
) -> None:
    if trend not in _VALID_TREND:
        raise InvalidModelConfigError(
            f"trend must be one of {_VALID_TREND}, got {trend!r}."
        )
    if not isinstance(damped_trend, bool):
        raise InvalidModelConfigError(
            f"damped_trend must be a bool, got {type(damped_trend).__name__}."
        )
    if damped_trend and trend is None:
        raise InvalidModelConfigError(
            "damped_trend=True requires a non-None trend component."
        )
    if seasonal not in _VALID_SEASONAL:
        raise InvalidModelConfigError(
            f"seasonal must be one of {_VALID_SEASONAL}, got {seasonal!r}."
        )
    if seasonal is not None:
        if not isinstance(seasonal_periods, int) or isinstance(seasonal_periods, bool) or seasonal_periods < 2:
            raise InvalidModelConfigError(
                f"seasonal_periods must be an integer ≥ 2 when seasonal is set, "
                f"got {seasonal_periods!r}."
            )
