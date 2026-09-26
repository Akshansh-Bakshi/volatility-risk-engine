"""SARIMA forecaster for classical time-series benchmarking.

**Model**: Seasonal ARIMA — SARIMA(p, d, q)(P, D, Q, s).

**Target series**: The adjusted close-price level (``MarketData.price``),
*not* log returns.

Seasonal structure in financial data
--------------------------------------
Daily equity price series do **not** automatically exhibit a strong seasonal
structure.  The seasonal period ``s`` must be chosen deliberately based on
domain knowledge (e.g. ``s=5`` for a weekly pattern in daily data, or
``s=252`` for an approximate annual cycle).  Without evidence of seasonality,
fitting a SARIMA model is not necessarily better than ARIMA — including
unnecessary seasonal terms can worsen out-of-sample performance.

This implementation makes the seasonal period fully configurable and does not
assume any fixed seasonality.  Users should examine the ACF/PACF plots (from
``src.statistics.diagnostics``) before choosing seasonal orders.

Usage::

    from src.forecasting.sarima import SARIMAForecaster

    model = SARIMAForecaster(order=(1, 1, 0), seasonal_order=(1, 0, 0, 5))
    model.fit(split.train_series)
    result = model.forecast(horizon=split.n_test)
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import pandas as pd

from src.exceptions import (
    ForecastingDataError,
    InvalidModelConfigError,
    ModelNotFittedError,
)
from src.forecasting.arima import _validate_series
from src.forecasting.result import ForecastResult
from src.logging_config import PACKAGE_LOGGER_NAME

logger = logging.getLogger(f"{PACKAGE_LOGGER_NAME}.forecasting.sarima")

_MIN_TRAIN = 20
_SERIES_DESCRIPTION = "adjusted close price"


class SARIMAForecaster:
    """Fit a SARIMA(p,d,q)(P,D,Q,s) model and produce forecasts.

    Args:
        order: Non-seasonal ARIMA order ``(p, d, q)``.
        seasonal_order: Seasonal order ``(P, D, Q, s)`` where ``s`` is the
            seasonal period.  All components must be non-negative integers.
            ``s`` must be ≥ 2 if any of ``P``, ``D``, ``Q`` are non-zero.
        ticker: Asset symbol (informational).
        series_description: Label for the modelled series.

    Raises:
        InvalidModelConfigError: If any order is invalid.
    """

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 1, 0),
        seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
        *,
        ticker: str = "UNKNOWN",
        series_description: str = _SERIES_DESCRIPTION,
    ) -> None:
        self._order = _validate_arima_order_3(order)
        self._seasonal_order = _validate_seasonal_order(seasonal_order)
        self._ticker = ticker
        self._series_description = series_description
        self._fitted_model: Any = None
        self._train_series: pd.Series | None = None
        self._aic: float | None = None
        self._bic: float | None = None
        self._notes: list[str] = []

    @property
    def order(self) -> tuple[int, int, int]:
        return self._order

    @property
    def seasonal_order(self) -> tuple[int, int, int, int]:
        return self._seasonal_order

    @property
    def is_fitted(self) -> bool:
        return self._fitted_model is not None

    @property
    def model_name(self) -> str:
        p, d, q = self._order
        P, D, Q, s = self._seasonal_order
        return f"SARIMA({p},{d},{q})({P},{D},{Q},{s})"

    def fit(self, train_series: pd.Series) -> "SARIMAForecaster":
        """Fit the SARIMA model to the training series.

        Args:
            train_series: Price series with a DatetimeIndex, no NaN, ≥ 20 obs.

        Returns:
            ``self`` (fluent interface).

        Raises:
            ForecastingDataError: If the series is invalid or fitting fails.
        """
        _validate_series(train_series, self._ticker, _MIN_TRAIN)
        self._train_series = train_series.copy()
        self._fitted_model = None
        self._aic = None
        self._bic = None
        self._notes = []

        from statsmodels.tsa.statespace.sarimax import SARIMAX as _SARIMAX

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                sm_model = _SARIMAX(
                    train_series,
                    order=self._order,
                    seasonal_order=self._seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                fit_result = sm_model.fit(disp=False)
                self._fitted_model = fit_result
                self._aic = float(fit_result.aic)
                self._bic = float(fit_result.bic)
            except Exception as exc:
                raise ForecastingDataError(
                    f"SARIMA fitting failed for {self._ticker}: {exc}"
                ) from exc

        for w in caught:
            msg = str(w.message)
            if "convergence" in msg.lower() or "maximum" in msg.lower():
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
        fc = self._fitted_model.forecast(steps=horizon)
        forecast_values = {
            str(pd.Timestamp(idx).date()): float(val)
            for idx, val in fc.items()
        }

        train = self._train_series
        p, d, q = self._order
        P, D, Q, s = self._seasonal_order
        return ForecastResult(
            model_name=self.model_name,
            model_type="sarima",
            config={
                "p": p, "d": d, "q": q,
                "P": P, "D": D, "Q": Q, "s": s,
            },
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
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_arima_order_3(order: Any) -> tuple[int, int, int]:
    if (
        not isinstance(order, (tuple, list))
        or len(order) != 3
        or not all(isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in order)
    ):
        raise InvalidModelConfigError(
            f"Non-seasonal order must be a 3-tuple of non-negative integers, got {order!r}."
        )
    return (int(order[0]), int(order[1]), int(order[2]))


def _validate_seasonal_order(seasonal_order: Any) -> tuple[int, int, int, int]:
    if (
        not isinstance(seasonal_order, (tuple, list))
        or len(seasonal_order) != 4
        or not all(isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in seasonal_order)
    ):
        raise InvalidModelConfigError(
            f"Seasonal order must be a 4-tuple of non-negative integers (P, D, Q, s), "
            f"got {seasonal_order!r}."
        )
    P, D, Q, s = (int(x) for x in seasonal_order)
    if (P > 0 or D > 0 or Q > 0) and s < 2:
        raise InvalidModelConfigError(
            f"Seasonal period s must be ≥ 2 when any of P, D, Q are non-zero "
            f"(got s={s}, P={P}, D={D}, Q={Q})."
        )
    return (P, D, Q, s)
