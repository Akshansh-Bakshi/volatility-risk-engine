"""ARIMA forecaster for classical time-series benchmarking.

**Model**: Autoregressive Integrated Moving Average — ARIMA(p, d, q).

**Target series**: The adjusted close-price level (``MarketData.price``),
*not* log returns.  ARIMA is appropriate for the price level because it
explicitly models integration (differencing order *d*) and does not require
the series to be stationary before fitting — the model handles that
internally.

**Not implemented here**: GARCH, EGARCH, volatility forecasting.  Those
belong to later stages.

Usage::

    from src.forecasting.arima import ARIMAForecaster

    model = ARIMAForecaster(order=(1, 1, 1))
    model.fit(split.train_series)
    result = model.forecast(horizon=split.n_test)

Configuration
-------------
``order = (p, d, q)``

- ``p``: number of autoregressive terms (≥ 0).
- ``d``: degree of differencing (≥ 0, typically 0 or 1 for price series).
- ``q``: number of moving-average terms (≥ 0).

Choosing orders
---------------
For a daily equity price series, ``(p=1, d=1, q=0)`` (random walk with AR)
or ``(p=0, d=1, q=0)`` (random walk) are common starting points.  A
systematic order-selection procedure (AIC/BIC grid search) belongs to a
later model-comparison stage.
"""

from __future__ import annotations

import logging
import warnings
from datetime import date, datetime, timezone
from typing import Any

import pandas as pd

from src.exceptions import (
    ForecastingDataError,
    InvalidModelConfigError,
    ModelNotFittedError,
)
from src.forecasting.result import ForecastResult
from src.logging_config import PACKAGE_LOGGER_NAME

logger = logging.getLogger(f"{PACKAGE_LOGGER_NAME}.forecasting.arima")

_MIN_TRAIN = 20
_SERIES_DESCRIPTION = "adjusted close price"


class ARIMAForecaster:
    """Fit an ARIMA(p, d, q) model to a price series and produce forecasts.

    The object follows a two-step lifecycle:

    1. **Construct** with model configuration.
    2. **Fit** by calling :meth:`fit` with a training series.
    3. **Forecast** by calling :meth:`forecast` with a horizon length.

    The :meth:`forecast` method returns a :class:`~src.forecasting.result.ForecastResult`
    that is frozen, self-describing and JSON-serialisable.

    Args:
        order: ARIMA order ``(p, d, q)``.  All components must be
            non-negative integers.
        ticker: Asset symbol (informational; stored in the result).
        series_description: Human-readable label for the series being
            modelled (default: ``"adjusted close price"``).

    Raises:
        InvalidModelConfigError: If the order is not a 3-tuple of
            non-negative integers.
    """

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 1, 0),
        *,
        ticker: str = "UNKNOWN",
        series_description: str = _SERIES_DESCRIPTION,
    ) -> None:
        self._order = _validate_arima_order(order)
        self._ticker = ticker
        self._series_description = series_description
        self._fitted_model: Any = None       # statsmodels ARIMAResultsWrapper
        self._train_series: pd.Series | None = None
        self._aic: float | None = None
        self._bic: float | None = None
        self._notes: list[str] = []

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def order(self) -> tuple[int, int, int]:
        """ARIMA order ``(p, d, q)``."""
        return self._order

    @property
    def is_fitted(self) -> bool:
        """``True`` after :meth:`fit` has completed successfully."""
        return self._fitted_model is not None

    @property
    def model_name(self) -> str:
        p, d, q = self._order
        return f"ARIMA({p},{d},{q})"

    # ------------------------------------------------------------------ #
    # Fit                                                                  #
    # ------------------------------------------------------------------ #

    def fit(self, train_series: pd.Series) -> "ARIMAForecaster":
        """Fit the ARIMA model to the training series.

        Args:
            train_series: The price series to fit on.  Must be a
                :class:`pandas.Series` with a :class:`pandas.DatetimeIndex`,
                at least :data:`_MIN_TRAIN` observations, and no NaN values.

        Returns:
            ``self`` (fluent interface).

        Raises:
            ForecastingDataError: If ``train_series`` is invalid or too short.
        """
        _validate_series(train_series, self._ticker, _MIN_TRAIN)
        self._train_series = train_series.copy()
        self._fitted_model = None
        self._aic = None
        self._bic = None
        self._notes = []

        from statsmodels.tsa.arima.model import ARIMA as _ARIMA  # local: heavy import

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                sm_model = _ARIMA(train_series, order=self._order)
                fit_result = sm_model.fit()
                self._fitted_model = fit_result
                self._aic = float(fit_result.aic)
                self._bic = float(fit_result.bic)
            except Exception as exc:
                raise ForecastingDataError(
                    f"ARIMA fitting failed for {self._ticker}: {exc}"
                ) from exc

        for w in caught:
            msg = str(w.message)
            if "convergence" in msg.lower() or "maximum iterations" in msg.lower():
                self._notes.append(f"Convergence warning: {msg}")

        logger.info(
            "%s fitted: ticker=%s n_train=%d aic=%.2f bic=%.2f",
            self.model_name, self._ticker,
            len(train_series),
            self._aic or float("nan"),
            self._bic or float("nan"),
        )
        return self

    # ------------------------------------------------------------------ #
    # Forecast                                                             #
    # ------------------------------------------------------------------ #

    def forecast(self, horizon: int) -> ForecastResult:
        """Generate an out-of-sample forecast.

        Args:
            horizon: Number of steps ahead to forecast.  Must be ≥ 1.

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
        return ForecastResult(
            model_name=self.model_name,
            model_type="arima",
            config={"p": p, "d": d, "q": q},
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
# Validation helpers (module-level, shared across forecasting modules)
# ---------------------------------------------------------------------------


def _validate_arima_order(order: Any) -> tuple[int, int, int]:
    """Validate and return an ARIMA order tuple ``(p, d, q)``."""
    if (
        not isinstance(order, (tuple, list))
        or len(order) != 3
        or not all(isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in order)
    ):
        raise InvalidModelConfigError(
            f"ARIMA order must be a 3-tuple of non-negative integers (p, d, q), got {order!r}."
        )
    return (int(order[0]), int(order[1]), int(order[2]))


def _validate_series(series: pd.Series, ticker: str, min_obs: int) -> None:
    """Raise ForecastingDataError if ``series`` cannot be used for modelling."""
    if not isinstance(series, pd.Series):
        raise ForecastingDataError(
            f"Training input for {ticker} must be a pandas Series, got {type(series).__name__}."
        )
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ForecastingDataError(
            f"Training series for {ticker} must have a DatetimeIndex."
        )
    if series.isnull().any():
        raise ForecastingDataError(
            f"Training series for {ticker} contains NaN values."
        )
    if len(series) < min_obs:
        raise ForecastingDataError(
            f"Training series for {ticker} has only {len(series)} observations "
            f"(minimum required: {min_obs})."
        )
