"""Shared mechanics for one-step forecasts from fitted ``arch`` models."""

from __future__ import annotations

import math
import warnings
from datetime import date, datetime
from typing import Any

import pandas as pd

from src.exceptions import VolatilityModelDataError, VolatilityModelFitError
from src.models.result import ModelFitResult, VarianceForecast


def make_one_step_forecast(
    arch_result: Any,
    fit_result: ModelFitResult,
    forecast_date: date,
) -> VarianceForecast:
    """Forecast next-observation residual variance and attach its date label."""
    if not isinstance(forecast_date, date) or isinstance(forecast_date, datetime):
        raise VolatilityModelDataError("forecast_date must be a calendar date.")
    if forecast_date <= fit_result.fit_end:
        raise VolatilityModelDataError(
            "forecast_date must be strictly after the final observation used for fitting."
        )

    notes = list(fit_result.notes)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            forecast = arch_result.forecast(horizon=1, reindex=False)
            mean = float(forecast.mean.iloc[-1, 0])
            variance = float(forecast.residual_variance.iloc[-1, 0])
        except Exception as exc:
            raise VolatilityModelFitError(
                f"{fit_result.model_name} one-step variance forecast failed: {exc}"
            ) from exc
    notes.extend(f"arch forecast warning: {warning.message}" for warning in caught)

    if not math.isfinite(mean):
        raise VolatilityModelFitError(
            f"{fit_result.model_name} produced a non-finite conditional mean forecast."
        )
    if not math.isfinite(variance) or variance <= 0.0:
        raise VolatilityModelFitError(
            f"{fit_result.model_name} produced a non-positive or non-finite variance forecast."
        )

    return VarianceForecast(
        model_name=fit_result.model_name,
        model_type=fit_result.model_type,
        config=fit_result.config.copy(),
        ticker=fit_result.ticker,
        return_scale=fit_result.return_scale,
        variance_scale=f"{fit_result.return_scale}^2",
        fit_start=fit_result.fit_start,
        fit_end=fit_result.fit_end,
        n_train=fit_result.n_observations,
        forecast_date=forecast_date,
        forecast_variance=variance,
        converged=fit_result.converged,
        notes=tuple(notes),
        forecast_mean=mean,
    )
