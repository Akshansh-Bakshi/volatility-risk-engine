"""Helpers for forecasting by observation step and labelling forecast dates."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.exceptions import ForecastingDataError


def resolve_forecast_index(
    train_series: pd.Series,
    horizon: int,
    forecast_index: pd.DatetimeIndex | None,
) -> pd.DatetimeIndex:
    """Resolve result labels without requiring a calendar frequency to forecast."""
    if forecast_index is not None:
        if not isinstance(forecast_index, pd.DatetimeIndex):
            raise ForecastingDataError("forecast_index must be a pandas DatetimeIndex.")
        if len(forecast_index) != horizon:
            raise ForecastingDataError(
                f"forecast_index has {len(forecast_index)} dates but horizon is {horizon}."
            )
        if forecast_index.tz is not None:
            raise ForecastingDataError("forecast_index must be timezone-naive.")
        if forecast_index.has_duplicates or not forecast_index.is_monotonic_increasing:
            raise ForecastingDataError(
                "forecast_index must be strictly increasing and contain no duplicates."
            )
        if forecast_index[0] <= train_series.index[-1]:
            raise ForecastingDataError("forecast_index must start after the training period.")
        return forecast_index.copy()

    freq = train_series.index.freq or train_series.index.inferred_freq
    if freq is None:
        raise ForecastingDataError(
            "forecast_index is required when the training DatetimeIndex has no fixed "
            "or inferable frequency; pass dates corresponding to the forecast steps."
        )
    return pd.date_range(
        start=train_series.index[-1], periods=horizon + 1, freq=freq,
        name=train_series.index.name,
    )[1:]


def forecast_values_by_date(
    forecast: object,
    forecast_index: pd.DatetimeIndex,
    horizon: int,
) -> dict[str, float]:
    """Pair step-space forecast values with their date labels."""
    values = np.asarray(forecast, dtype=float).reshape(-1)
    if len(values) != horizon:
        raise ForecastingDataError(
            f"model returned {len(values)} forecast values for horizon {horizon}."
        )
    return {
        timestamp.date().isoformat(): float(value)
        for timestamp, value in zip(forecast_index, values, strict=True)
    }
