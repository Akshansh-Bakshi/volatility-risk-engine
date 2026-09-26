"""Shared forecast result domain object.

:class:`ForecastResult` is the common output type for all classical
forecasting models (ARIMA, SARIMA, Holt-Winters).  It is:

- **Frozen**: immutable after construction.
- **JSON-serialisable**: all fields are plain Python types; no statsmodels
  objects are stored.
- **Self-describing**: the result knows which model produced it, what the
  configuration was, what period was forecast, and what the values are.

Design rationale
----------------
Storing raw statsmodels result objects would tightly couple the result layer
to statsmodels internals and make serialisation impossible.  Instead, we
extract only the information needed downstream (forecast values, metadata,
diagnostics) and store those as plain Python types.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ForecastResult:
    """The output of fitting and forecasting from a classical time-series model.

    Attributes:
        model_name: Human-readable model name (e.g. ``"ARIMA(1,1,1)"``).
        model_type: Short model family string: ``"arima"``, ``"sarima"``,
            or ``"holtwinters"``.
        config: Dictionary of model configuration parameters (orders, seasonal
            settings, etc.).  Must be JSON-serialisable.
        ticker: Asset symbol the model was fitted on.
        series_description: What the series represents (always the adjusted
            close price for this stage).
        frequency: Observation frequency (``"1d"`` for daily).
        train_start: First date of the training window.
        train_end: Last date of the training window.
        n_train: Number of training observations.
        forecast_start: First date of the forecast horizon.
        forecast_end: Last date of the forecast horizon.
        n_forecast: Number of forecast steps.
        forecast_values: Forecast values as a mapping
            ``{date_iso_str: float}`` (a ``dict`` so it is JSON-serialisable).
        fitted: Whether fitting converged without fatal errors.
        aic: Akaike information criterion (``None`` if unavailable).
        bic: Bayesian information criterion (``None`` if unavailable).
        notes: List of informational messages (warnings, convergence notices).
    """

    model_name: str
    model_type: str
    config: dict[str, Any]
    ticker: str
    series_description: str
    frequency: str
    train_start: date
    train_end: date
    n_train: int
    forecast_start: date
    forecast_end: date
    n_forecast: int
    forecast_values: dict[str, float]   # {ISO date str → float}
    fitted: bool
    aic: float | None = None
    bic: float | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    # ------------------------------------------------------------------ #
    # Convenience                                                          #
    # ------------------------------------------------------------------ #

    def forecast_series(self) -> pd.Series:
        """Return the forecast as a :class:`pandas.Series` with a DatetimeIndex.

        The index uses the same ``date`` name and timezone-naive convention as
        the rest of the pipeline.
        """
        dates = pd.to_datetime(list(self.forecast_values.keys()))
        values = list(self.forecast_values.values())
        return pd.Series(values, index=dates, name="forecast")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "config": self.config,
            "ticker": self.ticker,
            "series_description": self.series_description,
            "frequency": self.frequency,
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "n_train": self.n_train,
            "forecast_start": self.forecast_start.isoformat(),
            "forecast_end": self.forecast_end.isoformat(),
            "n_forecast": self.n_forecast,
            "forecast_values": self.forecast_values,
            "fitted": self.fitted,
            "aic": self.aic,
            "bic": self.bic,
            "notes": list(self.notes),
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Serialise to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
