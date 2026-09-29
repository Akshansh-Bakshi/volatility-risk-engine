"""Forecast evaluation metrics and evaluation-result domain objects.

This module provides the mathematical scoring layer for classical
forecasting models.  It is deliberately separate from the model-fitting
code so that:

- The same metrics can be reused by any model family.
- The evaluation logic can be tested independently of fitting.
- The result is a plain, frozen, JSON-serialisable object.

Metrics implemented
-------------------
* **RMSE** — Root Mean Squared Error.  Penalises large errors more than
  small ones.  Same unit as the target series (price).
* **MAE** — Mean Absolute Error.  Robust to outliers; same unit as the
  target series.
* **MAPE** — Mean Absolute Percentage Error.  Scale-free; expressed as a
  percentage (0–100).  Undefined when actual values are zero; such
  observations are excluded and a warning is recorded in the result notes.

Alignment policy
----------------
Actual and predicted series must have the same length.  Index values are
compared after sorting; a mismatch raises :class:`~src.exceptions.EvaluationError`.
The indices are aligned by position (not by label join) once length equality
is confirmed, which prevents accidental key-based mismatches caused by
business-day vs calendar-day conventions.

MAPE and zero values
--------------------
When an actual value is zero the per-step percentage error is infinite.
The module excludes these observations, computes MAPE over the remaining
steps, and adds a note to the result.  If *all* actuals are zero, MAPE
is reported as ``None`` with an explanatory note.

Design: no "best model" declaration
-------------------------------------
:class:`EvaluationResult` stores metrics; it does not declare a winner.
:class:`ComparisonResult` collects per-model results and provides a
convenience ranking, but callers must interpret it.  A lower error metric
is *not* always the only criterion.

Usage::

    from src.forecasting.evaluation import evaluate_forecast, compare_models
    from src.forecasting.evaluation import EvaluationResult, ComparisonResult

    result = evaluate_forecast(actual, forecast_result)
    comparison = compare_models([result_arima, result_sarima, result_hw])
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from src.exceptions import EvaluationError
from src.forecasting.result import ForecastResult


# ---------------------------------------------------------------------------
# Result objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvaluationResult:
    """Evaluation metrics for a single model's forecast against actual values.

    Attributes:
        model_name: Human-readable name of the model (e.g. ``"ARIMA(1,1,0)"``).
        model_type: Model family string (``"arima"``, ``"sarima"``, or
            ``"holtwinters"``).
        ticker: Asset symbol.
        series_description: Description of the series being forecast
            (e.g. ``"adjusted close price"``).
        frequency: Observation frequency (e.g. ``"1d"``).
        eval_start: First date of the evaluation window.
        eval_end: Last date of the evaluation window.
        n_observations: Number of actual/predicted pairs used in scoring.
        rmse: Root Mean Squared Error (same unit as the target series).
        mae: Mean Absolute Error (same unit as the target series).
        mape: Mean Absolute Percentage Error (percentage, 0–100), or
            ``None`` if all actual values were zero.
        n_zero_actuals: Number of actual observations excluded from MAPE
            because they were zero.
        notes: Tuple of informational strings (warnings, exclusions, etc.).
    """

    model_name: str
    model_type: str
    ticker: str
    series_description: str
    frequency: str
    eval_start: date
    eval_end: date
    n_observations: int
    rmse: float
    mae: float
    mape: float | None
    n_zero_actuals: int = 0
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "ticker": self.ticker,
            "series_description": self.series_description,
            "frequency": self.frequency,
            "eval_start": self.eval_start.isoformat(),
            "eval_end": self.eval_end.isoformat(),
            "n_observations": self.n_observations,
            "rmse": self.rmse,
            "mae": self.mae,
            "mape": self.mape,
            "n_zero_actuals": self.n_zero_actuals,
            "notes": list(self.notes),
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Serialise to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


@dataclass(frozen=True)
class ComparisonResult:
    """Side-by-side evaluation metrics for multiple classical models.

    Attributes:
        ticker: Asset symbol all models were evaluated on.
        series_description: Description of the target series.
        frequency: Observation frequency.
        eval_start: First date of the common evaluation window.
        eval_end: Last date of the common evaluation window.
        n_observations: Number of paired observations in the evaluation window.
        results: Tuple of per-model :class:`EvaluationResult` objects, in the
            order they were supplied.
        notes: Additional cross-model notes.

    Design note
    -----------
    This class surfaces the measured metrics.  It provides a convenience
    :meth:`ranked_by` method that sorts models by a chosen metric, but it
    does **not** declare any model "best" — that interpretation belongs to
    downstream analysis or the researcher.
    """

    ticker: str
    series_description: str
    frequency: str
    eval_start: date
    eval_end: date
    n_observations: int
    results: tuple[EvaluationResult, ...]
    notes: tuple[str, ...] = field(default_factory=tuple)

    def ranked_by(
        self,
        metric: str = "rmse",
        *,
        ascending: bool = True,
    ) -> list[EvaluationResult]:
        """Return the per-model results sorted by ``metric``.

        Args:
            metric: One of ``"rmse"``, ``"mae"``, ``"mape"``.
            ascending: If ``True`` (default), lower metric value ranks first.

        Returns:
            A new list of :class:`EvaluationResult` objects in the requested
            order.  Models for which the metric is ``None`` (e.g. MAPE when
            all actuals are zero) are placed at the end.
        """
        if metric not in ("rmse", "mae", "mape"):
            raise ValueError(f"metric must be 'rmse', 'mae' or 'mape'; got {metric!r}.")

        def _key(r: EvaluationResult) -> tuple[int, float]:
            val = getattr(r, metric)
            if val is None:
                return (1, 0.0)  # sort nulls last
            return (0, val)

        return sorted(self.results, key=_key, reverse=not ascending)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "ticker": self.ticker,
            "series_description": self.series_description,
            "frequency": self.frequency,
            "eval_start": self.eval_start.isoformat(),
            "eval_end": self.eval_end.isoformat(),
            "n_observations": self.n_observations,
            "results": [r.to_dict() for r in self.results],
            "notes": list(self.notes),
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Serialise to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def evaluate_forecast(
    actual: pd.Series,
    forecast_result: ForecastResult,
) -> EvaluationResult:
    """Score a :class:`~src.forecasting.result.ForecastResult` against actuals.

    Args:
        actual: The held-out test series (prices, same scale as the forecast).
            Must have a :class:`pandas.DatetimeIndex`, no NaN, and the same
            length as ``forecast_result.n_forecast``.
        forecast_result: The model's forecast output.  The forecast values
            must cover the same number of steps as ``len(actual)``.

    Returns:
        An :class:`EvaluationResult` containing RMSE, MAE and MAPE.

    Raises:
        EvaluationError: If the series are empty, lengths differ, indices do
            not match (after sorting), or the series contain non-finite values.
    """
    pred_series = forecast_result.forecast_series()
    actual_arr, pred_arr = _align_and_validate(actual, pred_series)
    rmse, mae, mape, n_zero, metric_notes = _compute_metrics(actual_arr, pred_arr)

    notes = tuple(metric_notes)
    n = len(actual_arr)
    return EvaluationResult(
        model_name=forecast_result.model_name,
        model_type=forecast_result.model_type,
        ticker=forecast_result.ticker,
        series_description=forecast_result.series_description,
        frequency=forecast_result.frequency,
        eval_start=pd.Timestamp(actual.index[0]).date(),
        eval_end=pd.Timestamp(actual.index[-1]).date(),
        n_observations=n,
        rmse=rmse,
        mae=mae,
        mape=mape,
        n_zero_actuals=n_zero,
        notes=notes,
    )


def compare_models(
    actual: pd.Series,
    forecast_results: list[ForecastResult],
) -> ComparisonResult:
    """Evaluate multiple models on the same actual test series.

    All models in ``forecast_results`` must have been forecast over the same
    horizon (i.e. ``n_forecast`` must equal ``len(actual)`` for each model).

    Args:
        actual: The single held-out test series used for all models.
        forecast_results: List of :class:`~src.forecasting.result.ForecastResult`
            objects from ARIMA, SARIMA and/or Holt-Winters.

    Returns:
        A :class:`ComparisonResult` with per-model :class:`EvaluationResult`
        objects inside.

    Raises:
        EvaluationError: If ``forecast_results`` is empty, or if any model's
            forecast cannot be aligned with ``actual``.
        ValueError: If models have inconsistent tickers or series descriptions.
    """
    if not forecast_results:
        raise EvaluationError("forecast_results must be a non-empty list.")

    eval_results = [evaluate_forecast(actual, fr) for fr in forecast_results]

    # Collect cross-model notes if any model had zero actuals
    notes: list[str] = []
    if any(r.n_zero_actuals > 0 for r in eval_results):
        notes.append(
            "One or more models encountered zero actual values; "
            "MAPE was computed over non-zero actuals only."
        )

    first = eval_results[0]
    return ComparisonResult(
        ticker=first.ticker,
        series_description=first.series_description,
        frequency=first.frequency,
        eval_start=first.eval_start,
        eval_end=first.eval_end,
        n_observations=first.n_observations,
        results=tuple(eval_results),
        notes=tuple(notes),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _align_and_validate(
    actual: pd.Series,
    predicted: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate alignment and return (actual_arr, predicted_arr) as float64 arrays.

    Length must be equal.  Both series are sorted by their index before
    positional alignment, so minor timestamp-precision differences (e.g.
    end-of-day vs. midnight timestamps produced by different statsmodels
    versions) are absorbed.  If the sorted indices still disagree the
    function raises :class:`~src.exceptions.EvaluationError`.

    Raises:
        EvaluationError: On empty series, length mismatch, index mismatch,
            or non-finite values.
    """
    if len(actual) == 0 or len(predicted) == 0:
        raise EvaluationError("actual and predicted series must be non-empty.")

    if len(actual) != len(predicted):
        raise EvaluationError(
            f"actual has {len(actual)} observations but predicted has "
            f"{len(predicted)}.  They must have the same length."
        )

    # Sort both series by index so positional alignment is meaningful
    actual_s = actual.sort_index()
    pred_s = predicted.sort_index()

    # Compare index dates (date-only) — absorbs time-of-day differences
    actual_dates = [pd.Timestamp(t).date() for t in actual_s.index]
    pred_dates = [pd.Timestamp(t).date() for t in pred_s.index]

    if actual_dates != pred_dates:
        raise EvaluationError(
            "actual and predicted series have different date indices after sorting. "
            f"First mismatch: actual={actual_dates[0]} vs predicted={pred_dates[0]}."
        )

    a = actual_s.to_numpy(dtype=float)
    p = pred_s.to_numpy(dtype=float)

    if not np.all(np.isfinite(a)):
        raise EvaluationError("actual series contains non-finite values (NaN or Inf).")
    if not np.all(np.isfinite(p)):
        raise EvaluationError("predicted series contains non-finite values (NaN or Inf).")

    return a, p


def _compute_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> tuple[float, float, float | None, int, list[str]]:
    """Compute RMSE, MAE, MAPE from aligned float64 arrays.

    Returns:
        ``(rmse, mae, mape_or_none, n_zero_actuals, notes)``

    MAPE formula (over non-zero actuals)::

        MAPE = 100 * mean(|actual_i - predicted_i| / |actual_i|)
    """
    errors = actual - predicted
    rmse = float(math.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(np.abs(errors)))

    notes: list[str] = []
    nonzero_mask = actual != 0.0
    n_zero = int(np.sum(~nonzero_mask))

    if n_zero > 0:
        notes.append(
            f"{n_zero} observation(s) with zero actual value excluded from MAPE "
            "to avoid division by zero."
        )

    if nonzero_mask.sum() == 0:
        notes.append(
            "MAPE is undefined: all actual values are zero. "
            "RMSE and MAE are still valid."
        )
        mape: float | None = None
    else:
        a_nz = actual[nonzero_mask]
        p_nz = predicted[nonzero_mask]
        mape = float(100.0 * np.mean(np.abs((a_nz - p_nz) / np.abs(a_nz))))

    return rmse, mae, mape, n_zero, notes
