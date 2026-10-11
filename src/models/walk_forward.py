"""Leakage-safe expanding-window volatility forecasts and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.exceptions import VolatilityModelDataError
from src.models.egarch import EGARCHModel
from src.models.garch import GARCHModel
from src.preprocessing.return_series import ReturnSeries


@dataclass(frozen=True)
class WalkForwardResult:
    """Per-observation forecasts plus shared-window summary metrics."""

    forecasts: pd.DataFrame
    metrics: pd.DataFrame
    evaluation_start: str
    evaluation_end: str
    evaluation_observations: int
    rolling_window: int
    garch_config: dict[str, Any]
    egarch_config: dict[str, Any]


def training_return_prefix(returns: ReturnSeries, end_position: int) -> ReturnSeries:
    """Build a validated ReturnSeries containing observations strictly before position."""
    if isinstance(end_position, bool) or not isinstance(end_position, int):
        raise VolatilityModelDataError("end_position must be an integer observation position.")
    if end_position < 1 or end_position > returns.return_observations:
        raise VolatilityModelDataError(
            f"end_position must be between 1 and {returns.return_observations}, got {end_position}."
        )
    return ReturnSeries(
        ticker=returns.ticker,
        frequency=returns.frequency,
        price_basis=returns.price_basis,
        decimal=returns.decimal.iloc[:end_position].copy(),
        spans_gap=returns.spans_gap.iloc[:end_position].copy(),
        first_price_date=returns.first_price_date,
        gap_tolerance_weekdays=returns.gap_tolerance_weekdays,
        provisional_bar=None,
        provisional_bar_excluded=False,
        ingestion_rows_dropped=returns.ingestion_rows_dropped,
        source=returns.source,
        fetched_at=returns.fetched_at,
    )


def run_expanding_volatility_forecast(
    returns: ReturnSeries,
    *,
    evaluation_size: int = 252,
    rolling_window: int = 20,
    evaluation_dates: pd.DatetimeIndex | None = None,
) -> WalkForwardResult:
    """Run GARCH, EGARCH and a prior-window variance benchmark without look-ahead.

    Flagged gap-spanning returns are retained according to the canonical
    ``ReturnSeries`` policy. At each origin, all model fits and the benchmark
    use only returns before the target observation.
    """
    if isinstance(evaluation_size, bool) or not isinstance(evaluation_size, int) or evaluation_size < 1:
        raise VolatilityModelDataError("evaluation_size must be a positive integer.")
    if isinstance(rolling_window, bool) or not isinstance(rolling_window, int) or rolling_window < 2:
        raise VolatilityModelDataError("rolling_window must be an integer of at least 2.")
    if evaluation_dates is None:
        first_position = returns.return_observations - evaluation_size
        positions = np.arange(first_position, returns.return_observations, dtype=int)
        target_index = returns.decimal.index[positions]
    else:
        if not isinstance(evaluation_dates, pd.DatetimeIndex):
            raise VolatilityModelDataError("evaluation_dates must be a pandas DatetimeIndex.")
        if len(evaluation_dates) == 0 or evaluation_dates.has_duplicates or not evaluation_dates.is_monotonic_increasing:
            raise VolatilityModelDataError(
                "evaluation_dates must be non-empty, unique and chronologically ordered."
            )
        positions = returns.decimal.index.get_indexer(evaluation_dates)
        if np.any(positions < 0):
            raise VolatilityModelDataError("One or more evaluation dates are missing from ReturnSeries.")
        target_index = evaluation_dates
    if np.any(positions < max(50, rolling_window)):
        raise VolatilityModelDataError(
            f"Each forecast origin needs at least {max(50, rolling_window)} prior returns; "
            f"earliest origin has {int(positions.min())}."
        )

    percent = returns.percent
    garch_model = GARCHModel(ticker=returns.ticker)
    egarch_model = EGARCHModel(ticker=returns.ticker)
    rows: list[dict[str, Any]] = []

    for target_timestamp, position_value in zip(target_index, positions, strict=True):
        position = int(position_value)
        target_date = target_timestamp.date()
        training = training_return_prefix(returns, position)

        garch_fit = garch_model.fit(training)
        garch_forecast = garch_model.forecast_one_step(target_date)
        egarch_fit = egarch_model.fit(training)
        egarch_forecast = egarch_model.forecast_one_step(target_date)

        benchmark_history = percent.iloc[position - rolling_window:position]
        if len(benchmark_history) != rolling_window:
            raise VolatilityModelDataError("Rolling variance benchmark has insufficient history.")
        rolling_variance = float(benchmark_history.var(ddof=1))
        realised_return = float(percent.iloc[position])
        realised_squared = realised_return**2

        forecasts = {
            "garch_variance_pct2": garch_forecast.forecast_variance,
            "egarch_variance_pct2": egarch_forecast.forecast_variance,
            "rolling_variance_pct2": rolling_variance,
        }
        if not all(np.isfinite(value) and value > 0.0 for value in forecasts.values()):
            raise VolatilityModelDataError(
                f"Non-positive or non-finite variance forecast for {target_date}."
            )
        if not (garch_forecast.fit_end < target_date and egarch_forecast.fit_end < target_date):
            raise VolatilityModelDataError(f"Look-ahead detected at forecast date {target_date}.")

        rows.append({
            "date": target_timestamp,
            "realised_return_pct": realised_return,
            "realised_squared_return_pct2": realised_squared,
            "garch_conditional_mean_pct": garch_forecast.forecast_mean,
            **forecasts,
            "garch_fit_end": pd.Timestamp(garch_fit.fit_end),
            "garch_n_train": garch_fit.n_observations,
            "garch_converged": garch_fit.converged,
            "garch_notes": " | ".join(garch_forecast.notes),
            "egarch_fit_end": pd.Timestamp(egarch_fit.fit_end),
            "egarch_conditional_mean_pct": egarch_forecast.forecast_mean,
            "egarch_n_train": egarch_fit.n_observations,
            "egarch_converged": egarch_fit.converged,
            "egarch_notes": " | ".join(egarch_forecast.notes),
            "rolling_window_start": benchmark_history.index[0],
            "rolling_window_end": benchmark_history.index[-1],
            "rolling_window_n": len(benchmark_history),
            "gap_flagged": bool(returns.spans_gap.iloc[position]),
        })

    frame = pd.DataFrame(rows).set_index("date")
    frame.index = pd.DatetimeIndex(frame.index, name="date")
    metrics = evaluate_variance_forecasts(frame)
    return WalkForwardResult(
        forecasts=frame,
        metrics=metrics,
        evaluation_start=frame.index[0].date().isoformat(),
        evaluation_end=frame.index[-1].date().isoformat(),
        evaluation_observations=len(frame),
        rolling_window=rolling_window,
        garch_config=garch_model.config,
        egarch_config=egarch_model.config,
    )


def evaluate_variance_forecasts(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Score variance predictions using QLIKE and squared-return proxy errors.

    QLIKE is ``mean(log(h_t) + RV_t / h_t)``; MAE and RMSE compare ``h_t``
    with ``RV_t = r_t**2``. All quantities are in percent-squared units.
    """
    actual = forecasts["realised_squared_return_pct2"].to_numpy(dtype=float)
    if len(actual) == 0 or not np.isfinite(actual).all() or np.any(actual < 0):
        raise VolatilityModelDataError("Realised squared returns must be finite and non-negative.")

    columns = {
        "GARCH(1,1)": "garch_variance_pct2",
        "EGARCH(1,1,1)": "egarch_variance_pct2",
        "20-session rolling variance": "rolling_variance_pct2",
    }
    rows: list[dict[str, float | str]] = []
    for model, column in columns.items():
        predicted = forecasts[column].to_numpy(dtype=float)
        if not np.isfinite(predicted).all() or np.any(predicted <= 0):
            raise VolatilityModelDataError(f"{model} forecasts must be positive and finite.")
        errors = predicted - actual
        rows.append({
            "model": model,
            "observations": len(actual),
            "qlike_mean_log_h_plus_rv_over_h": float(np.mean(np.log(predicted) + actual / predicted)),
            "mae_pct2": float(np.mean(np.abs(errors))),
            "rmse_pct2": float(np.sqrt(np.mean(errors**2))),
        })
    return pd.DataFrame(rows)
