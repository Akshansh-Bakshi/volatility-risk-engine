"""One-day-ahead parametric and historical-simulation Value-at-Risk."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.exceptions import VolatilityModelDataError
from src.preprocessing.return_series import ReturnSeries


def normal_var(mean_pct: float, variance_pct2: float, confidence: float) -> float:
    """Return positive-loss VaR in return percentage points under Normality.

    The lower-tail return quantile is ``mean + z_(1-c) * sqrt(variance)``;
    VaR is its negation so losses are reported as positive numbers.
    """
    _validate_confidence(confidence)
    if not math.isfinite(mean_pct):
        raise ValueError("mean_pct must be finite.")
    if not math.isfinite(variance_pct2) or variance_pct2 <= 0.0:
        raise ValueError("variance_pct2 must be positive and finite.")
    alpha = 1.0 - confidence
    return float(-(mean_pct + norm.ppf(alpha) * math.sqrt(variance_pct2)))


def historical_simulation_var(
    prior_returns_pct: Iterable[float],
    confidence: float,
    *,
    quantile_method: str = "linear",
) -> float:
    """Return empirical positive-loss VaR from prior percentage returns only."""
    _validate_confidence(confidence)
    values = np.asarray(list(prior_returns_pct), dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("prior_returns_pct must be non-empty and finite.")
    try:
        quantile = float(np.quantile(values, 1.0 - confidence, method=quantile_method))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unsupported quantile method {quantile_method!r}.") from exc
    return -quantile


def build_var_forecasts(
    returns: ReturnSeries,
    variance_forecasts: pd.DataFrame,
    *,
    historical_window: int = 250,
    confidence_levels: Iterable[float] = (0.95, 0.99),
    quantile_method: str = "linear",
) -> pd.DataFrame:
    """Align model distribution forecasts and rolling historical VaR by date.

    ``variance_forecasts`` is the Stage 9 walk-forward frame. Its forecast rows
    must be a strictly increasing subset of the validated return index. For
    each target, historical simulation sees only the preceding window of valid
    returns. Gap-flagged returns are retained, in line with ``ReturnSeries``.
    """
    if isinstance(historical_window, bool) or not isinstance(historical_window, int) or historical_window < 1:
        raise ValueError("historical_window must be a positive integer.")
    if not isinstance(variance_forecasts.index, pd.DatetimeIndex):
        raise ValueError("variance_forecasts must have a DatetimeIndex.")
    if variance_forecasts.empty or not variance_forecasts.index.is_unique or not variance_forecasts.index.is_monotonic_increasing:
        raise ValueError("variance_forecasts must be non-empty, unique and chronologically ordered.")
    levels = tuple(float(level) for level in confidence_levels)
    if not levels or len(set(levels)) != len(levels):
        raise ValueError("confidence_levels must contain unique confidence values.")
    for level in levels:
        _validate_confidence(level)

    required = (
        "garch_conditional_mean_pct", "garch_variance_pct2",
        "egarch_conditional_mean_pct", "egarch_variance_pct2",
    )
    missing = [column for column in required if column not in variance_forecasts]
    if missing:
        raise ValueError(f"Missing Stage 9 forecast columns: {', '.join(missing)}.")

    index = returns.decimal.index
    positions = index.get_indexer(variance_forecasts.index)
    if np.any(positions < 0) or np.any(np.diff(positions) <= 0):
        raise ValueError("Forecast dates must align in order with ReturnSeries observations.")
    pct = returns.percent
    rows: list[dict[str, float | bool | pd.Timestamp]] = []

    for timestamp, position in zip(variance_forecasts.index, positions, strict=True):
        start = int(position) - historical_window
        if start < 0:
            raise VolatilityModelDataError(
                f"Date {timestamp.date()} has fewer than {historical_window} prior valid returns."
            )
        historical = pct.iloc[start:int(position)]
        if len(historical) != historical_window:
            raise VolatilityModelDataError("Historical VaR window does not match requested size.")

        realised_return = float(pct.iloc[int(position)])
        realised_loss = -realised_return
        garch_mean = float(variance_forecasts.at[timestamp, "garch_conditional_mean_pct"])
        garch_var = float(variance_forecasts.at[timestamp, "garch_variance_pct2"])
        egarch_mean = float(variance_forecasts.at[timestamp, "egarch_conditional_mean_pct"])
        egarch_var = float(variance_forecasts.at[timestamp, "egarch_variance_pct2"])
        historical_mean = float(historical.mean())
        historical_var = float(historical.var(ddof=1)) if historical_window > 1 else 0.0

        row: dict[str, float | bool | pd.Timestamp] = {
            "date": timestamp,
            "realised_return_pct": realised_return,
            "realised_loss_pct": realised_loss,
            "garch_conditional_mean_pct": garch_mean,
            "garch_forecast_variance_pct2": garch_var,
            "garch_forecast_std_pct": math.sqrt(garch_var),
            "egarch_conditional_mean_pct": egarch_mean,
            "egarch_forecast_variance_pct2": egarch_var,
            "egarch_forecast_std_pct": math.sqrt(egarch_var),
            "historical_conditional_mean_pct": historical_mean,
            "historical_forecast_variance_pct2": historical_var,
            "historical_forecast_std_pct": math.sqrt(max(historical_var, 0.0)),
            "historical_window_start": historical.index[0],
            "historical_window_end": historical.index[-1],
            "historical_window_n": len(historical),
            "historical_window_gap_flagged_n": int(returns.spans_gap.iloc[start:int(position)].sum()),
            "gap_flagged": bool(returns.spans_gap.iloc[int(position)]),
        }
        for confidence in levels:
            label = int(round(confidence * 100))
            garch_var_threshold = normal_var(garch_mean, garch_var, confidence)
            egarch_var_threshold = normal_var(egarch_mean, egarch_var, confidence)
            historical_threshold = historical_simulation_var(
                historical.to_numpy(), confidence, quantile_method=quantile_method
            )
            thresholds = {
                "garch": garch_var_threshold,
                "egarch": egarch_var_threshold,
                "historical": historical_threshold,
            }
            if not all(math.isfinite(value) and value > 0.0 for value in thresholds.values()):
                raise VolatilityModelDataError(
                    f"VaR thresholds must be positive and finite on {timestamp.date()}."
                )
            row[f"garch_var{label}_pct"] = garch_var_threshold
            row[f"egarch_var{label}_pct"] = egarch_var_threshold
            row[f"historical_var{label}_pct"] = historical_threshold
            row[f"garch_breach{label}"] = realised_loss > garch_var_threshold
            row[f"egarch_breach{label}"] = realised_loss > egarch_var_threshold
            row[f"historical_breach{label}"] = realised_loss > historical_threshold
        rows.append(row)

    result = pd.DataFrame(rows).set_index("date")
    result.index = pd.DatetimeIndex(result.index, name="date")
    return result


def _validate_confidence(confidence: float) -> None:
    if not math.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be a finite probability strictly between 0 and 1.")
