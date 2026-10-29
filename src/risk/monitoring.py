"""Trailing realised risk measures and historical VaR monitoring."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.preprocessing.return_series import ReturnSeries


@dataclass(frozen=True)
class RiskMonitoringConfig:
    """Windows and annualisation used by historical risk monitoring."""

    realised_volatility_window: int = 20
    historical_var_window: int = 250
    breach_summary_window: int = 60
    confidence_levels: tuple[float, ...] = (0.95, 0.99)
    annualisation_factor: int = 252
    quantile_method: str = "linear"

    def __post_init__(self) -> None:
        for name in ("realised_volatility_window", "historical_var_window", "breach_summary_window", "annualisation_factor"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.realised_volatility_window < 2:
            raise ValueError("realised_volatility_window must be at least 2 for sample standard deviation.")
        levels = tuple(float(level) for level in self.confidence_levels)
        labels = [int(round(level * 100)) for level in levels]
        if not levels or len(levels) != len(set(levels)) or len(labels) != len(set(labels)) or any(not math.isfinite(x) or not 0 < x < 1 for x in levels):
            raise ValueError("confidence_levels must contain unique probabilities strictly between 0 and 1.")
        object.__setattr__(self, "confidence_levels", levels)
        try:
            np.quantile([0.0, 1.0], 0.5, method=self.quantile_method)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Unsupported quantile method {self.quantile_method!r}.") from exc


def calculate_risk_monitoring(
    returns: ReturnSeries,
    close_prices: pd.Series,
    config: RiskMonitoringConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Calculate trailing volatility, prior-window historical VaR/ES, breaches, and drawdowns.

    Returns are the project's decimal log returns; risk columns are in percent
    return points. Realised volatility at date *t* uses returns through *t* and
    is descriptive. VaR and ES at *t* use only the preceding historical window.
    Loss is ``-log_return`` and a breach is strictly ``realised_loss > VaR``.
    Drawdown is ``(close / prior_running_peak - 1) * 100`` through the date.
    """
    cfg = config or RiskMonitoringConfig()
    if not isinstance(returns, ReturnSeries):
        raise ValueError("returns must be a validated ReturnSeries.")
    if not isinstance(close_prices, pd.Series) or not isinstance(close_prices.index, pd.DatetimeIndex):
        raise ValueError("close_prices must be a Series with a DatetimeIndex.")
    if close_prices.empty or close_prices.index.hasnans or close_prices.index.tz is not None or close_prices.index.has_duplicates or not close_prices.index.is_monotonic_increasing:
        raise ValueError("close_prices must be non-empty, unique, and chronological.")
    closes = pd.to_numeric(close_prices, errors="coerce").astype("float64")
    if not np.isfinite(closes.to_numpy()).all() or (closes <= 0).any():
        raise ValueError("close_prices must be finite and strictly positive.")

    return_pct = returns.percent.astype("float64")
    if len(return_pct) == 0 or not return_pct.index.is_monotonic_increasing:
        raise ValueError("returns must contain chronological observations.")
    if not np.isfinite(return_pct.to_numpy()).all():
        raise ValueError("returns contain missing or non-finite observations.")
    if returns.first_price_date != close_prices.index[0].date() or not return_pct.index.isin(close_prices.index).all():
        raise ValueError("close_prices must include the ReturnSeries first price and every return date.")
    cutoff = return_pct.index[-1]
    closes = closes.loc[closes.index <= cutoff]
    drawdown_pct = (closes / closes.cummax() - 1.0) * 100.0

    frame = pd.DataFrame(index=return_pct.index)
    frame.index.name = "date"
    frame["log_return_pct"] = return_pct
    frame["realised_loss_pct"] = -return_pct
    volatility_name = f"realised_volatility_{cfg.realised_volatility_window}d_annualised_pct"
    frame[volatility_name] = (
        return_pct.rolling(cfg.realised_volatility_window, min_periods=cfg.realised_volatility_window)
        .std(ddof=1) * math.sqrt(cfg.annualisation_factor)
    )
    frame["drawdown_pct"] = drawdown_pct.reindex(frame.index)

    losses = -return_pct
    for confidence in cfg.confidence_levels:
        label = int(round(confidence * 100))
        var_values = np.full(len(frame), np.nan, dtype="float64")
        es_values = np.full(len(frame), np.nan, dtype="float64")
        breach_values = np.full(len(frame), np.nan, dtype="float64")
        for position in range(cfg.historical_var_window, len(frame)):
            prior_losses = losses.iloc[position - cfg.historical_var_window:position].to_numpy(dtype="float64")
            var_value = float(np.quantile(prior_losses, confidence, method=cfg.quantile_method))
            tail = prior_losses[prior_losses >= var_value]
            es_value = float(tail.mean())
            realised_loss = float(losses.iloc[position])
            var_values[position] = var_value
            es_values[position] = es_value
            breach_values[position] = float(realised_loss > var_value)
        frame[f"historical_var_{label}_pct"] = var_values
        frame[f"historical_es_{label}_pct"] = es_values
        breach_name = f"historical_breach_{label}"
        frame[breach_name] = pd.Series(breach_values, index=frame.index, dtype="Float64").astype("boolean")
        rolling_breach = pd.Series(breach_values, index=frame.index)
        frame[f"rolling_breach_count_{label}"] = rolling_breach.rolling(
            cfg.breach_summary_window, min_periods=cfg.breach_summary_window
        ).sum()
        frame[f"rolling_breach_rate_{label}"] = rolling_breach.rolling(
            cfg.breach_summary_window, min_periods=cfg.breach_summary_window
        ).mean()

    summary: dict[str, Any] = {
        "observations": len(frame),
        "start": frame.index[0].date().isoformat(),
        "end": frame.index[-1].date().isoformat(),
        "windows": {
            "realised_volatility_sessions": cfg.realised_volatility_window,
            "historical_var_es_prior_sessions": cfg.historical_var_window,
            "breach_summary_sessions": cfg.breach_summary_window,
            "annualisation_factor": cfg.annualisation_factor,
            "quantile_method": cfg.quantile_method,
        },
        "confidence_levels": list(cfg.confidence_levels),
        "warmup_observations": {
            "realised_volatility": cfg.realised_volatility_window - 1,
            "historical_var_and_es": cfg.historical_var_window,
            "note": "Warm-up rows remain in the daily file with missing metrics because no full trailing window exists yet.",
        },
        "maximum_drawdown_pct": float(frame["drawdown_pct"].min()),
        "latest": {},
        "breaches": {},
        "conventions": {
            "input_returns": "decimal log returns ln(P_t / P_(t-1))",
            "risk_return_scale": "percent log-return points",
            "realised_volatility": "sample standard deviation (ddof=1) of trailing returns through the date, annualised by sqrt(252)",
            "var": "linear empirical quantile at confidence of prior-window positive losses (-log return)",
            "expected_shortfall": "mean of prior-window losses greater than or equal to the empirical VaR threshold",
            "breach": "current realised loss > VaR estimated from prior observations only",
            "drawdown": "(adjusted_close / running_peak_adjusted_close - 1) * 100 through the date",
            "gap_policy": "gap-spanning returns are retained, as in ReturnSeries, and remain part of trailing windows",
        },
    }
    valid_vol = frame[volatility_name].dropna()
    if len(valid_vol):
        summary["latest"][volatility_name] = float(valid_vol.iloc[-1])
    for confidence in cfg.confidence_levels:
        label = int(round(confidence * 100))
        var_name, es_name = f"historical_var_{label}_pct", f"historical_es_{label}_pct"
        valid = frame[var_name].notna()
        observed = frame.loc[valid, f"historical_breach_{label}"].astype(bool)
        recent = observed.iloc[-cfg.breach_summary_window:]
        summary["latest"][var_name] = float(frame[var_name].dropna().iloc[-1]) if valid.any() else None
        summary["latest"][es_name] = float(frame[es_name].dropna().iloc[-1]) if valid.any() else None
        summary["breaches"][str(label)] = {
            "all_count": int(observed.sum()),
            "all_rate": float(observed.mean()) if len(observed) else None,
            "all_observations": int(len(observed)),
            "recent_count": int(recent.sum()),
            "recent_rate": float(recent.mean()) if len(recent) else None,
            "recent_observations": int(len(recent)),
            "recent_window_sessions": cfg.breach_summary_window,
        }
    return frame, summary
