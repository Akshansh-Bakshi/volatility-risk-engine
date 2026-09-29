"""Presentation metrics for the descriptive market dashboard.

This module contains small, reusable historical summaries that are specific to
the dashboard. It does not fit models or make forward-looking calculations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.market_data import MarketData
from src.preprocessing.return_series import ReturnSeries
from src.statistics.descriptive import ReturnStats


@dataclass(frozen=True)
class DashboardMetrics:
    """Historical KPIs and aligned series for presentation charts."""

    latest_price: float | None
    latest_completed_return_pct: float | None
    latest_completed_return_date: str | None
    rolling_volatility_20_pct: float | None
    trailing_20_session_return_pct: float | None
    maximum_drawdown_pct: float | None
    rolling_volatility_series_pct: pd.Series
    drawdown_series_pct: pd.Series
    peak_volatility_date: str | None


def calculate_dashboard_metrics(
    market_data: MarketData,
    returns: ReturnSeries,
) -> DashboardMetrics:
    """Calculate historical price/return metrics, excluding an open final bar."""
    close = market_data.price.astype("float64")
    latest_price = float(close.iloc[-1]) if len(close) else None

    completed = returns.decimal
    if returns.last_return_is_provisional:
        completed = completed.iloc[:-1]
    completed_pct = completed * 100.0
    latest_return = float(completed_pct.iloc[-1]) if len(completed_pct) else None
    latest_return_date = (
        completed_pct.index[-1].date().isoformat() if len(completed_pct) else None
    )

    rolling = completed.rolling(window=20, min_periods=20).std(ddof=1) * np.sqrt(252) * 100
    rolling = rolling.rename("20-session annualised volatility (%)")
    rolling_valid = rolling.dropna()
    volatility = float(rolling_valid.iloc[-1]) if len(rolling_valid) else None
    peak_volatility_date = (
        rolling_valid.idxmax().date().isoformat() if len(rolling_valid) else None
    )

    # For return, rolling volatility and drawdown KPIs, use completed closes.
    if returns.provisional_bar is not None:
        completed_close = close.loc[close.index.date < returns.provisional_bar]
    else:
        completed_close = close
    trailing = None
    if len(completed_close) >= 21:
        trailing = float(
            (completed_close.iloc[-1] / completed_close.iloc[-21] - 1.0) * 100.0
        )

    drawdown = (close / close.cummax() - 1.0) * 100.0
    drawdown = drawdown.rename("Drawdown (%)")
    max_drawdown = float(drawdown.min()) if len(drawdown) else None

    return DashboardMetrics(
        latest_price=latest_price,
        latest_completed_return_pct=latest_return,
        latest_completed_return_date=latest_return_date,
        rolling_volatility_20_pct=volatility,
        trailing_20_session_return_pct=trailing,
        maximum_drawdown_pct=max_drawdown,
        rolling_volatility_series_pct=rolling,
        drawdown_series_pct=drawdown,
        peak_volatility_date=peak_volatility_date,
    )


def factual_findings(stats: ReturnStats, metrics: DashboardMetrics) -> list[str]:
    """Create stable, factual observations from the selected historical sample."""
    skew = (
        "positive" if stats.skewness > 0.1
        else "negative" if stats.skewness < -0.1
        else "close to symmetric"
    )
    findings = [
        f"Across {stats.count:,} daily returns, the mean was {stats.mean_pct:.4f}% and the sample standard deviation was {stats.std_pct:.4f}%.",
        f"The observed daily return range was {stats.minimum_pct:.2f}% to {stats.maximum_pct:.2f}%; sample skewness was {stats.skewness:.3f} ({skew}).",
        f"Excess kurtosis was {stats.excess_kurtosis:.3f} for this sample.",
    ]
    if metrics.maximum_drawdown_pct is not None:
        findings.append(
            f"The largest peak-to-trough decline in the selected price history was {metrics.maximum_drawdown_pct:.2f}%."
        )
    if metrics.peak_volatility_date is not None:
        findings.append(
            f"The highest observed 20-session annualised volatility window ended {metrics.peak_volatility_date}."
        )
    return findings[:5]
