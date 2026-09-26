"""Forecast visualisation figures.

All functions return a :class:`matplotlib.figure.Figure` built with the
object-oriented API.  No pyplot global state is touched.  No Streamlit
imports live here.

Available figures
-----------------
* :func:`plot_forecast` — Training history, actual test values, and model
  forecast on a single axis.
"""

from __future__ import annotations

import pandas as pd

from src.forecasting.result import ForecastResult


def plot_forecast(
    result: ForecastResult,
    *,
    actual_test: pd.Series | None = None,
    n_train_show: int | None = 120,
    train_series: pd.Series | None = None,
    figsize: tuple[float, float] = (14, 5),
    title: str | None = None,
) -> "matplotlib.figure.Figure":  # type: ignore[name-defined]
    """Plot training history, optional actual test values, and forecast.

    Args:
        result: A :class:`~src.forecasting.result.ForecastResult` from any
            of the three model families.
        actual_test: Optional held-out test series (the actual observed values
            for the forecast period).  When provided the figure shows forecast
            vs. actual for the test window.
        n_train_show: How many of the most recent training observations to
            display.  ``None`` shows the full training history.
        train_series: The original training series to display.  If ``None``,
            only the forecast (and optionally the actual test) are shown.
        figsize: Figure width × height in inches.
        title: Override the auto-generated title.

    Returns:
        A ``matplotlib.figure.Figure``.
    """
    import matplotlib.figure as _mfig
    import matplotlib.patches as _mpatches

    fig = _mfig.Figure(figsize=figsize, tight_layout=True)
    ax = fig.add_subplot(1, 1, 1)

    # --- training history ---
    if train_series is not None:
        ts = train_series
        if n_train_show is not None and len(ts) > n_train_show:
            ts = ts.iloc[-n_train_show:]
        ax.plot(ts.index, ts.values, color="#1f77b4", linewidth=1.0, label="Training")

    # --- actual test values ---
    if actual_test is not None and len(actual_test) > 0:
        ax.plot(
            actual_test.index, actual_test.values,
            color="#2ca02c", linewidth=1.2, label="Actual (test)",
        )

    # --- forecast ---
    fc = result.forecast_series()
    ax.plot(
        fc.index, fc.values,
        color="#d62728", linewidth=1.2, linestyle="--", label="Forecast",
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.set_title(
        title or (
            f"{result.ticker} — {result.model_name} forecast "
            f"({result.n_forecast}-step, {result.series_description})"
        )
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return fig
