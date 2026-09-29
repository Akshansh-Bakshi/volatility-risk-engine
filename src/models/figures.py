"""Volatility model figure builders.

All functions return a :class:`matplotlib.figure.Figure` built with the
matplotlib object-oriented API.  No pyplot global state is touched.
No Streamlit imports live here.

Available figures
-----------------
* :func:`plot_conditional_volatility` — Daily and annualised conditional
  volatility from a fitted :class:`~src.models.result.ModelFitResult`.
* :func:`plot_volatility_comparison` — Overlaid conditional volatility
  from two :class:`~src.models.result.ModelFitResult` objects (e.g. GARCH
  vs EGARCH) for visual comparison.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from src.models.result import ModelFitResult


def plot_conditional_volatility(
    result: ModelFitResult,
    *,
    annualized: bool = True,
    figsize: tuple[float, float] = (14, 5),
    title: str | None = None,
) -> "matplotlib.figure.Figure":  # type: ignore[name-defined]
    """Plot conditional volatility from a fitted model.

    Args:
        result: A :class:`~src.models.result.ModelFitResult` (GARCH or EGARCH).
        annualized: If ``True`` (default), plot
            :attr:`~src.models.result.ModelFitResult.annualized_vol_pct`.
            If ``False``, plot
            :attr:`~src.models.result.ModelFitResult.daily_vol_pct`.
        figsize: Figure width × height in inches.
        title: Override the auto-generated title.

    Returns:
        A ``matplotlib.figure.Figure``.
    """
    import matplotlib.figure as _mfig

    series = result.annualized_vol_pct if annualized else result.daily_vol_pct
    unit_label = (
        f"Annualised volatility (%, √{result.annualization_factor} convention)"
        if annualized
        else "Daily volatility (% per day)"
    )

    fig = _mfig.Figure(figsize=figsize, tight_layout=True)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(series.index, series.values, color="#d62728", linewidth=1.0,
            label=result.model_name)
    ax.set_xlabel("Date")
    ax.set_ylabel(unit_label)
    ax.set_title(
        title or (
            f"{result.ticker} — {result.model_name} conditional volatility "
            f"(n={result.n_observations}, "
            f"return_scale={result.return_scale})"
        )
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return fig


def plot_volatility_comparison(
    results: Sequence[ModelFitResult],
    *,
    annualized: bool = True,
    figsize: tuple[float, float] = (14, 5),
    title: str | None = None,
) -> "matplotlib.figure.Figure":  # type: ignore[name-defined]
    """Overlay conditional volatility from multiple fitted models.

    Useful for visual GARCH vs EGARCH comparison.  No winner is declared —
    the plot simply overlays the series.

    Args:
        results: Sequence of :class:`~src.models.result.ModelFitResult` objects.
        annualized: If ``True``, use annualised volatility.
        figsize: Figure width × height in inches.
        title: Override the auto-generated title.

    Returns:
        A ``matplotlib.figure.Figure``.
    """
    import matplotlib.figure as _mfig

    _COLOURS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]

    fig = _mfig.Figure(figsize=figsize, tight_layout=True)
    ax = fig.add_subplot(1, 1, 1)

    for i, result in enumerate(results):
        series = result.annualized_vol_pct if annualized else result.daily_vol_pct
        ax.plot(
            series.index, series.values,
            color=_COLOURS[i % len(_COLOURS)],
            linewidth=1.0,
            label=result.model_name,
        )

    unit_label = "Annualised volatility (%)" if annualized else "Daily volatility (% per day)"
    tickers = ", ".join(dict.fromkeys(r.ticker for r in results))
    ax.set_xlabel("Date")
    ax.set_ylabel(unit_label)
    ax.set_title(title or f"{tickers} — volatility model comparison")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return fig
