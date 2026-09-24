"""Reusable figure builders for EDA visualisations.

Every function returns a ``matplotlib.figure.Figure``.  The caller owns the
figure and is responsible for displaying or saving it.  No Streamlit imports
live in this module, so the figures can be consumed by the dashboard, a
notebook, or a test with equal ease.

All figures are built with ``matplotlib``'s object-oriented API (``Figure`` +
``Axes``).  No global pyplot state is touched.

Available figures
-----------------
* :func:`plot_price_series` — historical close-price time series.
* :func:`plot_return_series` — log-return time series with gap flags.
* :func:`plot_return_histogram` — return histogram with a normal density overlay.
* :func:`plot_data_quality` — missing-data / data-quality bar chart.
"""

from __future__ import annotations

import math

import matplotlib.figure
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from src.data.market_data import PRICE_COLUMN, MarketData
from src.preprocessing.return_series import ReturnSeries
from src.statistics.descriptive import ReturnStats

# Colour palette — consistent across all figures.
_PRICE_COLOUR = "#1f77b4"    # muted blue
_RETURN_COLOUR = "#2ca02c"   # muted green
_GAP_COLOUR = "#d62728"      # muted red (flagged gaps)
_HIST_COLOUR = "#9467bd"     # muted purple
_NORMAL_COLOUR = "#ff7f0e"   # muted orange (normal density overlay)
_BAR_GOOD = "#2ca02c"
_BAR_WARN = "#d62728"

_DEFAULT_FIGSIZE = (12, 4)
_HIST_FIGSIZE = (8, 5)
_QUALITY_FIGSIZE = (8, 4)


def plot_price_series(
    market_data: MarketData,
    *,
    figsize: tuple[float, float] = _DEFAULT_FIGSIZE,
    title: str | None = None,
) -> matplotlib.figure.Figure:
    """Plot the historical close-price time series.

    Args:
        market_data: Validated market data from the loader.
        figsize: Width × height in inches.
        title: Override the auto-generated title.

    Returns:
        A ``matplotlib`` :class:`~matplotlib.figure.Figure`.
    """
    prices = market_data.prices[PRICE_COLUMN]
    fig = matplotlib.figure.Figure(figsize=figsize, tight_layout=True)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(prices.index, prices.to_numpy(), color=_PRICE_COLOUR, linewidth=0.9)
    ax.set_xlabel("Date")
    ax.set_ylabel("Close price")
    ax.set_title(
        title
        or f"{market_data.ticker} — Close price ({market_data.source})"
    )
    ax.grid(True, alpha=0.3)
    _rotate_xlabels(ax)
    return fig


def plot_return_series(
    returns: ReturnSeries,
    *,
    figsize: tuple[float, float] = _DEFAULT_FIGSIZE,
    title: str | None = None,
    scale: str = "percent",
) -> matplotlib.figure.Figure:
    """Plot the log-return time series, highlighting gap-spanning observations.

    Args:
        returns: Validated return series from the preprocessing layer.
        figsize: Width × height in inches.
        title: Override the auto-generated title.
        scale: ``"decimal"`` or ``"percent"`` (default).  Affects the y-axis
            label and the values shown.

    Returns:
        A ``matplotlib`` :class:`~matplotlib.figure.Figure`.
    """
    if scale not in ("decimal", "percent"):
        raise ValueError(f"scale must be 'decimal' or 'percent'; got {scale!r}.")

    values = returns.percent if scale == "percent" else returns.decimal
    unit_label = "Log return (%)" if scale == "percent" else "Log return (decimal)"
    gap_mask = returns.spans_gap.to_numpy()

    fig = matplotlib.figure.Figure(figsize=figsize, tight_layout=True)
    ax = fig.add_subplot(1, 1, 1)

    # Non-gap observations
    normal_idx = np.where(~gap_mask)[0]
    if len(normal_idx):
        ax.vlines(
            values.index[normal_idx],
            0,
            values.iloc[normal_idx],
            colors=_RETURN_COLOUR,
            linewidth=0.6,
            alpha=0.7,
        )

    # Gap-spanning observations — plotted on top in a contrasting colour
    gap_idx = np.where(gap_mask)[0]
    if len(gap_idx):
        ax.vlines(
            values.index[gap_idx],
            0,
            values.iloc[gap_idx],
            colors=_GAP_COLOUR,
            linewidth=0.9,
            label=f"Spans gap (n={len(gap_idx)})",
        )
        ax.legend(fontsize=8)

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Date")
    ax.set_ylabel(unit_label)
    ax.set_title(title or f"{returns.ticker} — Daily log returns")
    ax.grid(True, alpha=0.3)
    _rotate_xlabels(ax)
    return fig


def plot_return_histogram(
    returns: ReturnSeries,
    stats: ReturnStats,
    *,
    figsize: tuple[float, float] = _HIST_FIGSIZE,
    bins: int | str = "auto",
    title: str | None = None,
    scale: str = "percent",
) -> matplotlib.figure.Figure:
    """Plot a histogram of log returns with a fitted normal density overlay.

    The normal curve uses the empirical mean and standard deviation from
    ``stats``.  No distribution is fitted or tested.

    Args:
        returns: Validated return series.
        stats: Pre-computed descriptive statistics.  Assumed to match
            ``returns`` (same ticker and observations).
        figsize: Width × height in inches.
        bins: Number of bins or a string recognised by
            :func:`numpy.histogram_bin_edges` (default ``"auto"``).
        title: Override the auto-generated title.
        scale: ``"decimal"`` or ``"percent"`` (default).

    Returns:
        A ``matplotlib`` :class:`~matplotlib.figure.Figure`.
    """
    if scale not in ("decimal", "percent"):
        raise ValueError(f"scale must be 'decimal' or 'percent'; got {scale!r}.")

    values = (returns.percent if scale == "percent" else returns.decimal).to_numpy()
    unit_label = "Log return (%)" if scale == "percent" else "Log return (decimal)"
    factor = 100.0 if scale == "percent" else 1.0
    mu = stats.mean * factor
    sigma = stats.std * factor

    fig = matplotlib.figure.Figure(figsize=figsize, tight_layout=True)
    ax = fig.add_subplot(1, 1, 1)

    counts, edges, _ = ax.hist(
        values,
        bins=bins,
        density=True,
        color=_HIST_COLOUR,
        alpha=0.6,
        label="Empirical",
    )

    # Normal density overlay
    if sigma > 0.0:
        x = np.linspace(edges[0], edges[-1], 300)
        normal_pdf = (
            np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2.0 * math.pi))
        )
        ax.plot(x, normal_pdf, color=_NORMAL_COLOUR, linewidth=1.5, label="Normal")

    ax.set_xlabel(unit_label)
    ax.set_ylabel("Density")
    ax.set_title(
        title
        or (
            f"{returns.ticker} — Return distribution  "
            f"(n={stats.count}, skew={stats.skewness:.2f}, "
            f"kurt={stats.excess_kurtosis:.2f})"
        )
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    return fig


def plot_data_quality(
    market_data: MarketData,
    returns: ReturnSeries,
    *,
    figsize: tuple[float, float] = _QUALITY_FIGSIZE,
    title: str | None = None,
) -> matplotlib.figure.Figure:
    """Plot a bar chart summarising data-quality metrics.

    Shows: price observations, return observations, rows dropped for a missing
    close, and gap-flagged returns.  Bars that indicate problems are coloured
    differently to draw the eye.

    Args:
        market_data: Validated market data.
        returns: Return series derived from ``market_data``.
        figsize: Width × height in inches.
        title: Override the auto-generated title.

    Returns:
        A ``matplotlib`` :class:`~matplotlib.figure.Figure`.
    """
    labels = [
        "Price obs.",
        "Return obs.",
        "Rows dropped\n(missing close)",
        "Gap-flagged\nreturns",
    ]
    values = [
        market_data.observations,
        returns.return_observations,
        market_data.rows_dropped,
        returns.flagged_gaps,
    ]
    colours = [
        _BAR_GOOD,
        _BAR_GOOD,
        _BAR_WARN if market_data.rows_dropped > 0 else _BAR_GOOD,
        _BAR_WARN if returns.flagged_gaps > 0 else _BAR_GOOD,
    ]

    fig = matplotlib.figure.Figure(figsize=figsize, tight_layout=True)
    ax = fig.add_subplot(1, 1, 1)
    bars = ax.bar(labels, values, color=colours, edgecolor="white", width=0.5)

    # Value labels above bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + max(values) * 0.01,
            str(val),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    good_patch = mpatches.Patch(color=_BAR_GOOD, label="OK")
    warn_patch = mpatches.Patch(color=_BAR_WARN, label="Attention needed")
    ax.legend(handles=[good_patch, warn_patch], fontsize=8)
    ax.set_ylabel("Count")
    ax.set_title(title or f"{market_data.ticker} — Data quality summary")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, max(values) * 1.15 if max(values) > 0 else 10)
    return fig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _rotate_xlabels(ax: matplotlib.axes.Axes, rotation: int = 30) -> None:  # type: ignore[name-defined]
    """Rotate x-axis tick labels to prevent overlap on date axes."""
    for label in ax.get_xticklabels():
        label.set_rotation(rotation)
        label.set_ha("right")
