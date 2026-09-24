"""Tests for src.statistics.figures.

Figures are tested at the object level — we verify they are proper
``matplotlib.figure.Figure`` instances, that they have exactly the expected
number of axes, that axes carry sensible titles/labels, and that they include
the expected data series.  We never display or save figures.

All tests use synthetic data; no network access.
"""

from __future__ import annotations

import matplotlib.figure
import numpy as np
import pandas as pd
import pytest

from src.preprocessing.returns import build_return_series
from src.statistics.descriptive import compute_return_stats
from src.statistics.figures import (
    plot_data_quality,
    plot_price_series,
    plot_return_histogram,
    plot_return_series,
)
from tests.data_fakes import make_market_data, make_prices


def _make(periods: int = 300):
    md = make_market_data(make_prices(periods=periods))
    rs = build_return_series(md)
    stats = compute_return_stats(rs)
    return md, rs, stats


# ---------------------------------------------------------------------------
# plot_price_series
# ---------------------------------------------------------------------------


class TestPlotPriceSeries:
    def test_returns_a_figure(self) -> None:
        md, _, _ = _make()
        fig = plot_price_series(md)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_figure_has_one_axes(self) -> None:
        md, _, _ = _make()
        fig = plot_price_series(md)
        assert len(fig.axes) == 1

    def test_axes_has_data(self) -> None:
        md, _, _ = _make()
        fig = plot_price_series(md)
        ax = fig.axes[0]
        # The price series is drawn as a single line.
        assert len(ax.lines) == 1

    def test_title_contains_ticker(self) -> None:
        md, _, _ = _make()
        fig = plot_price_series(md)
        assert "TEST" in fig.axes[0].get_title()

    def test_custom_title_is_used(self) -> None:
        md, _, _ = _make()
        fig = plot_price_series(md, title="Custom Title")
        assert fig.axes[0].get_title() == "Custom Title"

    def test_ylabel_set(self) -> None:
        md, _, _ = _make()
        fig = plot_price_series(md)
        assert fig.axes[0].get_ylabel()

    def test_custom_figsize_is_accepted(self) -> None:
        md, _, _ = _make()
        fig = plot_price_series(md, figsize=(6, 3))
        w, h = fig.get_size_inches()
        assert w == pytest.approx(6.0)
        assert h == pytest.approx(3.0)

    def test_number_of_x_data_points_matches_price_rows(self) -> None:
        md, _, _ = _make()
        fig = plot_price_series(md)
        line = fig.axes[0].lines[0]
        assert len(line.get_xdata()) == md.observations


# ---------------------------------------------------------------------------
# plot_return_series
# ---------------------------------------------------------------------------


class TestPlotReturnSeries:
    def test_returns_a_figure(self) -> None:
        _, rs, _ = _make()
        fig = plot_return_series(rs)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_figure_has_one_axes(self) -> None:
        _, rs, _ = _make()
        fig = plot_return_series(rs)
        assert len(fig.axes) == 1

    def test_title_contains_ticker(self) -> None:
        _, rs, _ = _make()
        fig = plot_return_series(rs)
        assert "TEST" in fig.axes[0].get_title()

    def test_ylabel_mentions_return(self) -> None:
        _, rs, _ = _make()
        fig = plot_return_series(rs)
        assert "return" in fig.axes[0].get_ylabel().lower()

    def test_decimal_scale_ylabel(self) -> None:
        _, rs, _ = _make()
        fig = plot_return_series(rs, scale="decimal")
        assert "decimal" in fig.axes[0].get_ylabel().lower()

    def test_invalid_scale_raises(self) -> None:
        _, rs, _ = _make()
        with pytest.raises(ValueError, match="scale"):
            plot_return_series(rs, scale="basis_points")

    def test_custom_title_is_used(self) -> None:
        _, rs, _ = _make()
        fig = plot_return_series(rs, title="My Returns")
        assert fig.axes[0].get_title() == "My Returns"

    def test_gap_flagged_returns_have_extra_lines_or_legend(self) -> None:
        # Introduce gaps so gap-flag vlines appear.
        frame = make_prices()
        dropped = [frame.index[10].strftime("%Y-%m-%d"), frame.index[11].strftime("%Y-%m-%d")]
        md = make_market_data(frame, dropped_dates=dropped)
        rs = build_return_series(md)
        assert rs.flagged_gaps > 0
        fig = plot_return_series(rs)
        # A legend should be present when there are gap-flagged bars.
        assert fig.axes[0].get_legend() is not None


# ---------------------------------------------------------------------------
# plot_return_histogram
# ---------------------------------------------------------------------------


class TestPlotReturnHistogram:
    def test_returns_a_figure(self) -> None:
        _, rs, stats = _make()
        fig = plot_return_histogram(rs, stats)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_figure_has_one_axes(self) -> None:
        _, rs, stats = _make()
        fig = plot_return_histogram(rs, stats)
        assert len(fig.axes) == 1

    def test_title_contains_ticker(self) -> None:
        _, rs, stats = _make()
        fig = plot_return_histogram(rs, stats)
        assert "TEST" in fig.axes[0].get_title()

    def test_title_contains_skewness(self) -> None:
        _, rs, stats = _make()
        fig = plot_return_histogram(rs, stats)
        assert "skew" in fig.axes[0].get_title().lower()

    def test_histogram_patches_present(self) -> None:
        _, rs, stats = _make()
        fig = plot_return_histogram(rs, stats)
        ax = fig.axes[0]
        assert len(ax.patches) > 0

    def test_normal_overlay_line_present(self) -> None:
        _, rs, stats = _make()
        fig = plot_return_histogram(rs, stats)
        ax = fig.axes[0]
        # Normal density overlay drawn as a line.
        assert len(ax.lines) >= 1

    def test_legend_present(self) -> None:
        _, rs, stats = _make()
        fig = plot_return_histogram(rs, stats)
        assert fig.axes[0].get_legend() is not None

    def test_decimal_scale_xlabel(self) -> None:
        _, rs, stats = _make()
        fig = plot_return_histogram(rs, stats, scale="decimal")
        assert "decimal" in fig.axes[0].get_xlabel().lower()

    def test_percent_scale_xlabel(self) -> None:
        _, rs, stats = _make()
        fig = plot_return_histogram(rs, stats, scale="percent")
        assert "%" in fig.axes[0].get_xlabel()

    def test_invalid_scale_raises(self) -> None:
        _, rs, stats = _make()
        with pytest.raises(ValueError, match="scale"):
            plot_return_histogram(rs, stats, scale="bps")

    def test_integer_bin_count_is_accepted(self) -> None:
        _, rs, stats = _make()
        fig = plot_return_histogram(rs, stats, bins=30)
        assert len(fig.axes[0].patches) > 0

    def test_custom_figsize(self) -> None:
        _, rs, stats = _make()
        fig = plot_return_histogram(rs, stats, figsize=(6, 3))
        w, h = fig.get_size_inches()
        assert w == pytest.approx(6.0)
        assert h == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# plot_data_quality
# ---------------------------------------------------------------------------


class TestPlotDataQuality:
    def test_returns_a_figure(self) -> None:
        md, rs, _ = _make()
        fig = plot_data_quality(md, rs)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_figure_has_one_axes(self) -> None:
        md, rs, _ = _make()
        fig = plot_data_quality(md, rs)
        assert len(fig.axes) == 1

    def test_four_bars_present(self) -> None:
        md, rs, _ = _make()
        fig = plot_data_quality(md, rs)
        ax = fig.axes[0]
        assert len(ax.patches) == 4

    def test_title_contains_ticker(self) -> None:
        md, rs, _ = _make()
        fig = plot_data_quality(md, rs)
        assert "TEST" in fig.axes[0].get_title()

    def test_custom_title(self) -> None:
        md, rs, _ = _make()
        fig = plot_data_quality(md, rs, title="Quality Check")
        assert fig.axes[0].get_title() == "Quality Check"

    def test_bar_heights_match_counts(self) -> None:
        md, rs, _ = _make()
        fig = plot_data_quality(md, rs)
        ax = fig.axes[0]
        heights = [p.get_height() for p in ax.patches]
        expected_heights = [
            md.observations,
            rs.return_observations,
            md.rows_dropped,
            rs.flagged_gaps,
        ]
        for h, e in zip(heights, expected_heights):
            assert h == pytest.approx(e, abs=1e-6)

    def test_ylabel_set(self) -> None:
        md, rs, _ = _make()
        fig = plot_data_quality(md, rs)
        assert fig.axes[0].get_ylabel()
