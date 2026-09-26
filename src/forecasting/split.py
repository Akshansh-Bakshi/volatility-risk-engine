"""Chronological train/test splitting for time-series data.

Time-series data must **never** be split by random shuffling — that would
introduce look-ahead bias by leaking future observations into the training
window.  This module enforces strictly chronological splits only.

The split is defined by a ``test_horizon`` (number of test observations).
The last ``test_horizon`` observations form the test set; everything before
them is the training set.  The split point is therefore determined by the
data length, not by a calendar date, though the actual dates are recorded
in the :class:`TimeSeriesSplit` result for full auditability.

Usage::

    from src.forecasting.split import make_split

    split = make_split(market_data.price, test_horizon=60)
    # split.train_series — pd.Series (training prices)
    # split.test_series  — pd.Series (held-out test prices)
    # split.info()       — dict of auditable metadata
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd

from src.data.market_data import MarketData
from src.exceptions import ForecastingDataError
from src.logging_config import PACKAGE_LOGGER_NAME

logger = logging.getLogger(f"{PACKAGE_LOGGER_NAME}.forecasting.split")

_MIN_TRAIN = 30   # minimum training observations required
_MIN_TEST = 1     # minimum test observations required


@dataclass(frozen=True)
class TimeSeriesSplit:
    """An auditable chronological train/test split of a price series.

    Attributes:
        ticker: Asset symbol.
        train_series: Training price series (all observations *before* the
            split point, in chronological order).
        test_series: Test price series (the last ``test_horizon`` observations,
            in chronological order).  Never overlaps with ``train_series``.
        frequency: Observation frequency (e.g. ``"1d"``).
        series_description: Human-readable description of what the series
            represents (e.g. ``"adjusted close price"``) so downstream code
            is never ambiguous about what was split.

    Design note
    -----------
    The split is strictly chronological.  ``train_series`` ends immediately
    before ``test_series`` begins.  No shuffling, no random sampling.  This
    guarantees leakage-free evaluation: the model sees *only* the training
    window when it is fitted and *only* the test window when it is evaluated.
    """

    ticker: str
    train_series: pd.Series
    test_series: pd.Series
    frequency: str
    series_description: str

    # ------------------------------------------------------------------ #
    # Convenience properties                                               #
    # ------------------------------------------------------------------ #

    @property
    def n_train(self) -> int:
        """Number of training observations."""
        return len(self.train_series)

    @property
    def n_test(self) -> int:
        """Number of test observations."""
        return len(self.test_series)

    @property
    def train_start(self) -> date:
        return pd.Timestamp(self.train_series.index[0]).date()

    @property
    def train_end(self) -> date:
        return pd.Timestamp(self.train_series.index[-1]).date()

    @property
    def test_start(self) -> date:
        return pd.Timestamp(self.test_series.index[0]).date()

    @property
    def test_end(self) -> date:
        return pd.Timestamp(self.test_series.index[-1]).date()

    def info(self) -> dict[str, Any]:
        """Return an auditable JSON-serialisable summary of the split."""
        return {
            "ticker": self.ticker,
            "series_description": self.series_description,
            "frequency": self.frequency,
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "n_train": self.n_train,
            "test_start": self.test_start.isoformat(),
            "test_end": self.test_end.isoformat(),
            "n_test": self.n_test,
            "total_observations": self.n_train + self.n_test,
            "split_method": "chronological — no random shuffling",
        }

    def info_json(self, *, indent: int = 2) -> str:
        """Serialise :meth:`info` to a JSON string."""
        return json.dumps(self.info(), indent=indent)

    def no_leakage(self) -> bool:
        """``True`` if train and test periods are strictly non-overlapping.

        This should always be ``True`` for splits produced by :func:`make_split`.
        Callers and tests can use this as an invariant assertion.
        """
        return self.train_end < self.test_start


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_split(
    market_data: MarketData,
    *,
    test_horizon: int = 60,
    series_description: str = "adjusted close price",
) -> TimeSeriesSplit:
    """Split a price series into a chronological train/test pair.

    Args:
        market_data: A validated :class:`~src.data.market_data.MarketData`
            object.  The ``close`` price series is used.
        test_horizon: Number of observations to hold out as the test set.
            Must be ≥ 1.  The last ``test_horizon`` observations go into the
            test set; the rest form the training set.
        series_description: Human-readable label stored on the split object.

    Returns:
        A :class:`TimeSeriesSplit` with ``train_series`` and ``test_series``
        that are strictly non-overlapping and chronologically ordered.

    Raises:
        ForecastingDataError: If ``test_horizon`` is invalid, or if there are
            not enough observations to satisfy the minimum train/test sizes.
    """
    if not isinstance(test_horizon, int) or isinstance(test_horizon, bool):
        raise ForecastingDataError(
            f"test_horizon must be a positive integer, got {test_horizon!r}."
        )
    if test_horizon < _MIN_TEST:
        raise ForecastingDataError(
            f"test_horizon must be ≥ {_MIN_TEST}, got {test_horizon}."
        )

    price = market_data.price
    n = len(price)
    n_train = n - test_horizon

    if n_train < _MIN_TRAIN:
        raise ForecastingDataError(
            f"Not enough observations for {market_data.ticker}: "
            f"{n} total, {test_horizon} test → {n_train} train "
            f"(minimum required: {_MIN_TRAIN})."
        )

    train = price.iloc[:n_train].copy()
    test = price.iloc[n_train:].copy()

    split = TimeSeriesSplit(
        ticker=market_data.ticker,
        train_series=train,
        test_series=test,
        frequency=market_data.request.interval,
        series_description=series_description,
    )
    logger.info(
        "Split: ticker=%s train=%d [%s..%s] test=%d [%s..%s]",
        split.ticker,
        split.n_train,
        split.train_start,
        split.train_end,
        split.n_test,
        split.test_start,
        split.test_end,
    )
    return split
