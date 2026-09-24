"""Descriptive statistics for a return series.

:class:`ReturnStats` collects the standard summary statistics — mean, median,
standard deviation, min, max, quartiles, skewness and excess kurtosis — that
characterise the distribution of log returns.  All values are computed once on
construction from the decimal-scale series and stored; the percent-scale
equivalents are exposed as convenience properties.

The object is frozen, JSON-serialisable and carries no reference to the source
:class:`~src.preprocessing.return_series.ReturnSeries`; callers that want only
the numbers can safely discard the parent object.

No hypothesis tests and no model fitting are performed here.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.exceptions import InvalidProfileInputError
from src.logging_config import PACKAGE_LOGGER_NAME
from src.preprocessing.return_series import MODELING_SCALE, STORED_SCALE, ReturnSeries

logger = logging.getLogger(f"{PACKAGE_LOGGER_NAME}.statistics.descriptive")


@dataclass(frozen=True)
class ReturnStats:
    """Descriptive statistics of a log-return series.

    All scalar attributes are stored as **decimal** log returns (the
    canonical scale).  Use the ``_pct`` properties for the percent-scale
    equivalents expected by volatility models.

    Attributes:
        ticker: Asset symbol.
        count: Number of return observations.
        mean: Arithmetic mean of decimal log returns.
        median: Median of decimal log returns.
        std: Sample standard deviation (ddof=1) of decimal log returns.
        minimum: Minimum decimal log return.
        maximum: Maximum decimal log return.
        q25: 25th percentile (first quartile).
        q75: 75th percentile (third quartile).
        skewness: Fisher's moment coefficient of skewness (biased estimator).
        excess_kurtosis: Excess kurtosis (Fisher definition; 0 for Gaussian).
    """

    ticker: str
    count: int
    mean: float
    median: float
    std: float
    minimum: float
    maximum: float
    q25: float
    q75: float
    skewness: float
    excess_kurtosis: float

    # --- percent-scale convenience properties -----------------------------------

    @property
    def mean_pct(self) -> float:
        """Mean expressed in percent."""
        return self.mean * MODELING_SCALE.factor

    @property
    def median_pct(self) -> float:
        """Median expressed in percent."""
        return self.median * MODELING_SCALE.factor

    @property
    def std_pct(self) -> float:
        """Standard deviation expressed in percent."""
        return self.std * MODELING_SCALE.factor

    @property
    def minimum_pct(self) -> float:
        """Minimum expressed in percent."""
        return self.minimum * MODELING_SCALE.factor

    @property
    def maximum_pct(self) -> float:
        """Maximum expressed in percent."""
        return self.maximum * MODELING_SCALE.factor

    @property
    def q25_pct(self) -> float:
        """25th percentile expressed in percent."""
        return self.q25 * MODELING_SCALE.factor

    @property
    def q75_pct(self) -> float:
        """75th percentile expressed in percent."""
        return self.q75 * MODELING_SCALE.factor

    # --- serialisation ----------------------------------------------------------

    def to_dict(self, *, scale: str = "decimal") -> dict[str, Any]:
        """Return a JSON-serialisable dictionary.

        Args:
            scale: ``"decimal"`` (default) or ``"percent"``.  Selects which
                values are emitted for the numeric fields.

        Returns:
            A plain ``dict`` whose values are Python scalars.
        """
        if scale not in ("decimal", "percent"):
            raise ValueError(f"scale must be 'decimal' or 'percent'; got {scale!r}.")
        factor = MODELING_SCALE.factor if scale == "percent" else STORED_SCALE.factor
        return {
            "ticker": self.ticker,
            "scale": scale,
            "count": self.count,
            "mean": self.mean * factor,
            "median": self.median * factor,
            "std": self.std * factor,
            "min": self.minimum * factor,
            "max": self.maximum * factor,
            "q25": self.q25 * factor,
            "q75": self.q75 * factor,
            "skewness": self.skewness,
            "excess_kurtosis": self.excess_kurtosis,
        }

    def to_json(self, *, scale: str = "decimal", indent: int = 2) -> str:
        """Serialise to a JSON string."""
        return json.dumps(self.to_dict(scale=scale), indent=indent)

    def summary_frame(self, *, scale: str = "percent") -> pd.DataFrame:
        """Return a single-column DataFrame suitable for display or export.

        The index contains statistic names; the column is the ticker symbol.
        Skewness and excess kurtosis are dimensionless and are included as-is.
        """
        d = self.to_dict(scale=scale)
        # Preserve a readable display order.
        keys = ["count", "mean", "median", "std", "min", "q25", "q75", "max",
                "skewness", "excess_kurtosis"]
        labels = ["count", "mean", "median", "std", "min", "25%", "75%", "max",
                  "skewness", "excess kurtosis"]
        return pd.DataFrame(
            {self.ticker: [d[k] for k in keys]},
            index=labels,
        )


def compute_return_stats(returns: ReturnSeries) -> ReturnStats:
    """Compute descriptive statistics from a validated return series.

    Args:
        returns: Output of the preprocessing layer.  The decimal-scale series
            is used; nothing is modified.

    Returns:
        A frozen :class:`ReturnStats` instance.

    Raises:
        InvalidProfileInputError: If the series is empty or contains
            non-finite values (which :class:`~src.preprocessing.return_series.ReturnSeries`
            already prevents, so this guards against future API changes).
    """
    series = returns.decimal
    if len(series) == 0:
        raise InvalidProfileInputError(
            f"Cannot compute statistics for {returns.ticker}: the return series is empty."
        )
    values = series.to_numpy(dtype="float64")
    if not np.isfinite(values).all():
        raise InvalidProfileInputError(
            f"Cannot compute statistics for {returns.ticker}: series contains non-finite values."
        )

    n = len(values)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if n > 1 else 0.0

    # Skewness and excess kurtosis (Fisher definition, biased estimators,
    # consistent with scipy.stats and pandas.Series.kurt default).
    if n >= 3 and std > 0.0:
        z = (values - mean) / std
        skewness = float(np.mean(z ** 3))
        excess_kurtosis = float(np.mean(z ** 4) - 3.0)
    else:
        skewness = 0.0
        excess_kurtosis = 0.0

    stats = ReturnStats(
        ticker=returns.ticker,
        count=n,
        mean=mean,
        median=float(np.median(values)),
        std=std,
        minimum=float(np.min(values)),
        maximum=float(np.max(values)),
        q25=float(np.percentile(values, 25)),
        q75=float(np.percentile(values, 75)),
        skewness=skewness,
        excess_kurtosis=excess_kurtosis,
    )
    logger.info(
        "Computed return stats: ticker=%s n=%d mean=%.6f std=%.6f skew=%.4f kurt=%.4f",
        stats.ticker,
        stats.count,
        stats.mean,
        stats.std,
        stats.skewness,
        stats.excess_kurtosis,
    )
    return stats
